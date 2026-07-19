#include "impl/cuda_device.h"

#include "impl/cpu_storage.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require_defined(const Tensor& tensor, const char* name) {
    if (!tensor.defined())
        throw std::invalid_argument(std::string(name) + " tensor is undefined");
}
void require_float32(const Tensor& tensor, const char* name) {
    if (tensor.dtype() != DType::Float32)
        throw std::invalid_argument(std::string(name) + " tensor must be Float32");
}
void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape())
        throw std::invalid_argument("tensor shapes must match");
}

Shape broadcast_shape(const Shape& left, const Shape& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Shape output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto a = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto b = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (a != b && a != 1 && b != 1)
            throw std::invalid_argument("tensor shapes are not broadcastable");
        output[rank - 1 - offset] = std::max(a, b);
    }
    return output;
}

Strides contiguous_strides(const Shape& shape) {
    Strides result(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;)
        result[index - 1] = result[index] * shape[index];
    return result;
}
void require_2d_matmul_shapes(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::invalid_argument("matmul expects 2D tensors");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul inner dimensions must match");
}

// CUDA execution hierarchy:
//
//   kernel launch = grid
//   grid
//   +-- block 0 -- warp 0: threads 0..31, warp 1: threads 32..63, ...
//   +-- block 1 -- warp 0: threads 0..31, warp 1: threads 32..63, ...
//   +-- ...
//
// A thread is one logical execution of this function. A block is a cooperating
// group of threads with shared memory and block-level synchronization. Hardware
// executes threads from a block in warps of 32; a warp never crosses a block.
// The grid contains every block in this kernel launch. Blocks are scheduled
// onto the GPU's streaming multiprocessors (SMs), which execute their warps.
// A thread's unique 1D grid index is blockIdx.x * blockDim.x + threadIdx.x.
//
// Grid-stride element assignment, illustrated with two blocks of four threads
// and 21 elements (grid width/stride = 8):
//
//   Grid
//   +---------------- Block 0 ----------------+  +---------------- Block 1
//   ----------------+ |  local thread:   0     1     2     3   |  |  local
//   thread:   0     1     2     3   | |  global thread: T0    T1    T2    T3 |
//   |  global thread: T4    T5    T6    T7   |
//   +-----------------------------------------+
//   +-----------------------------------------+
//
//                  Pass 1     Pass 2     Pass 3
//                +----------+----------+----------+
//   Thread T0    |    0     |    8     |   16     |
//   Thread T1    |    1     |    9     |   17     |
//   Thread T2    |    2     |   10     |   18     |
//   Thread T3    |    3     |   11     |   19     |
//   Thread T4    |    4     |   12     |   20     |
//   Thread T5    |    5     |   13     |  stop    |
//   Thread T6    |    6     |   14     |  stop    |
//   Thread T7    |    7     |   15     |  stop    |
//                +----------+----------+----------+
//
// Each thread starts at its global index, processes that element, then advances
// by the full grid width. Adjacent threads access adjacent elements on every
// pass, preserving coalesced memory access. The i < count condition handles the
// uneven final pass without a separate tail kernel.
__global__ void add_f32(const float* a, const float* b, float* out, std::int64_t count) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto i = first; i < count; i += stride)
        out[i] = a[i] + b[i];
}
__global__ void sub_f32(const float* a, const float* b, float* out, std::int64_t count) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto i = first; i < count; i += stride)
        out[i] = a[i] - b[i];
}

__device__ float apply_elementwise(
    float a,
    float b,
    CudaElementwiseOperation operation) {
    switch (operation) {
        case CudaElementwiseOperation::Add: return a + b;
        case CudaElementwiseOperation::Subtract: return a - b;
        case CudaElementwiseOperation::Multiply: return a * b;
        case CudaElementwiseOperation::Divide: return a / b;
        case CudaElementwiseOperation::Maximum: return a < b ? b : a;
    }
    return 0.0f;
}

__global__ void broadcast_elementwise_f32(
    const float* a,
    const float* b,
    float* out,
    std::int64_t count,
    const std::int64_t* metadata,
    std::int64_t rank,
    CudaElementwiseOperation operation) {
    const auto* output_strides = metadata;
    const auto* output_shape = metadata + rank;
    const auto* a_shape = metadata + 2 * rank;
    const auto* a_strides = metadata + 3 * rank;
    const auto* b_shape = metadata + 4 * rank;
    const auto* b_strides = metadata + 5 * rank;
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto index = first; index < count; index += stride) {
        std::int64_t a_index = 0;
        std::int64_t b_index = 0;
        for (std::int64_t axis = 0; axis < rank; ++axis) {
            const auto coordinate =
                (index / output_strides[axis]) % output_shape[axis];
            if (a_shape[axis] != 1) a_index += coordinate * a_strides[axis];
            if (b_shape[axis] != 1) b_index += coordinate * b_strides[axis];
        }
        out[index] = apply_elementwise(a[a_index], b[b_index], operation);
    }
}

__global__ void scalar_elementwise_f32(
    const float* input,
    float scalar,
    float* out,
    std::int64_t count,
    CudaElementwiseOperation operation,
    bool scalar_is_left) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto index = first; index < count; index += stride) {
        const float value = input[index];
        out[index] = scalar_is_left ? apply_elementwise(scalar, value, operation)
                                    : apply_elementwise(value, scalar, operation);
    }
}

__global__ void reduce_f32(
    const float* input,
    float* output,
    std::int64_t output_count,
    std::int64_t reduced_count,
    const std::int64_t* metadata,
    std::int64_t rank,
    CudaReductionOperation operation) {
    const auto* shape = metadata;
    const auto* strides = metadata + rank;
    const auto* reduced = metadata + 2 * rank;
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto grid_stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto output_index = first; output_index < output_count;
         output_index += grid_stride) {
        std::int64_t remainder = output_index;
        std::int64_t base = 0;
        for (std::int64_t axis = rank; axis-- > 0;) {
            if (reduced[axis] == 0) {
                const auto coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                base += coordinate * strides[axis];
            }
        }

        float value = operation == CudaReductionOperation::Maximum
            ? -__int_as_float(0x7f800000)
            : 0.0f;
        float mean = 0.0f;
        if (operation == CudaReductionOperation::Variance && reduced_count != 0) {
            for (std::int64_t reduction_index = 0; reduction_index < reduced_count;
                 ++reduction_index) {
                std::int64_t reduction_remainder = reduction_index;
                std::int64_t input_index = base;
                for (std::int64_t axis = rank; axis-- > 0;) {
                    if (reduced[axis] != 0) {
                        const auto coordinate = reduction_remainder % shape[axis];
                        reduction_remainder /= shape[axis];
                        input_index += coordinate * strides[axis];
                    }
                }
                mean += input[input_index];
            }
            mean /= static_cast<float>(reduced_count);
        }

        for (std::int64_t reduction_index = 0; reduction_index < reduced_count;
             ++reduction_index) {
            std::int64_t reduction_remainder = reduction_index;
            std::int64_t input_index = base;
            for (std::int64_t axis = rank; axis-- > 0;) {
                if (reduced[axis] != 0) {
                    const auto coordinate = reduction_remainder % shape[axis];
                    reduction_remainder /= shape[axis];
                    input_index += coordinate * strides[axis];
                }
            }
            const float input_value = input[input_index];
            if (operation == CudaReductionOperation::Maximum)
                value = value < input_value ? input_value : value;
            else if (operation == CudaReductionOperation::Variance) {
                const float difference = input_value - mean;
                value += difference * difference;
            } else
                value += input_value;
        }
        if (operation == CudaReductionOperation::Mean ||
            operation == CudaReductionOperation::Variance)
            value /= static_cast<float>(reduced_count);
        output[output_index] = value;
    }
}

__global__ void unary_f32(
    const float* input,
    float* output,
    std::int64_t count,
    CudaUnaryOperation operation,
    float argument) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto index = first; index < count; index += stride) {
        const float value = input[index];
        switch (operation) {
            case CudaUnaryOperation::Exp: output[index] = expf(value); break;
            case CudaUnaryOperation::Sqrt: output[index] = sqrtf(value); break;
            case CudaUnaryOperation::Power: output[index] = powf(value, argument); break;
        }
    }
}
constexpr int matmul_tile_size = 16;

__global__ void matmul_f32(const float* a, const float* b, float* out, std::int64_t m,
                           std::int64_t k, std::int64_t n) {
    // These tiles are shared by all 256 threads in the block. Staging input
    // values here lets 16 output threads reuse each value loaded from global
    // memory instead of fetching it independently.
    __shared__ float a_tile[matmul_tile_size][matmul_tile_size];
    __shared__ float b_tile[matmul_tile_size][matmul_tile_size];

    // The block selects a 16x16 output tile; the thread selects one element
    // within that tile and owns its accumulator for the entire K dimension.
    const auto row = static_cast<std::int64_t>(blockIdx.y) * matmul_tile_size + threadIdx.y;
    const auto col = static_cast<std::int64_t>(blockIdx.x) * matmul_tile_size + threadIdx.x;
    float total = 0.0f;

    // Traverse K in 16-element slices. In each pass, every thread loads one A
    // value and one B value, cooperatively filling both shared-memory tiles.
    for (std::int64_t start = 0; start < k; start += matmul_tile_size) {
        const auto a_col = start + threadIdx.x;
        const auto b_row = start + threadIdx.y;
        // Zero padding keeps every thread participating in synchronization and
        // makes partial tiles behave as though the matrices were padded with 0.
        a_tile[threadIdx.y][threadIdx.x] = row < m && a_col < k ? a[row * k + a_col] : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = b_row < k && col < n ? b[b_row * n + col] : 0.0f;
        // Do not read a tile until every thread has finished loading it.
        __syncthreads();

        // Multiply this thread's row of A's tile by its column of B's tile.
#pragma unroll
        for (int inner = 0; inner < matmul_tile_size; ++inner) {
            total += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        // Do not overwrite the shared tiles for the next pass until every
        // thread has finished consuming the current values.
        __syncthreads();
    }

    // Threads in partial right/bottom blocks still reach both barriers above,
    // but only threads mapped to valid output elements write a result.
    if (row < m && col < n)
        out[row * n + col] = total;
}

struct BatchLayout {
    Shape shape;
    std::vector<std::int64_t> ao, bo;
};
BatchLayout batch_layout(const Tensor& a, const Tensor& b) {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    if (a.ndim() < 2 || b.ndim() < 2 || (a.ndim() == 2 && b.ndim() == 2))
        throw std::invalid_argument("batched_matmul expects at least one batched input");
    if (a.shape().back() != b.shape()[b.ndim() - 2])
        throw std::invalid_argument("batched_matmul inner dimensions must match");
    Shape ab(a.shape().begin(), a.shape().end() - 2), bb(b.shape().begin(), b.shape().end() - 2);
    const auto rank = std::max(ab.size(), bb.size());
    Shape ob(rank, 1);
    for (std::size_t off = 0; off < rank; ++off) {
        auto x = off < ab.size() ? ab[ab.size() - 1 - off] : 1;
        auto y = off < bb.size() ? bb[bb.size() - 1 - off] : 1;
        if (x != y && x != 1 && y != 1)
            throw std::invalid_argument("batched_matmul batch dimensions are not broadcastable");
        ob[rank - 1 - off] = std::max(x, y);
    }
    auto strides = [](const Shape& s) {
        Strides r(s.size(), 1);
        for (std::size_t i = s.size(); i-- > 1;)
            r[i - 1] = r[i] * s[i];
        return r;
    };
    auto os = strides(ob), as = strides(ab), bs = strides(bb);
    auto map = [&](std::int64_t batch, const Shape& s, const Strides& ss) {
        std::int64_t v = 0;
        auto pad = ob.size() - s.size();
        for (std::size_t i = pad; i < ob.size(); ++i) {
            auto c = (batch / os[i]) % ob[i];
            if (s[i - pad] != 1)
                v += c * ss[i - pad];
        }
        return v;
    };
    auto count = std::accumulate(ob.begin(), ob.end(), std::int64_t{1}, std::multiplies<>());
    BatchLayout l;
    l.shape = ob;
    l.shape.push_back(a.shape()[a.ndim() - 2]);
    l.shape.push_back(b.shape().back());
    l.ao.resize(count);
    l.bo.resize(count);
    auto am = a.shape()[a.ndim() - 2] * a.shape().back(),
         bm = b.shape()[b.ndim() - 2] * b.shape().back();
    for (std::int64_t i = 0; i < count; ++i) {
        l.ao[i] = map(i, ab, as) * am;
        l.bo[i] = map(i, bb, bs) * bm;
    }
    return l;
}

__global__ void batched_matmul_f32(const float* a, const float* b, float* out,
                                   const std::int64_t* ao, const std::int64_t* bo,
                                   std::int64_t batches, std::int64_t m, std::int64_t k,
                                   std::int64_t n) {
    __shared__ float at[matmul_tile_size][matmul_tile_size];
    __shared__ float bt[matmul_tile_size][matmul_tile_size];
    auto batch = static_cast<std::int64_t>(blockIdx.z);
    if (batch >= batches)
        return;
    auto row = static_cast<std::int64_t>(blockIdx.y) * matmul_tile_size + threadIdx.y,
         col = static_cast<std::int64_t>(blockIdx.x) * matmul_tile_size + threadIdx.x;
    float total = 0;
    const float* am = a + ao[batch];
    const float* bm = b + bo[batch];
    for (std::int64_t start = 0; start < k; start += matmul_tile_size) {
        auto ac = start + threadIdx.x, br = start + threadIdx.y;
        at[threadIdx.y][threadIdx.x] = row < m && ac < k ? am[row * k + ac] : 0;
        bt[threadIdx.y][threadIdx.x] = br < k && col < n ? bm[br * n + col] : 0;
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < matmul_tile_size; ++inner)
            total += at[threadIdx.y][inner] * bt[inner][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n)
        out[batch * m * n + row * n + col] = total;
}

float* data(CudaMemTensorStorageImpl& storage) {
    return static_cast<float*>(storage.handle().ptr);
}

} // namespace

CudaDeviceImpl::CudaDeviceImpl(int device_index)
    : device_index_(device_index), max_elementwise_blocks_(0) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "failed to query CUDA devices");
    if (device_index < 0 || device_index >= count)
        throw std::runtime_error("CUDA device index is not available");
    check_cuda(cudaSetDevice(device_index_), "failed to select CUDA device");
    int multiprocessor_count = 0;
    check_cuda(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount,
                                      device_index_),
               "failed to query CUDA multiprocessor count");
    // A bounded grid avoids launching one thread per element for very large
    // tensors. Thirty-two blocks per SM provides ample work for scheduling;
    // each thread advances by the full grid width until all elements are done.
    max_elementwise_blocks_ = multiprocessor_count * 32;
}

DeviceType CudaDeviceImpl::type() const {
    return DeviceType::CUDA;
}
int CudaDeviceImpl::device_index() const {
    return device_index_;
}

Tensor CudaDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<CudaMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype), dtype, device_index_);
    return Tensor(std::move(shape), dtype, Device::cuda(device_index_), std::move(storage));
}

Tensor CudaDeviceImpl::add(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    auto out = empty(a.shape(), a.dtype());
    add_out(a, b, out);
    return out;
}

void CudaDeviceImpl::add_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    require_defined(out, "output");
    require_float32(out, "output");
    require_same_shape(a, out);
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto count = a.numel();
    if (count != 0) {
        const auto required_blocks = (count + 255) / 256;
        const auto blocks =
            static_cast<unsigned>(std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        // <<<blocks, 256>>> launches a grid of `blocks` blocks, each containing 256
        // threads.
        add_f32<<<blocks, 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
                                 data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA add kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA add kernel failed");
}

Tensor CudaDeviceImpl::sub(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    auto out = empty(a.shape(), a.dtype());
    sub_out(a, b, out);
    return out;
}

Tensor CudaDeviceImpl::broadcast_elementwise(
    const Tensor& a,
    const Tensor& b,
    CudaElementwiseOperation operation) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    const Shape output_shape = broadcast_shape(a.shape(), b.shape());
    Tensor out = empty(output_shape, DType::Float32);
    if (out.numel() == 0) return out;

    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const std::size_t rank = output_shape.size();
    std::vector<std::int64_t> metadata(rank * 6, 0);
    const Strides output_strides = contiguous_strides(output_shape);
    const Strides a_strides = contiguous_strides(a.shape());
    const Strides b_strides = contiguous_strides(b.shape());
    std::copy(output_strides.begin(), output_strides.end(), metadata.begin());
    std::copy(output_shape.begin(), output_shape.end(), metadata.begin() + rank);
    const std::size_t a_padding = rank - a.ndim();
    const std::size_t b_padding = rank - b.ndim();
    std::fill(metadata.begin() + 2 * rank, metadata.begin() + 3 * rank, 1);
    std::fill(metadata.begin() + 4 * rank, metadata.begin() + 5 * rank, 1);
    std::copy(a.shape().begin(), a.shape().end(), metadata.begin() + 2 * rank + a_padding);
    std::copy(a_strides.begin(), a_strides.end(), metadata.begin() + 3 * rank + a_padding);
    std::copy(b.shape().begin(), b.shape().end(), metadata.begin() + 4 * rank + b_padding);
    std::copy(b_strides.begin(), b_strides.end(), metadata.begin() + 5 * rank + b_padding);

    std::int64_t* device_metadata = nullptr;
    const std::size_t metadata_bytes = metadata.size() * sizeof(std::int64_t);
    if (metadata_bytes != 0) {
        check_cuda(cudaMalloc(&device_metadata, metadata_bytes),
                   "failed to allocate CUDA broadcast metadata");
    }
    try {
        if (metadata_bytes != 0) {
            check_cuda(cudaMemcpy(device_metadata, metadata.data(), metadata_bytes,
                                  cudaMemcpyHostToDevice),
                       "failed to copy CUDA broadcast metadata");
        }
        const auto required_blocks = (out.numel() + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        broadcast_elementwise_f32<<<blocks, 256>>>(
            data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
            data(require_cuda_storage(*out.storage())), out.numel(), device_metadata,
            static_cast<std::int64_t>(rank), operation);
        check_cuda(cudaGetLastError(), "failed to launch CUDA broadcast elementwise kernel");
        check_cuda(cudaDeviceSynchronize(), "CUDA broadcast elementwise kernel failed");
    } catch (...) {
        if (device_metadata) cudaFree(device_metadata);
        throw;
    }
    if (device_metadata) cudaFree(device_metadata);
    return out;
}

Tensor CudaDeviceImpl::scalar_elementwise(
    const Tensor& tensor,
    float scalar,
    CudaElementwiseOperation operation,
    bool scalar_is_left) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    Tensor out = empty(tensor.shape(), DType::Float32);
    if (out.numel() == 0) return out;
    auto input = ensure_storage(tensor.storage(), ConversionPolicy::CopyToDevice);
    const auto required_blocks = (out.numel() + 255) / 256;
    const auto blocks = static_cast<unsigned>(
        std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
    scalar_elementwise_f32<<<blocks, 256>>>(
        data(require_cuda_storage(*input)), scalar,
        data(require_cuda_storage(*out.storage())), out.numel(), operation, scalar_is_left);
    check_cuda(cudaGetLastError(), "failed to launch CUDA scalar elementwise kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA scalar elementwise kernel failed");
    return out;
}

Tensor CudaDeviceImpl::reduce(
    const Tensor& tensor,
    const std::vector<std::int64_t>& dimensions,
    bool keepdim,
    CudaReductionOperation operation) const {
    require_defined(tensor, "reduction input");
    require_float32(tensor, "reduction input");
    std::vector<bool> reduced(tensor.ndim(), false);
    for (const std::int64_t dimension : dimensions) {
        if (dimension < 0 || dimension >= static_cast<std::int64_t>(tensor.ndim()))
            throw std::out_of_range("reduction dimension is out of range");
        if (reduced[static_cast<std::size_t>(dimension)])
            throw std::invalid_argument("reduction dimensions must be unique");
        reduced[static_cast<std::size_t>(dimension)] = true;
    }

    Shape output_shape;
    std::int64_t reduced_count = 1;
    for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
        if (reduced[axis]) {
            reduced_count *= tensor.shape()[axis];
            if (keepdim) output_shape.push_back(1);
        } else {
            output_shape.push_back(tensor.shape()[axis]);
        }
    }
    Tensor output = empty(output_shape, DType::Float32);
    if (output.numel() == 0) return output;

    const Strides input_strides = contiguous_strides(tensor.shape());
    const std::size_t rank = tensor.ndim();
    // Keep a valid metadata pointer for rank-zero scalar reductions even though
    // the kernel does not dereference any axis entries in that case.
    std::vector<std::int64_t> metadata(std::max<std::size_t>(rank * 3, 1), 0);
    std::copy(tensor.shape().begin(), tensor.shape().end(), metadata.begin());
    std::copy(input_strides.begin(), input_strides.end(), metadata.begin() + rank);
    for (std::size_t axis = 0; axis < rank; ++axis)
        metadata[2 * rank + axis] = reduced[axis] ? 1 : 0;

    std::int64_t* device_metadata = nullptr;
    const std::size_t metadata_bytes = metadata.size() * sizeof(std::int64_t);
    if (metadata_bytes != 0) {
        check_cuda(cudaMalloc(&device_metadata, metadata_bytes),
                   "failed to allocate CUDA reduction metadata");
    }
    auto input = ensure_storage(tensor.storage(), ConversionPolicy::CopyToDevice);
    try {
        if (metadata_bytes != 0) {
            check_cuda(cudaMemcpy(device_metadata, metadata.data(), metadata_bytes,
                                  cudaMemcpyHostToDevice),
                       "failed to copy CUDA reduction metadata");
        }
        const auto required_blocks = (output.numel() + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        reduce_f32<<<blocks, 256>>>(
            data(require_cuda_storage(*input)),
            data(require_cuda_storage(*output.storage())), output.numel(), reduced_count,
            device_metadata, static_cast<std::int64_t>(rank), operation);
        check_cuda(cudaGetLastError(), "failed to launch CUDA reduction kernel");
        check_cuda(cudaDeviceSynchronize(), "CUDA reduction kernel failed");
    } catch (...) {
        if (device_metadata) cudaFree(device_metadata);
        throw;
    }
    if (device_metadata) cudaFree(device_metadata);
    return output;
}

Tensor CudaDeviceImpl::unary(
    const Tensor& tensor,
    CudaUnaryOperation operation,
    float argument) const {
    require_defined(tensor, "unary input");
    require_float32(tensor, "unary input");
    Tensor output = empty(tensor.shape(), DType::Float32);
    if (output.numel() == 0) return output;
    auto input = ensure_storage(tensor.storage(), ConversionPolicy::CopyToDevice);
    const auto required_blocks = (output.numel() + 255) / 256;
    const auto blocks = static_cast<unsigned>(
        std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
    unary_f32<<<blocks, 256>>>(
        data(require_cuda_storage(*input)),
        data(require_cuda_storage(*output.storage())), output.numel(), operation, argument);
    check_cuda(cudaGetLastError(), "failed to launch CUDA unary kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA unary kernel failed");
    return output;
}

void CudaDeviceImpl::sub_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    require_defined(out, "output");
    require_float32(out, "output");
    require_same_shape(a, out);
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto count = a.numel();
    if (count != 0) {
        const auto required_blocks = (count + 255) / 256;
        const auto blocks =
            static_cast<unsigned>(std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        // <<<blocks, 256>>> launches a grid of `blocks` blocks, each containing 256
        // threads.
        sub_f32<<<blocks, 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
                                 data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA sub kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA sub kernel failed");
}

Tensor CudaDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);
    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    auto out = empty({m, n}, a.dtype());
    matmul_out(a, b, out);
    return out;
}

Tensor CudaDeviceImpl::batched_matmul(const Tensor& a, const Tensor& b) const {
    auto l = batch_layout(a, b);
    auto out = empty(l.shape, DType::Float32);
    if (out.numel() == 0)
        return out;
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice),
         bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    std::int64_t *dao = nullptr, *dbo = nullptr;
    auto bytes = l.ao.size() * sizeof(std::int64_t);
    check_cuda(cudaMalloc(&dao, bytes), "failed to allocate CUDA batch offsets");
    try {
        check_cuda(cudaMalloc(&dbo, bytes), "failed to allocate CUDA batch offsets");
        check_cuda(cudaMemcpy(dao, l.ao.data(), bytes, cudaMemcpyHostToDevice),
                   "failed to copy CUDA batch offsets");
        check_cuda(cudaMemcpy(dbo, l.bo.data(), bytes, cudaMemcpyHostToDevice),
                   "failed to copy CUDA batch offsets");
        auto m = a.shape()[a.ndim() - 2], n = b.shape().back();
        dim3 threads(matmul_tile_size, matmul_tile_size);
        dim3 blocks(static_cast<unsigned>((n + matmul_tile_size - 1) / matmul_tile_size),
                    static_cast<unsigned>((m + matmul_tile_size - 1) / matmul_tile_size),
                    static_cast<unsigned>(l.ao.size()));
        batched_matmul_f32<<<blocks, threads>>>(data(require_cuda_storage(*ap)),
                                                data(require_cuda_storage(*bp)),
                                                data(require_cuda_storage(*out.storage())), dao,
                                                dbo, l.ao.size(), m, a.shape().back(), n);
        check_cuda(cudaGetLastError(), "failed to launch CUDA batched_matmul kernel");
        check_cuda(cudaDeviceSynchronize(), "CUDA batched_matmul failed");
    } catch (...) {
        if (dbo)
            cudaFree(dbo);
        cudaFree(dao);
        throw;
    }
    cudaFree(dbo);
    cudaFree(dao);
    return out;
}

void CudaDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);
    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    require_defined(out, "output");
    require_float32(out, "output");
    if (out.shape() != Shape({m, n}))
        throw std::invalid_argument("matmul output shape must be [m, n]");
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    if (m != 0 && n != 0) {
        // One 16x16 thread block computes one 16x16 tile of the [m, n]
        // output. Each thread accumulates one output element while all threads
        // in the block cooperate through shared memory to reuse input tiles.
        const dim3 threads(matmul_tile_size, matmul_tile_size);

        // Ceiling division launches enough blocks to cover partial tiles at
        // the right and bottom edges. The kernel zero-pads out-of-range input
        // loads and suppresses out-of-range output stores.
        const dim3 blocks(static_cast<unsigned>((n + matmul_tile_size - 1) / matmul_tile_size),
                          static_cast<unsigned>((m + matmul_tile_size - 1) / matmul_tile_size));
        matmul_f32<<<blocks, threads>>>(data(require_cuda_storage(*ap)),
                                        data(require_cuda_storage(*bp)),
                                        data(require_cuda_storage(*out.storage())), m, k, n);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA matmul kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA matmul kernel failed");
}

TensorStoragePtr CudaDeviceImpl::ensure_storage(const TensorStoragePtr& storage,
                                                ConversionPolicy policy) const {
    if (!storage)
        throw std::invalid_argument("tensor has no storage");
    if (storage->type() == TensorStorageType::CudaMemory) {
        const auto& cuda = static_cast<const CudaMemTensorStorageImpl&>(*storage);
        if (cuda.device_index() == device_index_)
            return storage;
    }
    if (storage->type() == TensorStorageType::CpuMemory &&
        policy == ConversionPolicy::CopyToDevice) {
        const auto& cpu = static_cast<const CpuMemTensorStorageImpl&>(*storage);
        auto cuda =
            std::make_shared<CudaMemTensorStorageImpl>(cpu.nbytes(), cpu.dtype(), device_index_);
        cuda->copy_from_host(cpu.data(), cpu.nbytes());
        return cuda;
    }
    throw std::invalid_argument(
        "CudaDeviceImpl requires CudaMemory storage on the selected device");
}

const CudaMemTensorStorageImpl&
CudaDeviceImpl::require_cuda_storage(const ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CudaMemory)
        throw std::invalid_argument("CudaDeviceImpl requires CudaMemory storage");
    const auto& cuda = static_cast<const CudaMemTensorStorageImpl&>(storage);
    if (cuda.device_index() != device_index_)
        throw std::invalid_argument("CUDA storage is on a different device");
    return cuda;
}
CudaMemTensorStorageImpl& CudaDeviceImpl::require_cuda_storage(ITensorStorage& storage) const {
    return const_cast<CudaMemTensorStorageImpl&>(
        require_cuda_storage(static_cast<const ITensorStorage&>(storage)));
}

} // namespace citrius::impl
