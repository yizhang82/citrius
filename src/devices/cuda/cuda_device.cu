#include "impl/cuda_device.h"
#include "impl/cuda_allocation.h"
#include "impl/cuda_context.h"

#include "impl/cpu_storage.h"
#include "tensor_utils.h"

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
    (void)name;
    ENSURE_TENSOR_DEFINED(tensor);
}
void require_float32(const Tensor& tensor, const char* name) {
    (void)name;
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
}
void require_same_shape(const Tensor& a, const Tensor& b) {
    ENSURE_TENSOR_SHAPE(b, a.shape());
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
    ENSURE_TENSOR_DIM(a, 2);
    ENSURE_TENSOR_DIM(b, 2);
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

__global__ void argmax_f32(const float* input, std::int64_t* output,
                           std::int64_t output_count, std::int64_t reduction_size,
                           std::int64_t inner_size) {
    __shared__ float values[256];
    __shared__ std::int64_t indices[256];
    for (std::int64_t output_index = blockIdx.x; output_index < output_count;
         output_index += gridDim.x) {
        const auto outer = output_index / inner_size;
        const auto inner = output_index % inner_size;
        const auto base = outer * reduction_size * inner_size + inner;
        float best_value = -__int_as_float(0x7f800000);
        std::int64_t best_index = 0x7fffffffffffffffLL;
        for (std::int64_t index = threadIdx.x; index < reduction_size; index += blockDim.x) {
            const float value = input[base + index * inner_size];
            if (value > best_value || (value == best_value && index < best_index)) {
                best_value = value;
                best_index = index;
            }
        }
        values[threadIdx.x] = best_value;
        indices[threadIdx.x] = best_index;
        __syncthreads();
        for (unsigned offset = blockDim.x / 2; offset != 0; offset /= 2) {
            if (threadIdx.x < offset) {
                const float other_value = values[threadIdx.x + offset];
                const auto other_index = indices[threadIdx.x + offset];
                if (other_value > values[threadIdx.x] ||
                    (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                    values[threadIdx.x] = other_value;
                    indices[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) output[output_index] = indices[0];
        __syncthreads();
    }
}

__device__ float combine_reduction(
    float left,
    float right,
    CudaReductionOperation operation) {
    return operation == CudaReductionOperation::Maximum
        ? (left < right ? right : left)
        : left + right;
}

__global__ void reduce_last_dimension_f32(
    const float* input,
    float* output,
    std::int64_t row_count,
    std::int64_t row_size,
    CudaReductionOperation operation) {
    __shared__ float warp_values[32];
    const auto lane = threadIdx.x & 31;
    const auto warp = threadIdx.x >> 5;
    const auto warp_count = (blockDim.x + 31) / 32;
    const float identity = operation == CudaReductionOperation::Maximum
        ? -__int_as_float(0x7f800000)
        : 0.0f;
    for (std::int64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
        float value = identity;
        for (std::int64_t column = threadIdx.x; column < row_size; column += blockDim.x)
            value = combine_reduction(value, input[row * row_size + column], operation);
        for (int offset = 16; offset > 0; offset /= 2)
            value = combine_reduction(value, __shfl_down_sync(0xffffffff, value, offset), operation);
        if (lane == 0) warp_values[warp] = value;
        __syncthreads();
        if (warp == 0) {
            value = lane < warp_count ? warp_values[lane] : identity;
            for (int offset = 16; offset > 0; offset /= 2)
                value = combine_reduction(
                    value, __shfl_down_sync(0xffffffff, value, offset), operation);
            if (lane == 0) {
                if (operation == CudaReductionOperation::Mean)
                    value /= static_cast<float>(row_size);
                output[row] = value;
            }
        }
        __syncthreads();
    }
}

__global__ void gather_rows_f32(
    const float* table,
    const std::int64_t* indices,
    float* output,
    std::int64_t index_count,
    std::int64_t row_count,
    std::int64_t row_size,
    int* invalid_index) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    const auto output_count = index_count * row_size;
    for (auto output_index = first; output_index < output_count; output_index += stride) {
        const auto index_position = output_index / row_size;
        const auto row = indices[index_position];
        if (row < 0 || row >= row_count) {
            atomicExch(invalid_index, 1);
            continue;
        }
        output[output_index] = table[row * row_size + output_index % row_size];
    }
}

__global__ void copy_strided(
    const unsigned char* input,
    unsigned char* output,
    std::int64_t count,
    std::int64_t rank,
    const std::int64_t* shape,
    const std::int64_t* strides,
    std::int64_t element_size) {
    const auto thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto thread_count = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t output_index = thread; output_index < count; output_index += thread_count) {
        std::int64_t remainder = output_index;
        std::int64_t input_index = 0;
        for (std::int64_t axis = rank; axis-- > 0;) {
            const auto coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            input_index += coordinate * strides[axis];
        }
        for (std::int64_t byte = 0; byte < element_size; ++byte) {
            output[output_index * element_size + byte] =
                input[input_index * element_size + byte];
        }
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

__global__ void masked_fill_f32(
    const float* input,
    const std::uint8_t* mask,
    float* output,
    std::int64_t count,
    const std::int64_t* metadata,
    std::int64_t rank,
    float fill_value) {
    const auto* output_shape = metadata;
    const auto* output_strides = metadata + rank;
    const auto* mask_shape = metadata + 2 * rank;
    const auto* mask_strides = metadata + 3 * rank;
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto index = first; index < count; index += stride) {
        std::int64_t mask_index = 0;
        for (std::int64_t axis = 0; axis < rank; ++axis) {
            const auto coordinate =
                (index / output_strides[axis]) % output_shape[axis];
            if (mask_shape[axis] != 1)
                mask_index += coordinate * mask_strides[axis];
        }
        output[index] = mask[mask_index] != 0 ? fill_value : input[index];
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

cudaStream_t stream(const std::shared_ptr<CudaExecutionContext>& context) {
    return static_cast<cudaStream_t>(context->stream());
}

} // namespace

CudaDeviceImpl::CudaDeviceImpl(int device_index)
    : device_index_(device_index),
      max_elementwise_blocks_(0),
      context_(cuda_execution_context(device_index)) {
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
const std::shared_ptr<CudaExecutionContext>& CudaDeviceImpl::execution_context() const {
    return context_;
}

Tensor CudaDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<CudaMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype), dtype, context_);
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
        add_f32<<<blocks, 256, 0, stream(context_)>>>(
            data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
            data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA add kernel");
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

    const std::size_t metadata_bytes = metadata.size() * sizeof(std::int64_t);
    CudaAllocation metadata_allocation(metadata_bytes, context_);
    auto* device_metadata = metadata_allocation.data_as<std::int64_t>();
    {
        if (metadata_bytes != 0) {
            check_cuda(cudaMemcpyAsync(device_metadata, metadata.data(), metadata_bytes,
                                       cudaMemcpyHostToDevice, stream(context_)),
                       "failed to copy CUDA broadcast metadata");
        }
        const auto required_blocks = (out.numel() + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        broadcast_elementwise_f32<<<blocks, 256, 0, stream(context_)>>>(
            data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
            data(require_cuda_storage(*out.storage())), out.numel(), device_metadata,
            static_cast<std::int64_t>(rank), operation);
        check_cuda(cudaGetLastError(), "failed to launch CUDA broadcast elementwise kernel");
    }
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
    scalar_elementwise_f32<<<blocks, 256, 0, stream(context_)>>>(
        data(require_cuda_storage(*input)), scalar,
        data(require_cuda_storage(*out.storage())), out.numel(), operation, scalar_is_left);
    check_cuda(cudaGetLastError(), "failed to launch CUDA scalar elementwise kernel");
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

    auto input = ensure_storage(tensor.storage(), ConversionPolicy::CopyToDevice);
    auto& input_storage = require_cuda_storage(*input);
    const float* input_data = data(input_storage) + tensor.storage_offset();
    const bool specialized_last_dimension =
        tensor.ndim() != 0 && dimensions.size() == 1 &&
        dimensions.front() == static_cast<std::int64_t>(tensor.ndim() - 1) &&
        tensor.is_contiguous() && operation != CudaReductionOperation::Variance;
    if (specialized_last_dimension) {
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(output.numel(), max_elementwise_blocks_));
        reduce_last_dimension_f32<<<blocks, 256, 0, stream(context_)>>>(
            input_data, data(require_cuda_storage(*output.storage())), output.numel(),
            tensor.shape().back(), operation);
        check_cuda(cudaGetLastError(), "failed to launch CUDA last-dimension reduction kernel");
        return output;
    }

    const std::size_t rank = tensor.ndim();
    // Keep a valid metadata pointer for rank-zero scalar reductions even though
    // the kernel does not dereference any axis entries in that case.
    std::vector<std::int64_t> metadata(std::max<std::size_t>(rank * 3, 1), 0);
    std::copy(tensor.shape().begin(), tensor.shape().end(), metadata.begin());
    std::copy(tensor.strides().begin(), tensor.strides().end(), metadata.begin() + rank);
    for (std::size_t axis = 0; axis < rank; ++axis)
        metadata[2 * rank + axis] = reduced[axis] ? 1 : 0;

    const std::size_t metadata_bytes = metadata.size() * sizeof(std::int64_t);
    CudaAllocation metadata_allocation(metadata_bytes, context_);
    auto* device_metadata = metadata_allocation.data_as<std::int64_t>();
    {
        if (metadata_bytes != 0) {
            check_cuda(cudaMemcpyAsync(device_metadata, metadata.data(), metadata_bytes,
                                       cudaMemcpyHostToDevice, stream(context_)),
                       "failed to copy CUDA reduction metadata");
        }
        const auto required_blocks = (output.numel() + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        reduce_f32<<<blocks, 256, 0, stream(context_)>>>(
            input_data,
            data(require_cuda_storage(*output.storage())), output.numel(), reduced_count,
            device_metadata, static_cast<std::int64_t>(rank), operation);
        check_cuda(cudaGetLastError(), "failed to launch CUDA reduction kernel");
    }
    return output;
}

Tensor CudaDeviceImpl::argmax(const Tensor& tensor, std::int64_t dimension, bool keepdim) const {
    require_defined(tensor, "argmax input");
    require_float32(tensor, "argmax input");
    if (dimension < 0 || dimension >= static_cast<std::int64_t>(tensor.ndim()))
        throw std::out_of_range("argmax dimension is out of range");
    const auto axis = static_cast<std::size_t>(dimension);
    const auto reduction_size = tensor.shape()[axis];
    if (reduction_size == 0) throw std::invalid_argument("argmax cannot reduce an empty dimension");
    Shape output_shape = tensor.shape();
    if (keepdim) output_shape[axis] = 1;
    else output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    Tensor output = empty(output_shape, DType::Int64);
    if (output.numel() == 0) return output;
    const auto inner_size = std::accumulate(
        tensor.shape().begin() + static_cast<std::ptrdiff_t>(axis + 1), tensor.shape().end(),
        std::int64_t{1}, std::multiplies<>());
    const Tensor packed = tensor.is_contiguous() ? tensor : contiguous(tensor);
    auto input = ensure_storage(packed.storage(), ConversionPolicy::CopyToDevice);
    const auto blocks = static_cast<unsigned>(
        std::min<std::int64_t>(output.numel(), max_elementwise_blocks_));
    auto* output_data = static_cast<std::int64_t*>(
        require_cuda_storage(*output.storage()).handle().ptr);
    argmax_f32<<<blocks, 256, 0, stream(context_)>>>(
        data(require_cuda_storage(*input)) + packed.storage_offset(), output_data,
        output.numel(), reduction_size,
        inner_size);
    check_cuda(cudaGetLastError(), "failed to launch CUDA argmax kernel");
    return output;
}

Tensor CudaDeviceImpl::contiguous(const Tensor& tensor) const {
    require_defined(tensor, "contiguous input");
    if (tensor.device() != Device::cuda(device_index_)) {
        throw std::invalid_argument("contiguous input is on a different CUDA device");
    }
    if (tensor.is_contiguous()) return tensor;

    Tensor output = empty(tensor.shape(), tensor.dtype());
    if (tensor.numel() == 0) return output;
    const auto rank = static_cast<std::int64_t>(tensor.ndim());
    std::vector<std::int64_t> metadata;
    metadata.reserve(static_cast<std::size_t>(2 * rank));
    metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
    metadata.insert(metadata.end(), tensor.strides().begin(), tensor.strides().end());
    const auto metadata_bytes = metadata.size() * sizeof(std::int64_t);
    CudaAllocation device_metadata(metadata_bytes, context_);
    device_metadata.copy_from_host_async(metadata.data(), metadata_bytes);

    const auto& input_storage = require_cuda_storage(*tensor.storage());
    const auto* input_base = static_cast<const unsigned char*>(input_storage.handle().ptr);
    const auto element_size = static_cast<std::int64_t>(dtype_size(tensor.dtype()));
    const auto* input = input_base + tensor.storage_offset() * element_size;
    auto* destination = static_cast<unsigned char*>(
        require_cuda_storage(*output.storage()).handle().ptr);
    const auto blocks = static_cast<unsigned>(
        std::min<std::int64_t>((tensor.numel() + 255) / 256, max_elementwise_blocks_));
    copy_strided<<<blocks, 256, 0, stream(context_)>>>(
        input, destination, tensor.numel(), rank, device_metadata.data_as<std::int64_t>(),
        device_metadata.data_as<std::int64_t>() + rank, element_size);
    check_cuda(cudaGetLastError(), "failed to launch CUDA contiguous kernel");
    return output;
}

Tensor CudaDeviceImpl::gather_rows(const Tensor& table, const Tensor& indices) const {
    ENSURE_TENSOR_DEFINED(table);
    ENSURE_TENSOR_DEFINED(indices);
    ENSURE_TENSOR_DIM(table, 2);
    ENSURE_TENSOR_DTYPE(table, DType::Float32);
    ENSURE_TENSOR_DTYPE(indices, DType::Int64);
    ENSURE_TENSOR_DEVICE_MATCH_2(table, indices);
    if (table.device() != Device::cuda(device_index_))
        throw std::invalid_argument("gather_rows inputs must be on the selected CUDA device");

    const Tensor packed_table = contiguous(table);
    const Tensor packed_indices = contiguous(indices);
    Shape output_shape = indices.shape();
    output_shape.push_back(table.shape()[1]);
    Tensor output = empty(std::move(output_shape), DType::Float32);
    if (indices.numel() == 0 || table.shape()[1] == 0) return output;

    CudaAllocation invalid_index(sizeof(int), context_);
    int invalid = 0;
    invalid_index.copy_from_host_async(&invalid, sizeof(invalid));
    const auto& table_storage = require_cuda_storage(*packed_table.storage());
    const auto& index_storage = require_cuda_storage(*packed_indices.storage());
    auto& output_storage = require_cuda_storage(*output.storage());
    const auto required_blocks = (output.numel() + 255) / 256;
    const auto blocks = static_cast<unsigned>(
        std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
    gather_rows_f32<<<blocks, 256, 0, stream(context_)>>>(
        static_cast<const float*>(table_storage.handle().ptr),
        static_cast<const std::int64_t*>(index_storage.handle().ptr),
        data(output_storage), indices.numel(), table.shape()[0], table.shape()[1],
        invalid_index.data_as<int>());
    check_cuda(cudaGetLastError(), "failed to launch CUDA gather_rows kernel");
    invalid_index.copy_to_host_async(&invalid, sizeof(invalid));
    invalid_index.synchronize();
    if (invalid != 0) throw std::out_of_range("gather_rows index is out of range");
    return output;
}

Tensor CudaDeviceImpl::concat(
    const std::vector<Tensor>& tensors,
    std::int64_t dimension) const {
    if (tensors.empty()) throw std::invalid_argument("concat expects at least one tensor");
    Shape output_shape = tensors.front().shape();
    output_shape[static_cast<std::size_t>(dimension)] = 0;
    for (const Tensor& tensor : tensors)
        output_shape[static_cast<std::size_t>(dimension)] +=
            tensor.shape()[static_cast<std::size_t>(dimension)];
    Tensor output = empty(output_shape, tensors.front().dtype());
    const auto axis = static_cast<std::size_t>(dimension);
    const auto inner = std::accumulate(
        output_shape.begin() + static_cast<std::ptrdiff_t>(axis + 1), output_shape.end(),
        std::int64_t{1}, std::multiplies<>());
    const auto outer = std::accumulate(
        output_shape.begin(), output_shape.begin() + static_cast<std::ptrdiff_t>(axis),
        std::int64_t{1}, std::multiplies<>());
    const auto element_size = dtype_size(output.dtype());
    const auto output_pitch = static_cast<std::size_t>(output_shape[axis] * inner) * element_size;
    auto* output_data = static_cast<unsigned char*>(
        require_cuda_storage(*output.storage()).handle().ptr);
    std::int64_t dimension_offset = 0;
    for (const Tensor& tensor : tensors) {
        const Tensor packed = contiguous(tensor);
        const auto& storage = require_cuda_storage(*packed.storage());
        const auto* source = static_cast<const unsigned char*>(storage.handle().ptr) +
            packed.storage_offset() * element_size;
        const auto width = static_cast<std::size_t>(tensor.shape()[axis] * inner) * element_size;
        if (width != 0 && outer != 0) {
            check_cuda(cudaMemcpy2DAsync(
                output_data + dimension_offset * inner * element_size,
                output_pitch, source, width, width, static_cast<std::size_t>(outer),
                cudaMemcpyDeviceToDevice, stream(context_)),
                "failed to concatenate CUDA tensors");
        }
        dimension_offset += tensor.shape()[axis];
    }
    return output;
}

Tensor CudaDeviceImpl::argmax(const Tensor& tensor) const {
    require_defined(tensor, "argmax input");
    require_float32(tensor, "argmax input");
    if (tensor.numel() == 0) throw std::invalid_argument("argmax input cannot be empty");
    const Tensor flattened({tensor.numel()}, tensor.dtype(), tensor.device(), tensor.storage());
    return argmax(flattened, 0, false);
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
    unary_f32<<<blocks, 256, 0, stream(context_)>>>(
        data(require_cuda_storage(*input)),
        data(require_cuda_storage(*output.storage())), output.numel(), operation, argument);
    check_cuda(cudaGetLastError(), "failed to launch CUDA unary kernel");
    return output;
}

Tensor CudaDeviceImpl::masked_fill(
    const Tensor& tensor,
    const Tensor& mask,
    float value) const {
    require_defined(tensor, "masked_fill input");
    require_float32(tensor, "masked_fill input");
    require_defined(mask, "masked_fill mask");
    ENSURE_TENSOR_DTYPE(mask, DType::Bool);
    ENSURE_TENSOR_DEVICE_MATCH_2(tensor, mask);
    if (broadcast_shape(tensor.shape(), mask.shape()) != tensor.shape())
        throw std::invalid_argument("masked_fill mask must broadcast to the input shape");

    Tensor output = empty(tensor.shape(), DType::Float32);
    if (output.numel() == 0) return output;
    auto input = ensure_storage(tensor.storage(), ConversionPolicy::CopyToDevice);
    auto mask_storage = ensure_storage(mask.storage(), ConversionPolicy::CopyToDevice);
    const std::size_t rank = tensor.ndim();
    const Strides output_strides = contiguous_strides(tensor.shape());
    const Strides mask_strides = contiguous_strides(mask.shape());
    std::vector<std::int64_t> metadata(std::max<std::size_t>(rank * 4, 1), 0);
    std::copy(tensor.shape().begin(), tensor.shape().end(), metadata.begin());
    std::copy(output_strides.begin(), output_strides.end(), metadata.begin() + rank);
    std::fill(metadata.begin() + 2 * rank, metadata.begin() + 3 * rank, 1);
    const std::size_t padding = rank - mask.ndim();
    std::copy(mask.shape().begin(), mask.shape().end(), metadata.begin() + 2 * rank + padding);
    std::copy(mask_strides.begin(), mask_strides.end(), metadata.begin() + 3 * rank + padding);

    const std::size_t metadata_bytes = metadata.size() * sizeof(std::int64_t);
    CudaAllocation metadata_allocation(metadata_bytes, context_);
    auto* device_metadata = metadata_allocation.data_as<std::int64_t>();
    {
        check_cuda(cudaMemcpyAsync(device_metadata, metadata.data(), metadata_bytes,
                                   cudaMemcpyHostToDevice, stream(context_)),
                   "failed to copy CUDA masked_fill metadata");
        const auto required_blocks = (output.numel() + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        masked_fill_f32<<<blocks, 256, 0, stream(context_)>>>(
            data(require_cuda_storage(*input)),
            static_cast<const std::uint8_t*>(require_cuda_storage(*mask_storage).handle().ptr),
            data(require_cuda_storage(*output.storage())), output.numel(), device_metadata,
            static_cast<std::int64_t>(rank), value);
        check_cuda(cudaGetLastError(), "failed to launch CUDA masked_fill kernel");
    }
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
        sub_f32<<<blocks, 256, 0, stream(context_)>>>(
            data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
            data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA sub kernel");
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
    auto bytes = l.ao.size() * sizeof(std::int64_t);
    CudaAllocation a_offsets(bytes, context_);
    CudaAllocation b_offsets(bytes, context_);
    auto* dao = a_offsets.data_as<std::int64_t>();
    auto* dbo = b_offsets.data_as<std::int64_t>();
    {
        check_cuda(cudaMemcpyAsync(dao, l.ao.data(), bytes, cudaMemcpyHostToDevice,
                                   stream(context_)),
                   "failed to copy CUDA batch offsets");
        check_cuda(cudaMemcpyAsync(dbo, l.bo.data(), bytes, cudaMemcpyHostToDevice,
                                   stream(context_)),
                   "failed to copy CUDA batch offsets");
        auto m = a.shape()[a.ndim() - 2], n = b.shape().back();
        dim3 threads(matmul_tile_size, matmul_tile_size);
        dim3 blocks(static_cast<unsigned>((n + matmul_tile_size - 1) / matmul_tile_size),
                    static_cast<unsigned>((m + matmul_tile_size - 1) / matmul_tile_size),
                    static_cast<unsigned>(l.ao.size()));
        batched_matmul_f32<<<blocks, threads, 0, stream(context_)>>>(
            data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)),
            data(require_cuda_storage(*out.storage())), dao, dbo, l.ao.size(), m,
            a.shape().back(), n);
        check_cuda(cudaGetLastError(), "failed to launch CUDA batched_matmul kernel");
    }
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
    ENSURE_TENSOR_SHAPE(out, Shape({m, n}));
    const Tensor packed_a = a.is_contiguous() ? a : contiguous(a);
    const Tensor packed_b = b.is_contiguous() ? b : contiguous(b);
    auto ap = ensure_storage(packed_a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(packed_b.storage(), ConversionPolicy::CopyToDevice);
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
        matmul_f32<<<blocks, threads, 0, stream(context_)>>>(
            data(require_cuda_storage(*ap)) + packed_a.storage_offset(),
            data(require_cuda_storage(*bp)) + packed_b.storage_offset(),
            data(require_cuda_storage(*out.storage())), m, k, n);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA matmul kernel");
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
