#include "impl/cuda_device.h"

#include "impl/cpu_storage.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void require_defined(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) throw std::invalid_argument(std::string(name) + " tensor is undefined");
}
void require_float32(const Tensor& tensor, const char* name) {
    if (tensor.dtype() != DType::Float32) throw std::invalid_argument(std::string(name) + " tensor must be Float32");
}
void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) throw std::invalid_argument("tensor shapes must match");
}
void require_2d_matmul_shapes(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) throw std::invalid_argument("matmul expects 2D tensors");
    if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("matmul inner dimensions must match");
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
//   +---------------- Block 0 ----------------+  +---------------- Block 1 ----------------+
//   |  local thread:   0     1     2     3   |  |  local thread:   0     1     2     3   |
//   |  global thread: T0    T1    T2    T3   |  |  global thread: T4    T5    T6    T7   |
//   +-----------------------------------------+  +-----------------------------------------+
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
    for (auto i = first; i < count; i += stride) out[i] = a[i] + b[i];
}
__global__ void sub_f32(const float* a, const float* b, float* out, std::int64_t count) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto i = first; i < count; i += stride) out[i] = a[i] - b[i];
}
constexpr int matmul_tile_size = 16;

__global__ void matmul_f32(
    const float* a,
    const float* b,
    float* out,
    std::int64_t m,
    std::int64_t k,
    std::int64_t n) {
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
        a_tile[threadIdx.y][threadIdx.x] = row < m && a_col < k
            ? a[row * k + a_col]
            : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = b_row < k && col < n
            ? b[b_row * n + col]
            : 0.0f;
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
    if (row < m && col < n) out[row * n + col] = total;
}

float* data(CudaMemTensorStorageImpl& storage) { return static_cast<float*>(storage.handle().ptr); }

} // namespace

CudaDeviceImpl::CudaDeviceImpl(int device_index) : device_index_(device_index), max_elementwise_blocks_(0) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "failed to query CUDA devices");
    if (device_index < 0 || device_index >= count) throw std::runtime_error("CUDA device index is not available");
    check_cuda(cudaSetDevice(device_index_), "failed to select CUDA device");
    int multiprocessor_count = 0;
    check_cuda(
        cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device_index_),
        "failed to query CUDA multiprocessor count");
    // A bounded grid avoids launching one thread per element for very large
    // tensors. Thirty-two blocks per SM provides ample work for scheduling;
    // each thread advances by the full grid width until all elements are done.
    max_elementwise_blocks_ = multiprocessor_count * 32;
}

DeviceType CudaDeviceImpl::type() const { return DeviceType::CUDA; }
int CudaDeviceImpl::device_index() const { return device_index_; }

Tensor CudaDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<CudaMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype), dtype, device_index_);
    return Tensor(std::move(shape), dtype, Device::cuda(device_index_), std::move(storage));
}

Tensor CudaDeviceImpl::add(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_same_shape(a, b);
    auto out = empty(a.shape(), a.dtype());
    add_out(a, b, out);
    return out;
}

void CudaDeviceImpl::add_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_same_shape(a, b);
    require_defined(out, "output"); require_float32(out, "output"); require_same_shape(a, out);
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto count = a.numel();
    if (count != 0) {
        const auto required_blocks = (count + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        // <<<blocks, 256>>> launches a grid of `blocks` blocks, each containing 256 threads.
        add_f32<<<blocks, 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)), data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA add kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA add kernel failed");
}

Tensor CudaDeviceImpl::sub(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_same_shape(a, b);
    auto out = empty(a.shape(), a.dtype());
    sub_out(a, b, out);
    return out;
}

void CudaDeviceImpl::sub_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_same_shape(a, b);
    require_defined(out, "output"); require_float32(out, "output"); require_same_shape(a, out);
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto count = a.numel();
    if (count != 0) {
        const auto required_blocks = (count + 255) / 256;
        const auto blocks = static_cast<unsigned>(
            std::min<std::int64_t>(required_blocks, max_elementwise_blocks_));
        // <<<blocks, 256>>> launches a grid of `blocks` blocks, each containing 256 threads.
        sub_f32<<<blocks, 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)), data(require_cuda_storage(*out.storage())), count);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA sub kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA sub kernel failed");
}

Tensor CudaDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_2d_matmul_shapes(a, b);
    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    auto out = empty({m, n}, a.dtype());
    matmul_out(a, b, out);
    return out;
}

void CudaDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_defined(a, "left"); require_defined(b, "right");
    require_float32(a, "left"); require_float32(b, "right"); require_2d_matmul_shapes(a, b);
    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    require_defined(out, "output"); require_float32(out, "output");
    if (out.shape() != Shape({m, n})) throw std::invalid_argument("matmul output shape must be [m, n]");
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
        const dim3 blocks(
            static_cast<unsigned>((n + matmul_tile_size - 1) / matmul_tile_size),
            static_cast<unsigned>((m + matmul_tile_size - 1) / matmul_tile_size));
        matmul_f32<<<blocks, threads>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)), data(require_cuda_storage(*out.storage())), m, k, n);
    }
    check_cuda(cudaGetLastError(), "failed to launch CUDA matmul kernel");
    check_cuda(cudaDeviceSynchronize(), "CUDA matmul kernel failed");
}

TensorStoragePtr CudaDeviceImpl::ensure_storage(const TensorStoragePtr& storage, ConversionPolicy policy) const {
    if (!storage) throw std::invalid_argument("tensor has no storage");
    if (storage->type() == TensorStorageType::CudaMemory) {
        const auto& cuda = static_cast<const CudaMemTensorStorageImpl&>(*storage);
        if (cuda.device_index() == device_index_) return storage;
    }
    if (storage->type() == TensorStorageType::CpuMemory && policy == ConversionPolicy::CopyToDevice) {
        const auto& cpu = static_cast<const CpuMemTensorStorageImpl&>(*storage);
        auto cuda = std::make_shared<CudaMemTensorStorageImpl>(cpu.nbytes(), cpu.dtype(), device_index_);
        cuda->copy_from_host(cpu.data(), cpu.nbytes());
        return cuda;
    }
    throw std::invalid_argument("CudaDeviceImpl requires CudaMemory storage on the selected device");
}

const CudaMemTensorStorageImpl& CudaDeviceImpl::require_cuda_storage(const ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CudaMemory) throw std::invalid_argument("CudaDeviceImpl requires CudaMemory storage");
    const auto& cuda = static_cast<const CudaMemTensorStorageImpl&>(storage);
    if (cuda.device_index() != device_index_) throw std::invalid_argument("CUDA storage is on a different device");
    return cuda;
}
CudaMemTensorStorageImpl& CudaDeviceImpl::require_cuda_storage(ITensorStorage& storage) const {
    return const_cast<CudaMemTensorStorageImpl&>(require_cuda_storage(static_cast<const ITensorStorage&>(storage)));
}

} // namespace citrius::impl
