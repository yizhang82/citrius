#include "impl/cuda_device.h"

#include "impl/cpu_storage.h"

#include <cuda_runtime.h>

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

__global__ void add_f32(const float* a, const float* b, float* out, std::int64_t count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] + b[i];
}
__global__ void sub_f32(const float* a, const float* b, float* out, std::int64_t count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) out[i] = a[i] - b[i];
}
__global__ void matmul_f32(const float* a, const float* b, float* out, std::int64_t m, std::int64_t k, std::int64_t n) {
    const auto row = static_cast<std::int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const auto col = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= m || col >= n) return;
    float total = 0.0f;
    for (std::int64_t inner = 0; inner < k; ++inner) total += a[row * k + inner] * b[inner * n + col];
    out[row * n + col] = total;
}

float* data(CudaMemTensorStorageImpl& storage) { return static_cast<float*>(storage.handle().ptr); }

} // namespace

CudaDeviceImpl::CudaDeviceImpl(int device_index) : device_index_(device_index) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "failed to query CUDA devices");
    if (device_index < 0 || device_index >= count) throw std::runtime_error("CUDA device index is not available");
    check_cuda(cudaSetDevice(device_index_), "failed to select CUDA device");
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
    if (count != 0) add_f32<<<static_cast<unsigned>((count + 255) / 256), 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)), data(require_cuda_storage(*out.storage())), count);
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
    if (count != 0) sub_f32<<<static_cast<unsigned>((count + 255) / 256), 256>>>(data(require_cuda_storage(*ap)), data(require_cuda_storage(*bp)), data(require_cuda_storage(*out.storage())), count);
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
        const dim3 threads(16, 16);
        const dim3 blocks(static_cast<unsigned>((n + 15) / 16), static_cast<unsigned>((m + 15) / 16));
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
