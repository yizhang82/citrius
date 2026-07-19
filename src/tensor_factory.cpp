#include "tensor_factory.h"
#include "impl/tensor_factory.h"
#include "impl/cpu_device.h"
#include "impl/cpu_storage.h"
#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#include "impl/cuda_storage.h"
#endif
#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#include "impl/metal_storage.h"
#endif
#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace citrius::impl {
namespace {
using CpuDeviceImpl = citrius::impl::CpuDeviceImpl;
using CpuMemTensorStorageImpl = citrius::impl::CpuMemTensorStorageImpl;
#ifdef CITRIUS_HAS_CUDA
using CudaDeviceImpl = citrius::impl::CudaDeviceImpl;
using CudaMemTensorStorageImpl = citrius::impl::CudaMemTensorStorageImpl;
#endif
#ifdef CITRIUS_HAS_METAL
using MetalDeviceImpl = citrius::impl::MetalDeviceImpl;
using MetalMemTensorStorageImpl = citrius::impl::MetalMemTensorStorageImpl;
#endif

Tensor copy_to_cpu(const Tensor& tensor) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument("moving a non-contiguous tensor requires contiguous()");
    }
    if (tensor.device().type == DeviceType::CPU) return tensor;
    auto output = CpuDeviceImpl().empty(tensor.shape(), tensor.dtype());
    auto destination = std::static_pointer_cast<CpuMemTensorStorageImpl>(output.storage());
    const auto source_offset =
        static_cast<std::size_t>(tensor.storage_offset()) * dtype_size(tensor.dtype());
    switch (tensor.device().type) {
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA:
            std::static_pointer_cast<CudaMemTensorStorageImpl>(tensor.storage())->copy_to_host(
                destination->data(), destination->nbytes(), source_offset);
            return output;
#endif
#ifdef CITRIUS_HAS_METAL
        case DeviceType::Metal:
            std::static_pointer_cast<MetalMemTensorStorageImpl>(tensor.storage())->copy_to_host(
                destination->data(), destination->nbytes(), source_offset);
            return output;
#endif
        default: throw std::invalid_argument("source tensor backend is not enabled");
    }
}
} // namespace

Tensor TensorFactory::empty(Shape shape, DType dtype, Device device) {
    switch (device.type) {
        case DeviceType::CPU: {
            const auto count = std::accumulate(
                shape.begin(), shape.end(), std::int64_t{1},
                [](std::int64_t total, std::int64_t dimension) { return total * dimension; });
            auto storage = std::make_shared<CpuMemTensorStorageImpl>(
                static_cast<std::size_t>(count) * dtype_size(dtype), dtype);
            return Tensor(std::move(shape), dtype, device, std::move(storage));
        }
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA: return CudaDeviceImpl(device.index).empty(std::move(shape), dtype);
#endif
#ifdef CITRIUS_HAS_METAL
        case DeviceType::Metal:
            if (device.index != 0) throw std::invalid_argument("Metal device index is not available");
            return MetalDeviceImpl().empty(std::move(shape), dtype);
#endif
        default: throw std::invalid_argument("requested tensor backend is not enabled");
    }
}

Tensor TensorFactory::from_vector(const std::vector<float>& values, Device device) {
    const auto size = static_cast<std::int64_t>(values.size());
    return from_vector(values, {size}, device);
}

Tensor TensorFactory::from_vector(const std::vector<float>& values, Shape shape, Device device) {
    auto cpu = empty(std::move(shape), DType::Float32, Device::cpu());
    if (cpu.numel() != static_cast<std::int64_t>(values.size())) throw std::invalid_argument("tensor shape does not match vector size");
    auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu.storage());
    std::copy(values.begin(), values.end(), storage->data_as<float>());
    return to(cpu, device);
}

Tensor TensorFactory::from_vector(const std::vector<std::int64_t>& values, Device device) {
    return from_vector(values, {static_cast<std::int64_t>(values.size())}, device);
}

Tensor TensorFactory::from_vector(
    const std::vector<std::int64_t>& values,
    Shape shape,
    Device device) {
    auto cpu = empty(std::move(shape), DType::Int64, Device::cpu());
    if (cpu.numel() != static_cast<std::int64_t>(values.size())) {
        throw std::invalid_argument("tensor shape does not match vector size");
    }
    auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu.storage());
    std::copy(values.begin(), values.end(), storage->data_as<std::int64_t>());
    return to(cpu, device);
}

Tensor TensorFactory::from_vector(const std::vector<bool>& values, Device device) {
    return from_vector(values, {static_cast<std::int64_t>(values.size())}, device);
}

Tensor TensorFactory::from_vector(const std::vector<bool>& values, Shape shape, Device device) {
    auto cpu = empty(std::move(shape), DType::Bool, Device::cpu());
    if (cpu.numel() != static_cast<std::int64_t>(values.size())) {
        throw std::invalid_argument("tensor shape does not match vector size");
    }
    auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu.storage());
    auto* destination = storage->data_as<std::uint8_t>();
    for (std::size_t index = 0; index < values.size(); ++index) {
        destination[index] = values[index] ? 1 : 0;
    }
    return to(cpu, device);
}

Tensor TensorFactory::to(const Tensor& tensor, Device device) {
    if (!tensor.defined()) throw std::invalid_argument("cannot move an undefined tensor");
    if (tensor.device() == device) return tensor;
    auto cpu = copy_to_cpu(tensor);
    if (device.type == DeviceType::CPU) return cpu;
    auto output = empty(tensor.shape(), tensor.dtype(), device);
    const auto source = std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu.storage());
    const auto source_offset =
        static_cast<std::size_t>(cpu.storage_offset()) * dtype_size(cpu.dtype());
    const auto nbytes = static_cast<std::size_t>(cpu.numel()) * dtype_size(cpu.dtype());
    const auto* source_data = static_cast<const std::byte*>(source->data()) + source_offset;
    switch (device.type) {
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA:
            std::static_pointer_cast<CudaMemTensorStorageImpl>(output.storage())->copy_from_host(
                source_data, nbytes);
            return output;
#endif
#ifdef CITRIUS_HAS_METAL
        case DeviceType::Metal:
            std::static_pointer_cast<MetalMemTensorStorageImpl>(output.storage())->copy_from_host(
                source_data, nbytes);
            return output;
#endif
        default: throw std::invalid_argument("requested tensor backend is not enabled");
    }
}
} // namespace citrius::impl

namespace citrius {

Tensor empty(Shape shape, DType dtype, Device device) {
    return impl::TensorFactory::empty(std::move(shape), dtype, device);
}

Tensor from_vector(const std::vector<float>& values, Device device) {
    return impl::TensorFactory::from_vector(values, device);
}

Tensor from_vector(const std::vector<float>& values, Shape shape, Device device) {
    return impl::TensorFactory::from_vector(values, std::move(shape), device);
}

Tensor from_vector(const std::vector<std::int64_t>& values, Device device) {
    return impl::TensorFactory::from_vector(values, device);
}

Tensor from_vector(const std::vector<std::int64_t>& values, Shape shape, Device device) {
    return impl::TensorFactory::from_vector(values, std::move(shape), device);
}

Tensor from_vector(const std::vector<bool>& values, Device device) {
    return impl::TensorFactory::from_vector(values, device);
}

Tensor from_vector(const std::vector<bool>& values, Shape shape, Device device) {
    return impl::TensorFactory::from_vector(values, std::move(shape), device);
}

Tensor to(const Tensor& tensor, Device device) {
    return impl::TensorFactory::to(tensor, device);
}

} // namespace citrius
