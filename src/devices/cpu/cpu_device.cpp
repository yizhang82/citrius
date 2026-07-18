#include "impl/cpu_device.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace citrius::impl {

namespace {

void require_defined(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " tensor is undefined");
    }
}

void require_float32(const Tensor& tensor, const char* name) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " tensor must be Float32");
    }
}

void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("tensor shapes must match");
    }
}

void require_2d_matmul_shapes(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }
}

} // namespace

DeviceType CpuDeviceImpl::type() const {
    return DeviceType::CPU;
}

Tensor CpuDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<CpuMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype),
        dtype);
    return Tensor(std::move(shape), dtype, Device::cpu(), std::move(storage));
}

Tensor CpuDeviceImpl::add(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t i = 0; i < a.numel(); ++i) {
        output_data[i] = a_data[i] + b_data[i];
    }

    return output;
}

Tensor CpuDeviceImpl::sub(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t i = 0; i < a.numel(); ++i) {
        output_data[i] = a_data[i] - b_data[i];
    }

    return output;
}

Tensor CpuDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);

    const std::int64_t m = a.shape()[0];
    const std::int64_t k = a.shape()[1];
    const std::int64_t n = b.shape()[1];

    auto output = empty({m, n}, a.dtype());
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t col = 0; col < n; ++col) {
            float total = 0.0f;
            for (std::int64_t inner = 0; inner < k; ++inner) {
                total += a_data[row * k + inner] * b_data[inner * n + col];
            }
            output_data[row * n + col] = total;
        }
    }

    return output;
}

TensorStoragePtr CpuDeviceImpl::ensure_storage(
    const TensorStoragePtr& storage,
    ConversionPolicy policy) const {
    if (!storage) {
        throw std::invalid_argument("tensor has no storage");
    }

    if (storage->type() == TensorStorageType::CpuMemory) {
        return storage;
    }

    if (policy == ConversionPolicy::CopyToDevice) {
        throw std::invalid_argument("copying non-CPU storage to CPU is not implemented");
    }

    throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
}

const CpuMemTensorStorageImpl& CpuDeviceImpl::require_cpu_storage(
    const ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CpuMemory) {
        throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
    }

    return static_cast<const CpuMemTensorStorageImpl&>(storage);
}

CpuMemTensorStorageImpl& CpuDeviceImpl::require_cpu_storage(ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CpuMemory) {
        throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
    }

    return static_cast<CpuMemTensorStorageImpl&>(storage);
}

} // namespace citrius::impl
