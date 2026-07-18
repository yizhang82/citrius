#include "operations.h"

#include "impl/cpu_device.h"
#include "impl/multi_thread_cpu_device.h"
#include "impl/cpu_storage.h"
#include "exceptions.h"
#include "tensor_factory.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cutlass_cuda_device.h"
#endif

#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace citrius {
namespace {

using impl::CpuDeviceImpl;
using impl::MultiThreadCpuDeviceImpl;
#ifdef CITRIUS_HAS_CUDA
using impl::CudaDeviceImpl;
using impl::CublasCudaDeviceImpl;
using impl::CutlassCudaDeviceImpl;
#endif
#ifdef CITRIUS_HAS_METAL
using impl::MetalDeviceImpl;
#endif

void require_matching_devices(const Tensor& left, const Tensor& right) {
    if (left.device() != right.device()) {
        throw DeviceMismatchException(
            "tensor devices must match for binary operations");
    }
}

void require_float32(const Tensor& tensor) {
    if (!tensor.defined()) throw std::invalid_argument("tensor is undefined");
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument("operation currently supports Float32 only");
    }
}

Shape broadcast_shape(const Shape& left, const Shape& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Shape output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto left_dim = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto right_dim = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
            throw std::invalid_argument("tensor shapes are not broadcastable");
        }
        output[rank - 1 - offset] = std::max(left_dim, right_dim);
    }
    return output;
}

Strides row_major_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

std::int64_t broadcast_index(
    std::int64_t output_index,
    const Shape& output_shape,
    const Strides& output_strides,
    const Shape& input_shape,
    const Strides& input_strides) {
    const std::size_t padding = output_shape.size() - input_shape.size();
    std::int64_t input_index = 0;
    for (std::size_t axis = 0; axis < output_shape.size(); ++axis) {
        const std::int64_t coordinate =
            (output_index / output_strides[axis]) % output_shape[axis];
        if (axis >= padding && input_shape[axis - padding] != 1) {
            input_index += coordinate * input_strides[axis - padding];
        }
    }
    return input_index;
}

template <typename Operation>
Tensor broadcast_binary(const Tensor& left, const Tensor& right, Operation operation) {
    require_matching_devices(left, right);
    require_float32(left);
    require_float32(right);
    const Shape shape = broadcast_shape(left.shape(), right.shape());
    const Tensor cpu_left = left.to(Device::cpu());
    const Tensor cpu_right = right.to(Device::cpu());
    const auto left_storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_left.storage());
    const auto right_storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_right.storage());
    const float* left_values = left_storage->data_as<float>();
    const float* right_values = right_storage->data_as<float>();
    const Strides output_strides = row_major_strides(shape);
    const Strides left_strides = row_major_strides(left.shape());
    const Strides right_strides = row_major_strides(right.shape());
    std::int64_t count = 1;
    for (const auto dimension : shape) count *= dimension;
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::int64_t index = 0; index < count; ++index) {
        const auto left_index = broadcast_index(index, shape, output_strides, left.shape(), left_strides);
        const auto right_index = broadcast_index(index, shape, output_strides, right.shape(), right_strides);
        values[static_cast<std::size_t>(index)] = operation(left_values[left_index], right_values[right_index]);
    }
    return from_vector(values, shape, left.device());
}

template <typename Operation>
Tensor unary(const Tensor& tensor, Operation operation) {
    require_float32(tensor);
    const Tensor cpu = tensor.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu.storage());
    const float* input = storage->data_as<float>();
    std::vector<float> values(static_cast<std::size_t>(tensor.numel()));
    std::transform(input, input + tensor.numel(), values.begin(), operation);
    return from_vector(values, tensor.shape(), tensor.device());
}

#ifdef CITRIUS_HAS_CUDA
std::unique_ptr<impl::IDevice> cuda_device(int device_index) {
    const char* configured_backend = std::getenv("CITRIUS_CUDA_BACKEND");
    if (configured_backend) {
        const std::string_view backend(configured_backend);
        if (backend == "cublas") return std::make_unique<CublasCudaDeviceImpl>(device_index);
        if (backend == "reference") return std::make_unique<CudaDeviceImpl>(device_index);
        if (backend == "cutlass") return std::make_unique<CutlassCudaDeviceImpl>(device_index);
        throw CitriusException(
            "CITRIUS_CUDA_BACKEND must be 'cublas', 'cutlass', or 'reference', got '" +
            std::string(backend) + "'");
    }
#ifdef CITRIUS_CUDA_DEFAULT_CUBLAS
    return std::make_unique<CublasCudaDeviceImpl>(device_index);
#else
    return std::make_unique<CudaDeviceImpl>(device_index);
#endif
}
#endif

template <typename Operation>
Tensor dispatch(const Tensor& left, const Tensor& right, Operation operation) {
    require_matching_devices(left, right);
    const Device device = left.device();

    switch (device.type) {
        case DeviceType::CPU:
            return operation(MultiThreadCpuDeviceImpl(), left, right);
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA: {
            auto cuda = cuda_device(device.index);
            return operation(*cuda, left, right);
        }
#endif
#ifdef CITRIUS_HAS_METAL
        case DeviceType::Metal:
            if (device.index != 0) {
                throw CitriusException("Metal device index is not available");
            }
            return operation(MetalDeviceImpl(), left, right);
#endif
        default:
            throw CitriusException("tensor backend is not enabled");
    }
}

} // namespace

Tensor add(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
        return broadcast_binary(left, right, std::plus<float>());
    }
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.add(a, b);
    });
}

Tensor sub(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
        return broadcast_binary(left, right, std::minus<float>());
    }
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.sub(a, b);
    });
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.matmul(a, b);
    });
}

Tensor mul(const Tensor& left, const Tensor& right) {
    return broadcast_binary(left, right, std::multiplies<float>());
}

Tensor div(const Tensor& left, const Tensor& right) {
    return broadcast_binary(left, right, std::divides<float>());
}

Tensor maximum(const Tensor& left, const Tensor& right) {
    return broadcast_binary(left, right, [](float a, float b) { return std::max(a, b); });
}

Tensor exp(const Tensor& tensor) {
    return unary(tensor, [](float value) { return std::exp(value); });
}

Tensor sqrt(const Tensor& tensor) {
    return unary(tensor, [](float value) { return std::sqrt(value); });
}

Tensor pow(const Tensor& tensor, float exponent) {
    return unary(tensor, [exponent](float value) { return std::pow(value, exponent); });
}

Tensor masked_fill(const Tensor& tensor, const Tensor& mask, float value) {
    require_float32(tensor);
    if (!mask.defined() || mask.dtype() != DType::Bool) {
        throw std::invalid_argument("masked_fill mask must be a defined Bool tensor");
    }
    require_matching_devices(tensor, mask);
    if (broadcast_shape(tensor.shape(), mask.shape()) != tensor.shape()) {
        throw std::invalid_argument("masked_fill mask must broadcast to the input shape");
    }
    const Tensor cpu_tensor = tensor.to(Device::cpu());
    const Tensor cpu_mask = mask.to(Device::cpu());
    const auto tensor_storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_tensor.storage());
    const auto mask_storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_mask.storage());
    const float* input = tensor_storage->data_as<float>();
    const auto* mask_values = mask_storage->data_as<std::uint8_t>();
    const Strides output_strides = row_major_strides(tensor.shape());
    const Strides mask_strides = row_major_strides(mask.shape());
    std::vector<float> values(static_cast<std::size_t>(tensor.numel()));
    for (std::int64_t index = 0; index < tensor.numel(); ++index) {
        const auto mask_index = broadcast_index(index, tensor.shape(), output_strides, mask.shape(), mask_strides);
        values[static_cast<std::size_t>(index)] = mask_values[mask_index] ? value : input[index];
    }
    return from_vector(values, tensor.shape(), tensor.device());
}

Tensor add(const Tensor& tensor, float scalar) { return unary(tensor, [scalar](float x) { return x + scalar; }); }
Tensor add(float scalar, const Tensor& tensor) { return add(tensor, scalar); }
Tensor sub(const Tensor& tensor, float scalar) { return unary(tensor, [scalar](float x) { return x - scalar; }); }
Tensor sub(float scalar, const Tensor& tensor) { return unary(tensor, [scalar](float x) { return scalar - x; }); }
Tensor mul(const Tensor& tensor, float scalar) { return unary(tensor, [scalar](float x) { return x * scalar; }); }
Tensor mul(float scalar, const Tensor& tensor) { return mul(tensor, scalar); }
Tensor div(const Tensor& tensor, float scalar) { return unary(tensor, [scalar](float x) { return x / scalar; }); }
Tensor div(float scalar, const Tensor& tensor) { return unary(tensor, [scalar](float x) { return scalar / x; }); }

Tensor operator+(const Tensor& left, const Tensor& right) {
    return add(left, right);
}

Tensor operator-(const Tensor& left, const Tensor& right) {
    return sub(left, right);
}

Tensor operator*(const Tensor& left, const Tensor& right) {
    return matmul(left, right);
}

Tensor operator+(const Tensor& tensor, float scalar) { return add(tensor, scalar); }
Tensor operator+(float scalar, const Tensor& tensor) { return add(scalar, tensor); }
Tensor operator-(const Tensor& tensor, float scalar) { return sub(tensor, scalar); }
Tensor operator-(float scalar, const Tensor& tensor) { return sub(scalar, tensor); }
Tensor operator*(const Tensor& tensor, float scalar) { return mul(tensor, scalar); }
Tensor operator*(float scalar, const Tensor& tensor) { return mul(scalar, tensor); }
Tensor operator/(const Tensor& left, const Tensor& right) { return div(left, right); }
Tensor operator/(const Tensor& tensor, float scalar) { return div(tensor, scalar); }
Tensor operator/(float scalar, const Tensor& tensor) { return div(scalar, tensor); }

} // namespace citrius
