#include "operations.h"

#include "impl/cpu_device.h"
#include "impl/multi_thread_cpu_device.h"
#include "impl/cpu_storage.h"
#include "exceptions.h"
#include "indexing_operations.h"
#include "reduction_operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

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
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
    ENSURE_TENSOR_DEVICE_MATCH_2(left, right);
}

void require_float32(const Tensor& tensor) {
    ENSURE_TENSOR_DEFINED(tensor);
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
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
std::shared_ptr<impl::IDevice> cuda_device(int device_index) {
    std::string backend;
    const char* configured_backend = std::getenv("CITRIUS_CUDA_BACKEND");
    if (configured_backend) {
        backend = configured_backend;
        if (backend != "cublas" && backend != "reference" && backend != "cutlass") {
            throw CitriusException(
                "CITRIUS_CUDA_BACKEND must be 'cublas', 'cutlass', or 'reference', got '" +
                backend + "'");
        }
    } else {
#ifdef CITRIUS_CUDA_DEFAULT_CUBLAS
        backend = "cublas";
#else
        backend = "reference";
#endif
    }

    static std::shared_mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<impl::IDevice>> devices;
    const std::string key = backend + ':' + std::to_string(device_index);
    {
        std::shared_lock lock(mutex);
        if (const auto existing = devices.find(key); existing != devices.end())
            return existing->second;
    }

    std::unique_lock lock(mutex);
    if (const auto existing = devices.find(key); existing != devices.end())
        return existing->second;

    std::shared_ptr<impl::IDevice> device;
    if (backend == "cublas")
        device = std::make_shared<CublasCudaDeviceImpl>(device_index);
    else if (backend == "cutlass")
        device = std::make_shared<CutlassCudaDeviceImpl>(device_index);
    else
        device = std::make_shared<CudaDeviceImpl>(device_index);
    devices.emplace(key, device);
    return device;
}

Tensor cuda_broadcast_elementwise(
    const Tensor& left,
    const Tensor& right,
    impl::CudaElementwiseOperation operation) {
    require_matching_devices(left, right);
    require_float32(left);
    require_float32(right);
    auto device = cuda_device(left.device().index);
    return static_cast<impl::CudaDeviceImpl&>(*device).broadcast_elementwise(
        left, right, operation);
}

Tensor cuda_scalar_elementwise(
    const Tensor& tensor,
    float scalar,
    impl::CudaElementwiseOperation operation,
    bool scalar_is_left = false) {
    auto device = cuda_device(tensor.device().index);
    return static_cast<impl::CudaDeviceImpl&>(*device).scalar_elementwise(
        tensor, scalar, operation, scalar_is_left);
}
#endif

template <typename Operation>
auto dispatch(const Tensor& left, const Tensor& right, Operation operation) {
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
#ifdef CITRIUS_HAS_CUDA
        if (left.device().type == DeviceType::CUDA)
            return cuda_broadcast_elementwise(
                left, right, impl::CudaElementwiseOperation::Add);
#endif
        return broadcast_binary(left, right, std::plus<float>());
    }
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.add(a, b);
    });
}

Tensor sub(const Tensor& left, const Tensor& right) {
    if (left.shape() != right.shape()) {
#ifdef CITRIUS_HAS_CUDA
        if (left.device().type == DeviceType::CUDA)
            return cuda_broadcast_elementwise(
                left, right, impl::CudaElementwiseOperation::Subtract);
#endif
        return broadcast_binary(left, right, std::minus<float>());
    }
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.sub(a, b);
    });
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    if (left.ndim() != 2 || right.ndim() != 2) {
        const Tensor packed_left = left.is_contiguous() ? left : contiguous(left);
        const Tensor packed_right = right.is_contiguous() ? right : contiguous(right);
        return dispatch(packed_left, packed_right, [](const auto& device, const Tensor& a, const Tensor& b) {
            return device.batched_matmul(a, b);
        });
    }
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.matmul(a, b);
    });
}

Tensor cast(const Tensor& tensor, DType dtype) {
    ENSURE_TENSOR_DEFINED(tensor);
    if (!is_floating_point(tensor.dtype()) || tensor.dtype() == DType::Float64 ||
        !is_floating_point(dtype) || dtype == DType::Float64)
        throw std::invalid_argument("cast supports Float16, BFloat16, and Float32");
    if (tensor.dtype() == dtype) return tensor;
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return CudaDeviceImpl(tensor.device().index).cast(tensor, dtype);
#endif
    const Tensor packed = tensor.is_contiguous() ? tensor : contiguous(tensor);
    const Tensor cpu = packed.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu.storage());
    std::vector<float> values(static_cast<std::size_t>(cpu.numel()));
    if (cpu.dtype() == DType::Float32) {
        std::copy_n(storage->data_as<float>(), values.size(), values.begin());
    } else {
        const auto* bits = storage->data_as<std::uint16_t>();
        for (std::size_t index = 0; index < values.size(); ++index)
            values[index] = cpu.dtype() == DType::Float16
                ? float16_to_float(bits[index]) : bfloat16_to_float(bits[index]);
    }
    return from_vector(values, packed.shape(), dtype, tensor.device());
}

Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon) {
    require_matching_devices(input, weight);
    require_float32(input);
    require_float32(weight);
    if (input.ndim() == 0 || weight.shape() != Shape{input.shape().back()})
        throw std::invalid_argument("rms_norm weight must match the input's last dimension");
    if (!(epsilon > 0.0f))
        throw std::invalid_argument("rms_norm epsilon must be positive");

    const auto optimized = dispatch(
        input, weight, [epsilon](const auto& device, const Tensor& value, const Tensor& scale) {
            return device.try_rms_norm(value, scale, epsilon);
        });
    if (optimized) return *optimized;

    const Tensor variance = mean(pow(input, 2.0f), -1, true);
    return mul(div(input, sqrt(add(variance, epsilon))), weight);
}

Tensor swiglu(const Tensor& gate, const Tensor& up) {
    require_matching_devices(gate, up);
    require_float32(gate);
    require_float32(up);
    if (gate.shape() != up.shape())
        throw std::invalid_argument("swiglu inputs must have identical shapes");

    const auto optimized = dispatch(
        gate, up, [](const auto& device, const Tensor& gate_value, const Tensor& up_value) {
            return device.try_swiglu(gate_value, up_value);
        });
    if (optimized) return *optimized;
    return mul(div(gate, add(exp(mul(gate, -1.0f)), 1.0f)), up);
}

Tensor rms_norm_rope(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    float theta) {
    require_matching_devices(input, weight);
    require_float32(input);
    require_float32(weight);
    if (input.ndim() != 4 || input.shape().back() % 2 != 0 ||
        weight.shape() != Shape{input.shape().back()}) {
        throw std::invalid_argument(
            "rms_norm_rope requires [batch, sequence, heads, even head_dim] input");
    }
    if (!(epsilon > 0.0f) || !(theta > 0.0f))
        throw std::invalid_argument("rms_norm_rope epsilon and theta must be positive");

    const auto optimized = dispatch(
        input, weight,
        [epsilon, theta](const auto& device, const Tensor& value, const Tensor& scale) {
            return device.try_rms_norm_rope(value, scale, epsilon, theta);
        });
    if (optimized) return *optimized;

    const Tensor normalized = rms_norm(input, weight, epsilon);
    const Tensor packed = contiguous(permute(normalized, {0, 2, 1, 3}));
    const auto sequence = packed.shape()[2];
    const auto head_dim = packed.shape()[3];
    const auto half = head_dim / 2;
    std::vector<float> cosine(static_cast<std::size_t>(sequence * head_dim));
    std::vector<float> sine(cosine.size());
    for (std::int64_t position = 0; position < sequence; ++position) {
        for (std::int64_t index = 0; index < half; ++index) {
            const float frequency =
                1.0f / std::pow(theta, static_cast<float>(2 * index) / head_dim);
            const float angle = static_cast<float>(position) * frequency;
            cosine[static_cast<std::size_t>(position * head_dim + index)] = std::cos(angle);
            cosine[static_cast<std::size_t>(position * head_dim + half + index)] = std::cos(angle);
            sine[static_cast<std::size_t>(position * head_dim + index)] = std::sin(angle);
            sine[static_cast<std::size_t>(position * head_dim + half + index)] = std::sin(angle);
        }
    }
    const Tensor cos_tensor = from_vector(
        cosine, {1, 1, sequence, head_dim}, packed.device());
    const Tensor sin_tensor = from_vector(
        sine, {1, 1, sequence, head_dim}, packed.device());
    const Tensor first_half = contiguous(
        packed.index({indexing::Ellipsis, indexing::Slice(0, half)}));
    const Tensor second_half = contiguous(
        packed.index({indexing::Ellipsis, indexing::Slice(half, head_dim)}));
    const Tensor rotated = concat({mul(second_half, -1.0f), first_half}, -1);
    return add(mul(packed, cos_tensor), mul(rotated, sin_tensor));
}

AddRmsNormResult add_rms_norm(
    const Tensor& left,
    const Tensor& right,
    const Tensor& weight,
    float epsilon) {
    require_matching_devices(left, right);
    require_matching_devices(left, weight);
    require_float32(left);
    require_float32(right);
    require_float32(weight);
    if (left.shape() != right.shape() || left.ndim() == 0 ||
        weight.shape() != Shape{left.shape().back()})
        throw std::invalid_argument("add_rms_norm inputs have incompatible shapes");
    if (!(epsilon > 0.0f))
        throw std::invalid_argument("add_rms_norm epsilon must be positive");

    const auto optimized = dispatch(
        left, right,
        [&weight, epsilon](const auto& device, const Tensor& a, const Tensor& b) {
            return device.try_add_rms_norm(a, b, weight, epsilon);
        });
    if (optimized) return {optimized->first, optimized->second};
    Tensor residual = add(left, right);
    return {residual, rms_norm(residual, weight, epsilon)};
}

Tensor mul(const Tensor& left, const Tensor& right) {
#ifdef CITRIUS_HAS_CUDA
    if (left.device().type == DeviceType::CUDA)
        return cuda_broadcast_elementwise(
            left, right, impl::CudaElementwiseOperation::Multiply);
#endif
    return broadcast_binary(left, right, std::multiplies<float>());
}

Tensor div(const Tensor& left, const Tensor& right) {
#ifdef CITRIUS_HAS_CUDA
    if (left.device().type == DeviceType::CUDA)
        return cuda_broadcast_elementwise(
            left, right, impl::CudaElementwiseOperation::Divide);
#endif
    return broadcast_binary(left, right, std::divides<float>());
}

Tensor maximum(const Tensor& left, const Tensor& right) {
#ifdef CITRIUS_HAS_CUDA
    if (left.device().type == DeviceType::CUDA)
        return cuda_broadcast_elementwise(
            left, right, impl::CudaElementwiseOperation::Maximum);
#endif
    return broadcast_binary(left, right, [](float a, float b) { return std::max(a, b); });
}

Tensor exp(const Tensor& tensor) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA) {
        auto device = cuda_device(tensor.device().index);
        return static_cast<impl::CudaDeviceImpl&>(*device).unary(
            tensor, impl::CudaUnaryOperation::Exp);
    }
#endif
    return unary(tensor, [](float value) { return std::exp(value); });
}

Tensor sqrt(const Tensor& tensor) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA) {
        auto device = cuda_device(tensor.device().index);
        return static_cast<impl::CudaDeviceImpl&>(*device).unary(
            tensor, impl::CudaUnaryOperation::Sqrt);
    }
#endif
    return unary(tensor, [](float value) { return std::sqrt(value); });
}

Tensor pow(const Tensor& tensor, float exponent) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA) {
        auto device = cuda_device(tensor.device().index);
        return static_cast<impl::CudaDeviceImpl&>(*device).unary(
            tensor, impl::CudaUnaryOperation::Power, exponent);
    }
#endif
    return unary(tensor, [exponent](float value) { return std::pow(value, exponent); });
}

Tensor masked_fill(const Tensor& tensor, const Tensor& mask, float value) {
    require_float32(tensor);
    ENSURE_TENSOR_DEFINED(mask);
    ENSURE_TENSOR_DTYPE(mask, DType::Bool);
    require_matching_devices(tensor, mask);
    if (broadcast_shape(tensor.shape(), mask.shape()) != tensor.shape()) {
        throw std::invalid_argument("masked_fill mask must broadcast to the input shape");
    }
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA) {
        auto device = cuda_device(tensor.device().index);
        return static_cast<impl::CudaDeviceImpl&>(*device).masked_fill(tensor, mask, value);
    }
#endif
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

Tensor add(const Tensor& tensor, float scalar) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(tensor, scalar, impl::CudaElementwiseOperation::Add);
#endif
    return unary(tensor, [scalar](float x) { return x + scalar; });
}
Tensor add(float scalar, const Tensor& tensor) { return add(tensor, scalar); }
Tensor sub(const Tensor& tensor, float scalar) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(tensor, scalar, impl::CudaElementwiseOperation::Subtract);
#endif
    return unary(tensor, [scalar](float x) { return x - scalar; });
}
Tensor sub(float scalar, const Tensor& tensor) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(
            tensor, scalar, impl::CudaElementwiseOperation::Subtract, true);
#endif
    return unary(tensor, [scalar](float x) { return scalar - x; });
}
Tensor mul(const Tensor& tensor, float scalar) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(tensor, scalar, impl::CudaElementwiseOperation::Multiply);
#endif
    return unary(tensor, [scalar](float x) { return x * scalar; });
}
Tensor mul(float scalar, const Tensor& tensor) { return mul(tensor, scalar); }
Tensor div(const Tensor& tensor, float scalar) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(tensor, scalar, impl::CudaElementwiseOperation::Divide);
#endif
    return unary(tensor, [scalar](float x) { return x / scalar; });
}
Tensor div(float scalar, const Tensor& tensor) {
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return cuda_scalar_elementwise(
            tensor, scalar, impl::CudaElementwiseOperation::Divide, true);
#endif
    return unary(tensor, [scalar](float x) { return scalar / x; });
}

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
