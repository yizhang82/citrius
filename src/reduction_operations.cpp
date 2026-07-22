#include "reduction_operations.h"

#include "impl/cpu_storage.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace citrius {
namespace {

enum class Reduction { Sum, Mean, Max, Variance };

void require_float32(const Tensor& tensor) {
    ENSURE_TENSOR_DEFINED(tensor);
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
}

std::vector<std::int64_t> all_dims(const Tensor& tensor) {
    std::vector<std::int64_t> dims(tensor.ndim());
    std::iota(dims.begin(), dims.end(), std::int64_t{0});
    return dims;
}

std::vector<std::int64_t> normalize_dims(
    const Tensor& tensor,
    std::vector<std::int64_t> dims) {
    if (dims.empty()) {
        if (tensor.ndim() == 0) return dims;
        throw std::invalid_argument("reduction dimensions cannot be empty");
    }
    for (auto& dim : dims) {
        if (dim < 0) dim += static_cast<std::int64_t>(tensor.ndim());
        if (dim < 0 || dim >= static_cast<std::int64_t>(tensor.ndim())) {
            throw std::out_of_range("reduction dimension is out of range");
        }
    }
    std::sort(dims.begin(), dims.end());
    if (std::adjacent_find(dims.begin(), dims.end()) != dims.end()) {
        throw std::invalid_argument("reduction dimensions must be unique");
    }
    return dims;
}

Strides strides(const Shape& shape) {
    Strides result(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        result[index - 1] = result[index] * shape[index];
    }
    return result;
}

Tensor reduce(
    const Tensor& tensor,
    std::vector<std::int64_t> dims,
    bool keepdim,
    Reduction reduction) {
    require_float32(tensor);
    dims = normalize_dims(tensor, std::move(dims));
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA) {
        impl::CudaReductionOperation cuda_operation;
        switch (reduction) {
            case Reduction::Sum: cuda_operation = impl::CudaReductionOperation::Sum; break;
            case Reduction::Mean: cuda_operation = impl::CudaReductionOperation::Mean; break;
            case Reduction::Max: cuda_operation = impl::CudaReductionOperation::Maximum; break;
            case Reduction::Variance:
                cuda_operation = impl::CudaReductionOperation::Variance;
                break;
        }
        return impl::CudaDeviceImpl(tensor.device().index)
            .reduce(tensor, dims, keepdim, cuda_operation);
    }
#endif
    std::vector<bool> reduced(tensor.ndim(), false);
    for (const auto dim : dims) reduced[static_cast<std::size_t>(dim)] = true;

    Shape output_shape;
    for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
        if (reduced[axis]) {
            if (keepdim) output_shape.push_back(1);
        } else {
            output_shape.push_back(tensor.shape()[axis]);
        }
    }
    std::int64_t output_count = 1;
    for (const auto size : output_shape) output_count *= size;
    const float initial = reduction == Reduction::Max
        ? -std::numeric_limits<float>::infinity()
        : 0.0f;
    std::vector<float> output(static_cast<std::size_t>(output_count), initial);
    std::vector<float> means;
    if (reduction == Reduction::Variance) {
        means.assign(static_cast<std::size_t>(output_count), 0.0f);
    }

    const Tensor cpu = tensor.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu.storage());
    const float* input = storage->data_as<float>();
    const Strides input_strides = strides(tensor.shape());
    const Strides output_strides = strides(output_shape);
    std::int64_t reduced_count = 1;
    for (const auto dim : dims) reduced_count *= tensor.shape()[static_cast<std::size_t>(dim)];

    auto output_index_for = [&](std::int64_t input_index) {
        std::int64_t output_index = 0;
        std::size_t output_axis = 0;
        for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
            const auto coordinate = (input_index / input_strides[axis]) % tensor.shape()[axis];
            if (reduced[axis]) {
                if (keepdim) ++output_axis;
            } else {
                output_index += coordinate * output_strides[output_axis++];
            }
        }
        return output_index;
    };

    if (reduction == Reduction::Variance) {
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            means[static_cast<std::size_t>(output_index_for(index))] += input[index];
        }
        for (float& value : means) value /= static_cast<float>(reduced_count);
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            const auto output_index = output_index_for(index);
            const float difference = input[index] - means[static_cast<std::size_t>(output_index)];
            output[static_cast<std::size_t>(output_index)] += difference * difference;
        }
        for (float& value : output) value /= static_cast<float>(reduced_count);
    } else {
        for (std::int64_t index = 0; index < tensor.numel(); ++index) {
            float& destination = output[static_cast<std::size_t>(output_index_for(index))];
            if (reduction == Reduction::Max) destination = std::max(destination, input[index]);
            else destination += input[index];
        }
        if (reduction == Reduction::Mean) {
            for (float& value : output) value /= static_cast<float>(reduced_count);
        }
    }
    return from_vector(output, output_shape, tensor.device());
}

Tensor argmax_cpu(const Tensor& tensor, std::int64_t dim, bool keepdim) {
    const auto normalized = normalize_dims(tensor, {dim});
    const auto axis = static_cast<std::size_t>(normalized.front());
    if (tensor.shape()[axis] == 0) throw std::invalid_argument("argmax cannot reduce an empty dimension");

    Shape output_shape = tensor.shape();
    if (keepdim) output_shape[axis] = 1;
    else output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    const Tensor packed = contiguous(tensor);
    const Tensor cpu = packed.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu.storage());
    const float* input = storage->data_as<float>() + cpu.storage_offset();
    const auto inner = std::accumulate(
        tensor.shape().begin() + static_cast<std::ptrdiff_t>(axis + 1), tensor.shape().end(),
        std::int64_t{1}, std::multiplies<>());
    const auto reduction_size = tensor.shape()[axis];
    std::vector<std::int64_t> output(static_cast<std::size_t>(tensor.numel() / reduction_size));
    for (std::int64_t output_index = 0; output_index < static_cast<std::int64_t>(output.size()); ++output_index) {
        const auto base = (output_index / inner) * reduction_size * inner + output_index % inner;
        float best_value = input[base];
        std::int64_t best_index = 0;
        for (std::int64_t index = 1; index < reduction_size; ++index) {
            const float value = input[base + index * inner];
            if (value > best_value) {
                best_value = value;
                best_index = index;
            }
        }
        output[static_cast<std::size_t>(output_index)] = best_index;
    }
    return from_vector(output, output_shape, tensor.device());
}

} // namespace

#define CITRIUS_REDUCTION_OVERLOADS(name, kind) \
Tensor name(const Tensor& tensor) { return reduce(tensor, all_dims(tensor), false, Reduction::kind); } \
Tensor name(const Tensor& tensor, std::int64_t dim, bool keepdim) { return reduce(tensor, {dim}, keepdim, Reduction::kind); } \
Tensor name(const Tensor& tensor, std::vector<std::int64_t> dims, bool keepdim) { return reduce(tensor, std::move(dims), keepdim, Reduction::kind); }

CITRIUS_REDUCTION_OVERLOADS(sum, Sum)
CITRIUS_REDUCTION_OVERLOADS(mean, Mean)
CITRIUS_REDUCTION_OVERLOADS(max, Max)
CITRIUS_REDUCTION_OVERLOADS(variance, Variance)

#undef CITRIUS_REDUCTION_OVERLOADS

Tensor argmax(const Tensor& tensor) {
    require_float32(tensor);
    if (tensor.numel() == 0) throw std::invalid_argument("argmax input cannot be empty");
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return impl::CudaDeviceImpl(tensor.device().index).argmax(tensor);
#endif
    const Tensor flattened = reshape(contiguous(tensor), {tensor.numel()});
    return argmax_cpu(flattened, 0, false);
}

Tensor argmax(const Tensor& tensor, std::int64_t dim, bool keepdim) {
    require_float32(tensor);
    const auto normalized = normalize_dims(tensor, {dim});
#ifdef CITRIUS_HAS_CUDA
    if (tensor.device().type == DeviceType::CUDA)
        return impl::CudaDeviceImpl(tensor.device().index).argmax(tensor, normalized.front(), keepdim);
#endif
    return argmax_cpu(tensor, normalized.front(), keepdim);
}

} // namespace citrius
