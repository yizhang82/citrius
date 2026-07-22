#include "nn/functional.h"

#include "exceptions.h"
#include "indexing_operations.h"
#include "operations.h"
#include "reduction_operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#endif
#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#endif

#include <cmath>
#include <stdexcept>
#include <vector>

namespace citrius::nn::functional {

namespace {

Tensor stable_tanh(const Tensor& tensor) {
    const Tensor zero = mul(tensor, 0.0f);
    const Tensor positive = maximum(tensor, zero);
    const Tensor negative = maximum(mul(tensor, -1.0f), zero);

    const auto tanh_magnitude = [](const Tensor& magnitude) {
        const Tensor exponential = exp(mul(magnitude, -2.0f));
        return div(sub(1.0f, exponential), add(1.0f, exponential));
    };

    return sub(tanh_magnitude(positive), tanh_magnitude(negative));
}

} // namespace

Tensor relu(const Tensor& tensor) {
    return maximum(tensor, mul(tensor, 0.0f));
}

Tensor gelu(const Tensor& tensor) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float cubic_coefficient = 0.044715f;

    const Tensor cubic = pow(tensor, 3.0f);
    const Tensor inner = mul(add(tensor, mul(cubic, cubic_coefficient)), sqrt_two_over_pi);
    return mul(mul(tensor, 0.5f), add(stable_tanh(inner), 1.0f));
}

Tensor layer_norm(
    const Tensor& tensor,
    const Shape& normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    float eps) {
    ENSURE_TENSOR_DEFINED(tensor);
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
    if (normalized_shape.empty()) {
        throw std::invalid_argument("layer_norm normalized_shape cannot be empty");
    }
    if (!(eps > 0.0f)) throw std::invalid_argument("layer_norm epsilon must be positive");
    if (normalized_shape.size() > tensor.ndim()) {
        throw std::invalid_argument("layer_norm normalized_shape exceeds input rank");
    }

    const std::size_t offset = tensor.ndim() - normalized_shape.size();
    std::vector<std::int64_t> dims;
    dims.reserve(normalized_shape.size());
    for (std::size_t index = 0; index < normalized_shape.size(); ++index) {
        if (normalized_shape[index] <= 0) {
            throw std::invalid_argument("layer_norm normalized dimensions must be positive");
        }
        if (tensor.shape()[offset + index] != normalized_shape[index]) {
            throw std::invalid_argument(
                "layer_norm normalized_shape must match the input's trailing dimensions");
        }
        dims.push_back(static_cast<std::int64_t>(offset + index));
    }

    const auto validate_affine = [&](const Tensor& parameter, const char* name) {
        if (!parameter.defined()) return;
        ENSURE_TENSOR_DTYPE(parameter, DType::Float32);
        ENSURE_TENSOR_SHAPE(parameter, normalized_shape);
        ENSURE_TENSOR_DEVICE_MATCH_2(parameter, tensor);
    };
    validate_affine(weight, "weight");
    validate_affine(bias, "bias");

    const Tensor centered = sub(tensor, mean(tensor, dims, true));
    Tensor output = div(centered, sqrt(add(variance(tensor, dims, true), eps)));
    if (weight.defined()) output = mul(output, weight);
    if (bias.defined()) output = add(output, bias);
    return output;
}

Tensor softmax(const Tensor& tensor, std::int64_t dim) {
    ENSURE_TENSOR_DEFINED(tensor);
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
    if (dim < 0) dim += static_cast<std::int64_t>(tensor.ndim());
    if (dim < 0 || dim >= static_cast<std::int64_t>(tensor.ndim()))
        throw std::out_of_range("softmax dimension is out of range");
#ifdef CITRIUS_HAS_METAL
    if (tensor.device().type == DeviceType::Metal &&
        dim == static_cast<std::int64_t>(tensor.ndim()) - 1 && tensor.is_contiguous())
        return impl::MetalDeviceImpl().softmax_last_dimension(tensor);
#endif
    const Tensor shifted = sub(tensor, citrius::max(tensor, dim, true));
    const Tensor exponentials = citrius::exp(shifted);
    return div(exponentials, sum(exponentials, dim, true));
}

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask,
    bool is_causal) {
    ENSURE_TENSOR_DEFINED(query);
    ENSURE_TENSOR_DTYPE(query, DType::Float32);
    if (query.shape().size() < 2) {
        throw std::invalid_argument("query must have at least 2 dimensions");
    }
    ENSURE_TENSOR_DEFINED(key);
    ENSURE_TENSOR_DTYPE(key, DType::Float32);
    if (key.shape().size() < 2) {
        throw std::invalid_argument("key must have at least 2 dimensions");
    }
    ENSURE_TENSOR_DEFINED(value);
    ENSURE_TENSOR_DTYPE(value, DType::Float32);
    if (value.shape().size() < 2) {
        throw std::invalid_argument("value must have at least 2 dimensions");
    }
    ENSURE_TENSOR_DEVICE_MATCH_3(query, key, value);
#ifdef CITRIUS_HAS_CUDA
    if (query.device().type == DeviceType::CUDA) {
        impl::CudaDeviceImpl device(query.device().index);
        if (auto optimized =
                device.try_scaled_dot_product_attention(
                    query, key, value, attn_mask, is_causal))
            return *optimized;
    }
#endif
    Tensor effective_key = key;
    Tensor effective_value = value;
    if (query.ndim() == 4 && key.ndim() == 4 && value.ndim() == 4 &&
        query.shape()[0] == key.shape()[0] && key.shape()[1] == value.shape()[1] &&
        query.shape()[1] != key.shape()[1]) {
        if (query.shape()[1] % key.shape()[1] != 0)
            throw std::invalid_argument("query heads must be divisible by key/value heads");
        const auto repetitions = query.shape()[1] / key.shape()[1];
        const auto repeat_heads = [repetitions](const Tensor& tensor) {
            const Tensor packed = contiguous(tensor);
            std::vector<Tensor> repeated;
            repeated.reserve(static_cast<std::size_t>(packed.shape()[1] * repetitions));
            for (std::int64_t head = 0; head < packed.shape()[1]; ++head) {
                const Tensor view = packed.index(
                    {indexing::Slice(), indexing::Slice(head, head + 1)});
                for (std::int64_t index = 0; index < repetitions; ++index)
                    repeated.push_back(view);
            }
            return concat(repeated, 1);
        };
        effective_key = repeat_heads(key);
        effective_value = repeat_heads(value);
    }
    // key -> [..., K, H]
    // key_transposed -> [..., H, K]
    auto key_transposed = citrius::transpose(effective_key, -2, -1);
    // query -> [..., Q, H]
    // scores -> [..., Q, K]
    auto scores = citrius::matmul(query, key_transposed); 

    if (attn_mask.defined()) {
        ENSURE_TENSOR_DTYPE(attn_mask, DType::Bool);
        ENSURE_TENSOR_DEVICE_MATCH_2(attn_mask, scores);

        // mask based on the attention (casual attention, self attention, etc)
        scores = citrius::masked_fill(
            scores,
            attn_mask,
            -std::numeric_limits<float>::infinity());
    }

    if (is_causal) {
        const auto query_length = query.shape()[query.ndim() - 2];
        const auto key_length = key.shape()[key.ndim() - 2];
        std::vector<bool> causal_values(
            static_cast<std::size_t>(query_length * key_length), false);
        for (std::int64_t row = 0; row < query_length; ++row) {
            for (std::int64_t column = row + 1; column < key_length; ++column)
                causal_values[static_cast<std::size_t>(row * key_length + column)] = true;
        }
        scores = citrius::masked_fill(
            scores, from_vector(causal_values, {query_length, key_length}, query.device()),
            -std::numeric_limits<float>::infinity());
    }

    auto scores_div = citrius::div(scores, std::sqrt(static_cast<float>(query.shape().back())));

    // probabilities -> [..., Q, K]
    auto probabilities = citrius::nn::functional::softmax(scores_div, -1);

    // value -> [..., K, V]
    // output -> [..., Q, V]
    return citrius::matmul(probabilities, effective_value);
}

} // namespace citrius::nn::functional
