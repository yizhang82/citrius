#pragma once

#include "tensor.h"

#include <cstdint>

namespace citrius::nn::functional {

/// Applies the rectified linear unit function element by element.
/// @param tensor Defined Float32 input tensor.
/// @return A new tensor with the input shape and device containing `max(0, x)`.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
Tensor relu(const Tensor& tensor);

/// Applies the tanh approximation of the Gaussian error linear unit element by element.
/// The approximation is `0.5 * x * (1 + tanh(sqrt(2 / pi) *
/// (x + 0.044715 * x^3)))`.
/// @param tensor Defined Float32 input tensor.
/// @return A new tensor with the input shape and device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
Tensor gelu(const Tensor& tensor);

/// Normalizes the trailing dimensions of a Float32 tensor using population variance.
/// Optional weight and bias tensors must exactly match `normalized_shape` and are
/// broadcast over the leading dimensions.
/// @param tensor Defined Float32 input tensor.
/// @param normalized_shape Nonempty positive shape matching the input's trailing dimensions.
/// @param weight Optional elementwise scale tensor.
/// @param bias Optional elementwise bias tensor.
/// @param eps Positive value added to the variance for numerical stability.
/// @return A new tensor with the input shape and device.
/// @throws std::invalid_argument If shapes, dtypes, devices, or epsilon are invalid.
Tensor layer_norm(
    const Tensor& tensor,
    const Shape& normalized_shape,
    const Tensor& weight = Tensor(),
    const Tensor& bias = Tensor(),
    float eps = 1e-5f);

/// Computes numerically stable softmax along one dimension.
/// @param tensor Defined Float32 input tensor.
/// @param dim Dimension normalized into probabilities; negative dimensions are accepted.
/// @return A new tensor with the input shape and device whose slices sum to approximately one.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor probabilities = softmax(logits, -1);
/// @endcode
Tensor softmax(const Tensor& tensor, std::int64_t dim);

/// Computes scaled dot-product attention.
/// Query, key, and value may include any number of leading batch dimensions. Their
/// trailing dimensions must be `[query_length, head_dim]`,
/// `[key_length, head_dim]`, and `[key_length, value_dim]`, respectively.
/// @param attn_mask Optional Bool mask broadcastable to
///        `[..., query_length, key_length]`; `true` entries cannot be attended to.
/// @return A tensor shaped `[..., query_length, value_dim]`.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, non-Float32, or has
///         incompatible dimensions.
/// @code
/// Tensor output = scaled_dot_product_attention(query, key, value);
/// @endcode
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& attn_mask = Tensor(),
    bool is_causal = false);

} // namespace citrius::nn::functional
