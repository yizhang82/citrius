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

} // namespace citrius::nn::functional
