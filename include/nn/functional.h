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
