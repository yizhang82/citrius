#pragma once

#include "tensor.h"

#include <cstdint>

namespace citrius::nn::functional {

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
