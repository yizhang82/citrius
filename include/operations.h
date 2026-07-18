#pragma once

#include "tensor.h"

namespace citrius {

/// Adds two tensors element by element.
/// @param left Left input tensor.
/// @param right Right input tensor with the same shape, dtype, and device as `left`.
/// @return A new tensor containing `left + right`, with the inputs' shape and device.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, has an unsupported dtype,
///         or has a different shape from the other input.
/// @code
/// Tensor result = add(from_vector({1, 2}), from_vector({10, 20})); // [11, 22]
/// @endcode
Tensor add(const Tensor& left, const Tensor& right);

/// Subtracts two tensors element by element.
/// @param left Left input tensor.
/// @param right Right input tensor with the same shape, dtype, and device as `left`.
/// @return A new tensor containing `left - right`, with the inputs' shape and device.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, has an unsupported dtype,
///         or has a different shape from the other input.
/// @code
/// Tensor result = sub(from_vector({10, 20}), from_vector({1, 2})); // [9, 18]
/// @endcode
Tensor sub(const Tensor& left, const Tensor& right);

/// Multiplies two matrices.
/// @param left A 2D input shaped `[M, K]`.
/// @param right A 2D input shaped `[K, N]` on the same device as `left`.
/// @return A new tensor shaped `[M, N]` on the inputs' device.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, is not 2D Float32, or
///         has an incompatible inner dimension.
/// @code
/// Tensor a = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
/// Tensor b = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
/// Tensor result = matmul(a, b); // shape [2, 2]
/// @endcode
Tensor matmul(const Tensor& left, const Tensor& right);

/// Adds two tensors by delegating to `citrius::add`.
/// @param left Left input tensor.
/// @param right Right input tensor.
/// @return The result returned by `citrius::add`.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs do not satisfy `citrius::add`.
/// @code
/// Tensor result = left + right;
/// @endcode
Tensor operator+(const Tensor& left, const Tensor& right);

/// Subtracts two tensors by delegating to `citrius::sub`.
/// @param left Left input tensor.
/// @param right Right input tensor.
/// @return The result returned by `citrius::sub`.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs do not satisfy `citrius::sub`.
/// @code
/// Tensor result = left - right;
/// @endcode
Tensor operator-(const Tensor& left, const Tensor& right);

/// Multiplies two tensors by delegating to `citrius::matmul`.
/// @param left Left matrix input.
/// @param right Right matrix input.
/// @return The result returned by `citrius::matmul`.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs do not satisfy `citrius::matmul`.
/// @code
/// Tensor result = left * right; // Matrix multiplication, not elementwise multiplication.
/// @endcode
Tensor operator*(const Tensor& left, const Tensor& right);

} // namespace citrius
