#pragma once

#include "tensor.h"

namespace citrius {

struct AddRmsNormResult {
    Tensor residual;
    Tensor normalized;
};

/// Adds two tensors element by element.
/// @param left Left input tensor.
/// @param right Right input tensor with the same shape, dtype, and device as `left`.
/// @return A new tensor containing `left + right`, with the inputs' shape and device.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, has an unsupported dtype,
///         or has a different shape from the other input.
/// @code
/// Tensor result = add(from_vector(std::vector<float>{1, 2}),
///                     from_vector(std::vector<float>{10, 20})); // [11, 22]
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
/// Tensor result = sub(from_vector(std::vector<float>{10, 20}),
///                     from_vector(std::vector<float>{1, 2})); // [9, 18]
/// @endcode
Tensor sub(const Tensor& left, const Tensor& right);

/// Multiplies matrices, dispatching to regular or batched device execution.
/// @param left A Float32 input shaped `[M, K]` or `[..., M, K]`.
/// @param right A Float32 input shaped `[K, N]` or `[..., K, N]` on the same device.
/// @return A new tensor shaped `[M, N]` for 2D inputs or `[..., M, N]` with
///         broadcasted leading batch dimensions.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If an input is undefined, has rank below two, is
///         non-Float32, or has incompatible inner or batch dimensions.
/// @throws std::runtime_error If batched execution is unavailable on the device.
/// @code
/// Tensor a = from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
/// Tensor b = from_vector(std::vector<float>{1, 2, 3, 4, 5, 6}, {3, 2});
/// Tensor result = matmul(a, b); // shape [2, 2]
/// Tensor scores = matmul(query, transpose(key, -2, -1)); // [B, H, S, S]
/// @endcode
Tensor matmul(const Tensor& left, const Tensor& right);

/// Converts floating-point tensor elements to another floating-point dtype.
Tensor cast(const Tensor& tensor, DType dtype);

/// Applies root-mean-square normalization over the last dimension.
/// A device may provide a fused implementation; otherwise this uses the
/// portable tensor-operation composition.
Tensor rms_norm(const Tensor& input, const Tensor& weight, float epsilon);

/// Applies the SwiGLU activation `silu(gate) * up` elementwise.
/// A device may provide a fused implementation.
Tensor swiglu(const Tensor& gate, const Tensor& up);

/// Applies per-head RMSNorm and rotary position embedding to a contiguous
/// `[batch, sequence, heads, head_dim]` tensor, returning
/// `[batch, heads, sequence, head_dim]` in attention-ready layout.
Tensor rms_norm_rope(
    const Tensor& input,
    const Tensor& weight,
    float epsilon,
    float theta);

/// Adds two tensors and RMS-normalizes the sum, returning both values so a
/// residual connection does not require a second materialization.
AddRmsNormResult add_rms_norm(
    const Tensor& left,
    const Tensor& right,
    const Tensor& weight,
    float epsilon);

/// Multiplies two tensors element by element using trailing-dimension broadcasting.
/// @param left Left Float32 input tensor.
/// @param right Right Float32 input tensor on the same device.
/// @return A new tensor with the broadcasted shape.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs are undefined, non-Float32, or not broadcastable.
/// @code
/// Tensor result = mul(from_vector(std::vector<float>{1, 2}),
///                     from_vector(std::vector<float>{10, 20})); // [10, 40]
/// @endcode
Tensor mul(const Tensor& left, const Tensor& right);

/// Divides two tensors element by element using trailing-dimension broadcasting.
/// @param left Numerator Float32 tensor.
/// @param right Denominator Float32 tensor on the same device.
/// @return A new tensor with the broadcasted shape.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs are undefined, non-Float32, or not broadcastable.
/// @code
/// Tensor result = div(from_vector(std::vector<float>{10, 20}),
///                     from_vector(std::vector<float>{2})); // [5, 10]
/// @endcode
Tensor div(const Tensor& left, const Tensor& right);

/// Returns the elementwise maximum using trailing-dimension broadcasting.
/// @param left Left Float32 input tensor.
/// @param right Right Float32 input tensor on the same device.
/// @return A new tensor containing the broadcasted elementwise maxima.
/// @throws DeviceMismatchException If the input devices differ.
/// @throws std::invalid_argument If the inputs are undefined, non-Float32, or not broadcastable.
/// @code
/// Tensor result = maximum(input, from_vector(std::vector<float>{0}));
/// @endcode
Tensor maximum(const Tensor& left, const Tensor& right);

/// Computes the elementwise exponential of a Float32 tensor.
/// @param tensor Defined Float32 input tensor.
/// @return A new tensor with the same shape and device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor result = exp(from_vector(std::vector<float>{0, 1})); // [1, e]
/// @endcode
Tensor exp(const Tensor& tensor);

/// Computes the elementwise square root of a Float32 tensor.
/// @param tensor Defined Float32 input tensor.
/// @return A new tensor with the same shape and device; negative inputs produce NaN.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor result = sqrt(from_vector(std::vector<float>{1, 4})); // [1, 2]
/// @endcode
Tensor sqrt(const Tensor& tensor);

/// Raises each element of a Float32 tensor to `exponent`.
/// @param tensor Defined Float32 input tensor.
/// @param exponent Scalar exponent applied to every element.
/// @return A new tensor with the same shape and device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor result = pow(from_vector(std::vector<float>{2, 3}), 2.0f); // [4, 9]
/// @endcode
Tensor pow(const Tensor& tensor, float exponent);

/// Replaces input values where a broadcastable Bool mask is true.
/// @param tensor Defined Float32 input tensor.
/// @param mask Defined Bool tensor on the same device that broadcasts to `tensor.shape()`.
/// @param value Replacement value.
/// @return A new Float32 tensor with the input shape and device.
/// @throws DeviceMismatchException If the tensor and mask devices differ.
/// @throws std::invalid_argument If the inputs have invalid dtypes or incompatible shapes.
/// @code
/// Tensor result = masked_fill(input, from_vector(std::vector<bool>{false, true}), -1.0f);
/// @endcode
Tensor masked_fill(const Tensor& tensor, const Tensor& mask, float value);

/// Adds a scalar to every tensor element.
/// @return A new Float32 tensor with the input shape and device.
/// @throws std::invalid_argument If the tensor is undefined or non-Float32.
Tensor add(const Tensor& tensor, float scalar);
Tensor add(float scalar, const Tensor& tensor);
/// Subtracts a scalar from every tensor element.
/// @return A new Float32 tensor with the input shape and device.
/// @throws std::invalid_argument If the tensor is undefined or non-Float32.
Tensor sub(const Tensor& tensor, float scalar);
Tensor sub(float scalar, const Tensor& tensor);
/// Multiplies every tensor element by a scalar.
/// @return A new Float32 tensor with the input shape and device.
/// @throws std::invalid_argument If the tensor is undefined or non-Float32.
Tensor mul(const Tensor& tensor, float scalar);
Tensor mul(float scalar, const Tensor& tensor);
/// Divides every tensor element by a scalar.
/// @return A new Float32 tensor with the input shape and device.
/// @throws std::invalid_argument If the tensor is undefined or non-Float32.
/// @code
/// Tensor scaled = div(input, 2.0f); // Equivalent to input / 2.0f.
/// @endcode
Tensor div(const Tensor& tensor, float scalar);
Tensor div(float scalar, const Tensor& tensor);

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
Tensor operator+(const Tensor& tensor, float scalar);
Tensor operator+(float scalar, const Tensor& tensor);
Tensor operator-(const Tensor& tensor, float scalar);
Tensor operator-(float scalar, const Tensor& tensor);
Tensor operator*(const Tensor& tensor, float scalar);
Tensor operator*(float scalar, const Tensor& tensor);
Tensor operator/(const Tensor& left, const Tensor& right);
Tensor operator/(const Tensor& tensor, float scalar);
Tensor operator/(float scalar, const Tensor& tensor);

} // namespace citrius
