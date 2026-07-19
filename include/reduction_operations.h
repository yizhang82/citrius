#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace citrius {

/// Sums all elements of a Float32 tensor.
/// @param tensor Defined Float32 input tensor.
/// @return A new Float32 tensor on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor total = sum(from_vector(std::vector<float>{1, 2, 3})); // scalar 6
/// @endcode
Tensor sum(const Tensor& tensor);

/// Sums a Float32 tensor along one dimension.
/// @param tensor Defined Float32 input tensor.
/// @param dim Dimension to reduce; negative dimensions are accepted.
/// @param keepdim If true, retains `dim` with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If the input is undefined or non-Float32.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor columns = sum(matrix, 0); // [M, N] -> [N]
/// @endcode
Tensor sum(const Tensor& tensor, std::int64_t dim, bool keepdim = false);

/// Sums a Float32 tensor along multiple unique dimensions.
/// @param tensor Defined Float32 input tensor.
/// @param dims Nonempty unique dimensions; negative dimensions are accepted.
/// @param keepdim If true, retains reduced dimensions with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If the input or dimension list is invalid.
/// @throws std::out_of_range If a dimension is outside the tensor rank.
/// @code
/// Tensor totals = sum(input, std::vector<std::int64_t>{1, 2}, true);
/// @endcode
Tensor sum(const Tensor& tensor, std::vector<std::int64_t> dims, bool keepdim = false);

/// Computes the arithmetic mean of all tensor elements.
/// @param tensor Defined Float32 input tensor.
/// @return A scalar Float32 tensor on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor average = mean(input);
/// @endcode
Tensor mean(const Tensor& tensor);
/// Computes the arithmetic mean along one dimension.
/// @param tensor Defined Float32 input tensor.
/// @param dim Dimension to reduce; negative dimensions are accepted.
/// @param keepdim If true, retains `dim` with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor row_means = mean(matrix, -1, true);
/// @endcode
Tensor mean(const Tensor& tensor, std::int64_t dim, bool keepdim = false);
/// Computes the arithmetic mean along multiple unique dimensions.
/// @param tensor Defined Float32 input tensor.
/// @param dims Nonempty unique dimensions; negative dimensions are accepted.
/// @param keepdim If true, retains reduced dimensions with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If the input or dimension list is invalid.
/// @throws std::out_of_range If a dimension is outside the tensor rank.
/// @code
/// Tensor average = mean(input, std::vector<std::int64_t>{0, 2});
/// @endcode
Tensor mean(const Tensor& tensor, std::vector<std::int64_t> dims, bool keepdim = false);

/// Returns the maximum value across all tensor elements.
/// @param tensor Defined Float32 input tensor.
/// @return A scalar Float32 tensor containing the maximum.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor largest = max(input);
/// @endcode
Tensor max(const Tensor& tensor);
/// Returns maximum values along one dimension.
/// @param tensor Defined Float32 input tensor.
/// @param dim Dimension to reduce; negative dimensions are accepted.
/// @param keepdim If true, retains `dim` with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor row_maxima = max(matrix, -1, true);
/// @endcode
Tensor max(const Tensor& tensor, std::int64_t dim, bool keepdim = false);
/// Returns maximum values along multiple unique dimensions.
/// @param tensor Defined Float32 input tensor.
/// @param dims Nonempty unique dimensions; negative dimensions are accepted.
/// @param keepdim If true, retains reduced dimensions with size one.
/// @return A new reduced tensor on the input device.
/// @throws std::invalid_argument If the input or dimension list is invalid.
/// @throws std::out_of_range If a dimension is outside the tensor rank.
/// @code
/// Tensor maxima = max(input, std::vector<std::int64_t>{1, 2});
/// @endcode
Tensor max(const Tensor& tensor, std::vector<std::int64_t> dims, bool keepdim = false);

/// Returns the index of the maximum element in the flattened tensor.
/// Ties are resolved by returning the first index.
/// @throws std::invalid_argument If the Float32 input is undefined or empty.
Tensor argmax(const Tensor& tensor);

/// Returns indices of maximum values along one dimension as Int64 values.
/// Negative dimensions are accepted and ties return the first index.
/// @throws std::invalid_argument If the Float32 input is undefined or the reduced dimension is empty.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
Tensor argmax(const Tensor& tensor, std::int64_t dim, bool keepdim = false);

/// Computes population variance across all tensor elements.
/// @param tensor Defined Float32 input tensor.
/// @return A scalar Float32 population variance on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @code
/// Tensor spread = variance(input);
/// @endcode
Tensor variance(const Tensor& tensor);
/// Computes population variance along one dimension.
/// @param tensor Defined Float32 input tensor.
/// @param dim Dimension to reduce; negative dimensions are accepted.
/// @param keepdim If true, retains `dim` with size one.
/// @return Population variance along `dim`, dividing by its element count.
/// @throws std::invalid_argument If `tensor` is undefined or non-Float32.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor row_variance = variance(matrix, -1, true);
/// @endcode
Tensor variance(const Tensor& tensor, std::int64_t dim, bool keepdim = false);
/// Computes population variance along multiple unique dimensions.
/// @param tensor Defined Float32 input tensor.
/// @param dims Nonempty unique dimensions; negative dimensions are accepted.
/// @param keepdim If true, retains reduced dimensions with size one.
/// @return Population variance, dividing by the number of reduced elements.
/// @throws std::invalid_argument If the input or dimension list is invalid.
/// @throws std::out_of_range If a dimension is outside the tensor rank.
/// @code
/// Tensor stats = variance(input, {-2, -1}, true);
/// @endcode
Tensor variance(const Tensor& tensor, std::vector<std::int64_t> dims, bool keepdim = false);

} // namespace citrius
