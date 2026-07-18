#pragma once

#include "tensor.h"

#include <cstdint>
#include <vector>

namespace citrius {

/// Changes a tensor's shape without changing its element order.
/// @param tensor Defined contiguous input tensor.
/// @param shape Requested output shape; at most one dimension may be `-1` for inference.
/// @return A tensor with the requested shape that shares storage with `tensor`.
/// @throws std::invalid_argument If `tensor` is undefined, dimensions are invalid, or
///         the requested shape does not preserve the element count.
/// @code
/// Tensor matrix = reshape(from_vector({1, 2, 3, 4, 5, 6}), {2, -1}); // [2, 3]
/// @endcode
Tensor reshape(const Tensor& tensor, Shape shape);

/// Creates a storage-sharing view with a different shape.
/// @param tensor Defined contiguous input tensor.
/// @param shape Requested output shape; at most one dimension may be `-1` for inference.
/// @return A tensor sharing storage with `tensor`; currently equivalent to `reshape`.
/// @throws std::invalid_argument If `tensor` is undefined or `shape` is incompatible.
/// @code
/// Tensor flat = view(matrix, {6}); // Shares storage with matrix.
/// @endcode
Tensor view(const Tensor& tensor, Shape shape);

/// Collapses a consecutive range of dimensions.
/// @param tensor Defined input tensor.
/// @param start_dim First dimension to collapse; negative indices are accepted.
/// @param end_dim Last dimension to collapse, inclusive; negative indices are accepted.
/// @return A storage-sharing tensor with the selected dimensions replaced by their product.
/// @throws std::invalid_argument If `tensor` is undefined or `start_dim` follows `end_dim`.
/// @throws std::out_of_range If either dimension is outside the tensor rank.
/// @code
/// Tensor output = flatten(input, 1, 2); // [2, 3, 4] becomes [2, 12].
/// @endcode
Tensor flatten(const Tensor& tensor, std::int64_t start_dim = 0, std::int64_t end_dim = -1);

/// Inserts a size-one dimension.
/// @param tensor Defined input tensor.
/// @param dim Position at which to insert the dimension; negative indices are accepted.
/// @return A storage-sharing tensor with one additional dimension.
/// @throws std::invalid_argument If `tensor` is undefined.
/// @throws std::out_of_range If `dim` is not a valid insertion position.
/// @code
/// Tensor output = unsqueeze(input, 0); // [3, 4] becomes [1, 3, 4].
/// @endcode
Tensor unsqueeze(const Tensor& tensor, std::int64_t dim);

/// Removes all size-one dimensions.
/// @param tensor Defined input tensor.
/// @return A storage-sharing tensor with every dimension of size one removed.
/// @throws std::invalid_argument If `tensor` is undefined.
/// @code
/// Tensor output = squeeze(input); // [1, 3, 1] becomes [3].
/// @endcode
Tensor squeeze(const Tensor& tensor);

/// Removes one size-one dimension when possible.
/// @param tensor Defined input tensor.
/// @param dim Dimension to inspect; negative indices are accepted.
/// @return A storage-sharing tensor without `dim` when its size is one; otherwise
///         returns a shallow copy of `tensor` unchanged.
/// @throws std::invalid_argument If `tensor` is undefined.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor output = squeeze(input, 0); // [1, 3, 4] becomes [3, 4].
/// @endcode
Tensor squeeze(const Tensor& tensor, std::int64_t dim);

/// Reorders all tensor dimensions.
/// @param tensor Defined input tensor.
/// @param dimensions A permutation containing each input dimension exactly once;
///        negative indices are accepted.
/// @return A materialized contiguous tensor with reordered shape and values on the
///         same device as `tensor`.
/// @throws std::invalid_argument If `tensor` is undefined, the permutation rank is
///         wrong, or a dimension occurs more than once.
/// @throws std::out_of_range If a dimension is outside the tensor rank.
/// @code
/// Tensor heads = permute(input, {0, 2, 1, 3}); // [B, S, H, D] -> [B, H, S, D].
/// @endcode
Tensor permute(const Tensor& tensor, std::vector<std::int64_t> dimensions);

/// Exchanges two tensor dimensions.
/// @param tensor Defined input tensor.
/// @param dim0 First dimension; negative indices are accepted.
/// @param dim1 Second dimension; negative indices are accepted.
/// @return A materialized contiguous tensor with the two dimensions exchanged.
/// @throws std::invalid_argument If `tensor` is undefined.
/// @throws std::out_of_range If either dimension is outside the tensor rank.
/// @code
/// Tensor keys_t = transpose(keys, -2, -1); // [B, H, S, D] -> [B, H, D, S].
/// @endcode
Tensor transpose(const Tensor& tensor, std::int64_t dim0, std::int64_t dim1);

/// Produces a tensor with contiguous element storage.
/// @param tensor Defined input tensor.
/// @return A shallow copy sharing storage with `tensor`, because all current Citrius
///         tensors are already contiguous.
/// @throws std::invalid_argument If `tensor` is undefined.
/// @code
/// Tensor packed = contiguous(tensor);
/// @endcode
Tensor contiguous(const Tensor& tensor);

/// Splits a tensor into sections of a fixed maximum size.
/// @param tensor Defined input tensor.
/// @param split_size Maximum length of each section along `dim`; must be positive.
/// @param dim Dimension along which to split; negative indices are accepted.
/// @return Materialized contiguous tensors on the input device. The final tensor may
///         be shorter than `split_size`.
/// @throws std::invalid_argument If `tensor` is undefined or `split_size` is not positive.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// std::vector<Tensor> parts = split(input, 2, 1); // Split dimension 1 in groups of 2.
/// @endcode
std::vector<Tensor> split(const Tensor& tensor, std::int64_t split_size, std::int64_t dim = 0);

/// Splits a tensor into approximately equal sections.
/// @param tensor Defined input tensor.
/// @param chunks Maximum requested number of sections; must be positive.
/// @param dim Dimension along which to split; negative indices are accepted.
/// @return At most `chunks` materialized contiguous tensors on the input device.
/// @throws std::invalid_argument If `tensor` is undefined or `chunks` is not positive.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// std::vector<Tensor> heads = chunk(input, 8, -1); // Split the last dimension into heads.
/// @endcode
std::vector<Tensor> chunk(const Tensor& tensor, std::int64_t chunks, std::int64_t dim = 0);

/// Joins tensors along one dimension.
/// @param tensors Nonempty inputs with matching ranks, dtypes, devices, and shapes
///        outside `dim`.
/// @param dim Dimension along which to join; negative indices are accepted.
/// @return A materialized contiguous tensor on the inputs' device whose size along
///         `dim` is the sum of the input sizes.
/// @throws std::invalid_argument If the input list is empty, an input is undefined,
///         or input metadata and shapes are incompatible.
/// @throws std::out_of_range If `dim` is outside the tensor rank.
/// @code
/// Tensor joined = concat({left, right}, -1); // Join tensors along the last dimension.
/// @endcode
Tensor concat(const std::vector<Tensor>& tensors, std::int64_t dim = 0);

} // namespace citrius
