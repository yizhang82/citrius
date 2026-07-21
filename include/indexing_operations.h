#pragma once

#include "tensor.h"

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <variant>

namespace citrius {

namespace indexing {

/// A Python-style half-open slice. Omitted bounds are resolved according to the
/// direction of `step`, and negative bounds are relative to the dimension size.
struct Slice {
    std::optional<std::int64_t> start;
    std::optional<std::int64_t> stop;
    std::int64_t step = 1;

    Slice() = default;
    explicit Slice(std::int64_t stop);
    Slice(std::int64_t start, std::int64_t stop, std::int64_t step = 1);
    Slice(
        std::optional<std::int64_t> start,
        std::optional<std::int64_t> stop,
        std::int64_t step = 1);
};

struct EllipsisIndex {};
struct NoneIndex {};

inline constexpr EllipsisIndex Ellipsis{};
inline constexpr NoneIndex None{};

} // namespace indexing

/// One component of a basic tensor index.
class TensorIndex {
public:
    TensorIndex(std::int64_t index);
    TensorIndex(indexing::Slice slice);
    TensorIndex(indexing::EllipsisIndex);
    TensorIndex(indexing::NoneIndex);

private:
    using Value = std::variant<
        std::int64_t,
        indexing::Slice,
        indexing::EllipsisIndex,
        indexing::NoneIndex>;
    Value value_;

    friend Tensor index(const Tensor&, std::initializer_list<TensorIndex>);
};

/// Applies Python/PyTorch-style basic indexing and returns a storage-sharing view.
/// Integer indices remove a dimension, slices preserve it, `indexing::None` inserts
/// a size-one dimension, and one `indexing::Ellipsis` may stand for omitted dimensions.
Tensor index(const Tensor& tensor, std::initializer_list<TensorIndex> indices);

/// Selects rows from a two-dimensional Float32 table using Int64 indices.
/// The output shape is `indices.shape() + [table.shape()[1]]`.
/// Operations without a native backend kernel stage through CPU while preserving
/// the table's device for the returned tensor.
/// @throws std::invalid_argument If tensors, dtypes, shapes, or devices are invalid.
/// @throws std::out_of_range If an index is outside `[0, table.shape()[0])`.
Tensor gather_rows(const Tensor& table, const Tensor& indices);

} // namespace citrius
