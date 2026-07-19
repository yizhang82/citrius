#pragma once

#include "tensor.h"

namespace citrius {

/// Selects rows from a two-dimensional Float32 table using Int64 indices.
/// The output shape is `indices.shape() + [table.shape()[1]]`.
/// Operations without a native backend kernel stage through CPU while preserving
/// the table's device for the returned tensor.
/// @throws std::invalid_argument If tensors, dtypes, shapes, or devices are invalid.
/// @throws std::out_of_range If an index is outside `[0, table.shape()[0])`.
Tensor gather_rows(const Tensor& table, const Tensor& indices);

} // namespace citrius
