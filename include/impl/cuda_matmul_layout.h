#pragma once

#include "tensor.h"

#include <cstdint>

namespace citrius::impl {

enum class CudaMatrixLayoutType {
    Unsupported,
    RowMajor,
    ColumnMajor,
};

struct CudaMatrixLayout {
    CudaMatrixLayoutType type = CudaMatrixLayoutType::Unsupported;
    std::int64_t leading_dimension = 0;
    std::int64_t storage_offset = 0;

    bool supported() const { return type != CudaMatrixLayoutType::Unsupported; }
};

inline CudaMatrixLayout cuda_matrix_layout(const Tensor& tensor) {
    if (!tensor.defined() || tensor.ndim() != 2) return {};
    const auto rows = tensor.shape()[0];
    const auto columns = tensor.shape()[1];
    const auto row_stride = tensor.strides()[0];
    const auto column_stride = tensor.strides()[1];

    if ((columns <= 1 || column_stride == 1) &&
        (rows <= 1 || row_stride >= columns)) {
        return {CudaMatrixLayoutType::RowMajor, row_stride, tensor.storage_offset()};
    }
    if ((rows <= 1 || row_stride == 1) &&
        (columns <= 1 || column_stride >= rows)) {
        return {CudaMatrixLayoutType::ColumnMajor, column_stride, tensor.storage_offset()};
    }
    return {};
}

} // namespace citrius::impl
