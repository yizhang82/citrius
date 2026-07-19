#include "indexing_operations.h"

#include "impl/cpu_storage.h"
#include "tensor_factory.h"

#include <cstring>
#include <stdexcept>

namespace citrius {

Tensor gather_rows(const Tensor& table, const Tensor& indices) {
    if (!table.defined() || !indices.defined()) {
        throw std::invalid_argument("gather_rows inputs must be defined");
    }
    if (table.dtype() != DType::Float32 || table.ndim() != 2) {
        throw std::invalid_argument("gather_rows table must be a two-dimensional Float32 tensor");
    }
    if (indices.dtype() != DType::Int64) {
        throw std::invalid_argument("gather_rows indices must be Int64");
    }
    if (table.device() != indices.device()) {
        throw std::invalid_argument("gather_rows inputs must be on the same device");
    }

    const Tensor cpu_table = table.to(Device::cpu());
    const Tensor cpu_indices = indices.to(Device::cpu());
    Shape output_shape = indices.shape();
    output_shape.push_back(table.shape()[1]);
    Tensor output = empty(output_shape, DType::Float32, Device::cpu());

    const auto table_storage =
        std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_table.storage());
    const auto index_storage =
        std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(cpu_indices.storage());
    auto output_storage =
        std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(output.storage());
    const float* table_data = table_storage->data_as<float>();
    const std::int64_t* index_data = index_storage->data_as<std::int64_t>();
    float* output_data = output_storage->data_as<float>();
    const std::int64_t row_size = table.shape()[1];

    for (std::int64_t position = 0; position < indices.numel(); ++position) {
        const std::int64_t row = index_data[position];
        if (row < 0 || row >= table.shape()[0]) {
            throw std::out_of_range("gather_rows index is out of range");
        }
        std::memcpy(
            output_data + position * row_size,
            table_data + row * row_size,
            static_cast<std::size_t>(row_size) * sizeof(float));
    }
    return output.to(table.device());
}

} // namespace citrius
