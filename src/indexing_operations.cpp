#include "indexing_operations.h"

#include "impl/cpu_storage.h"
#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#endif
#include "tensor_factory.h"
#include "tensor_utils.h"

#include <cstring>
#include <stdexcept>

namespace citrius {

Tensor gather_rows(const Tensor& table, const Tensor& indices) {
    ENSURE_TENSOR_DEFINED(table);
    ENSURE_TENSOR_DEFINED(indices);
    ENSURE_TENSOR_DIM(table, 2);
    ENSURE_TENSOR_DTYPE(table, DType::Float32);
    ENSURE_TENSOR_DTYPE(indices, DType::Int64);
    ENSURE_TENSOR_DEVICE_MATCH_2(table, indices);

#ifdef CITRIUS_HAS_CUDA
    if (table.device().type == DeviceType::CUDA) {
        return impl::CudaDeviceImpl(table.device().index).gather_rows(table, indices);
    }
#endif

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
