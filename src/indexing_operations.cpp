#include "indexing_operations.h"

#include "impl/cpu_storage.h"
#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#endif
#include "tensor_factory.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace citrius {
namespace {

struct NormalizedSlice {
    std::int64_t start;
    std::int64_t length;
    std::int64_t step;
};

NormalizedSlice normalize_slice(const indexing::Slice& slice, std::int64_t size) {
    if (slice.step == 0) throw std::invalid_argument("slice step cannot be zero");
    if (slice.step == std::numeric_limits<std::int64_t>::min())
        throw std::invalid_argument("slice step magnitude is too large");

    if (slice.step > 0) {
        auto start = slice.start.value_or(0);
        auto stop = slice.stop.value_or(size);
        if (slice.start && start < 0) start += size;
        if (slice.stop && stop < 0) stop += size;
        start = std::clamp<std::int64_t>(start, 0, size);
        stop = std::clamp<std::int64_t>(stop, 0, size);
        const auto length = stop <= start ? 0 : 1 + (stop - start - 1) / slice.step;
        return {start, length, slice.step};
    }

    auto start = slice.start.value_or(size - 1);
    auto stop = slice.stop.value_or(-1);
    if (slice.start && start < 0) start += size;
    if (slice.stop && stop < 0) stop += size;
    start = std::clamp<std::int64_t>(start, -1, size - 1);
    stop = std::clamp<std::int64_t>(stop, -1, size - 1);
    const auto stride = -slice.step;
    const auto length = start <= stop ? 0 : 1 + (start - stop - 1) / stride;
    return {start, length, slice.step};
}

} // namespace

indexing::Slice::Slice(std::int64_t stop) : stop(stop) {}

indexing::Slice::Slice(std::int64_t start, std::int64_t stop, std::int64_t step)
    : start(start), stop(stop), step(step) {}

indexing::Slice::Slice(
    std::optional<std::int64_t> start,
    std::optional<std::int64_t> stop,
    std::int64_t step)
    : start(start), stop(stop), step(step) {}

TensorIndex::TensorIndex(std::int64_t index) : value_(index) {}
TensorIndex::TensorIndex(indexing::Slice slice) : value_(std::move(slice)) {}
TensorIndex::TensorIndex(indexing::EllipsisIndex value) : value_(value) {}
TensorIndex::TensorIndex(indexing::NoneIndex value) : value_(value) {}

Tensor index(const Tensor& tensor, std::initializer_list<TensorIndex> indices) {
    ENSURE_TENSOR_DEFINED(tensor);

    std::size_t consuming = 0;
    std::size_t ellipses = 0;
    for (const auto& component : indices) {
        if (std::holds_alternative<indexing::EllipsisIndex>(component.value_)) ++ellipses;
        else if (!std::holds_alternative<indexing::NoneIndex>(component.value_)) ++consuming;
    }
    if (ellipses > 1) throw std::invalid_argument("a tensor index may contain only one ellipsis");
    if (consuming > tensor.ndim()) throw std::out_of_range("too many indices for tensor");

    Shape shape;
    Strides strides;
    shape.reserve(tensor.ndim() + indices.size());
    strides.reserve(tensor.ndim() + indices.size());
    std::int64_t offset = tensor.storage_offset();
    std::size_t axis = 0;

    const auto append_full_dimensions = [&](std::size_t count) {
        for (std::size_t current = 0; current < count; ++current, ++axis) {
            shape.push_back(tensor.shape()[axis]);
            strides.push_back(tensor.strides()[axis]);
        }
    };

    for (const auto& component : indices) {
        std::visit(
            [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::int64_t>) {
                    auto selected = value;
                    const auto size = tensor.shape()[axis];
                    if (selected < 0) selected += size;
                    if (selected < 0 || selected >= size)
                        throw std::out_of_range("tensor index is out of range");
                    offset += selected * tensor.strides()[axis++];
                } else if constexpr (std::is_same_v<T, indexing::Slice>) {
                    const auto normalized = normalize_slice(value, tensor.shape()[axis]);
                    if (normalized.length != 0)
                        offset += normalized.start * tensor.strides()[axis];
                    shape.push_back(normalized.length);
                    strides.push_back(tensor.strides()[axis++] * normalized.step);
                } else if constexpr (std::is_same_v<T, indexing::NoneIndex>) {
                    shape.push_back(1);
                    strides.push_back(0);
                } else {
                    append_full_dimensions(tensor.ndim() - consuming);
                }
            },
            component.value_);
    }
    append_full_dimensions(tensor.ndim() - axis);

    return Tensor(std::move(shape), std::move(strides), offset, tensor.dtype(),
                  tensor.device(), tensor.storage());
}

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
