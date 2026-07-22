#include "tensor.h"

#include "indexing_operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "impl/cpu_storage.h"
#include "impl/storage.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace citrius {

namespace {

using impl::CpuMemTensorStorageImpl;

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::Bool: return "bool";
    }
    return "unknown";
}

std::string device_name(Device device) {
    switch (device.type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::Metal: return "metal:" + std::to_string(device.index);
        case DeviceType::CUDA: return "cuda:" + std::to_string(device.index);
    }
    return "unknown";
}

template <typename T>
void append_values(std::ostringstream& stream, const T* values, std::int64_t count) {
    for (std::int64_t index = 0; index < count; ++index) {
        if (index != 0) stream << ", ";
        stream << values[index];
    }
}

Strides make_contiguous_strides(const Shape& shape) {
    Strides strides(shape.size());
    std::int64_t stride = 1;
    for (std::size_t index = shape.size(); index-- > 0;) {
        strides[index] = stride;
        const auto dimension = shape[index];
        if (dimension > 1 && stride > std::numeric_limits<std::int64_t>::max() / dimension) {
            throw std::invalid_argument("tensor shape is too large");
        }
        stride *= std::max<std::int64_t>(dimension, 1);
    }
    return strides;
}

void validate_layout(const Shape& shape, const Strides& strides, std::int64_t storage_offset,
                     DType dtype, Device,
                     const std::shared_ptr<impl::ITensorStorage>& storage) {
    if (!storage) throw std::invalid_argument("defined tensor requires storage");
    if (shape.size() != strides.size()) {
        throw std::invalid_argument("tensor shape and strides must have the same rank");
    }
    if (storage_offset < 0) throw std::invalid_argument("tensor storage offset cannot be negative");
    bool empty = false;
    std::int64_t minimum_offset = storage_offset;
    std::int64_t maximum_offset = storage_offset;
    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
        if (shape[dimension] < 0) throw std::invalid_argument("tensor dimensions cannot be negative");
        if (shape[dimension] == 0) {
            empty = true;
            continue;
        }
        const auto extent = shape[dimension] - 1;
        if (strides[dimension] >= 0) {
            if (extent != 0 && strides[dimension] >
                                   (std::numeric_limits<std::int64_t>::max() - maximum_offset) / extent) {
                throw std::invalid_argument("tensor layout offset overflows");
            }
            maximum_offset += extent * strides[dimension];
        } else {
            if (strides[dimension] == std::numeric_limits<std::int64_t>::min() ||
                (extent != 0 && -strides[dimension] >
                                    minimum_offset / extent)) {
                throw std::invalid_argument("tensor layout offset underflows");
            }
            minimum_offset += extent * strides[dimension];
        }
    }
    if (empty) return;
    if (minimum_offset < 0) throw std::invalid_argument("tensor layout precedes storage");

    const auto element_size = dtype_size(dtype);
    const auto maximum_size = static_cast<std::uint64_t>(maximum_offset) + 1;
    if (maximum_size > storage->nbytes() / element_size) {
        throw std::invalid_argument("tensor layout exceeds storage bounds");
    }
}

} // namespace

Tensor::Tensor() = default;

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : Tensor(citrius::empty(std::move(shape), dtype, device)) {}

Tensor::Tensor(const std::vector<float>& values, Device device)
    : Tensor(citrius::from_vector(values, device)) {}

Tensor::Tensor(const std::vector<float>& values, Shape shape, Device device)
    : Tensor(citrius::from_vector(values, std::move(shape), device)) {}

Tensor::Tensor(
    Shape shape,
    DType dtype,
    Device device,
    std::shared_ptr<impl::ITensorStorage> storage)
    : Tensor(shape, make_contiguous_strides(shape), 0, dtype, device, std::move(storage)) {}

Tensor::Tensor(
    Shape shape,
    Strides strides,
    std::int64_t storage_offset,
    DType dtype,
    Device device,
    std::shared_ptr<impl::ITensorStorage> storage)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      storage_offset_(storage_offset),
      dtype_(dtype),
      device_(device),
      storage_(std::move(storage)),
      defined_(true) {
    validate_layout(shape_, strides_, storage_offset_, dtype_, device_, storage_);
}

const Shape& Tensor::shape() const {
    return shape_;
}

const Strides& Tensor::strides() const {
    return strides_;
}

std::int64_t Tensor::storage_offset() const {
    return storage_offset_;
}

bool Tensor::is_contiguous() const {
    if (!defined()) return false;
    if (numel() == 0) return true;
    std::int64_t expected = 1;
    for (std::size_t index = shape_.size(); index-- > 0;) {
        if (shape_[index] == 1) continue;
        if (strides_[index] != expected) return false;
        expected *= std::max<std::int64_t>(shape_[index], 1);
    }
    return true;
}

DType Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

std::shared_ptr<impl::ITensorStorage> Tensor::storage() const {
    return storage_;
}

std::size_t Tensor::ndim() const {
    return shape().size();
}

std::int64_t Tensor::numel() const {
    const auto& dims = shape();
    return std::accumulate(
        dims.begin(),
        dims.end(),
        std::int64_t{1},
        [](std::int64_t total, std::int64_t dim) {
            return total * dim;
        });
}

bool Tensor::defined() const {
    return defined_;
}

Tensor Tensor::index(std::initializer_list<TensorIndex> indices) const {
    return citrius::index(*this, indices);
}

Tensor Tensor::operator[](std::int64_t index) const {
    return select(0, index);
}

Tensor Tensor::operator[](std::initializer_list<TensorIndex> indices) const {
    return citrius::index(*this, indices);
}

Tensor Tensor::select(std::int64_t dim, std::int64_t index) const {
    return citrius::select(*this, dim, index);
}

Tensor Tensor::slice(
    std::int64_t dim,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) const {
    return citrius::slice(*this, dim, start, end, step);
}

Tensor Tensor::copy() const {
    if (!defined()) {
        throw std::invalid_argument("cannot copy an undefined tensor");
    }

    return Tensor(shape_, strides_, storage_offset_, dtype_, device_, storage_->clone());
}

Tensor Tensor::to(Device device) const {
    return citrius::to(*this, device);
}

void Tensor::copy_item_to_host(void* destination, DType expected) const {
    if (!defined()) throw std::invalid_argument("cannot materialize an undefined tensor");
    if (numel() != 1) throw std::invalid_argument("Tensor::item requires exactly one element");
    if (dtype() != expected) throw std::invalid_argument("Tensor::item type does not match dtype");
    const Tensor cpu = to(Device::cpu());
    const auto cpu_storage =
        std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu.storage());
    const auto byte_offset =
        static_cast<std::size_t>(cpu.storage_offset()) * dtype_size(dtype());
    const auto* source = static_cast<const std::byte*>(cpu_storage->data()) + byte_offset;
    std::memcpy(destination, source, dtype_size(dtype()));
}

std::string Tensor::to_string() const {
    if (!defined()) return "tensor(undefined)";
    if (!is_contiguous()) {
        throw std::invalid_argument("printing a non-contiguous tensor requires contiguous()");
    }

    const Tensor cpu_tensor = to(Device::cpu());
    const auto cpu_storage =
        std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu_tensor.storage());
    const auto byte_offset =
        static_cast<std::size_t>(cpu_tensor.storage_offset()) * dtype_size(dtype());
    const auto* values = static_cast<const std::byte*>(cpu_storage->data()) + byte_offset;
    std::ostringstream stream;
    stream << "tensor([";
    switch (dtype()) {
        case DType::Float16:
        case DType::BFloat16: {
            const auto* bits = reinterpret_cast<const std::uint16_t*>(values);
            for (std::int64_t index = 0; index < numel(); ++index) {
                if (index != 0) stream << ", ";
                stream << (dtype() == DType::Float16
                                   ? float16_to_float(bits[index])
                                   : bfloat16_to_float(bits[index]));
            }
            break;
        }
        case DType::Float32:
            append_values(stream, reinterpret_cast<const float*>(values), numel());
            break;
        case DType::Float64:
            append_values(stream, reinterpret_cast<const double*>(values), numel());
            break;
        case DType::Int32:
            append_values(stream, reinterpret_cast<const std::int32_t*>(values), numel());
            break;
        case DType::Int64:
            append_values(stream, reinterpret_cast<const std::int64_t*>(values), numel());
            break;
        case DType::Bool: {
            const auto* bool_values = reinterpret_cast<const std::uint8_t*>(values);
            for (std::int64_t index = 0; index < numel(); ++index) {
                if (index != 0) stream << ", ";
                stream << (bool_values[index] == 0 ? "false" : "true");
            }
            break;
        }
    }
    stream << "], shape=[";
    for (std::size_t index = 0; index < shape().size(); ++index) {
        if (index != 0) stream << ", ";
        stream << shape()[index];
    }
    stream << "], dtype=" << dtype_name(dtype())
           << ", device=" << device_name(device()) << ')';
    return stream.str();
}

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor) {
    return stream << tensor.to_string();
}

} // namespace citrius
