#include "tensor.h"

#include "tensor_factory.h"
#include "cpu_storage.h"
#include "storage.h"

#include <cstdint>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace citrius {

namespace {

const char* dtype_name(DType dtype) {
    switch (dtype) {
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

} // namespace

Tensor::Tensor() = default;

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : Tensor(TensorFactory::empty(std::move(shape), dtype, device)) {}

Tensor::Tensor(const std::vector<float>& values, Device device)
    : Tensor(TensorFactory::from_vector(values, device)) {}

Tensor::Tensor(const std::vector<float>& values, Shape shape, Device device)
    : Tensor(TensorFactory::from_vector(values, std::move(shape), device)) {}

Tensor::Tensor(Shape shape, DType dtype, Device device, std::shared_ptr<ITensorStorage> storage)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device),
      storage_(std::move(storage)),
      defined_(true) {}

const Shape& Tensor::shape() const {
    return shape_;
}

DType Tensor::dtype() const {
    return dtype_;
}

Device Tensor::device() const {
    return device_;
}

std::shared_ptr<ITensorStorage> Tensor::storage() const {
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

Tensor Tensor::copy() const {
    if (!defined()) {
        throw std::invalid_argument("cannot copy an undefined tensor");
    }

    return Tensor(shape_, dtype_, device_, storage_->clone());
}

Tensor Tensor::to(Device device) const {
    return TensorFactory::to(*this, device);
}

std::string Tensor::to_string() const {
    if (!defined()) return "tensor(undefined)";

    const Tensor cpu_tensor = to(Device::cpu());
    const auto cpu_storage =
        std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu_tensor.storage());
    std::ostringstream stream;
    stream << "tensor([";
    switch (dtype()) {
        case DType::Float32:
            append_values(stream, cpu_storage->data_as<float>(), numel());
            break;
        case DType::Float64:
            append_values(stream, cpu_storage->data_as<double>(), numel());
            break;
        case DType::Int32:
            append_values(stream, cpu_storage->data_as<std::int32_t>(), numel());
            break;
        case DType::Int64:
            append_values(stream, cpu_storage->data_as<std::int64_t>(), numel());
            break;
        case DType::Bool: {
            const auto* values = cpu_storage->data_as<std::uint8_t>();
            for (std::int64_t index = 0; index < numel(); ++index) {
                if (index != 0) stream << ", ";
                stream << (values[index] == 0 ? "false" : "true");
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
