#include "tensor.h"

#include "cpu_storage.h"
#include "storage.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace citrius {

Tensor::Tensor() = default;

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : Tensor(
          shape,
          dtype,
          device,
          device.type == DeviceType::CPU
              ? std::make_shared<CpuMemTensorStorageImpl>(
                    static_cast<std::size_t>(
                        std::accumulate(
                            shape.begin(),
                            shape.end(),
                            std::int64_t{1},
                            [](std::int64_t total, std::int64_t dim) {
                                return total * dim;
                            })) *
                        dtype_size(dtype),
                    dtype)
              : nullptr) {
    if (device.type != DeviceType::CPU) {
        throw std::invalid_argument("only CPU tensor allocation is implemented");
    }
}

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

} // namespace citrius
