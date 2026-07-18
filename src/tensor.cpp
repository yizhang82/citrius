#include "tensor.h"

#include "tensor_factory.h"
#include "storage.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace citrius {

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

} // namespace citrius
