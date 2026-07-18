#include "tensor.h"

#include <numeric>
#include <utility>

namespace citrius {

class TensorImpl {
public:
    TensorImpl(Shape shape, DType dtype, Device device)
        : shape_(std::move(shape)),
          dtype_(dtype),
          device_(device) {}

    const Shape& shape() const {
        return shape_;
    }

    DType dtype() const {
        return dtype_;
    }

    Device device() const {
        return device_;
    }

private:
    Shape shape_;
    DType dtype_;
    Device device_;
};

Device Device::cpu() {
    return Device{DeviceType::CPU, 0};
}

Device Device::cuda(int index) {
    return Device{DeviceType::CUDA, index};
}

Tensor::Tensor() = default;

Tensor::Tensor(Shape shape, DType dtype, Device device)
    : impl_(std::make_shared<TensorImpl>(std::move(shape), dtype, device)) {}

const Shape& Tensor::shape() const {
    return impl_->shape();
}

DType Tensor::dtype() const {
    return impl_->dtype();
}

Device Tensor::device() const {
    return impl_->device();
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
    return impl_ != nullptr;
}

} // namespace citrius
