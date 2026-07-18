#include "tensor.h"

#include "cpu_storage.h"

#include <numeric>
#include <stdexcept>
#include <utility>

namespace citrius {

class TensorImpl {
public:
    TensorImpl(Shape shape, DType dtype, Device device, std::shared_ptr<ITensorStorage> storage)
        : shape_(std::move(shape)),
          dtype_(dtype),
          device_(device),
          storage_(std::move(storage)) {}

    const Shape& shape() const {
        return shape_;
    }

    DType dtype() const {
        return dtype_;
    }

    Device device() const {
        return device_;
    }

    std::shared_ptr<ITensorStorage> storage() const {
        return storage_;
    }

private:
    Shape shape_;
    DType dtype_;
    Device device_;
    std::shared_ptr<ITensorStorage> storage_;
};

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
    : impl_(std::make_shared<TensorImpl>(std::move(shape), dtype, device, std::move(storage))) {}

const Shape& Tensor::shape() const {
    return impl_->shape();
}

DType Tensor::dtype() const {
    return impl_->dtype();
}

Device Tensor::device() const {
    return impl_->device();
}

std::shared_ptr<ITensorStorage> Tensor::storage() const {
    return impl_->storage();
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
