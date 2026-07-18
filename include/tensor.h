#pragma once

#include "types.h"

#include <cstddef>
#include <memory>

namespace citrius {

class ITensorStorage;

class Tensor {
public:
    Tensor();
    Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    Tensor(Shape shape, DType dtype, Device device, std::shared_ptr<ITensorStorage> storage);

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;
    std::shared_ptr<ITensorStorage> storage() const;

    std::size_t ndim() const;
    std::int64_t numel() const;
    bool defined() const;
    Tensor copy() const;

private:
    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::cpu();
    std::shared_ptr<ITensorStorage> storage_;
    bool defined_ = false;
};

} // namespace citrius
