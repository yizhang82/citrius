#pragma once

#include "types.h"

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace citrius {

class ITensorStorage;

class Tensor {
public:
    Tensor();
    Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    explicit Tensor(const std::vector<float>& values, Device device = Device::cpu());
    Tensor(const std::vector<float>& values, Shape shape, Device device = Device::cpu());
    Tensor(Shape shape, DType dtype, Device device, std::shared_ptr<ITensorStorage> storage);

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;
    std::shared_ptr<ITensorStorage> storage() const;

    std::size_t ndim() const;
    std::int64_t numel() const;
    bool defined() const;
    Tensor copy() const;
    Tensor to(Device device) const;
    std::string to_string() const;

private:
    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::cpu();
    std::shared_ptr<ITensorStorage> storage_;
    bool defined_ = false;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

} // namespace citrius
