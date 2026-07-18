#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace citrius {

enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
};

enum class DeviceType {
    CPU,
    CUDA,
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    static Device cpu();
    static Device cuda(int index = 0);

    bool operator==(const Device& other) const = default;
};

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

class Storage;
class TensorImpl;

class Tensor {
public:
    Tensor();
    Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;

    std::size_t ndim() const;
    std::int64_t numel() const;
    bool defined() const;

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace citrius
