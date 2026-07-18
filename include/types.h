#pragma once

#include <cstdint>
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

std::size_t dtype_size(DType dtype);

} // namespace citrius
