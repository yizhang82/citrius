#pragma once

#include <cstdint>
#include <vector>

namespace citrius {

enum class DType {
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
};

enum class DeviceType {
    CPU,
    Metal,
    CUDA,
};

struct Device {
    DeviceType type = DeviceType::CPU;
    int index = 0;

    static Device cpu();
    static Device metal(int index = 0);
    static Device cuda(int index = 0);

    bool operator==(const Device& other) const = default;
};

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;

std::size_t dtype_size(DType dtype);
bool is_floating_point(DType dtype);
std::uint16_t float_to_float16(float value);
float float16_to_float(std::uint16_t value);
std::uint16_t float_to_bfloat16(float value);
float bfloat16_to_float(std::uint16_t value);

} // namespace citrius
