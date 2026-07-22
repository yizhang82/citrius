#include "types.h"

#include <bit>
#include <stdexcept>

namespace citrius {

Device Device::cpu() {
    return Device{DeviceType::CPU, 0};
}

Device Device::metal(int index) {
    return Device{DeviceType::Metal, index};
}

Device Device::cuda(int index) {
    return Device{DeviceType::CUDA, index};
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float16:
        case DType::BFloat16:
            return 2;
        case DType::Float32:
            return 4;
        case DType::Float64:
            return 8;
        case DType::Int32:
            return 4;
        case DType::Int64:
            return 8;
        case DType::Bool:
            return 1;
    }

    throw std::invalid_argument("unknown dtype");
}

bool is_floating_point(DType dtype) {
    return dtype == DType::Float16 || dtype == DType::BFloat16 ||
        dtype == DType::Float32 || dtype == DType::Float64;
}

std::uint16_t float_to_bfloat16(float value) {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + bias) >> 16);
}

float bfloat16_to_float(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

std::uint16_t float_to_float16(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exponent = (bits >> 23) & 0xffu;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    const int half_exponent = static_cast<int>(exponent) - 112;
    if (half_exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return static_cast<std::uint16_t>(sign);
        const std::uint32_t significand = mantissa | 0x800000u;
        const int shift = 14 - half_exponent;
        std::uint32_t rounded = significand >> shift;
        const std::uint32_t remainder = significand & ((std::uint32_t{1} << shift) - 1);
        const std::uint32_t halfway = std::uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    std::uint32_t rounded_mantissa = mantissa >> 13;
    const std::uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_mantissa & 1u))) {
        if (++rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            if (half_exponent + 1 >= 31)
                return static_cast<std::uint16_t>(sign | 0x7c00u);
            return static_cast<std::uint16_t>(sign | ((half_exponent + 1) << 10));
        }
    }
    return static_cast<std::uint16_t>(sign | (half_exponent << 10) | rounded_mantissa);
}

float float16_to_float(std::uint16_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            bits = sign | ((127 - 14 - shift) << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    return std::bit_cast<float>(bits);
}

} // namespace citrius
