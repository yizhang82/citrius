#include "safetensors.h"

#include "tensor_factory.h"

#include <bit>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace citrius {
namespace {

struct TensorInfo {
    std::string name;
    std::string dtype;
    Shape shape;
    std::uint64_t begin;
    std::uint64_t end;
};

std::uint64_t read_u64(const unsigned char* bytes) {
    std::uint64_t value = 0;
    for (int index = 0; index < 8; ++index) value |= std::uint64_t{bytes[index]} << (8 * index);
    return value;
}

std::vector<std::uint64_t> parse_numbers(const std::string& text) {
    std::vector<std::uint64_t> values;
    const std::regex number(R"((\d+))");
    for (auto iterator = std::sregex_iterator(text.begin(), text.end(), number);
         iterator != std::sregex_iterator(); ++iterator) {
        values.push_back(std::stoull((*iterator)[1].str()));
    }
    return values;
}

std::vector<TensorInfo> parse_header(const std::string& header) {
    std::vector<TensorInfo> tensors;
    std::size_t cursor = 0;
    while ((cursor = header.find('"', cursor)) != std::string::npos) {
        const std::size_t name_end = header.find('"', cursor + 1);
        if (name_end == std::string::npos) throw std::runtime_error("invalid safetensors JSON header");
        const std::string name = header.substr(cursor + 1, name_end - cursor - 1);
        std::size_t colon = header.find(':', name_end + 1);
        if (colon == std::string::npos) throw std::runtime_error("invalid safetensors JSON entry");
        std::size_t object_begin = header.find_first_not_of(" \t\r\n", colon + 1);
        if (object_begin == std::string::npos || header[object_begin] != '{') {
            cursor = name_end + 1;
            continue;
        }
        int depth = 0;
        std::size_t object_end = object_begin;
        for (; object_end < header.size(); ++object_end) {
            if (header[object_end] == '{') ++depth;
            if (header[object_end] == '}' && --depth == 0) break;
        }
        if (object_end == header.size()) throw std::runtime_error("unterminated safetensors JSON object");
        if (name != "__metadata__") {
            const std::string object = header.substr(object_begin, object_end - object_begin + 1);
            std::smatch match;
            const std::regex dtype_pattern(R"json("dtype"\s*:\s*"([^"]+)")json");
            const std::regex shape_pattern(R"json("shape"\s*:\s*\[([^\]]*)\])json");
            const std::regex offsets_pattern(
                R"json("data_offsets"\s*:\s*\[([^\]]*)\])json");
            if (!std::regex_search(object, match, dtype_pattern)) throw std::runtime_error("safetensors tensor is missing dtype");
            const std::string dtype = match[1].str();
            if (!std::regex_search(object, match, shape_pattern)) throw std::runtime_error("safetensors tensor is missing shape");
            const auto shape_numbers = parse_numbers(match[1].str());
            Shape shape;
            shape.reserve(shape_numbers.size());
            for (const auto dimension : shape_numbers) {
                if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) throw std::runtime_error("safetensors shape is too large");
                shape.push_back(static_cast<std::int64_t>(dimension));
            }
            if (!std::regex_search(object, match, offsets_pattern)) throw std::runtime_error("safetensors tensor is missing offsets");
            const auto offsets = parse_numbers(match[1].str());
            if (offsets.size() != 2 || offsets[1] < offsets[0]) throw std::runtime_error("invalid safetensors data offsets");
            tensors.push_back({name, dtype, std::move(shape), offsets[0], offsets[1]});
        }
        cursor = object_end + 1;
    }
    return tensors;
}

float half_to_float(std::uint16_t half) {
    const std::uint32_t sign = (half & 0x8000u) << 16;
    std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t mantissa = half & 0x03ffu;
    std::uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x03ffu;
            bits = sign | ((127 - 14 - shift) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return std::bit_cast<float>(bits);
}

std::size_t element_size(const std::string& dtype) {
    if (dtype == "F32") return 4;
    if (dtype == "F16" || dtype == "BF16") return 2;
    throw std::invalid_argument("unsupported safetensors dtype: " + dtype);
}

} // namespace

TensorMap load_safetensors(const std::filesystem::path& path, Device device) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open safetensors file: " + path.string());
    const auto file_size = static_cast<std::uint64_t>(stream.tellg());
    if (file_size < 8) throw std::runtime_error("safetensors file is too short");
    stream.seekg(0);
    unsigned char length_bytes[8];
    stream.read(reinterpret_cast<char*>(length_bytes), 8);
    const std::uint64_t header_length = read_u64(length_bytes);
    if (header_length > file_size - 8) throw std::runtime_error("invalid safetensors header length");
    std::string header(static_cast<std::size_t>(header_length), '\0');
    stream.read(header.data(), static_cast<std::streamsize>(header.size()));
    const std::uint64_t data_start = 8 + header_length;

    TensorMap result;
    for (const TensorInfo& info : parse_header(header)) {
        std::int64_t count = 1;
        for (const auto dimension : info.shape) count *= dimension;
        const std::size_t width = element_size(info.dtype);
        const std::uint64_t expected = static_cast<std::uint64_t>(count) * width;
        if (info.end - info.begin != expected || data_start + info.end > file_size) {
            throw std::runtime_error("safetensors tensor byte size does not match shape: " + info.name);
        }
        std::vector<unsigned char> bytes(static_cast<std::size_t>(expected));
        stream.seekg(static_cast<std::streamoff>(data_start + info.begin));
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("failed reading safetensors tensor: " + info.name);
        std::vector<float> values(static_cast<std::size_t>(count));
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (info.dtype == "F32") {
                std::uint32_t bits;
                std::memcpy(&bits, bytes.data() + index * 4, 4);
                values[index] = std::bit_cast<float>(bits);
            } else {
                const std::uint16_t bits = static_cast<std::uint16_t>(bytes[index * 2]) |
                    (static_cast<std::uint16_t>(bytes[index * 2 + 1]) << 8);
                values[index] = info.dtype == "BF16"
                    ? std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16)
                    : half_to_float(bits);
            }
        }
        result.emplace(info.name, from_vector(values, info.shape, device));
    }
    return result;
}

} // namespace citrius
