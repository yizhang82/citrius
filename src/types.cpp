#include "types.h"

#include <stdexcept>

namespace citrius {

Device Device::cpu() {
    return Device{DeviceType::CPU, 0};
}

Device Device::cuda(int index) {
    return Device{DeviceType::CUDA, index};
}

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
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

} // namespace citrius
