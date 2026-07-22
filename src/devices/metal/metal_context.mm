#include "metal_context.h"

#include <stdexcept>

namespace citrius::impl {

id<MTLDevice> shared_metal_device() {
    static id<MTLDevice> device = [] {
        id<MTLDevice> result = MTLCreateSystemDefaultDevice();
        if (result == nil) throw std::runtime_error("Metal device is not available");
        return result;
    }();
    return device;
}

} // namespace citrius::impl
