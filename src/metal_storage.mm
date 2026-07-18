#include "metal_storage.h"

#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>

namespace citrius {

class MetalMemTensorStorageImpl::Impl {
public:
    Impl(std::size_t nbytes, DType dtype)
        : dtype(dtype),
          nbytes(nbytes) {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("Metal device is not available");
        }

        buffer = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error("failed to allocate Metal buffer");
        }
    }

    id<MTLDevice> device = nil;
    id<MTLBuffer> buffer = nil;
    DType dtype;
    std::size_t nbytes;
};

MetalMemTensorStorageImpl::MetalMemTensorStorageImpl(std::size_t nbytes, DType dtype)
    : impl_(std::make_unique<Impl>(nbytes, dtype)) {}

MetalMemTensorStorageImpl::~MetalMemTensorStorageImpl() = default;

TensorStorageType MetalMemTensorStorageImpl::type() const {
    return TensorStorageType::MetalMemory;
}

DeviceType MetalMemTensorStorageImpl::device_type() const {
    return DeviceType::Metal;
}

DType MetalMemTensorStorageImpl::dtype() const {
    return impl_->dtype;
}

std::size_t MetalMemTensorStorageImpl::nbytes() const {
    return impl_->nbytes;
}

StorageHandle MetalMemTensorStorageImpl::handle() {
    return StorageHandle{(__bridge void*)impl_->buffer, impl_->nbytes};
}

StorageHandle MetalMemTensorStorageImpl::handle() const {
    return StorageHandle{(__bridge void*)impl_->buffer, impl_->nbytes};
}

void MetalMemTensorStorageImpl::copy_from_host(const void* data, std::size_t nbytes) {
    if (nbytes > impl_->nbytes) {
        throw std::invalid_argument("host data is larger than Metal storage");
    }

    std::memcpy([impl_->buffer contents], data, nbytes);
}

void MetalMemTensorStorageImpl::copy_to_host(void* data, std::size_t nbytes) const {
    if (nbytes > impl_->nbytes) {
        throw std::invalid_argument("host destination is larger than Metal storage");
    }

    std::memcpy(data, [impl_->buffer contents], nbytes);
}

} // namespace citrius
