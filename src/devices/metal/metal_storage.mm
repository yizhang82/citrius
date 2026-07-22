#include "impl/metal_storage.h"
#include "impl/metal_context.h"

#import <Metal/Metal.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace citrius::impl {

class MetalMemTensorStorageImpl::Impl {
public:
    Impl(std::size_t nbytes, DType dtype, std::shared_ptr<MetalExecutionContext> context)
        : context(std::move(context)),
          dtype(dtype),
          nbytes(nbytes) {
        if (!this->context)
            throw std::invalid_argument("Metal storage requires an execution context");
        id<MTLDevice> device = (__bridge id<MTLDevice>)this->context->device();
        buffer = [device
            newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            throw std::runtime_error("failed to allocate Metal buffer");
        }
    }

    std::shared_ptr<MetalExecutionContext> context;
    id<MTLBuffer> buffer = nil;
    DType dtype;
    std::size_t nbytes;
};

MetalMemTensorStorageImpl::MetalMemTensorStorageImpl(std::size_t nbytes, DType dtype)
    : MetalMemTensorStorageImpl(nbytes, dtype, metal_execution_context()) {}

MetalMemTensorStorageImpl::MetalMemTensorStorageImpl(
    std::size_t nbytes, DType dtype, std::shared_ptr<MetalExecutionContext> context)
    : impl_(std::make_unique<Impl>(nbytes, dtype, std::move(context))) {}

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

std::shared_ptr<ITensorStorage> MetalMemTensorStorageImpl::clone() const {
    impl_->context->synchronize();
    auto copied = std::make_shared<MetalMemTensorStorageImpl>(
        nbytes(), dtype(), impl_->context);
    copied->copy_from_host([impl_->buffer contents], nbytes());
    return copied;
}

void MetalMemTensorStorageImpl::copy_from_host(const void* data, std::size_t nbytes) {
    if (nbytes > impl_->nbytes) {
        throw std::invalid_argument("host data is larger than Metal storage");
    }

    impl_->context->synchronize();
    std::memcpy([impl_->buffer contents], data, nbytes);
}

void MetalMemTensorStorageImpl::copy_to_host(
    void* data,
    std::size_t nbytes,
    std::size_t source_offset) const {
    if (source_offset > impl_->nbytes || nbytes > impl_->nbytes - source_offset) {
        throw std::invalid_argument("host destination is larger than Metal storage");
    }

    impl_->context->synchronize();
    const auto* source = static_cast<const std::byte*>([impl_->buffer contents]) + source_offset;
    std::memcpy(data, source, nbytes);
}

const std::shared_ptr<MetalExecutionContext>&
MetalMemTensorStorageImpl::execution_context() const {
    return impl_->context;
}

} // namespace citrius::impl
