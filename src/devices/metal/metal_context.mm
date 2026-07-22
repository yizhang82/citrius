#include "impl/metal_context.h"

#import <Metal/Metal.h>

#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace citrius::impl {

class MetalExecutionContext::Impl {
public:
    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) throw std::runtime_error("Metal device is not available");
        queue = [device newCommandQueue];
        if (queue == nil) throw std::runtime_error("failed to create Metal command queue");
    }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    mutable std::mutex submission_mutex;
    mutable id<MTLCommandBuffer> active_command_buffer = nil;
    mutable std::size_t active_encoding_count = 0;
    mutable std::vector<id<MTLCommandBuffer>> pending_command_buffers;
};

MetalExecutionContext::MetalExecutionContext() : impl_(std::make_unique<Impl>()) {}
MetalExecutionContext::~MetalExecutionContext() = default;

void* MetalExecutionContext::device() const { return (__bridge void*)impl_->device; }
void* MetalExecutionContext::command_queue() const { return (__bridge void*)impl_->queue; }

void* MetalExecutionContext::command_buffer() const {
    std::lock_guard lock(impl_->submission_mutex);
    if (impl_->active_command_buffer == nil)
        impl_->active_command_buffer = [impl_->queue commandBuffer];
    return (__bridge void*)impl_->active_command_buffer;
}

void MetalExecutionContext::submit(void* command_buffer_handle) const {
    id<MTLCommandBuffer> command_buffer =
        (__bridge id<MTLCommandBuffer>)command_buffer_handle;
    std::lock_guard lock(impl_->submission_mutex);
    if (command_buffer != impl_->active_command_buffer)
        throw std::runtime_error("Metal command buffer submission is out of order");
    constexpr std::size_t max_encodings_per_command_buffer = 64;
    if (++impl_->active_encoding_count >= max_encodings_per_command_buffer) {
        [command_buffer commit];
        impl_->pending_command_buffers.push_back(command_buffer);
        impl_->active_command_buffer = nil;
        impl_->active_encoding_count = 0;
    }
}

void MetalExecutionContext::synchronize() const {
    std::vector<id<MTLCommandBuffer>> command_buffers;
    {
        std::lock_guard lock(impl_->submission_mutex);
        if (impl_->active_command_buffer != nil) {
            [impl_->active_command_buffer commit];
            impl_->pending_command_buffers.push_back(impl_->active_command_buffer);
            impl_->active_command_buffer = nil;
            impl_->active_encoding_count = 0;
        }
        command_buffers.swap(impl_->pending_command_buffers);
    }
    for (id<MTLCommandBuffer> command_buffer : command_buffers) {
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            const char* description = command_buffer.error
                ? [[command_buffer.error localizedDescription] UTF8String]
                : "unknown error";
            throw std::runtime_error(
                std::string("Metal command buffer execution failed: ") + description);
        }
    }
}

std::shared_ptr<MetalExecutionContext> metal_execution_context() {
    static auto context = std::make_shared<MetalExecutionContext>();
    return context;
}

} // namespace citrius::impl
