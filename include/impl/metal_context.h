#pragma once

#include <memory>

namespace citrius::impl {

/// Shared execution state for the process-wide Metal device.
/// Objective-C Metal types are hidden from public C++ headers.
class MetalExecutionContext final {
public:
    MetalExecutionContext();
    ~MetalExecutionContext();

    MetalExecutionContext(const MetalExecutionContext&) = delete;
    MetalExecutionContext& operator=(const MetalExecutionContext&) = delete;

    void* device() const;
    void* command_queue() const;
    void submit(void* command_buffer) const;
    void synchronize() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Returns the process-wide Metal execution context.
std::shared_ptr<MetalExecutionContext> metal_execution_context();

} // namespace citrius::impl
