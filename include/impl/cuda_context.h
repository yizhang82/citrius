#pragma once

#include <memory>

namespace citrius::impl {

/// Shared execution state for one CUDA device.
/// CUDA runtime types are hidden so public C++ headers do not require CUDA headers.
class CudaExecutionContext final {
public:
    explicit CudaExecutionContext(int device_index);
    ~CudaExecutionContext();

    CudaExecutionContext(const CudaExecutionContext&) = delete;
    CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;

    int device_index() const;
    void* stream() const;

private:
    int device_index_;
    void* stream_ = nullptr;
};

/// Returns the process-wide shared context for a CUDA device index.
std::shared_ptr<CudaExecutionContext> cuda_execution_context(int device_index);

} // namespace citrius::impl
