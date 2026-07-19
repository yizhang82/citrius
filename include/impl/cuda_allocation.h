#pragma once

#include <cstddef>
#include <memory>

namespace citrius::impl {

class CudaExecutionContext;

/// Move-only ownership of a cudaMalloc allocation.
class CudaAllocation final {
public:
    CudaAllocation() = default;
    CudaAllocation(
        std::size_t nbytes,
        std::shared_ptr<CudaExecutionContext> context);
    ~CudaAllocation();

    CudaAllocation(const CudaAllocation&) = delete;
    CudaAllocation& operator=(const CudaAllocation&) = delete;
    CudaAllocation(CudaAllocation&& other) noexcept;
    CudaAllocation& operator=(CudaAllocation&& other) noexcept;

    void* data() const;
    std::size_t nbytes() const;

    template <typename T>
    T* data_as() const {
        return static_cast<T*>(data_);
    }

private:
    void reset() noexcept;

    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    std::shared_ptr<CudaExecutionContext> context_;
};

} // namespace citrius::impl
