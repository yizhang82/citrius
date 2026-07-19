#pragma once

#include <cstddef>
#include <memory>

namespace citrius::impl {

class CudaExecutionContext;

/// Move-only ownership of a stream-ordered CUDA allocation.
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

    void copy_from_host_async(const void* source, std::size_t nbytes);
    void copy_to_host_async(
        void* destination,
        std::size_t nbytes,
        std::size_t source_offset = 0) const;
    void copy_from_device_async(const CudaAllocation& source, std::size_t nbytes);
    void synchronize() const;

    template <typename T>
    T* data_as() const {
        return static_cast<T*>(data_);
    }

private:
    void validate_copy_range(std::size_t offset, std::size_t nbytes) const;
    void reset() noexcept;

    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    std::shared_ptr<CudaExecutionContext> context_;
};

} // namespace citrius::impl
