#pragma once

#include "storage.h"

#include <memory>

namespace citrius::impl {

class CudaExecutionContext;

class CudaMemTensorStorageImpl final : public ITensorStorage {
public:
    CudaMemTensorStorageImpl(std::size_t nbytes, DType dtype, int device_index = 0);
    CudaMemTensorStorageImpl(
        std::size_t nbytes,
        DType dtype,
        std::shared_ptr<CudaExecutionContext> context);
    ~CudaMemTensorStorageImpl() override;

    CudaMemTensorStorageImpl(const CudaMemTensorStorageImpl&) = delete;
    CudaMemTensorStorageImpl& operator=(const CudaMemTensorStorageImpl&) = delete;

    TensorStorageType type() const override;
    DeviceType device_type() const override;
    DType dtype() const override;
    std::size_t nbytes() const override;
    StorageHandle handle() override;
    StorageHandle handle() const override;
    std::shared_ptr<ITensorStorage> clone() const override;

    int device_index() const;
    const std::shared_ptr<CudaExecutionContext>& execution_context() const;
    void copy_from_host(const void* data, std::size_t nbytes);
    void copy_to_host(void* data, std::size_t nbytes) const;

private:
    void* data_ = nullptr;
    std::size_t nbytes_ = 0;
    DType dtype_;
    int device_index_ = 0;
    std::shared_ptr<CudaExecutionContext> context_;
};

} // namespace citrius::impl
