#pragma once

#include "storage.h"

#include <memory>

namespace citrius::impl {

class MetalMemTensorStorageImpl final : public ITensorStorage {
public:
    MetalMemTensorStorageImpl(std::size_t nbytes, DType dtype);
    ~MetalMemTensorStorageImpl() override;

    MetalMemTensorStorageImpl(const MetalMemTensorStorageImpl&) = delete;
    MetalMemTensorStorageImpl& operator=(const MetalMemTensorStorageImpl&) = delete;

    TensorStorageType type() const override;
    DeviceType device_type() const override;
    DType dtype() const override;
    std::size_t nbytes() const override;
    StorageHandle handle() override;
    StorageHandle handle() const override;
    std::shared_ptr<ITensorStorage> clone() const override;

    void copy_from_host(const void* data, std::size_t nbytes);
    void copy_to_host(void* data, std::size_t nbytes, std::size_t source_offset = 0) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace citrius::impl
