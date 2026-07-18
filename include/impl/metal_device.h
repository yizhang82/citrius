#pragma once

#include "device.h"
#include "metal_storage.h"

#include <memory>

namespace citrius::impl {

class MetalDeviceImpl final : public IDevice {
public:
    MetalDeviceImpl();
    ~MetalDeviceImpl() override;

    MetalDeviceImpl(const MetalDeviceImpl&) = delete;
    MetalDeviceImpl& operator=(const MetalDeviceImpl&) = delete;

    DeviceType type() const override;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    const MetalMemTensorStorageImpl& require_metal_storage(const ITensorStorage& storage) const;
    MetalMemTensorStorageImpl& require_metal_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
