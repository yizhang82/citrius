#pragma once

#include "cpu_storage.h"
#include "device.h"

#include <memory>

namespace citrius {

class CpuDeviceImpl final : public IDevice {
public:
    DeviceType type() const override;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    const CpuMemTensorStorageImpl& require_cpu_storage(const ITensorStorage& storage) const;
    CpuMemTensorStorageImpl& require_cpu_storage(ITensorStorage& storage) const;
};

} // namespace citrius
