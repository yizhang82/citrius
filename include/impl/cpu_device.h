#pragma once

#include "cpu_storage.h"
#include "device.h"

#include <memory>

namespace citrius::impl {

class CpuDeviceImpl : public IDevice {
public:
    DeviceType type() const override;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    virtual void add_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void sub_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    const CpuMemTensorStorageImpl& require_cpu_storage(const ITensorStorage& storage) const;
    CpuMemTensorStorageImpl& require_cpu_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
