#pragma once

#include "cuda_storage.h"
#include "device.h"

namespace citrius::impl {

class CudaDeviceImpl : public IDevice {
public:
    explicit CudaDeviceImpl(int device_index = 0);

    DeviceType type() const override;
    int device_index() const;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    int device_index_;

    const CudaMemTensorStorageImpl& require_cuda_storage(const ITensorStorage& storage) const;
    CudaMemTensorStorageImpl& require_cuda_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
