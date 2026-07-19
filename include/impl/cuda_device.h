#pragma once

#include "cuda_storage.h"
#include "device.h"

namespace citrius::impl {

enum class CudaElementwiseOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
};

class CudaDeviceImpl : public IDevice {
public:
    explicit CudaDeviceImpl(int device_index = 0);

    DeviceType type() const override;
    int device_index() const;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    Tensor batched_matmul(const Tensor& a, const Tensor& b) const override;
    Tensor broadcast_elementwise(
        const Tensor& a,
        const Tensor& b,
        CudaElementwiseOperation operation) const;
    Tensor scalar_elementwise(
        const Tensor& tensor,
        float scalar,
        CudaElementwiseOperation operation,
        bool scalar_is_left = false) const;
    virtual void add_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void sub_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    int device_index_;
    int max_elementwise_blocks_;

    const CudaMemTensorStorageImpl& require_cuda_storage(const ITensorStorage& storage) const;
    CudaMemTensorStorageImpl& require_cuda_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
