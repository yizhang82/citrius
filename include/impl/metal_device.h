#pragma once

#include "device.h"
#include "metal_storage.h"

#include <cstdint>
#include <memory>

namespace citrius::impl {

enum class MetalElementwiseOperation : std::uint32_t {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
};

enum class MetalUnaryOperation : std::uint32_t {
    Exp,
    Sqrt,
    Power,
};

enum class MetalReductionOperation : std::uint32_t {
    Sum,
    Mean,
    Maximum,
    Variance,
};

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
    Tensor batched_matmul(const Tensor& a, const Tensor& b) const override;
    Tensor broadcast_elementwise(
        const Tensor& a,
        const Tensor& b,
        MetalElementwiseOperation operation) const;
    Tensor scalar_elementwise(
        const Tensor& tensor,
        float scalar,
        MetalElementwiseOperation operation,
        bool scalar_is_left = false) const;
    Tensor unary(
        const Tensor& tensor,
        MetalUnaryOperation operation,
        float argument = 0.0f) const;
    Tensor masked_fill(const Tensor& tensor, const Tensor& mask, float value) const;
    Tensor contiguous(const Tensor& tensor) const;
    Tensor concat(const std::vector<Tensor>& tensors, std::int64_t dimension) const;
    Tensor gather_rows(const Tensor& table, const Tensor& indices) const;
    Tensor reduce(
        const Tensor& tensor,
        const std::vector<std::int64_t>& dimensions,
        bool keepdim,
        MetalReductionOperation operation) const;
    Tensor argmax(const Tensor& tensor, std::int64_t dimension, bool keepdim) const;
    Tensor argmax(const Tensor& tensor) const;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    class Impl;
    std::shared_ptr<Impl> impl_;

    const MetalMemTensorStorageImpl& require_metal_storage(const ITensorStorage& storage) const;
    MetalMemTensorStorageImpl& require_metal_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
