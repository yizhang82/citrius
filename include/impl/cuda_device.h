#pragma once

#include "cuda_storage.h"
#include "device.h"

#include <memory>

namespace citrius::impl {

class CudaExecutionContext;

enum class CudaElementwiseOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
};

enum class CudaReductionOperation {
    Sum,
    Mean,
    Maximum,
    Variance,
};

enum class CudaUnaryOperation {
    Exp,
    Sqrt,
    Power,
};

class CudaDeviceImpl : public IDevice {
public:
    explicit CudaDeviceImpl(int device_index = 0);

    DeviceType type() const override;
    int device_index() const;
    const std::shared_ptr<CudaExecutionContext>& execution_context() const;
    Tensor empty(Shape shape, DType dtype) const override;
    Tensor add(const Tensor& a, const Tensor& b) const override;
    Tensor sub(const Tensor& a, const Tensor& b) const override;
    Tensor matmul(const Tensor& a, const Tensor& b) const override;
    Tensor batched_matmul(const Tensor& a, const Tensor& b) const override;
    std::optional<Tensor> try_rms_norm(
        const Tensor& input,
        const Tensor& weight,
        float epsilon) const override;
    std::optional<Tensor> try_swiglu(
        const Tensor& gate,
        const Tensor& up) const override;
    std::optional<Tensor> try_rms_norm_rope(
        const Tensor& input,
        const Tensor& weight,
        float epsilon,
        float theta) const override;
    std::optional<Tensor> try_scaled_dot_product_attention(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& mask,
        bool is_causal) const override;
    std::optional<std::pair<Tensor, Tensor>> try_add_rms_norm(
        const Tensor& left,
        const Tensor& right,
        const Tensor& weight,
        float epsilon) const override;
    Tensor broadcast_elementwise(
        const Tensor& a,
        const Tensor& b,
        CudaElementwiseOperation operation) const;
    Tensor scalar_elementwise(
        const Tensor& tensor,
        float scalar,
        CudaElementwiseOperation operation,
        bool scalar_is_left = false) const;
    Tensor reduce(
        const Tensor& tensor,
        const std::vector<std::int64_t>& dimensions,
        bool keepdim,
        CudaReductionOperation operation) const;
    Tensor argmax(const Tensor& tensor, std::int64_t dimension, bool keepdim) const;
    Tensor argmax(const Tensor& tensor) const;
    Tensor unary(
        const Tensor& tensor,
        CudaUnaryOperation operation,
        float argument = 0.0f) const;
    Tensor masked_fill(const Tensor& tensor, const Tensor& mask, float value) const;
    Tensor contiguous(const Tensor& tensor) const;
    Tensor concat(const std::vector<Tensor>& tensors, std::int64_t dimension) const;
    Tensor gather_rows(const Tensor& table, const Tensor& indices) const;
    Tensor cast(const Tensor& tensor, DType dtype) const;
    virtual void add_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void sub_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    virtual void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const;
    TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const override;

private:
    int device_index_;
    int max_elementwise_blocks_;
    std::shared_ptr<CudaExecutionContext> context_;

    const CudaMemTensorStorageImpl& require_cuda_storage(const ITensorStorage& storage) const;
    CudaMemTensorStorageImpl& require_cuda_storage(ITensorStorage& storage) const;
};

} // namespace citrius::impl
