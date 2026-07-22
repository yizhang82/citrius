#pragma once

#include "storage.h"
#include "tensor.h"

#include <memory>
#include <optional>
#include <utility>

namespace citrius::impl {

enum class ConversionPolicy {
    Error,
    CopyToDevice,
};

class IDevice {
public:
    virtual ~IDevice() = default;

    virtual DeviceType type() const = 0;
    virtual Tensor empty(Shape shape, DType dtype) const = 0;
    virtual Tensor add(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor sub(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor matmul(const Tensor& a, const Tensor& b) const = 0;
    virtual Tensor batched_matmul(const Tensor& a, const Tensor& b) const = 0;
    virtual std::optional<Tensor> try_rms_norm(
        const Tensor& input,
        const Tensor& weight,
        float epsilon) const {
        return std::nullopt;
    }
    virtual std::optional<Tensor> try_swiglu(
        const Tensor& gate,
        const Tensor& up) const {
        return std::nullopt;
    }
    virtual std::optional<Tensor> try_rms_norm_rope(
        const Tensor& input,
        const Tensor& weight,
        float epsilon,
        float theta) const {
        return std::nullopt;
    }
    virtual std::optional<Tensor> try_scaled_dot_product_attention(
        const Tensor& query,
        const Tensor& key,
        const Tensor& value,
        const Tensor& mask,
        bool is_causal) const {
        return std::nullopt;
    }
    virtual std::optional<std::pair<Tensor, Tensor>> try_add_rms_norm(
        const Tensor& left,
        const Tensor& right,
        const Tensor& weight,
        float epsilon) const {
        return std::nullopt;
    }
    virtual TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const = 0;
};

} // namespace citrius::impl
