#pragma once

#include "storage.h"
#include "tensor.h"

#include <memory>
#include <optional>

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
    virtual TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const = 0;
};

} // namespace citrius::impl
