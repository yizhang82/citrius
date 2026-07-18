#pragma once

#include "storage.h"
#include "tensor.h"

#include <memory>

namespace citrius {

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
    virtual TensorStoragePtr ensure_storage(
        const TensorStoragePtr& storage,
        ConversionPolicy policy = ConversionPolicy::Error) const = 0;
};

} // namespace citrius
