#pragma once

#include "nn/module.h"

namespace citrius::nn {

class LayerNorm final : public Module {
public:
    explicit LayerNorm(
        std::int64_t normalized_size,
        float eps = 1e-5f,
        bool elementwise_affine = true,
        Device device = Device::cpu());
    explicit LayerNorm(
        Shape normalized_shape,
        float eps = 1e-5f,
        bool elementwise_affine = true,
        Device device = Device::cpu());

    Tensor forward(const Tensor& input) override;

    const Shape& normalized_shape() const;
    float eps() const;
    bool has_elementwise_affine() const;

    Tensor& weight();
    const Tensor& weight() const;
    Tensor& bias();
    const Tensor& bias() const;

private:
    Shape normalized_shape_;
    float eps_;
    bool elementwise_affine_;
};

} // namespace citrius::nn
