#pragma once

#include "nn/module.h"

#include <cstdint>

namespace citrius::nn {

class Linear final : public Module {
public:
    Linear(
        std::int64_t in_features,
        std::int64_t out_features,
        bool bias = true,
        Device device = Device::cpu(),
        DType dtype = DType::Float32);

    Tensor forward(const Tensor& input) override;

    std::int64_t in_features() const;
    std::int64_t out_features() const;
    bool has_bias() const;

    Tensor& weight();
    const Tensor& weight() const;
    Tensor& bias();
    const Tensor& bias() const;

private:
    std::int64_t in_features_;
    std::int64_t out_features_;
    bool has_bias_;
};

} // namespace citrius::nn
