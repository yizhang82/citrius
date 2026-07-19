#pragma once

#include "nn/module.h"

namespace citrius::nn {

/// Inference-only dropout placeholder. Forward returns the input unchanged.
class Dropout final : public Module {
public:
    explicit Dropout(float probability = 0.5f);

    Tensor forward(const Tensor& input) override;
    float probability() const;

private:
    float probability_;
};

} // namespace citrius::nn
