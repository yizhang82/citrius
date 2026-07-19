#pragma once

#include "nn/linear.h"
#include "nn/module.h"

#include <cstdint>
#include <memory>

namespace citrius::nn {

/// Transformer feed-forward network applied independently to each token.
class FeedForward final : public Module {
public:
    FeedForward(
        std::int64_t embed_dim,
        std::int64_t hidden_dim,
        bool bias = true,
        Device device = Device::cpu());

    Tensor forward(const Tensor& input) override;

    std::int64_t embed_dim() const;
    std::int64_t hidden_dim() const;
    Linear& input_projection();
    Linear& output_projection();

private:
    std::int64_t embed_dim_;
    std::int64_t hidden_dim_;
    std::shared_ptr<Linear> input_projection_;
    std::shared_ptr<Linear> output_projection_;
};

} // namespace citrius::nn
