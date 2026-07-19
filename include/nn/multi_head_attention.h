#pragma once

#include "nn/linear.h"
#include "nn/module.h"

#include <cstdint>
#include <memory>

namespace citrius::nn {

/// Applies multi-head self-attention to `[batch, sequence, embed_dim]` inputs.
class MultiHeadAttention final : public Module {
public:
    MultiHeadAttention(
        std::int64_t embed_dim,
        std::int64_t num_heads,
        bool bias = true,
        Device device = Device::cpu());

    Tensor forward(const Tensor& input) override;

    std::int64_t embed_dim() const;
    std::int64_t num_heads() const;
    std::int64_t head_dim() const;

    Linear& query_projection();
    Linear& key_projection();
    Linear& value_projection();
    Linear& output_projection();

private:
    std::int64_t embed_dim_;
    std::int64_t num_heads_;
    std::int64_t head_dim_;
    std::shared_ptr<Linear> query_;
    std::shared_ptr<Linear> key_;
    std::shared_ptr<Linear> value_;
    std::shared_ptr<Linear> output_;
};

} // namespace citrius::nn
