#pragma once

#include "nn/feed_forward.h"
#include "nn/layer_norm.h"
#include "nn/module.h"
#include "nn/multi_head_attention.h"

#include <cstdint>
#include <memory>

namespace citrius::nn {

/// Pre-normalization Transformer block with self-attention and a feed-forward network.
class TransformerBlock final : public Module {
public:
    TransformerBlock(
        std::int64_t embed_dim,
        std::int64_t num_heads,
        std::int64_t hidden_dim,
        float layer_norm_eps = 1e-5f,
        bool bias = true,
        Device device = Device::cpu());

    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor& attn_mask);

    std::int64_t embed_dim() const;
    std::int64_t num_heads() const;
    std::int64_t hidden_dim() const;

    LayerNorm& attention_norm();
    MultiHeadAttention& attention();
    LayerNorm& feed_forward_norm();
    FeedForward& feed_forward();

private:
    std::int64_t embed_dim_;
    std::int64_t num_heads_;
    std::int64_t hidden_dim_;
    std::shared_ptr<LayerNorm> attention_norm_;
    std::shared_ptr<MultiHeadAttention> attention_;
    std::shared_ptr<LayerNorm> feed_forward_norm_;
    std::shared_ptr<FeedForward> feed_forward_;
};

} // namespace citrius::nn
