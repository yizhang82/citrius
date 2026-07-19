#include "nn/transformer_block.h"

#include "operations.h"
#include "tensor_utils.h"

#include <stdexcept>

namespace citrius::nn {

TransformerBlock::TransformerBlock(
    std::int64_t embed_dim,
    std::int64_t num_heads,
    std::int64_t hidden_dim,
    float layer_norm_eps,
    bool bias,
    Device device)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      hidden_dim_(hidden_dim) {
    if (embed_dim <= 0) {
        throw std::invalid_argument("TransformerBlock embed_dim must be positive");
    }
    if (num_heads <= 0) {
        throw std::invalid_argument("TransformerBlock num_heads must be positive");
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument("TransformerBlock hidden_dim must be positive");
    }
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "TransformerBlock embed_dim must be divisible by num_heads");
    }
    if (!(layer_norm_eps > 0.0f)) {
        throw std::invalid_argument("TransformerBlock layer_norm_eps must be positive");
    }

    attention_norm_ = register_module(
        "attention_norm",
        std::make_shared<LayerNorm>(embed_dim, layer_norm_eps, true, device));
    attention_ = register_module(
        "attention",
        std::make_shared<MultiHeadAttention>(embed_dim, num_heads, bias, device));
    feed_forward_norm_ = register_module(
        "feed_forward_norm",
        std::make_shared<LayerNorm>(embed_dim, layer_norm_eps, true, device));
    feed_forward_ = register_module(
        "feed_forward",
        std::make_shared<FeedForward>(embed_dim, hidden_dim, bias, device));
}

Tensor TransformerBlock::forward(const Tensor& input) {
    return forward(input, Tensor());
}

Tensor TransformerBlock::forward(const Tensor& input, const Tensor& attn_mask) {
    ENSURE_TENSOR_DEFINED(input);
    ENSURE_TENSOR_DIM(input, 3);
    if (input.shape().back() != embed_dim_) {
        throw std::invalid_argument(
            "TransformerBlock input's last dimension must equal embed_dim");
    }

    const Tensor attended = attention_->forward((*attention_norm_)(input), attn_mask);
    const Tensor attention_residual = add(input, attended);
    const Tensor transformed = (*feed_forward_)((*feed_forward_norm_)(attention_residual));
    return add(attention_residual, transformed);
}

std::int64_t TransformerBlock::embed_dim() const {
    return embed_dim_;
}

std::int64_t TransformerBlock::num_heads() const {
    return num_heads_;
}

std::int64_t TransformerBlock::hidden_dim() const {
    return hidden_dim_;
}

LayerNorm& TransformerBlock::attention_norm() {
    return *attention_norm_;
}

MultiHeadAttention& TransformerBlock::attention() {
    return *attention_;
}

LayerNorm& TransformerBlock::feed_forward_norm() {
    return *feed_forward_norm_;
}

FeedForward& TransformerBlock::feed_forward() {
    return *feed_forward_;
}

} // namespace citrius::nn
