#include "nn/multi_head_attention.h"

#include "shape_operations.h"
#include "operations.h"
#include "nn/functional.h"

#include <stdexcept>

namespace citrius::nn {

MultiHeadAttention::MultiHeadAttention(
    std::int64_t embed_dim,
    std::int64_t num_heads,
    bool bias,
    Device device)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(num_heads > 0 ? embed_dim / num_heads : 0) {
    if (embed_dim <= 0) {
        throw std::invalid_argument("MultiHeadAttention embed_dim must be positive");
    }
    if (num_heads <= 0) {
        throw std::invalid_argument("MultiHeadAttention num_heads must be positive");
    }
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument(
            "MultiHeadAttention embed_dim must be divisible by num_heads");
    }

    query_ = register_module(
        "query",
        std::make_shared<Linear>(embed_dim, embed_dim, bias, device));
    key_ = register_module(
        "key",
        std::make_shared<Linear>(embed_dim, embed_dim, bias, device));
    value_ = register_module(
        "value",
        std::make_shared<Linear>(embed_dim, embed_dim, bias, device));
    output_ = register_module(
        "output",
        std::make_shared<Linear>(embed_dim, embed_dim, bias, device));
}

Tensor MultiHeadAttention::forward(const Tensor& input) {
    // TODO: Implement the following steps:
    // 1. Validate input shape `[batch, sequence, embed_dim]`.
    // 2. Project input independently into query, key, and value.
    // 3. Reshape `[B, S, E]` to `[B, S, H, D]`, then permute to `[B, H, S, D]`.
    // 4. Call functional::scaled_dot_product_attention(query, key, value).
    // 5. Permute back to `[B, S, H, D]` and reshape to `[B, S, E]`.
    // 6. Apply the output projection.

    // input [B, S, E]
    if (!input.defined()) {
        throw std::invalid_argument("MultiHeadAttention input must be defined");
    }
    if (input.shape().size() != 3) {
        throw std::invalid_argument("MultiHeadAttention input must have 3 dimensions");
    }

    // [B, S, E] x [E, E] -> [B, S, E] 
    auto query = (*query_)(input);
    auto key = (*key_)(input);
    auto value = (*value_)(input);

    // [B, S, H, D]
    auto query_reshaped = citrius::reshape(query, {input.shape()[0], input.shape()[1], num_heads_, head_dim_});
    auto key_reshaped = citrius::reshape(key, {input.shape()[0], input.shape()[1], num_heads_, head_dim_});
    auto value_reshaped = citrius::reshape(value, {input.shape()[0], input.shape()[1], num_heads_, head_dim_});

    // [B, S, H, D] -> [B, H, S, D]
    auto query_permuted = citrius::permute(query_reshaped, {0, 2, 1, 3});
    auto key_permuted = citrius::permute(key_reshaped, {0, 2, 1, 3});
    auto value_permuted = citrius::permute(value_reshaped, {0, 2, 1, 3});

    // Call functional::scaled_dot_product_attention
    auto attn_output = citrius::nn::functional::scaled_dot_product_attention(query_permuted, key_permuted, value_permuted);

    // Permute back to [B, S, H, D] and reshape to [B, S, E]
    auto attn_output_permuted = citrius::permute(attn_output, {0, 2, 1, 3});
    auto attn_output_reshaped = citrius::reshape(attn_output_permuted, {input.shape()[0], input.shape()[1], embed_dim_});

    // Apply the output projection
    auto output = (*output_)(attn_output_reshaped);
    return output;
}

std::int64_t MultiHeadAttention::embed_dim() const {
    return embed_dim_;
}

std::int64_t MultiHeadAttention::num_heads() const {
    return num_heads_;
}

std::int64_t MultiHeadAttention::head_dim() const {
    return head_dim_;
}

Linear& MultiHeadAttention::query_projection() {
    return *query_;
}

Linear& MultiHeadAttention::key_projection() {
    return *key_;
}

Linear& MultiHeadAttention::value_projection() {
    return *value_;
}

Linear& MultiHeadAttention::output_projection() {
    return *output_;
}

} // namespace citrius::nn
