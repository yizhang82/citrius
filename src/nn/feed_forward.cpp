#include "nn/feed_forward.h"

#include "nn/functional.h"
#include "tensor_utils.h"

#include <stdexcept>

namespace citrius::nn {

FeedForward::FeedForward(
    std::int64_t embed_dim,
    std::int64_t hidden_dim,
    bool bias,
    Device device)
    : embed_dim_(embed_dim),
      hidden_dim_(hidden_dim) {
    if (embed_dim <= 0) {
        throw std::invalid_argument("FeedForward embed_dim must be positive");
    }
    if (hidden_dim <= 0) {
        throw std::invalid_argument("FeedForward hidden_dim must be positive");
    }

    input_projection_ = register_module(
        "input_projection",
        std::make_shared<Linear>(embed_dim, hidden_dim, bias, device));
    output_projection_ = register_module(
        "output_projection",
        std::make_shared<Linear>(hidden_dim, embed_dim, bias, device));
}

Tensor FeedForward::forward(const Tensor& input) {
    CITRIUS_ENSURE_TENSOR_DEFINED(input);
    CITRIUS_ENSURE_TENSOR_DIM(input, 3); 
    if (input.shape()[2] != embed_dim_) {
        throw std::invalid_argument("FeedForward input dim #2 must match embed_dim");
    }
    
    // input -> [ B, S, E ]
    // hidden -> [ B, S, E] x [ E, H ] = [ B, S, H ]
    auto hidden = (*input_projection_)(input);

    // hidden -> [ B, S, H ] 
    hidden = citrius::nn::functional::gelu(hidden);

    // output -> [ B, S, H ] x [ H, E ] = [ B, S, E ]
    return (*output_projection_)(hidden);
}

std::int64_t FeedForward::embed_dim() const {
    return embed_dim_;
}

std::int64_t FeedForward::hidden_dim() const {
    return hidden_dim_;
}

Linear& FeedForward::input_projection() {
    return *input_projection_;
}

Linear& FeedForward::output_projection() {
    return *output_projection_;
}

} // namespace citrius::nn
