#pragma once

#include "nn/module.h"

#include <cstdint>

namespace citrius::nn {

class Embedding final : public Module {
public:
    Embedding(
        std::int64_t num_embeddings,
        std::int64_t embedding_dim,
        Device device = Device::cpu(),
        DType dtype = DType::Float32);

    Tensor forward(const Tensor& input) override;

    std::int64_t num_embeddings() const;
    std::int64_t embedding_dim() const;
    Tensor& weight();
    const Tensor& weight() const;

private:
    std::int64_t num_embeddings_;
    std::int64_t embedding_dim_;
};

} // namespace citrius::nn
