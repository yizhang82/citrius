#include "nn/embedding.h"

#include "indexing_operations.h"

#include <random>
#include <stdexcept>
#include <vector>

namespace citrius::nn {
namespace {

std::vector<float> random_values(std::int64_t count) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> values(static_cast<std::size_t>(count));
    for (float& value : values) value = distribution(generator);
    return values;
}

} // namespace

Embedding::Embedding(
    std::int64_t num_embeddings,
    std::int64_t embedding_dim,
    Device device)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim) {
    if (num_embeddings <= 0 || embedding_dim <= 0) {
        throw std::invalid_argument("Embedding dimensions must be positive");
    }
    register_parameter(
        "weight",
        Tensor(
            random_values(num_embeddings * embedding_dim),
            {num_embeddings, embedding_dim},
            device));
}

Tensor Embedding::forward(const Tensor& input) {
    return gather_rows(weight(), input);
}

std::int64_t Embedding::num_embeddings() const {
    return num_embeddings_;
}

std::int64_t Embedding::embedding_dim() const {
    return embedding_dim_;
}

Tensor& Embedding::weight() {
    return parameter("weight");
}

const Tensor& Embedding::weight() const {
    return parameter("weight");
}

} // namespace citrius::nn
