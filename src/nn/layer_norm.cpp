#include "nn/layer_norm.h"

#include "nn/functional.h"

#include <stdexcept>
#include <utility>
#include <vector>

namespace citrius::nn {
namespace {

std::int64_t parameter_count(const Shape& shape) {
    if (shape.empty()) throw std::invalid_argument("LayerNorm normalized_shape cannot be empty");
    std::int64_t count = 1;
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument("LayerNorm normalized dimensions must be positive");
        }
        count *= dimension;
    }
    return count;
}

} // namespace

LayerNorm::LayerNorm(
    std::int64_t normalized_size,
    float eps,
    bool elementwise_affine,
    Device device)
    : LayerNorm(Shape{normalized_size}, eps, elementwise_affine, device) {}

LayerNorm::LayerNorm(
    Shape normalized_shape,
    float eps,
    bool elementwise_affine,
    Device device)
    : normalized_shape_(std::move(normalized_shape)),
      eps_(eps),
      elementwise_affine_(elementwise_affine) {
    const std::int64_t count = parameter_count(normalized_shape_);
    if (!(eps > 0.0f)) throw std::invalid_argument("LayerNorm epsilon must be positive");

    register_parameter(
        "weight",
        elementwise_affine
            ? Tensor(std::vector<float>(static_cast<std::size_t>(count), 1.0f), normalized_shape_, device)
            : Tensor());
    register_parameter(
        "bias",
        elementwise_affine
            ? Tensor(std::vector<float>(static_cast<std::size_t>(count), 0.0f), normalized_shape_, device)
            : Tensor());
}

Tensor LayerNorm::forward(const Tensor& input) {
    return functional::layer_norm(input, normalized_shape_, weight(), bias(), eps_);
}

const Shape& LayerNorm::normalized_shape() const {
    return normalized_shape_;
}

float LayerNorm::eps() const {
    return eps_;
}

bool LayerNorm::has_elementwise_affine() const {
    return elementwise_affine_;
}

Tensor& LayerNorm::weight() {
    return parameter("weight");
}

const Tensor& LayerNorm::weight() const {
    return parameter("weight");
}

Tensor& LayerNorm::bias() {
    return parameter("bias");
}

const Tensor& LayerNorm::bias() const {
    return parameter("bias");
}

} // namespace citrius::nn
