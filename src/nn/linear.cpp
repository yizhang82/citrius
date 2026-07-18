#include "nn/linear.h"

#include "operations.h"
#include "shape_operations.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace citrius::nn {
namespace {

std::vector<float> random_values(std::int64_t count, float bound) {
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_real_distribution<float> distribution(-bound, bound);
    std::vector<float> values(static_cast<std::size_t>(count));
    for (float& value : values) value = distribution(generator);
    return values;
}

} // namespace

Linear::Linear(
    std::int64_t in_features,
    std::int64_t out_features,
    bool use_bias,
    Device device)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(use_bias) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Linear features must be positive");
    }

    const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
    register_parameter(
        "weight",
        Tensor(
            random_values(in_features * out_features, bound),
            {out_features, in_features},
            device));
    register_parameter(
        "bias",
        use_bias
            ? Tensor(random_values(out_features, bound), {out_features}, device)
            : Tensor());
}

Tensor Linear::forward(const Tensor& input) {
    if (!input.defined()) throw std::invalid_argument("Linear input must be defined");
    if (input.dtype() != DType::Float32) {
        throw std::invalid_argument("Linear currently supports Float32 only");
    }
    if (input.ndim() == 0 || input.shape().back() != in_features_) {
        throw std::invalid_argument("Linear input's last dimension must equal in_features");
    }
    if (input.device() != weight().device()) {
        throw std::invalid_argument("Linear input and parameters must be on the same device");
    }

    const std::int64_t rows = input.numel() / in_features_;
    const Tensor matrix_input(
        {rows, in_features_},
        input.dtype(),
        input.device(),
        input.storage());
    Tensor output = matmul(matrix_input, transpose(weight(), 0, 1));
    if (has_bias_) output = add(output, bias());

    Shape output_shape = input.shape();
    output_shape.back() = out_features_;
    return Tensor(
        std::move(output_shape),
        output.dtype(),
        output.device(),
        output.storage());
}

std::int64_t Linear::in_features() const {
    return in_features_;
}

std::int64_t Linear::out_features() const {
    return out_features_;
}

bool Linear::has_bias() const {
    return has_bias_;
}

Tensor& Linear::weight() {
    return parameter("weight");
}

const Tensor& Linear::weight() const {
    return parameter("weight");
}

Tensor& Linear::bias() {
    return parameter("bias");
}

const Tensor& Linear::bias() const {
    return parameter("bias");
}

} // namespace citrius::nn
