#include "nn/linear.h"

#include "impl/cpu_storage.h"
#include "operations.h"

#include <cmath>
#include <memory>
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

Tensor transpose_weight(const Tensor& weight) {
    const Tensor cpu_weight = weight.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(
        cpu_weight.storage());
    const float* source = storage->data_as<float>();
    const std::int64_t out_features = weight.shape()[0];
    const std::int64_t in_features = weight.shape()[1];
    std::vector<float> values(static_cast<std::size_t>(weight.numel()));

    for (std::int64_t output = 0; output < out_features; ++output) {
        for (std::int64_t input = 0; input < in_features; ++input) {
            values[static_cast<std::size_t>(input * out_features + output)] =
                source[output * in_features + input];
        }
    }
    return Tensor(values, {in_features, out_features}, weight.device());
}

Tensor expanded_bias(const Tensor& bias, std::int64_t rows) {
    const Tensor cpu_bias = bias.to(Device::cpu());
    const auto storage = std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(
        cpu_bias.storage());
    const float* source = storage->data_as<float>();
    const std::int64_t out_features = bias.numel();
    std::vector<float> values(static_cast<std::size_t>(rows * out_features));

    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t output = 0; output < out_features; ++output) {
            values[static_cast<std::size_t>(row * out_features + output)] = source[output];
        }
    }
    return Tensor(values, {rows, out_features}, bias.device());
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
    Tensor output = matmul(matrix_input, transpose_weight(weight()));
    if (has_bias_) output = add(output, expanded_bias(bias(), rows));

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
