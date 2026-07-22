#include "nn/linear.h"

#include "operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

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
    Device device,
    DType dtype)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(use_bias) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument("Linear features must be positive");
    }
    if (!is_floating_point(dtype)) {
        throw std::invalid_argument("Linear parameters must use a floating-point dtype");
    }

    const float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
    register_parameter(
        "weight",
        from_vector(random_values(in_features * out_features, bound),
                    {out_features, in_features}, dtype, device));
    register_parameter(
        "bias",
        use_bias
            ? from_vector(random_values(out_features, bound), {out_features}, dtype, device)
            : Tensor());
}

Tensor Linear::forward(const Tensor& input) {
    ENSURE_TENSOR_DEFINED(input);
    ENSURE_TENSOR_DTYPE(input, DType::Float32);
    if (input.ndim() == 0 || input.shape().back() != in_features_) {
        throw std::invalid_argument("Linear input's last dimension must equal in_features");
    }
    ENSURE_TENSOR_DEVICE_MATCH_2(input, weight());

    const std::int64_t rows = input.numel() / in_features_;
    const Tensor matrix_input(
        {rows, in_features_},
        input.dtype(),
        input.device(),
        input.storage());
    const DType gemm_dtype = input.device().type == DeviceType::CUDA
        ? weight().dtype()
        : input.dtype();
    const Tensor gemm_input = matrix_input.dtype() == gemm_dtype
        ? matrix_input
        : cast(matrix_input, gemm_dtype);
    Tensor gemm_weight = transpose(weight(), 0, 1);
    if (gemm_weight.dtype() != gemm_dtype) gemm_weight = cast(gemm_weight, gemm_dtype);
    Tensor output = matmul(gemm_input, gemm_weight);
    if (output.dtype() != input.dtype()) output = cast(output, input.dtype());
    if (has_bias_) {
        const Tensor output_bias = bias().dtype() == output.dtype()
            ? bias()
            : cast(bias(), output.dtype());
        output = add(output, output_bias);
    }

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
