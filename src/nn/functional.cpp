#include "nn/functional.h"

#include "operations.h"
#include "reduction_operations.h"

namespace citrius::nn::functional {

namespace {

Tensor stable_tanh(const Tensor& tensor) {
    const Tensor zero = mul(tensor, 0.0f);
    const Tensor positive = maximum(tensor, zero);
    const Tensor negative = maximum(mul(tensor, -1.0f), zero);

    const auto tanh_magnitude = [](const Tensor& magnitude) {
        const Tensor exponential = exp(mul(magnitude, -2.0f));
        return div(sub(1.0f, exponential), add(1.0f, exponential));
    };

    return sub(tanh_magnitude(positive), tanh_magnitude(negative));
}

} // namespace

Tensor relu(const Tensor& tensor) {
    return maximum(tensor, mul(tensor, 0.0f));
}

Tensor gelu(const Tensor& tensor) {
    constexpr float sqrt_two_over_pi = 0.7978845608028654f;
    constexpr float cubic_coefficient = 0.044715f;

    const Tensor cubic = pow(tensor, 3.0f);
    const Tensor inner = mul(add(tensor, mul(cubic, cubic_coefficient)), sqrt_two_over_pi);
    return mul(mul(tensor, 0.5f), add(stable_tanh(inner), 1.0f));
}

Tensor softmax(const Tensor& tensor, std::int64_t dim) {
    const Tensor shifted = sub(tensor, citrius::max(tensor, dim, true));
    const Tensor exponentials = citrius::exp(shifted);
    return div(exponentials, sum(exponentials, dim, true));
}

} // namespace citrius::nn::functional
