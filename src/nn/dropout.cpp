#include "nn/dropout.h"
#include "tensor_utils.h"

#include <stdexcept>

namespace citrius::nn {

Dropout::Dropout(float probability) : probability_(probability) {
    if (probability < 0.0f || probability > 1.0f) {
        throw std::invalid_argument("Dropout probability must be between zero and one");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    ENSURE_TENSOR_DEFINED(input);
    return input;
}

float Dropout::probability() const {
    return probability_;
}

} // namespace citrius::nn
