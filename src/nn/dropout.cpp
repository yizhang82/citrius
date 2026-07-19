#include "nn/dropout.h"

#include <stdexcept>

namespace citrius::nn {

Dropout::Dropout(float probability) : probability_(probability) {
    if (probability < 0.0f || probability > 1.0f) {
        throw std::invalid_argument("Dropout probability must be between zero and one");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    if (!input.defined()) throw std::invalid_argument("Dropout input must be defined");
    return input;
}

float Dropout::probability() const {
    return probability_;
}

} // namespace citrius::nn
