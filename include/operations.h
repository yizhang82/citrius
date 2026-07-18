#pragma once

#include "tensor.h"

namespace citrius {

Tensor add(const Tensor& left, const Tensor& right);
Tensor sub(const Tensor& left, const Tensor& right);
Tensor matmul(const Tensor& left, const Tensor& right);

Tensor operator+(const Tensor& left, const Tensor& right);
Tensor operator-(const Tensor& left, const Tensor& right);
Tensor operator*(const Tensor& left, const Tensor& right);

} // namespace citrius
