#pragma once

#include "tensor.h"

#include <vector>

namespace citrius {

Tensor empty(
    Shape shape,
    DType dtype = DType::Float32,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<float>& values,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<float>& values,
    Shape shape,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<float>& values,
    Shape shape,
    DType dtype,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<std::int64_t>& values,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<std::int64_t>& values,
    Shape shape,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<bool>& values,
    Device device = Device::cpu());
Tensor from_vector(
    const std::vector<bool>& values,
    Shape shape,
    Device device = Device::cpu());
Tensor to(const Tensor& tensor, Device device);

} // namespace citrius
