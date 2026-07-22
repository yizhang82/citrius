#pragma once

#include "tensor.h"

#include <filesystem>
#include <string>
#include <unordered_map>

namespace citrius {

using TensorMap = std::unordered_map<std::string, Tensor>;

/// Loads F32, F16, or BF16 tensors from one unsharded safetensors file.
/// Floating-point values are converted to Citrius Float32 tensors.
TensorMap load_safetensors(
    const std::filesystem::path& path,
    Device device = Device::cpu(),
    DType dtype = DType::Float32);

} // namespace citrius
