#pragma once

#include "cuda_device.h"

namespace citrius::impl {

// CUDA device variant backed by CUTLASS's row-major Float32 GEMM.
class CutlassCudaDeviceImpl final : public CudaDeviceImpl {
public:
    explicit CutlassCudaDeviceImpl(int device_index = 0);

    Tensor matmul(const Tensor& a, const Tensor& b) const override;
};

} // namespace citrius::impl
