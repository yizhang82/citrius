#pragma once

#include "cuda_device.h"

namespace citrius::impl {

// CUDA device variant backed by CUTLASS's row-major Float32 GEMM.
class CutlassCudaDeviceImpl final : public CudaDeviceImpl {
public:
    explicit CutlassCudaDeviceImpl(int device_index = 0);

    void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const override;
};

} // namespace citrius::impl
