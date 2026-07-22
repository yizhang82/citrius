#pragma once

#include "cuda_device.h"

#include <memory>

namespace citrius::impl {

// CUDA device variant that uses NVIDIA cuBLAS for matrix multiplication while
// inheriting the reference CUDA implementations of the other operations.
class CublasCudaDeviceImpl final : public CudaDeviceImpl {
  public:
    explicit CublasCudaDeviceImpl(int device_index = 0);
    ~CublasCudaDeviceImpl() override;

    CublasCudaDeviceImpl(const CublasCudaDeviceImpl&) = delete;
    CublasCudaDeviceImpl& operator=(const CublasCudaDeviceImpl&) = delete;

    void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const override;
    Tensor matmul_float32_output(const Tensor& a, const Tensor& b) const;
    Tensor batched_matmul(const Tensor& a, const Tensor& b) const override;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace citrius::impl
