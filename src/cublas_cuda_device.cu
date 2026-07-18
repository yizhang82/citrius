#include "impl/cublas_cuda_device.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

void require_matmul_inputs(const Tensor& a, const Tensor& b) {
    if (!a.defined()) throw std::invalid_argument("left tensor is undefined");
    if (!b.defined()) throw std::invalid_argument("right tensor is undefined");
    if (a.dtype() != DType::Float32) throw std::invalid_argument("left tensor must be Float32");
    if (b.dtype() != DType::Float32) throw std::invalid_argument("right tensor must be Float32");
    if (a.ndim() != 2 || b.ndim() != 2) throw std::invalid_argument("matmul expects 2D tensors");
    if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("matmul inner dimensions must match");
}

float* data(ITensorStorage& storage) {
    return static_cast<float*>(storage.handle().ptr);
}

} // namespace

class CublasCudaDeviceImpl::Impl {
public:
    explicit Impl(int device_index) {
        check_cuda(cudaSetDevice(device_index), "failed to select CUDA device");
        check_cublas(cublasCreate(&handle), "failed to create cuBLAS handle");
    }

    ~Impl() {
        if (handle) cublasDestroy(handle);
    }

    cublasHandle_t handle = nullptr;
};

CublasCudaDeviceImpl::CublasCudaDeviceImpl(int device_index)
    : CudaDeviceImpl(device_index), impl_(std::make_unique<Impl>(device_index)) {}

CublasCudaDeviceImpl::~CublasCudaDeviceImpl() = default;

Tensor CublasCudaDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_matmul_inputs(a, b);

    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    const auto n = b.shape()[1];
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX) {
        throw std::invalid_argument("cuBLAS matmul dimensions are too large");
    }

    auto out = empty({m, n}, DType::Float32);
    if (m == 0 || n == 0) return out;

    check_cuda(cudaSetDevice(device_index()), "failed to select CUDA device");
    if (k == 0) {
        check_cuda(cudaMemset(data(*out.storage()), 0, out.storage()->nbytes()), "failed to clear CUDA matmul output");
        return out;
    }

    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS uses column-major matrices. Reversing the row-major operands
    // computes C^T = B^T A^T without allocating physical transposes.
    check_cublas(
        cublasSgemm(
            impl_->handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            data(*bp),
            static_cast<int>(n),
            data(*ap),
            static_cast<int>(k),
            &beta,
            data(*out.storage()),
            static_cast<int>(n)),
        "cuBLAS matmul failed");
    check_cuda(cudaDeviceSynchronize(), "CUDA cuBLAS matmul failed");
    return out;
}

} // namespace citrius::impl
