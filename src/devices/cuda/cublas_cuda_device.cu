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

    // Citrius stores A[m x k], B[k x n], and C[m x n] contiguously in
    // row-major order, while this cuBLAS API interprets buffers as column-major.
    // The same bytes that represent row-major A represent column-major A^T,
    // and likewise for B and C. We therefore reverse the operands and compute
    // C^T[n x m] = B^T[n x k] * A^T[k x m]. The resulting column-major C^T
    // bytes are exactly row-major C, so no physical transpose is needed.
    // This accounts for the (n, m, k) problem dimensions and leading
    // dimensions n for B^T, k for A^T, and n for C^T below.
    check_cublas(
        cublasSgemm(
            impl_->handle,                 // cuBLAS context for the selected CUDA device
            CUBLAS_OP_N,                   // use the column-major B^T view without another transpose
            CUBLAS_OP_N,                   // use the column-major A^T view without another transpose
            static_cast<int>(n),           // rows of the column-major result C^T
            static_cast<int>(m),           // columns of the column-major result C^T
            static_cast<int>(k),           // shared dimension of B^T and A^T
            &alpha,                        // scale applied to B^T * A^T (1.0)
            data(*bp),                     // row-major B bytes, interpreted as column-major B^T
            static_cast<int>(n),           // leading dimension of B^T
            data(*ap),                     // row-major A bytes, interpreted as column-major A^T
            static_cast<int>(k),           // leading dimension of A^T
            &beta,                         // scale applied to the previous output contents (0.0)
            data(*out.storage()),          // column-major C^T bytes, equivalent to row-major C
            static_cast<int>(n)),          // leading dimension of C^T
        "cuBLAS matmul failed");
    check_cuda(cudaDeviceSynchronize(), "CUDA cuBLAS matmul failed");
    return out;
}

} // namespace citrius::impl
