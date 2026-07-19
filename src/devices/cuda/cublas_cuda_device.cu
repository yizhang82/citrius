#include "impl/batched_matmul_layout.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cuda_allocation.h"
#include "impl/cuda_context.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void require_matmul_inputs(const Tensor& a, const Tensor& b) {
    if (!a.defined())
        throw std::invalid_argument("left tensor is undefined");
    if (!b.defined())
        throw std::invalid_argument("right tensor is undefined");
    if (a.dtype() != DType::Float32)
        throw std::invalid_argument("left tensor must be Float32");
    if (b.dtype() != DType::Float32)
        throw std::invalid_argument("right tensor must be Float32");
    if (a.ndim() != 2 || b.ndim() != 2)
        throw std::invalid_argument("matmul expects 2D tensors");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul inner dimensions must match");
}

float* data(ITensorStorage& storage) {
    return static_cast<float*>(storage.handle().ptr);
}

cudaStream_t stream(const std::shared_ptr<CudaExecutionContext>& context) {
    return static_cast<cudaStream_t>(context->stream());
}

} // namespace

class CublasCudaDeviceImpl::Impl {
  public:
    explicit Impl(const std::shared_ptr<CudaExecutionContext>& context) {
        check_cuda(cudaSetDevice(context->device_index()), "failed to select CUDA device");
        check_cublas(cublasCreate(&handle), "failed to create cuBLAS handle");
        check_cublas(cublasSetStream(handle, stream(context)), "failed to set cuBLAS stream");
    }

    ~Impl() {
        if (handle)
            cublasDestroy(handle);
    }

    cublasHandle_t handle = nullptr;
};

CublasCudaDeviceImpl::CublasCudaDeviceImpl(int device_index)
    : CudaDeviceImpl(device_index), impl_(std::make_unique<Impl>(execution_context())) {}

CublasCudaDeviceImpl::~CublasCudaDeviceImpl() = default;

void CublasCudaDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_matmul_inputs(a, b);

    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    const auto n = b.shape()[1];
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX) {
        throw std::invalid_argument("cuBLAS matmul dimensions are too large");
    }

    if (out.shape() != Shape({m, n}))
        throw std::invalid_argument("matmul output shape must be [m, n]");
    if (m == 0 || n == 0)
        return;

    check_cuda(cudaSetDevice(device_index()), "failed to select CUDA device");
    if (k == 0) {
        check_cuda(cudaMemsetAsync(data(*out.storage()), 0, out.storage()->nbytes(),
                                   stream(execution_context())),
                   "failed to clear CUDA matmul output");
        return;
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
    check_cublas(cublasSgemm(impl_->handle,       // cuBLAS context for the selected CUDA device
                             CUBLAS_OP_N,         // use the column-major B^T view without another
                                                  // transpose
                             CUBLAS_OP_N,         // use the column-major A^T view without another
                                                  // transpose
                             static_cast<int>(n), // rows of the column-major result C^T
                             static_cast<int>(m), // columns of the column-major result C^T
                             static_cast<int>(k), // shared dimension of B^T and A^T
                             &alpha,              // scale applied to B^T * A^T (1.0)
                             data(*bp), // row-major B bytes, interpreted as column-major B^T
                             static_cast<int>(n), // leading dimension of B^T
                             data(*ap), // row-major A bytes, interpreted as column-major A^T
                             static_cast<int>(k), // leading dimension of A^T
                             &beta, // scale applied to the previous output contents (0.0)
                             data(*out.storage()), // column-major C^T bytes, equivalent to
                                                   // row-major C
                             static_cast<int>(n)), // leading dimension of C^T
                 "cuBLAS matmul failed");
}

Tensor CublasCudaDeviceImpl::batched_matmul(const Tensor& a, const Tensor& b) const {
    auto l = make_batched_layout(a, b);
    auto out = empty(l.output, DType::Float32);
    if (out.numel() == 0)
        return out;
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice),
         bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    auto m = a.shape()[a.ndim() - 2], k = a.shape().back(), n = b.shape().back();
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX || l.a_offsets.size() > INT_MAX)
        throw std::invalid_argument("cuBLAS batched dimensions are too large");
    std::vector<float*> ah(l.a_offsets.size()), bh(l.b_offsets.size()), ch(l.a_offsets.size());
    auto* ad = data(*ap);
    auto* bd = data(*bp);
    auto* cd = data(*out.storage());
    for (std::size_t i = 0; i < ah.size(); ++i) {
        ah[i] = ad + l.a_offsets[i];
        bh[i] = bd + l.b_offsets[i];
        ch[i] = cd + i * m * n;
    }
    auto bytes = ah.size() * sizeof(float*);
    CudaAllocation a_pointers(bytes, execution_context());
    CudaAllocation b_pointers(bytes, execution_context());
    CudaAllocation c_pointers(bytes, execution_context());
    auto** da = a_pointers.data_as<float*>();
    auto** db = b_pointers.data_as<float*>();
    auto** dc = c_pointers.data_as<float*>();
    check_cuda(cudaMemcpyAsync(da, ah.data(), bytes, cudaMemcpyHostToDevice,
                               stream(execution_context())), "cuBLAS batch pointers");
    check_cuda(cudaMemcpyAsync(db, bh.data(), bytes, cudaMemcpyHostToDevice,
                               stream(execution_context())), "cuBLAS batch pointers");
    check_cuda(cudaMemcpyAsync(dc, ch.data(), bytes, cudaMemcpyHostToDevice,
                               stream(execution_context())), "cuBLAS batch pointers");
    float alpha = 1, beta = 0;
    auto status = cublasSgemmBatched(impl_->handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)n, (int)m,
                                     (int)k, &alpha, (const float**)db, (int)n, (const float**)da,
                                     (int)k, &beta, dc, (int)n, (int)ah.size());
    check_cublas(status, "cuBLAS batched_matmul failed");
    return out;
}

} // namespace citrius::impl
