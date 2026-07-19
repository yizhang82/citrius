#include "impl/batched_matmul_layout.h"
#include "impl/cutlass_cuda_device.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#include <cuda_runtime.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cutlass(cutlass::Status status, const char* operation) {
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
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

} // namespace

CutlassCudaDeviceImpl::CutlassCudaDeviceImpl(int device_index) : CudaDeviceImpl(device_index) {}

void CutlassCudaDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const {
    require_matmul_inputs(a, b);

    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    const auto n = b.shape()[1];
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX) {
        throw std::invalid_argument("CUTLASS matmul dimensions are too large");
    }

    if (out.shape() != Shape({m, n}))
        throw std::invalid_argument("matmul output shape must be [m, n]");
    if (m == 0 || n == 0)
        return;

    check_cuda(cudaSetDevice(device_index()), "failed to select CUDA device");
    if (k == 0) {
        check_cuda(cudaMemset(data(*out.storage()), 0, out.storage()->nbytes()),
                   "failed to clear CUDA matmul output");
        return;
    }

    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);

    // Unlike the traditional cuBLAS API, CUTLASS lets us declare the physical
    // layout of every operand explicitly. All three layouts are RowMajor, so
    // CUTLASS computes C[m x n] = A[m x k] * B[k x n] directly; no operand
    // reversal or transpose reinterpretation is required.
    using RowMajor = cutlass::layout::RowMajor;
    // Template arguments describe the element type and layout of A, B, and C.
    // The accumulator defaults to float for this Float32 GEMM specialization.
    using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

    // CUTLASS represents each matrix as {device pointer, leading dimension}.
    // For packed row-major storage, the leading dimension is the number of
    // columns: k for A and n for B, the source C, and destination D.
    typename Gemm::Arguments arguments(
        {static_cast<int>(m), static_cast<int>(n),
         static_cast<int>(k)},                       // GEMM problem size {m, n, k}
        {data(*ap), static_cast<int>(k)},            // A[m x k], row stride k
        {data(*bp), static_cast<int>(n)},            // B[k x n], row stride n
        {data(*out.storage()), static_cast<int>(n)}, // source C[m x n], row stride n
        {data(*out.storage()), static_cast<int>(n)}, // destination D[m x n], row stride n
        {1.0f, 0.0f});                               // D = 1 * (A * B) + 0 * C

    Gemm gemm;
    // Reject shapes, alignments, or device capabilities unsupported by this
    // generated kernel before attempting to launch it.
    check_cutlass(gemm.can_implement(arguments), "CUTLASS cannot implement matmul");
    // Initialize and launch the selected CUTLASS GEMM kernel on the default
    // stream. The call reports setup/launch errors through cutlass::Status.
    check_cutlass(gemm(arguments), "CUTLASS matmul failed");
    // Citrius operations are currently eager and synchronous, so do not return
    // until the kernel has completed and asynchronous CUDA errors are visible.
    check_cuda(cudaDeviceSynchronize(), "CUDA CUTLASS matmul failed");
}

Tensor CutlassCudaDeviceImpl::batched_matmul(const Tensor& a, const Tensor& b) const {
    auto l = make_batched_layout(a, b);
    auto out = empty(l.output, DType::Float32);
    if (out.numel() == 0)
        return out;
    auto ap = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice),
         bp = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    auto m = a.shape()[a.ndim() - 2], k = a.shape().back(), n = b.shape().back();
    using RowMajor = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
    auto* ad = data(*ap);
    auto* bd = data(*bp);
    auto* cd = data(*out.storage());
    for (std::size_t i = 0; i < l.a_offsets.size(); ++i) {
        typename Gemm::Arguments args({(int)m, (int)n, (int)k}, {ad + l.a_offsets[i], (int)k},
                                      {bd + l.b_offsets[i], (int)n}, {cd + i * m * n, (int)n},
                                      {cd + i * m * n, (int)n}, {1, 0});
        Gemm gemm;
        check_cutlass(gemm.can_implement(args), "CUTLASS cannot implement batched_matmul");
        check_cutlass(gemm(args), "CUTLASS batched_matmul failed");
    }
    check_cuda(cudaDeviceSynchronize(), "CUDA CUTLASS batched_matmul failed");
    return out;
}

} // namespace citrius::impl
