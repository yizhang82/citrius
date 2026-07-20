#include "impl/batched_matmul_layout.h"
#include "impl/cutlass_cuda_device.h"
#include "impl/cuda_matmul_layout.h"
#include "tensor_utils.h"
#include "impl/cuda_context.h"

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
    ENSURE_TENSOR_DEFINED(a);
    ENSURE_TENSOR_DEFINED(b);
    ENSURE_TENSOR_DTYPE(a, DType::Float32);
    ENSURE_TENSOR_DTYPE(b, DType::Float32);
    ENSURE_TENSOR_DIM(a, 2);
    ENSURE_TENSOR_DIM(b, 2);
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul inner dimensions must match");
}

float* data(ITensorStorage& storage) {
    return static_cast<float*>(storage.handle().ptr);
}

cudaStream_t stream(const std::shared_ptr<CudaExecutionContext>& context) {
    return static_cast<cudaStream_t>(context->stream());
}

template <typename LayoutA, typename LayoutB>
void launch_gemm(
    float* a, int lda, float* b, int ldb, float* output,
    int m, int n, int k, cudaStream_t execution_stream) {
    using RowMajor = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, float, RowMajor>;
    typename Gemm::Arguments arguments(
        {m, n, k}, {a, lda}, {b, ldb}, {output, n}, {output, n}, {1.0f, 0.0f});
    Gemm gemm;
    check_cutlass(gemm.can_implement(arguments), "CUTLASS cannot implement matmul");
    check_cutlass(gemm(arguments, nullptr, execution_stream), "CUTLASS matmul failed");
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

    ENSURE_TENSOR_SHAPE(out, Shape({m, n}));
    if (m == 0 || n == 0)
        return;

    check_cuda(cudaSetDevice(device_index()), "failed to select CUDA device");
    if (k == 0) {
        check_cuda(cudaMemsetAsync(data(*out.storage()), 0, out.storage()->nbytes(),
                                   stream(execution_context())),
                   "failed to clear CUDA matmul output");
        return;
    }

    Tensor packed_a = cuda_matrix_layout(a).supported() ? a : contiguous(a);
    Tensor packed_b = cuda_matrix_layout(b).supported() ? b : contiguous(b);
    const auto a_layout = cuda_matrix_layout(packed_a);
    const auto b_layout = cuda_matrix_layout(packed_b);
    if (a_layout.leading_dimension > INT_MAX || b_layout.leading_dimension > INT_MAX)
        throw std::invalid_argument("CUTLASS matmul strides are too large");
    auto ap = ensure_storage(packed_a.storage(), ConversionPolicy::CopyToDevice);
    auto bp = ensure_storage(packed_b.storage(), ConversionPolicy::CopyToDevice);
    auto* a_data = data(*ap) + a_layout.storage_offset;
    auto* b_data = data(*bp) + b_layout.storage_offset;

    // Unlike the traditional cuBLAS API, CUTLASS lets us declare the physical
    // layout of every operand explicitly. All three layouts are RowMajor, so
    // CUTLASS computes C[m x n] = A[m x k] * B[k x n] directly; no operand
    // reversal or transpose reinterpretation is required.
    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;
    const auto lda = static_cast<int>(a_layout.leading_dimension);
    const auto ldb = static_cast<int>(b_layout.leading_dimension);
    auto* output = data(*out.storage());
    const auto execution_stream = stream(execution_context());
    if (a_layout.type == CudaMatrixLayoutType::RowMajor &&
        b_layout.type == CudaMatrixLayoutType::RowMajor) {
        launch_gemm<RowMajor, RowMajor>(a_data, lda, b_data, ldb, output, m, n, k, execution_stream);
    } else if (a_layout.type == CudaMatrixLayoutType::RowMajor) {
        launch_gemm<RowMajor, ColumnMajor>(a_data, lda, b_data, ldb, output, m, n, k, execution_stream);
    } else if (b_layout.type == CudaMatrixLayoutType::RowMajor) {
        launch_gemm<ColumnMajor, RowMajor>(a_data, lda, b_data, ldb, output, m, n, k, execution_stream);
    } else {
        launch_gemm<ColumnMajor, ColumnMajor>(a_data, lda, b_data, ldb, output, m, n, k, execution_stream);
    }
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
        check_cutlass(gemm(args, nullptr, stream(execution_context())),
                      "CUTLASS batched_matmul failed");
    }
    return out;
}

} // namespace citrius::impl
