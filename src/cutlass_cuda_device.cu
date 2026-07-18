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
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cutlass(cutlass::Status status, const char* operation) {
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
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

CutlassCudaDeviceImpl::CutlassCudaDeviceImpl(int device_index)
    : CudaDeviceImpl(device_index) {}

Tensor CutlassCudaDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_matmul_inputs(a, b);

    const auto m = a.shape()[0];
    const auto k = a.shape()[1];
    const auto n = b.shape()[1];
    if (m > INT_MAX || k > INT_MAX || n > INT_MAX) {
        throw std::invalid_argument("CUTLASS matmul dimensions are too large");
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
    using RowMajor = cutlass::layout::RowMajor;
    using Gemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

    typename Gemm::Arguments arguments(
        {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
        {data(*ap), static_cast<int>(k)},
        {data(*bp), static_cast<int>(n)},
        {data(*out.storage()), static_cast<int>(n)},
        {data(*out.storage()), static_cast<int>(n)},
        {1.0f, 0.0f});

    Gemm gemm;
    check_cutlass(gemm.can_implement(arguments), "CUTLASS cannot implement matmul");
    check_cutlass(gemm(arguments), "CUTLASS matmul failed");
    check_cuda(cudaDeviceSynchronize(), "CUDA CUTLASS matmul failed");
    return out;
}

} // namespace citrius::impl
