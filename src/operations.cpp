#include "operations.h"

#include "impl/cpu_device.h"
#include "impl/multi_thread_cpu_device.h"
#include "exceptions.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cutlass_cuda_device.h"
#endif

#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#endif

#include <string>
#include <cstdlib>
#include <memory>
#include <string_view>

namespace citrius {
namespace {

using impl::CpuDeviceImpl;
using impl::MultiThreadCpuDeviceImpl;
#ifdef CITRIUS_HAS_CUDA
using impl::CudaDeviceImpl;
using impl::CublasCudaDeviceImpl;
using impl::CutlassCudaDeviceImpl;
#endif
#ifdef CITRIUS_HAS_METAL
using impl::MetalDeviceImpl;
#endif

void require_matching_devices(const Tensor& left, const Tensor& right) {
    if (left.device() != right.device()) {
        throw DeviceMismatchException(
            "tensor devices must match for binary operations");
    }
}

#ifdef CITRIUS_HAS_CUDA
std::unique_ptr<impl::IDevice> cuda_device(int device_index) {
    const char* configured_backend = std::getenv("CITRIUS_CUDA_BACKEND");
    if (configured_backend) {
        const std::string_view backend(configured_backend);
        if (backend == "cublas") return std::make_unique<CublasCudaDeviceImpl>(device_index);
        if (backend == "reference") return std::make_unique<CudaDeviceImpl>(device_index);
        if (backend == "cutlass") return std::make_unique<CutlassCudaDeviceImpl>(device_index);
        throw CitriusException(
            "CITRIUS_CUDA_BACKEND must be 'cublas', 'cutlass', or 'reference', got '" +
            std::string(backend) + "'");
    }
#ifdef CITRIUS_CUDA_DEFAULT_CUBLAS
    return std::make_unique<CublasCudaDeviceImpl>(device_index);
#else
    return std::make_unique<CudaDeviceImpl>(device_index);
#endif
}
#endif

template <typename Operation>
Tensor dispatch(const Tensor& left, const Tensor& right, Operation operation) {
    require_matching_devices(left, right);
    const Device device = left.device();

    switch (device.type) {
        case DeviceType::CPU:
            return operation(MultiThreadCpuDeviceImpl(), left, right);
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA: {
            auto cuda = cuda_device(device.index);
            return operation(*cuda, left, right);
        }
#endif
#ifdef CITRIUS_HAS_METAL
        case DeviceType::Metal:
            if (device.index != 0) {
                throw CitriusException("Metal device index is not available");
            }
            return operation(MetalDeviceImpl(), left, right);
#endif
        default:
            throw CitriusException("tensor backend is not enabled");
    }
}

} // namespace

Tensor add(const Tensor& left, const Tensor& right) {
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.add(a, b);
    });
}

Tensor sub(const Tensor& left, const Tensor& right) {
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.sub(a, b);
    });
}

Tensor matmul(const Tensor& left, const Tensor& right) {
    return dispatch(left, right, [](const auto& device, const Tensor& a, const Tensor& b) {
        return device.matmul(a, b);
    });
}

Tensor operator+(const Tensor& left, const Tensor& right) {
    return add(left, right);
}

Tensor operator-(const Tensor& left, const Tensor& right) {
    return sub(left, right);
}

Tensor operator*(const Tensor& left, const Tensor& right) {
    return matmul(left, right);
}

} // namespace citrius
