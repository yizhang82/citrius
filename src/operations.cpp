#include "operations.h"

#include "impl/cpu_device.h"
#include "exceptions.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cuda_device.h"
#endif

#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#endif

#include <string>

namespace citrius {
namespace {

using impl::CpuDeviceImpl;
#ifdef CITRIUS_HAS_CUDA
using impl::CudaDeviceImpl;
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

template <typename Operation>
Tensor dispatch(const Tensor& left, const Tensor& right, Operation operation) {
    require_matching_devices(left, right);
    const Device device = left.device();

    switch (device.type) {
        case DeviceType::CPU:
            return operation(CpuDeviceImpl(), left, right);
#ifdef CITRIUS_HAS_CUDA
        case DeviceType::CUDA:
            return operation(CudaDeviceImpl(device.index), left, right);
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
