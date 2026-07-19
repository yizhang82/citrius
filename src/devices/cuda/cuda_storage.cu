#include "impl/cuda_context.h"
#include "impl/cuda_storage.h"

#include <stdexcept>
#include <utility>

namespace citrius::impl {

CudaMemTensorStorageImpl::CudaMemTensorStorageImpl(std::size_t nbytes, DType dtype,
                                                   int device_index)
    : CudaMemTensorStorageImpl(nbytes, dtype, cuda_execution_context(device_index)) {}

CudaMemTensorStorageImpl::CudaMemTensorStorageImpl(std::size_t nbytes, DType dtype,
                                                   std::shared_ptr<CudaExecutionContext> context)
    : nbytes_(nbytes), dtype_(dtype), device_index_(context ? context->device_index() : 0),
      context_(std::move(context)), allocation_(nbytes_, context_) {
    if (!context_)
        throw std::invalid_argument("CUDA storage requires an execution context");
}

CudaMemTensorStorageImpl::~CudaMemTensorStorageImpl() = default;

TensorStorageType CudaMemTensorStorageImpl::type() const {
    return TensorStorageType::CudaMemory;
}
DeviceType CudaMemTensorStorageImpl::device_type() const {
    return DeviceType::CUDA;
}
DType CudaMemTensorStorageImpl::dtype() const {
    return dtype_;
}
std::size_t CudaMemTensorStorageImpl::nbytes() const {
    return nbytes_;
}
StorageHandle CudaMemTensorStorageImpl::handle() {
    return {allocation_.data(), nbytes_};
}
StorageHandle CudaMemTensorStorageImpl::handle() const {
    return {allocation_.data(), nbytes_};
}
int CudaMemTensorStorageImpl::device_index() const {
    return device_index_;
}
const std::shared_ptr<CudaExecutionContext>& CudaMemTensorStorageImpl::execution_context() const {
    return context_;
}

std::shared_ptr<ITensorStorage> CudaMemTensorStorageImpl::clone() const {
    auto copied = std::make_shared<CudaMemTensorStorageImpl>(nbytes_, dtype_, context_);
    copied->allocation_.copy_from_device_async(allocation_, nbytes_);
    return copied;
}

void CudaMemTensorStorageImpl::copy_from_host(const void* data, std::size_t nbytes) {
    if (nbytes > nbytes_)
        throw std::invalid_argument("host data is larger than CUDA storage");
    if (nbytes != 0) {
        allocation_.copy_from_host_async(data, nbytes);
        allocation_.synchronize();
    }
}

void CudaMemTensorStorageImpl::copy_to_host(
    void* data,
    std::size_t nbytes,
    std::size_t source_offset) const {
    if (source_offset > nbytes_ || nbytes > nbytes_ - source_offset)
        throw std::invalid_argument("host destination is larger than CUDA storage");
    if (nbytes != 0) {
        allocation_.copy_to_host_async(data, nbytes, source_offset);
        allocation_.synchronize();
    }
}

} // namespace citrius::impl
