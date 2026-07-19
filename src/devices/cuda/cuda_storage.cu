#include "impl/cuda_storage.h"
#include "impl/cuda_context.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void select_device(int index) {
    check_cuda(cudaSetDevice(index), "failed to select CUDA device");
}

cudaStream_t stream(const std::shared_ptr<CudaExecutionContext>& context) {
    return static_cast<cudaStream_t>(context->stream());
}

} // namespace

CudaMemTensorStorageImpl::CudaMemTensorStorageImpl(
    std::size_t nbytes,
    DType dtype,
    int device_index)
    : CudaMemTensorStorageImpl(nbytes, dtype, cuda_execution_context(device_index)) {}

CudaMemTensorStorageImpl::CudaMemTensorStorageImpl(
    std::size_t nbytes,
    DType dtype,
    std::shared_ptr<CudaExecutionContext> context)
    : nbytes_(nbytes),
      dtype_(dtype),
      device_index_(context ? context->device_index() : 0),
      context_(std::move(context)),
      allocation_(nbytes_, context_) {
    if (!context_) throw std::invalid_argument("CUDA storage requires an execution context");
}

CudaMemTensorStorageImpl::~CudaMemTensorStorageImpl() = default;

TensorStorageType CudaMemTensorStorageImpl::type() const { return TensorStorageType::CudaMemory; }
DeviceType CudaMemTensorStorageImpl::device_type() const { return DeviceType::CUDA; }
DType CudaMemTensorStorageImpl::dtype() const { return dtype_; }
std::size_t CudaMemTensorStorageImpl::nbytes() const { return nbytes_; }
StorageHandle CudaMemTensorStorageImpl::handle() { return {allocation_.data(), nbytes_}; }
StorageHandle CudaMemTensorStorageImpl::handle() const { return {allocation_.data(), nbytes_}; }
int CudaMemTensorStorageImpl::device_index() const { return device_index_; }
const std::shared_ptr<CudaExecutionContext>& CudaMemTensorStorageImpl::execution_context() const {
    return context_;
}

std::shared_ptr<ITensorStorage> CudaMemTensorStorageImpl::clone() const {
    auto copied = std::make_shared<CudaMemTensorStorageImpl>(nbytes_, dtype_, context_);
    select_device(device_index_);
    if (nbytes_ != 0) {
        check_cuda(cudaMemcpyAsync(copied->allocation_.data(), allocation_.data(), nbytes_,
                                   cudaMemcpyDeviceToDevice,
                                   stream(context_)),
                   "failed to copy CUDA memory");
        check_cuda(cudaStreamSynchronize(stream(context_)), "CUDA storage copy failed");
    }
    return copied;
}

void CudaMemTensorStorageImpl::copy_from_host(const void* data, std::size_t nbytes) {
    if (nbytes > nbytes_) throw std::invalid_argument("host data is larger than CUDA storage");
    select_device(device_index_);
    if (nbytes != 0) {
        check_cuda(cudaMemcpyAsync(allocation_.data(), data, nbytes, cudaMemcpyHostToDevice,
                                   stream(context_)),
                   "failed to copy data to CUDA");
        check_cuda(cudaStreamSynchronize(stream(context_)), "CUDA host-to-device copy failed");
    }
}

void CudaMemTensorStorageImpl::copy_to_host(void* data, std::size_t nbytes) const {
    if (nbytes > nbytes_) throw std::invalid_argument("host destination is larger than CUDA storage");
    select_device(device_index_);
    if (nbytes != 0) {
        check_cuda(cudaMemcpyAsync(data, allocation_.data(), nbytes, cudaMemcpyDeviceToHost,
                                   stream(context_)),
                   "failed to copy data from CUDA");
        check_cuda(cudaStreamSynchronize(stream(context_)), "CUDA device-to-host copy failed");
    }
}

} // namespace citrius::impl
