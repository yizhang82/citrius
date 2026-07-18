#include "impl/cuda_storage.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

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

} // namespace

CudaMemTensorStorageImpl::CudaMemTensorStorageImpl(
    std::size_t nbytes,
    DType dtype,
    int device_index)
    : nbytes_(nbytes), dtype_(dtype), device_index_(device_index) {
    select_device(device_index_);
    if (nbytes_ != 0) {
        check_cuda(cudaMalloc(&data_, nbytes_), "failed to allocate CUDA memory");
    }
}

CudaMemTensorStorageImpl::~CudaMemTensorStorageImpl() {
    if (data_ != nullptr) {
        cudaSetDevice(device_index_);
        cudaFree(data_);
    }
}

TensorStorageType CudaMemTensorStorageImpl::type() const { return TensorStorageType::CudaMemory; }
DeviceType CudaMemTensorStorageImpl::device_type() const { return DeviceType::CUDA; }
DType CudaMemTensorStorageImpl::dtype() const { return dtype_; }
std::size_t CudaMemTensorStorageImpl::nbytes() const { return nbytes_; }
StorageHandle CudaMemTensorStorageImpl::handle() { return {data_, nbytes_}; }
StorageHandle CudaMemTensorStorageImpl::handle() const { return {data_, nbytes_}; }
int CudaMemTensorStorageImpl::device_index() const { return device_index_; }

std::shared_ptr<ITensorStorage> CudaMemTensorStorageImpl::clone() const {
    auto copied = std::make_shared<CudaMemTensorStorageImpl>(nbytes_, dtype_, device_index_);
    select_device(device_index_);
    if (nbytes_ != 0) {
        check_cuda(cudaMemcpy(copied->data_, data_, nbytes_, cudaMemcpyDeviceToDevice),
                   "failed to copy CUDA memory");
    }
    return copied;
}

void CudaMemTensorStorageImpl::copy_from_host(const void* data, std::size_t nbytes) {
    if (nbytes > nbytes_) throw std::invalid_argument("host data is larger than CUDA storage");
    select_device(device_index_);
    if (nbytes != 0) check_cuda(cudaMemcpy(data_, data, nbytes, cudaMemcpyHostToDevice), "failed to copy data to CUDA");
}

void CudaMemTensorStorageImpl::copy_to_host(void* data, std::size_t nbytes) const {
    if (nbytes > nbytes_) throw std::invalid_argument("host destination is larger than CUDA storage");
    select_device(device_index_);
    if (nbytes != 0) check_cuda(cudaMemcpy(data, data_, nbytes, cudaMemcpyDeviceToHost), "failed to copy data from CUDA");
}

} // namespace citrius::impl
