#include "impl/cuda_allocation.h"

#include "impl/cuda_context.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

} // namespace

CudaAllocation::CudaAllocation(
    std::size_t nbytes,
    std::shared_ptr<CudaExecutionContext> context)
    : nbytes_(nbytes), context_(std::move(context)) {
    if (!context_) throw std::invalid_argument("CUDA allocation requires an execution context");
    check_cuda(cudaSetDevice(context_->device_index()), "failed to select CUDA device");
    if (nbytes_ != 0) {
        check_cuda(cudaMallocAsync(&data_, nbytes_, static_cast<cudaStream_t>(context_->stream())),
                   "failed to allocate CUDA memory");
    }
}

CudaAllocation::~CudaAllocation() { reset(); }

CudaAllocation::CudaAllocation(CudaAllocation&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      nbytes_(std::exchange(other.nbytes_, 0)),
      context_(std::move(other.context_)) {}

CudaAllocation& CudaAllocation::operator=(CudaAllocation&& other) noexcept {
    if (this != &other) {
        reset();
        data_ = std::exchange(other.data_, nullptr);
        nbytes_ = std::exchange(other.nbytes_, 0);
        context_ = std::move(other.context_);
    }
    return *this;
}

void* CudaAllocation::data() const { return data_; }
std::size_t CudaAllocation::nbytes() const { return nbytes_; }

void CudaAllocation::copy_from_host_async(const void* source, std::size_t nbytes) {
    validate_copy_size(nbytes);
    if (nbytes == 0) return;
    if (!source) throw std::invalid_argument("CUDA copy source cannot be null");
    check_cuda(cudaSetDevice(context_->device_index()), "failed to select CUDA device");
    check_cuda(cudaMemcpyAsync(data_, source, nbytes, cudaMemcpyHostToDevice,
                               static_cast<cudaStream_t>(context_->stream())),
               "failed to copy data to CUDA");
}

void CudaAllocation::copy_to_host_async(void* destination, std::size_t nbytes) const {
    validate_copy_size(nbytes);
    if (nbytes == 0) return;
    if (!destination) throw std::invalid_argument("CUDA copy destination cannot be null");
    check_cuda(cudaSetDevice(context_->device_index()), "failed to select CUDA device");
    check_cuda(cudaMemcpyAsync(destination, data_, nbytes, cudaMemcpyDeviceToHost,
                               static_cast<cudaStream_t>(context_->stream())),
               "failed to copy data from CUDA");
}

void CudaAllocation::copy_from_device_async(const CudaAllocation& source, std::size_t nbytes) {
    validate_copy_size(nbytes);
    source.validate_copy_size(nbytes);
    if (nbytes == 0) return;
    if (context_->device_index() != source.context_->device_index()) {
        throw std::invalid_argument("CUDA device copy requires allocations on the same device");
    }
    check_cuda(cudaSetDevice(context_->device_index()), "failed to select CUDA device");
    check_cuda(cudaMemcpyAsync(data_, source.data_, nbytes, cudaMemcpyDeviceToDevice,
                               static_cast<cudaStream_t>(context_->stream())),
               "failed to copy CUDA memory");
}

void CudaAllocation::synchronize() const {
    if (!context_) throw std::invalid_argument("cannot synchronize an empty CUDA allocation");
    check_cuda(cudaSetDevice(context_->device_index()), "failed to select CUDA device");
    check_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(context_->stream())),
               "CUDA stream synchronization failed");
}

void CudaAllocation::validate_copy_size(std::size_t nbytes) const {
    if (nbytes > nbytes_) throw std::invalid_argument("copy is larger than CUDA allocation");
    if (nbytes != 0 && !data_) throw std::invalid_argument("cannot copy using an empty CUDA allocation");
}

void CudaAllocation::reset() noexcept {
    if (data_) {
        cudaSetDevice(context_->device_index());
        cudaFreeAsync(data_, static_cast<cudaStream_t>(context_->stream()));
        data_ = nullptr;
    }
    nbytes_ = 0;
}

} // namespace citrius::impl
