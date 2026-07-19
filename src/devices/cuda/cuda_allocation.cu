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

void CudaAllocation::reset() noexcept {
    if (data_) {
        cudaSetDevice(context_->device_index());
        cudaFreeAsync(data_, static_cast<cudaStream_t>(context_->stream()));
        data_ = nullptr;
    }
    nbytes_ = 0;
}

} // namespace citrius::impl
