#include "impl/cuda_context.h"

#include <cuda_runtime.h>

#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace citrius::impl {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

} // namespace

CudaExecutionContext::CudaExecutionContext(int device_index) : device_index_(device_index) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "failed to query CUDA devices");
    if (device_index_ < 0 || device_index_ >= count)
        throw std::runtime_error("CUDA device index is not available");
    check_cuda(cudaSetDevice(device_index_), "failed to select CUDA device");
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "failed to create CUDA execution stream");
    stream_ = stream;
}

CudaExecutionContext::~CudaExecutionContext() {
    if (stream_) {
        cudaSetDevice(device_index_);
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
    }
}

int CudaExecutionContext::device_index() const { return device_index_; }
void* CudaExecutionContext::stream() const { return stream_; }

std::shared_ptr<CudaExecutionContext> cuda_execution_context(int device_index) {
    static std::shared_mutex mutex;
    static std::unordered_map<int, std::shared_ptr<CudaExecutionContext>> contexts;

    {
        std::shared_lock lock(mutex);
        if (const auto existing = contexts.find(device_index); existing != contexts.end())
            return existing->second;
    }

    std::unique_lock lock(mutex);
    if (const auto existing = contexts.find(device_index); existing != contexts.end())
        return existing->second;
    auto context = std::make_shared<CudaExecutionContext>(device_index);
    contexts[device_index] = context;
    return context;
}

} // namespace citrius::impl
