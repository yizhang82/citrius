#pragma once

#include "types.h"

#include <cstddef>
#include <memory>

namespace citrius {

enum class TensorStorageType {
    CpuMemory,
    MetalMemory,
    CudaMemory,
};

struct StorageHandle {
    void* ptr = nullptr;
    std::size_t nbytes = 0;
};

class ITensorStorage {
public:
    virtual ~ITensorStorage() = default;

    virtual TensorStorageType type() const = 0;
    virtual DeviceType device_type() const = 0;
    virtual DType dtype() const = 0;
    virtual std::size_t nbytes() const = 0;
    virtual StorageHandle handle() = 0;
    virtual StorageHandle handle() const = 0;
    virtual std::shared_ptr<ITensorStorage> clone() const = 0;
};

using TensorStoragePtr = std::shared_ptr<ITensorStorage>;

} // namespace citrius
