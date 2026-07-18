#pragma once

#include "storage.h"

#include <cstddef>
#include <vector>

namespace citrius {

class CpuMemTensorStorageImpl final : public ITensorStorage {
public:
    CpuMemTensorStorageImpl(std::size_t nbytes, DType dtype);

    TensorStorageType type() const override;
    DeviceType device_type() const override;
    DType dtype() const override;
    std::size_t nbytes() const override;
    StorageHandle handle() override;
    StorageHandle handle() const override;
    std::shared_ptr<ITensorStorage> clone() const override;

    void* data();
    const void* data() const;

    template <typename T>
    T* data_as() {
        return static_cast<T*>(data());
    }

    template <typename T>
    const T* data_as() const {
        return static_cast<const T*>(data());
    }

private:
    std::vector<std::byte> buffer_;
    DType dtype_;
};

} // namespace citrius
