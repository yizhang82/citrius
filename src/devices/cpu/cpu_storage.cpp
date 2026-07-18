#include "impl/cpu_storage.h"

#include <cstring>
#include <utility>

namespace citrius::impl {

CpuMemTensorStorageImpl::CpuMemTensorStorageImpl(std::size_t nbytes, DType dtype)
    : buffer_(nbytes),
      dtype_(dtype) {}

TensorStorageType CpuMemTensorStorageImpl::type() const {
    return TensorStorageType::CpuMemory;
}

DeviceType CpuMemTensorStorageImpl::device_type() const {
    return DeviceType::CPU;
}

DType CpuMemTensorStorageImpl::dtype() const {
    return dtype_;
}

std::size_t CpuMemTensorStorageImpl::nbytes() const {
    return buffer_.size();
}

StorageHandle CpuMemTensorStorageImpl::handle() {
    return StorageHandle{buffer_.data(), buffer_.size()};
}

StorageHandle CpuMemTensorStorageImpl::handle() const {
    return StorageHandle{const_cast<std::byte*>(buffer_.data()), buffer_.size()};
}

std::shared_ptr<ITensorStorage> CpuMemTensorStorageImpl::clone() const {
    auto copied = std::make_shared<CpuMemTensorStorageImpl>(nbytes(), dtype_);
    std::memcpy(copied->data(), data(), nbytes());
    return copied;
}

void* CpuMemTensorStorageImpl::data() {
    return buffer_.data();
}

const void* CpuMemTensorStorageImpl::data() const {
    return buffer_.data();
}

} // namespace citrius::impl
