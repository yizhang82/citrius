#pragma once

#include "types.h"

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace citrius {

namespace impl {
class ITensorStorage;
}

class Tensor {
public:
    Tensor();
    Tensor(Shape shape, DType dtype = DType::Float32, Device device = Device::cpu());
    explicit Tensor(const std::vector<float>& values, Device device = Device::cpu());
    Tensor(const std::vector<float>& values, Shape shape, Device device = Device::cpu());
    Tensor(
        Shape shape,
        DType dtype,
        Device device,
        std::shared_ptr<impl::ITensorStorage> storage);

    const Shape& shape() const;
    DType dtype() const;
    Device device() const;
    std::shared_ptr<impl::ITensorStorage> storage() const;

    std::size_t ndim() const;
    std::int64_t numel() const;
    bool defined() const;
    Tensor copy() const;
    Tensor to(Device device) const;
    template <typename T>
    T item() const {
        DType expected;
        if constexpr (std::is_same_v<T, float>) expected = DType::Float32;
        else if constexpr (std::is_same_v<T, double>) expected = DType::Float64;
        else if constexpr (std::is_same_v<T, std::int32_t>) expected = DType::Int32;
        else if constexpr (std::is_same_v<T, std::int64_t>) expected = DType::Int64;
        else if constexpr (std::is_same_v<T, bool>) expected = DType::Bool;
        else static_assert(!sizeof(T), "Tensor::item does not support this type");
        T value{};
        copy_item_to_host(&value, expected);
        return value;
    }
    std::string to_string() const;

private:
    void copy_item_to_host(void* destination, DType expected) const;

    Shape shape_;
    DType dtype_ = DType::Float32;
    Device device_ = Device::cpu();
    std::shared_ptr<impl::ITensorStorage> storage_;
    bool defined_ = false;
};

std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

} // namespace citrius
