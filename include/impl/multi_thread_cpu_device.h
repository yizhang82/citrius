#pragma once

#include "cpu_device.h"

#include <cstddef>

namespace citrius::impl {

class MultiThreadCpuDeviceImpl final : public CpuDeviceImpl {
public:
    explicit MultiThreadCpuDeviceImpl(std::size_t thread_count = 0);

    std::size_t thread_count() const;
    void add_out(const Tensor& a, const Tensor& b, Tensor& out) const override;
    void sub_out(const Tensor& a, const Tensor& b, Tensor& out) const override;
    void matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const override;
    void batched_matmul_out(const Tensor& a, const Tensor& b, Tensor& out) const override;

private:
    std::size_t thread_count_;
};

} // namespace citrius::impl
