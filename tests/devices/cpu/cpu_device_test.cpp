#include "impl/cpu_device.h"
#include "impl/multi_thread_cpu_device.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using citrius::impl::CpuDeviceImpl;
using citrius::impl::CpuMemTensorStorageImpl;

citrius::Tensor make_cpu_tensor(
    const CpuDeviceImpl& device,
    citrius::Shape shape,
    const std::vector<float>& values) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(tensor.storage());
    float* data = storage->data_as<float>();

    for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }

    return tensor;
}

std::vector<float> tensor_values(const citrius::Tensor& tensor) {
    auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(CpuDeviceTest, UsesExplicitThreadCount) {
    const citrius::impl::MultiThreadCpuDeviceImpl device(3);
    EXPECT_EQ(device.thread_count(), 3u);
}

TEST(CpuDeviceTest, MultiThreadAddAndSubMatchReferenceForUnevenPartitions) {
    const CpuDeviceImpl reference;
    const citrius::impl::MultiThreadCpuDeviceImpl multi_thread(3);
    constexpr std::size_t count = 131'073;
    std::vector<float> a_values(count), b_values(count);
    for (std::size_t i = 0; i < count; ++i) {
        a_values[i] = static_cast<float>(i % 31);
        b_values[i] = static_cast<float>(i % 17);
    }
    auto a = make_cpu_tensor(reference, {static_cast<std::int64_t>(count)}, a_values);
    auto b = make_cpu_tensor(reference, {static_cast<std::int64_t>(count)}, b_values);

    EXPECT_EQ(tensor_values(multi_thread.add(a, b)), tensor_values(reference.add(a, b)));
    EXPECT_EQ(tensor_values(multi_thread.sub(a, b)), tensor_values(reference.sub(a, b)));
}

TEST(CpuDeviceTest, MultiThreadMatmulMatchesReferenceForUnevenRows) {
    const CpuDeviceImpl reference;
    const citrius::impl::MultiThreadCpuDeviceImpl multi_thread(3);
    std::vector<float> a_values(7 * 11), b_values(11 * 5);
    for (std::size_t i = 0; i < a_values.size(); ++i) a_values[i] = static_cast<float>(i % 13) - 6.0f;
    for (std::size_t i = 0; i < b_values.size(); ++i) b_values[i] = static_cast<float>(i % 7) - 3.0f;
    auto a = make_cpu_tensor(reference, {7, 11}, a_values);
    auto b = make_cpu_tensor(reference, {11, 5}, b_values);

    EXPECT_EQ(tensor_values(multi_thread.matmul(a, b)), tensor_values(reference.matmul(a, b)));
}

TEST(CpuDeviceTest, EmptyAllocatesCpuStorage) {
    const CpuDeviceImpl device;

    auto tensor = device.empty({2, 3}, citrius::DType::Float32);

    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(tensor.device(), citrius::Device::cpu());
    ASSERT_NE(tensor.storage(), nullptr);
    EXPECT_EQ(tensor.storage()->type(), citrius::impl::TensorStorageType::CpuMemory);
    EXPECT_EQ(tensor.storage()->nbytes(), 6 * sizeof(float));
}

TEST(CpuDeviceTest, AddsFloat32Tensors) {
    const CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_cpu_tensor(device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

    auto result = device.add(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(CpuDeviceTest, SubtractsFloat32Tensors) {
    const CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto b = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    auto result = device.sub(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({9.0f, 18.0f, 27.0f, 36.0f}));
}

TEST(CpuDeviceTest, MultipliesTwoDimensionalFloat32Matrices) {
    const CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b = make_cpu_tensor(device, {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    auto result = device.matmul(a, b);

    EXPECT_EQ(result.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(tensor_values(result), std::vector<float>({58.0f, 64.0f, 139.0f, 154.0f}));
}

TEST(CpuDeviceTest, RejectsShapeMismatchForAdd) {
    const CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_cpu_tensor(device, {4}, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_THROW(device.add(a, b), std::invalid_argument);
}
