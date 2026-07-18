#include "cpu_device.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

citrius::Tensor make_cpu_tensor(
    const citrius::CpuDeviceImpl& device,
    citrius::Shape shape,
    const std::vector<float>& values) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage = std::static_pointer_cast<citrius::CpuMemTensorStorageImpl>(tensor.storage());
    float* data = storage->data_as<float>();

    for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = values[i];
    }

    return tensor;
}

std::vector<float> tensor_values(const citrius::Tensor& tensor) {
    auto storage = std::static_pointer_cast<citrius::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(CpuDeviceTest, EmptyAllocatesCpuStorage) {
    const citrius::CpuDeviceImpl device;

    auto tensor = device.empty({2, 3}, citrius::DType::Float32);

    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(tensor.device(), citrius::Device::cpu());
    ASSERT_NE(tensor.storage(), nullptr);
    EXPECT_EQ(tensor.storage()->type(), citrius::TensorStorageType::CpuMemory);
    EXPECT_EQ(tensor.storage()->nbytes(), 6 * sizeof(float));
}

TEST(CpuDeviceTest, AddsFloat32Tensors) {
    const citrius::CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_cpu_tensor(device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

    auto result = device.add(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(CpuDeviceTest, SubtractsFloat32Tensors) {
    const citrius::CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto b = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    auto result = device.sub(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({9.0f, 18.0f, 27.0f, 36.0f}));
}

TEST(CpuDeviceTest, MultipliesTwoDimensionalFloat32Matrices) {
    const citrius::CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b = make_cpu_tensor(device, {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    auto result = device.matmul(a, b);

    EXPECT_EQ(result.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(tensor_values(result), std::vector<float>({58.0f, 64.0f, 139.0f, 154.0f}));
}

TEST(CpuDeviceTest, RejectsShapeMismatchForAdd) {
    const citrius::CpuDeviceImpl device;
    auto a = make_cpu_tensor(device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_cpu_tensor(device, {4}, {1.0f, 2.0f, 3.0f, 4.0f});

    EXPECT_THROW(device.add(a, b), std::invalid_argument);
}
