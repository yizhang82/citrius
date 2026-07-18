#include "cpu_device.h"
#include "cpu_storage.h"
#include "metal_device.h"
#include "metal_storage.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::unique_ptr<citrius::MetalDeviceImpl> make_metal_device(std::string* error_message) {
    try {
        return std::make_unique<citrius::MetalDeviceImpl>();
    } catch (const std::runtime_error& error) {
        *error_message = error.what();
        return nullptr;
    }
}

citrius::Tensor make_metal_tensor(
    const citrius::MetalDeviceImpl& device,
    citrius::Shape shape,
    const std::vector<float>& values) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage = std::static_pointer_cast<citrius::MetalMemTensorStorageImpl>(tensor.storage());
    storage->copy_from_host(values.data(), values.size() * sizeof(float));
    return tensor;
}

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
    auto storage = std::static_pointer_cast<citrius::MetalMemTensorStorageImpl>(tensor.storage());
    std::vector<float> values(static_cast<std::size_t>(tensor.numel()));
    storage->copy_to_host(values.data(), values.size() * sizeof(float));
    return values;
}

} // namespace

TEST(MetalDeviceTest, EmptyAllocatesMetalStorage) {
    std::string error_message;
    auto device = make_metal_device(&error_message);
    if (!device) {
        GTEST_SKIP() << error_message;
    }

    auto tensor = device->empty({2, 3}, citrius::DType::Float32);

    ASSERT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(tensor.device(), citrius::Device::metal());
    ASSERT_NE(tensor.storage(), nullptr);
    EXPECT_EQ(tensor.storage()->type(), citrius::TensorStorageType::MetalMemory);
    EXPECT_EQ(tensor.storage()->nbytes(), 6 * sizeof(float));
}

TEST(MetalDeviceTest, AddsFloat32Tensors) {
    std::string error_message;
    auto device = make_metal_device(&error_message);
    if (!device) {
        GTEST_SKIP() << error_message;
    }
    auto a = make_metal_tensor(*device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_metal_tensor(*device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

    auto result = device->add(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(MetalDeviceTest, SubtractsFloat32Tensors) {
    std::string error_message;
    auto device = make_metal_device(&error_message);
    if (!device) {
        GTEST_SKIP() << error_message;
    }
    auto a = make_metal_tensor(*device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    auto b = make_metal_tensor(*device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    auto result = device->sub(a, b);

    EXPECT_EQ(tensor_values(result), std::vector<float>({9.0f, 18.0f, 27.0f, 36.0f}));
}

TEST(MetalDeviceTest, MultipliesTwoDimensionalFloat32Matrices) {
    std::string error_message;
    auto device = make_metal_device(&error_message);
    if (!device) {
        GTEST_SKIP() << error_message;
    }
    auto a = make_metal_tensor(*device, {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto b = make_metal_tensor(*device, {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    auto result = device->matmul(a, b);

    EXPECT_EQ(result.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(tensor_values(result), std::vector<float>({58.0f, 64.0f, 139.0f, 154.0f}));
}

TEST(MetalDeviceTest, CopiesCpuStorageToMetalForOps) {
    const citrius::CpuDeviceImpl cpu_device;
    std::string error_message;
    auto metal_device = make_metal_device(&error_message);
    if (!metal_device) {
        GTEST_SKIP() << error_message;
    }
    auto a = make_cpu_tensor(cpu_device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto b = make_cpu_tensor(cpu_device, {2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});

    auto result = metal_device->add(a, b);

    EXPECT_EQ(result.storage()->type(), citrius::TensorStorageType::MetalMemory);
    EXPECT_EQ(tensor_values(result), std::vector<float>({11.0f, 22.0f, 33.0f, 44.0f}));
}

TEST(MetalDeviceTest, TensorCopyCreatesDeepCopyOfMetalStorage) {
    std::string error_message;
    auto device = make_metal_device(&error_message);
    if (!device) {
        GTEST_SKIP() << error_message;
    }
    auto tensor = make_metal_tensor(*device, {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});

    auto copied = tensor.copy();
    auto original_storage =
        std::static_pointer_cast<citrius::MetalMemTensorStorageImpl>(tensor.storage());

    const std::vector<float> updated_values = {100.0f, 2.0f, 3.0f, 4.0f};
    original_storage->copy_from_host(updated_values.data(), updated_values.size() * sizeof(float));

    EXPECT_EQ(copied.storage()->type(), citrius::TensorStorageType::MetalMemory);
    EXPECT_NE(copied.storage(), tensor.storage());
    EXPECT_EQ(tensor_values(tensor), std::vector<float>({100.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(tensor_values(copied), std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(MetalDeviceTest, RejectsCpuStorageWhenConversionPolicyIsError) {
    const citrius::CpuDeviceImpl cpu_device;
    std::string error_message;
    auto metal_device = make_metal_device(&error_message);
    if (!metal_device) {
        GTEST_SKIP() << error_message;
    }
    auto tensor = make_cpu_tensor(cpu_device, {2}, {1.0f, 2.0f});

    EXPECT_THROW(metal_device->ensure_storage(tensor.storage()), std::invalid_argument);
}
