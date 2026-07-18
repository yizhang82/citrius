#include "cpu_device.h"
#include "cpu_storage.h"
#include "cuda_device.h"
#include "cuda_storage.h"
#include "operations.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::unique_ptr<citrius::CudaDeviceImpl> make_cuda_device(std::string* error) {
    try { return std::make_unique<citrius::CudaDeviceImpl>(); }
    catch (const std::runtime_error& exception) { *error = exception.what(); return nullptr; }
}
citrius::Tensor make_cuda_tensor(const citrius::CudaDeviceImpl& device, citrius::Shape shape, const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    std::static_pointer_cast<citrius::CudaMemTensorStorageImpl>(tensor.storage())->copy_from_host(input.data(), input.size() * sizeof(float));
    return tensor;
}
citrius::Tensor make_cpu_tensor(const citrius::CpuDeviceImpl& device, citrius::Shape shape, const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage = std::static_pointer_cast<citrius::CpuMemTensorStorageImpl>(tensor.storage());
    std::copy(input.begin(), input.end(), storage->data_as<float>());
    return tensor;
}
std::vector<float> values(const citrius::Tensor& tensor) {
    std::vector<float> result(static_cast<std::size_t>(tensor.numel()));
    std::static_pointer_cast<citrius::CudaMemTensorStorageImpl>(tensor.storage())->copy_to_host(result.data(), result.size() * sizeof(float));
    return result;
}
} // namespace

TEST(CudaDeviceTest, EmptyAllocatesCudaStorage) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto tensor = device->empty({2, 3}, citrius::DType::Float32);
    EXPECT_EQ(tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(tensor.storage()->type(), citrius::TensorStorageType::CudaMemory);
    EXPECT_EQ(tensor.storage()->nbytes(), 6 * sizeof(float));
}
TEST(CudaDeviceTest, AddsAndSubtractsFloat32Tensors) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto a = make_cuda_tensor(*device, {2, 2}, {10, 20, 30, 40}); auto b = make_cuda_tensor(*device, {2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(values(device->add(a, b)), std::vector<float>({11, 22, 33, 44}));
    EXPECT_EQ(values(device->sub(a, b)), std::vector<float>({9, 18, 27, 36}));
}
TEST(CudaDeviceTest, MultipliesTwoDimensionalFloat32Matrices) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto a = make_cuda_tensor(*device, {2, 3}, {1, 2, 3, 4, 5, 6}); auto b = make_cuda_tensor(*device, {3, 2}, {7, 8, 9, 10, 11, 12});
    auto result = device->matmul(a, b);
    EXPECT_EQ(result.shape(), citrius::Shape({2, 2})); EXPECT_EQ(values(result), std::vector<float>({58, 64, 139, 154}));
}
TEST(CudaDeviceTest, CopiesCpuStorageForOperations) {
    citrius::CpuDeviceImpl cpu; std::string error; auto cuda = make_cuda_device(&error); if (!cuda) GTEST_SKIP() << error;
    auto a = make_cpu_tensor(cpu, {2}, {1, 2}); auto b = make_cpu_tensor(cpu, {2}, {10, 20});
    EXPECT_EQ(values(cuda->add(a, b)), std::vector<float>({11, 22}));
    EXPECT_THROW(cuda->ensure_storage(a.storage()), std::invalid_argument);
}
TEST(CudaDeviceTest, TensorCopyCreatesDeepCopy) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto tensor = make_cuda_tensor(*device, {2}, {1, 2}); auto copied = tensor.copy(); const std::vector<float> changed = {100, 2};
    std::static_pointer_cast<citrius::CudaMemTensorStorageImpl>(tensor.storage())->copy_from_host(changed.data(), changed.size() * sizeof(float));
    EXPECT_NE(tensor.storage(), copied.storage()); EXPECT_EQ(values(copied), std::vector<float>({1, 2}));
}

TEST(CudaDeviceTest, TensorConstructorAndToTransferValues) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const std::vector<float> input = {1, 2, 3, 4};
    const citrius::Tensor cuda_tensor(input, {2, 2}, citrius::Device::cuda());

    EXPECT_EQ(cuda_tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(cuda_tensor.storage()->type(), citrius::TensorStorageType::CudaMemory);
    const auto cpu_tensor = cuda_tensor.to(citrius::Device::cpu());
    auto storage = std::static_pointer_cast<citrius::CpuMemTensorStorageImpl>(cpu_tensor.storage());
    EXPECT_EQ(std::vector<float>(storage->data_as<float>(), storage->data_as<float>() + 4), input);
}

TEST(CudaDeviceTest, TopLevelOperationsDispatchToCuda) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const citrius::Tensor left(std::vector<float>{1, 2}, citrius::Device::cuda());
    const citrius::Tensor right(std::vector<float>{10, 20}, citrius::Device::cuda());

    EXPECT_EQ(values(citrius::add(left, right)), std::vector<float>({11, 22}));
}
