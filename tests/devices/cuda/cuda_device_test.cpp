#include "impl/cpu_device.h"
#include "impl/cpu_storage.h"
#include "impl/cuda_device.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cutlass_cuda_device.h"
#include "impl/cuda_storage.h"
#include "operations.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::unique_ptr<citrius::impl::CudaDeviceImpl> make_cuda_device(std::string* error) {
    try { return std::make_unique<citrius::impl::CudaDeviceImpl>(); }
    catch (const std::runtime_error& exception) { *error = exception.what(); return nullptr; }
}
citrius::Tensor make_cuda_tensor(const citrius::impl::CudaDeviceImpl& device, citrius::Shape shape, const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())->copy_from_host(input.data(), input.size() * sizeof(float));
    return tensor;
}
citrius::Tensor make_cpu_tensor(const citrius::impl::CpuDeviceImpl& device, citrius::Shape shape, const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    std::copy(input.begin(), input.end(), storage->data_as<float>());
    return tensor;
}
std::vector<float> values(const citrius::Tensor& tensor) {
    std::vector<float> result(static_cast<std::size_t>(tensor.numel()));
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())->copy_to_host(result.data(), result.size() * sizeof(float));
    return result;
}
} // namespace

TEST(CudaDeviceTest, EmptyAllocatesCudaStorage) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto tensor = device->empty({2, 3}, citrius::DType::Float32);
    EXPECT_EQ(tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(tensor.storage()->type(), citrius::impl::TensorStorageType::CudaMemory);
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
    citrius::impl::CpuDeviceImpl cpu; std::string error; auto cuda = make_cuda_device(&error); if (!cuda) GTEST_SKIP() << error;
    auto a = make_cpu_tensor(cpu, {2}, {1, 2}); auto b = make_cpu_tensor(cpu, {2}, {10, 20});
    EXPECT_EQ(values(cuda->add(a, b)), std::vector<float>({11, 22}));
    EXPECT_THROW(cuda->ensure_storage(a.storage()), std::invalid_argument);
}
TEST(CudaDeviceTest, TensorCopyCreatesDeepCopy) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    auto tensor = make_cuda_tensor(*device, {2}, {1, 2}); auto copied = tensor.copy(); const std::vector<float> changed = {100, 2};
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())->copy_from_host(changed.data(), changed.size() * sizeof(float));
    EXPECT_NE(tensor.storage(), copied.storage()); EXPECT_EQ(values(copied), std::vector<float>({1, 2}));
}

TEST(CudaDeviceTest, CublasMatmulMatchesReferenceForNonSquareMatrices) {
    std::string error; auto baseline = make_cuda_device(&error); if (!baseline) GTEST_SKIP() << error;
    citrius::impl::CublasCudaDeviceImpl cublas;
    auto a = make_cuda_tensor(*baseline, {3, 5}, {
        1, 2, 3, 4, 5,
        -1, 0, 2, -3, 4,
        0.5f, 1.5f, -2, 3, 0,
    });
    auto b = make_cuda_tensor(*baseline, {5, 2}, {
        2, -1,
        0, 3,
        4, 0.5f,
        -2, 1,
        1.5f, -4,
    });

    const auto expected = values(baseline->matmul(a, b));
    const auto actual = values(cublas.matmul(a, b));
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaDeviceTest, CutlassMatmulMatchesReferenceForNonSquareMatrices) {
    std::string error; auto reference = make_cuda_device(&error); if (!reference) GTEST_SKIP() << error;
    citrius::impl::CutlassCudaDeviceImpl cutlass;
    auto a = make_cuda_tensor(*reference, {3, 5}, {
        1, 2, 3, 4, 5,
        -1, 0, 2, -3, 4,
        0.5f, 1.5f, -2, 3, 0,
    });
    auto b = make_cuda_tensor(*reference, {5, 2}, {
        2, -1,
        0, 3,
        4, 0.5f,
        -2, 1,
        1.5f, -4,
    });

    const auto expected = values(reference->matmul(a, b));
    const auto actual = values(cutlass.matmul(a, b));
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaDeviceTest, TensorConstructorAndToTransferValues) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const std::vector<float> input = {1, 2, 3, 4};
    const citrius::Tensor cuda_tensor(input, {2, 2}, citrius::Device::cuda());

    EXPECT_EQ(cuda_tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(cuda_tensor.storage()->type(), citrius::impl::TensorStorageType::CudaMemory);
    const auto cpu_tensor = cuda_tensor.to(citrius::Device::cpu());
    auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(cpu_tensor.storage());
    EXPECT_EQ(std::vector<float>(storage->data_as<float>(), storage->data_as<float>() + 4), input);
}

TEST(CudaDeviceTest, TopLevelOperationsDispatchToCuda) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const citrius::Tensor left(std::vector<float>{1, 2}, citrius::Device::cuda());
    const citrius::Tensor right(std::vector<float>{10, 20}, citrius::Device::cuda());

    EXPECT_EQ(values(citrius::add(left, right)), std::vector<float>({11, 22}));
}

TEST(CudaDeviceTest, TopLevelMatmulUsesConfiguredCudaBackend) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const citrius::Tensor a(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, citrius::Device::cuda());
    const citrius::Tensor b(std::vector<float>{7, 8, 9, 10, 11, 12}, {3, 2}, citrius::Device::cuda());

    EXPECT_EQ(values(citrius::matmul(a, b)), std::vector<float>({58, 64, 139, 154}));
}

TEST(CudaDeviceTest, ToStringCopiesValuesForDisplay) {
    std::string error; auto device = make_cuda_device(&error); if (!device) GTEST_SKIP() << error;
    const citrius::Tensor tensor(std::vector<float>{1, 2}, citrius::Device::cuda());

    EXPECT_EQ(
        tensor.to_string(),
        "tensor([1, 2], shape=[2], dtype=float32, device=cuda:0)");
}
