#include "impl/cpu_device.h"
#include "impl/cpu_storage.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cuda_device.h"
#include "impl/cuda_storage.h"
#include "impl/cutlass_cuda_device.h"
#include "operations.h"
#include "reduction_operations.h"
#include "tensor_factory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::unique_ptr<citrius::impl::CudaDeviceImpl> make_cuda_device(std::string* error) {
    try {
        return std::make_unique<citrius::impl::CudaDeviceImpl>();
    } catch (const std::runtime_error& exception) {
        *error = exception.what();
        return nullptr;
    }
}
citrius::Tensor make_cuda_tensor(const citrius::impl::CudaDeviceImpl& device, citrius::Shape shape,
                                 const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())
        ->copy_from_host(input.data(), input.size() * sizeof(float));
    return tensor;
}
citrius::Tensor make_cpu_tensor(const citrius::impl::CpuDeviceImpl& device, citrius::Shape shape,
                                const std::vector<float>& input) {
    auto tensor = device.empty(std::move(shape), citrius::DType::Float32);
    auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    std::copy(input.begin(), input.end(), storage->data_as<float>());
    return tensor;
}
std::vector<float> values(const citrius::Tensor& tensor) {
    std::vector<float> result(static_cast<std::size_t>(tensor.numel()));
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())
        ->copy_to_host(result.data(), result.size() * sizeof(float));
    return result;
}
} // namespace

TEST(CudaDeviceTest, EmptyAllocatesCudaStorage) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    auto tensor = device->empty({2, 3}, citrius::DType::Float32);
    EXPECT_EQ(tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(tensor.storage()->type(), citrius::impl::TensorStorageType::CudaMemory);
    EXPECT_EQ(tensor.storage()->nbytes(), 6 * sizeof(float));
}
TEST(CudaDeviceTest, AddsAndSubtractsFloat32Tensors) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    auto a = make_cuda_tensor(*device, {2, 2}, {10, 20, 30, 40});
    auto b = make_cuda_tensor(*device, {2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(values(device->add(a, b)), std::vector<float>({11, 22, 33, 44}));
    EXPECT_EQ(values(device->sub(a, b)), std::vector<float>({9, 18, 27, 36}));
}
TEST(CudaDeviceTest, MultipliesTwoDimensionalFloat32Matrices) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    auto a = make_cuda_tensor(*device, {2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = make_cuda_tensor(*device, {3, 2}, {7, 8, 9, 10, 11, 12});
    auto result = device->matmul(a, b);
    EXPECT_EQ(result.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(values(result), std::vector<float>({58, 64, 139, 154}));
}

TEST(CudaDeviceTest, BatchedMatmulMatchesAcrossBackends) {
    std::string error;
    auto reference = make_cuda_device(&error);
    if (!reference)
        GTEST_SKIP() << error;
    auto a = make_cuda_tensor(*reference, {2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    auto b = make_cuda_tensor(*reference, {2, 2, 2}, {1, 0, 0, 1, 2, 0, 0, 2});
    const std::vector<float> expected{1, 2, 3, 4, 10, 12, 14, 16};
    EXPECT_EQ(values(reference->batched_matmul(a, b)), expected);
    citrius::impl::CublasCudaDeviceImpl cublas;
    citrius::impl::CutlassCudaDeviceImpl cutlass;
    EXPECT_EQ(values(cublas.batched_matmul(a, b)), expected);
    EXPECT_EQ(values(cutlass.batched_matmul(a, b)), expected);
}
TEST(CudaDeviceTest, CopiesCpuStorageForOperations) {
    citrius::impl::CpuDeviceImpl cpu;
    std::string error;
    auto cuda = make_cuda_device(&error);
    if (!cuda)
        GTEST_SKIP() << error;
    auto a = make_cpu_tensor(cpu, {2}, {1, 2});
    auto b = make_cpu_tensor(cpu, {2}, {10, 20});
    EXPECT_EQ(values(cuda->add(a, b)), std::vector<float>({11, 22}));
    EXPECT_THROW(cuda->ensure_storage(a.storage()), std::invalid_argument);
}
TEST(CudaDeviceTest, TensorCopyCreatesDeepCopy) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    auto tensor = make_cuda_tensor(*device, {2}, {1, 2});
    auto copied = tensor.copy();
    const std::vector<float> changed = {100, 2};
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())
        ->copy_from_host(changed.data(), changed.size() * sizeof(float));
    EXPECT_NE(tensor.storage(), copied.storage());
    EXPECT_EQ(values(copied), std::vector<float>({1, 2}));
}
TEST(CudaDeviceTest, GridStrideAddAndSubCoverLargeTensorAndTail) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    constexpr std::size_t count = 4'000'003;
    std::vector<float> a_values(count), b_values(count);
    for (std::size_t i = 0; i < count; ++i) {
        a_values[i] = static_cast<float>(i % 31);
        b_values[i] = static_cast<float>(i % 17);
    }
    auto a = make_cuda_tensor(*device, {static_cast<std::int64_t>(count)}, a_values);
    auto b = make_cuda_tensor(*device, {static_cast<std::int64_t>(count)}, b_values);
    const auto sums = values(device->add(a, b));
    const auto differences = values(device->sub(a, b));
    for (std::size_t i = 0; i < count; ++i) {
        ASSERT_EQ(sums[i], a_values[i] + b_values[i]);
        ASSERT_EQ(differences[i], a_values[i] - b_values[i]);
    }
}

TEST(CudaDeviceTest, CublasMatmulMatchesReferenceForNonSquareMatrices) {
    std::string error;
    auto baseline = make_cuda_device(&error);
    if (!baseline)
        GTEST_SKIP() << error;
    citrius::impl::CublasCudaDeviceImpl cublas;
    auto a = make_cuda_tensor(*baseline, {3, 5},
                              {
                                  1,
                                  2,
                                  3,
                                  4,
                                  5,
                                  -1,
                                  0,
                                  2,
                                  -3,
                                  4,
                                  0.5f,
                                  1.5f,
                                  -2,
                                  3,
                                  0,
                              });
    auto b = make_cuda_tensor(*baseline, {5, 2},
                              {
                                  2,
                                  -1,
                                  0,
                                  3,
                                  4,
                                  0.5f,
                                  -2,
                                  1,
                                  1.5f,
                                  -4,
                              });

    const auto expected = values(baseline->matmul(a, b));
    const auto actual = values(cublas.matmul(a, b));
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaDeviceTest, CutlassMatmulMatchesReferenceForNonSquareMatrices) {
    std::string error;
    auto reference = make_cuda_device(&error);
    if (!reference)
        GTEST_SKIP() << error;
    citrius::impl::CutlassCudaDeviceImpl cutlass;
    auto a = make_cuda_tensor(*reference, {3, 5},
                              {
                                  1,
                                  2,
                                  3,
                                  4,
                                  5,
                                  -1,
                                  0,
                                  2,
                                  -3,
                                  4,
                                  0.5f,
                                  1.5f,
                                  -2,
                                  3,
                                  0,
                              });
    auto b = make_cuda_tensor(*reference, {5, 2},
                              {
                                  2,
                                  -1,
                                  0,
                                  3,
                                  4,
                                  0.5f,
                                  -2,
                                  1,
                                  1.5f,
                                  -4,
                              });

    const auto expected = values(reference->matmul(a, b));
    const auto actual = values(cutlass.matmul(a, b));
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
        EXPECT_NEAR(actual[i], expected[i], 1e-5f);
}

TEST(CudaDeviceTest, TensorConstructorAndToTransferValues) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const std::vector<float> input = {1, 2, 3, 4};
    const citrius::Tensor cuda_tensor(input, {2, 2}, citrius::Device::cuda());

    EXPECT_EQ(cuda_tensor.device(), citrius::Device::cuda());
    EXPECT_EQ(cuda_tensor.storage()->type(), citrius::impl::TensorStorageType::CudaMemory);
    const auto cpu_tensor = cuda_tensor.to(citrius::Device::cpu());
    auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(cpu_tensor.storage());
    EXPECT_EQ(std::vector<float>(storage->data_as<float>(), storage->data_as<float>() + 4), input);
}

TEST(CudaDeviceTest, TopLevelOperationsDispatchToCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor left(std::vector<float>{1, 2}, citrius::Device::cuda());
    const citrius::Tensor right(std::vector<float>{10, 20}, citrius::Device::cuda());

    EXPECT_EQ(values(citrius::add(left, right)), std::vector<float>({11, 22}));
}

TEST(CudaDeviceTest, BroadcastElementwiseOperationsStayOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor left(
        std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 1, 3}, citrius::Device::cuda());
    const citrius::Tensor right(
        std::vector<float>{10, 20, 30, 40}, {1, 4, 1}, citrius::Device::cuda());

    const auto added = citrius::add(left, right);
    const auto subtracted = citrius::sub(right, left);
    const auto multiplied = citrius::mul(left, right);
    const auto divided = citrius::div(right, left);
    const auto maxima = citrius::maximum(left, right);

    EXPECT_EQ(added.device(), citrius::Device::cuda());
    EXPECT_EQ(added.shape(), citrius::Shape({2, 4, 3}));
    EXPECT_EQ(
        values(added),
        (std::vector<float>{11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43,
                            14, 15, 16, 24, 25, 26, 34, 35, 36, 44, 45, 46}));
    EXPECT_EQ(
        values(subtracted),
        (std::vector<float>{9, 8, 7, 19, 18, 17, 29, 28, 27, 39, 38, 37,
                            6, 5, 4, 16, 15, 14, 26, 25, 24, 36, 35, 34}));
    EXPECT_EQ(
        values(multiplied),
        (std::vector<float>{10, 20, 30, 20, 40, 60, 30, 60, 90, 40, 80, 120,
                            40, 50, 60, 80, 100, 120, 120, 150, 180, 160, 200, 240}));
    const auto quotients = values(divided);
    EXPECT_NEAR(quotients[0], 10.0f, 1e-6f);
    EXPECT_NEAR(quotients[23], 40.0f / 6.0f, 1e-6f);
    EXPECT_EQ(values(maxima), values(right + citrius::mul(left, 0.0f)));
    EXPECT_THROW(
        citrius::add(
            left,
            citrius::Tensor(
                std::vector<float>{1, 2, 3, 4}, {2, 2}, citrius::Device::cuda())),
        std::invalid_argument);
}

TEST(CudaDeviceTest, ScalarElementwiseOperationsStayOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor input(std::vector<float>{1, 2, 4}, citrius::Device::cuda());

    EXPECT_EQ(values(input + 2.0f), std::vector<float>({3, 4, 6}));
    EXPECT_EQ(values(input - 2.0f), std::vector<float>({-1, 0, 2}));
    EXPECT_EQ(values(10.0f - input), std::vector<float>({9, 8, 6}));
    EXPECT_EQ(values(input * 3.0f), std::vector<float>({3, 6, 12}));
    EXPECT_EQ(values(input / 2.0f), std::vector<float>({0.5f, 1, 2}));
    EXPECT_EQ(values(8.0f / input), std::vector<float>({8, 4, 2}));
}

TEST(CudaDeviceTest, ReductionsSupportDimensionsAndKeepdim) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
        {2, 2, 3}, citrius::Device::cuda());

    const auto sums = citrius::sum(input, -1, true);
    const auto means = citrius::mean(input, std::vector<std::int64_t>{0, 2});
    const auto maxima = citrius::max(input, 1);
    const auto variances = citrius::variance(input, -1);

    EXPECT_EQ(sums.device(), citrius::Device::cuda());
    EXPECT_EQ(sums.shape(), citrius::Shape({2, 2, 1}));
    EXPECT_EQ(values(sums), std::vector<float>({6, 15, 24, 33}));
    EXPECT_EQ(values(means), std::vector<float>({5, 8}));
    EXPECT_EQ(values(maxima), std::vector<float>({4, 5, 6, 10, 11, 12}));
    for (const float value : values(variances))
        EXPECT_NEAR(value, 2.0f / 3.0f, 1e-6f);
}

TEST(CudaDeviceTest, FullReductionsMatchCpuReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const std::vector<float> input_values{-3, 1, 4, 1, 5, 9};
    const citrius::Tensor cuda_input(input_values, {2, 3}, citrius::Device::cuda());
    EXPECT_EQ(values(citrius::sum(cuda_input)), std::vector<float>({17.0f}));
    EXPECT_NEAR(values(citrius::mean(cuda_input))[0], 17.0f / 6.0f, 1e-6f);
    EXPECT_EQ(values(citrius::max(cuda_input)), std::vector<float>({9.0f}));
    EXPECT_NEAR(values(citrius::variance(cuda_input))[0], 14.138889f, 1e-5f);
}

TEST(CudaDeviceTest, UnaryMathOperationsStayOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;

    const citrius::Tensor exponential_input(
        std::vector<float>{-1.0f, 0.0f, 1.0f}, citrius::Device::cuda());
    const citrius::Tensor sqrt_input(
        std::vector<float>{0.0f, 1.0f, 4.0f, 9.0f}, citrius::Device::cuda());
    const citrius::Tensor power_input(
        std::vector<float>{-2.0f, 3.0f, 0.5f}, citrius::Device::cuda());

    const auto exponentials = citrius::exp(exponential_input);
    const auto roots = citrius::sqrt(sqrt_input);
    const auto powers = citrius::pow(power_input, 3.0f);

    EXPECT_EQ(exponentials.device(), citrius::Device::cuda());
    const auto exponential_values = values(exponentials);
    EXPECT_NEAR(exponential_values[0], std::exp(-1.0f), 1e-6f);
    EXPECT_NEAR(exponential_values[1], 1.0f, 1e-6f);
    EXPECT_NEAR(exponential_values[2], std::exp(1.0f), 1e-6f);
    EXPECT_EQ(values(roots), std::vector<float>({0, 1, 2, 3}));
    EXPECT_EQ(values(powers), std::vector<float>({-8, 27, 0.125f}));
}

TEST(CudaDeviceTest, MaskedFillBroadcastsBoolMaskOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, citrius::Device::cuda());
    const citrius::Tensor mask = citrius::from_vector(
        std::vector<bool>{false, true, false}, {1, 3}, citrius::Device::cuda());

    const auto output = citrius::masked_fill(input, mask, -100.0f);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(values(output), std::vector<float>({1, -100, 3, 4, -100, 6}));
    EXPECT_THROW(
        citrius::masked_fill(
            input,
            citrius::from_vector(
                std::vector<bool>{true, false, true, false}, {2, 2},
                citrius::Device::cuda()),
            0.0f),
        std::invalid_argument);
}

TEST(CudaDeviceTest, TopLevelMatmulUsesConfiguredCudaBackend) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor a(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, citrius::Device::cuda());
    const citrius::Tensor b(std::vector<float>{7, 8, 9, 10, 11, 12}, {3, 2},
                            citrius::Device::cuda());

    EXPECT_EQ(values(citrius::matmul(a, b)), std::vector<float>({58, 64, 139, 154}));
}

TEST(CudaDeviceTest, ToStringCopiesValuesForDisplay) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor tensor(std::vector<float>{1, 2}, citrius::Device::cuda());

    EXPECT_EQ(tensor.to_string(), "tensor([1, 2], shape=[2], dtype=float32, device=cuda:0)");
}
