#include "impl/cpu_device.h"
#include "impl/cpu_storage.h"
#include "impl/cuda_allocation.h"
#include "impl/cuda_context.h"
#include "impl/cuda_matmul_layout.h"
#include "impl/cublas_cuda_device.h"
#include "impl/cuda_device.h"
#include "impl/cuda_storage.h"
#include "impl/cutlass_cuda_device.h"
#include "indexing_operations.h"
#include "operations.h"
#include "reduction_operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "nn/functional.h"
#include "nn/layer_norm.h"
#include "models/qwen3.h"

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
std::vector<float> floating_values(const citrius::Tensor& tensor) {
    if (tensor.dtype() == citrius::DType::Float32) return values(tensor);
    const auto cpu = tensor.to(citrius::Device::cpu());
    const auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(
        cpu.storage());
    const auto* bits = storage->data_as<std::uint16_t>();
    std::vector<float> result(static_cast<std::size_t>(cpu.numel()));
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = cpu.dtype() == citrius::DType::Float16
            ? citrius::float16_to_float(bits[index])
            : citrius::bfloat16_to_float(bits[index]);
    return result;
}
std::vector<std::int64_t> indices(const citrius::Tensor& tensor) {
    std::vector<std::int64_t> result(static_cast<std::size_t>(tensor.numel()));
    std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage())
        ->copy_to_host(result.data(), result.size() * sizeof(std::int64_t));
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
TEST(CudaDeviceTest, SharesOneExecutionContextPerDevice) {
    std::string error;
    auto first = make_cuda_device(&error);
    if (!first)
        GTEST_SKIP() << error;
    auto second = make_cuda_device(&error);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->execution_context(), second->execution_context());
    EXPECT_NE(first->execution_context()->stream(), nullptr);

    auto tensor = first->empty({1}, citrius::DType::Float32);
    const auto storage =
        std::static_pointer_cast<citrius::impl::CudaMemTensorStorageImpl>(tensor.storage());
    EXPECT_EQ(storage->execution_context(), first->execution_context());
}
TEST(CudaDeviceTest, CudaAllocationIsMoveOnlyAndRetainsMemory) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    citrius::impl::CudaAllocation allocation(64, device->execution_context());
    ASSERT_NE(allocation.data(), nullptr);
    EXPECT_EQ(allocation.nbytes(), 64);

    citrius::impl::CudaAllocation moved(std::move(allocation));
    EXPECT_EQ(allocation.data(), nullptr);
    EXPECT_EQ(allocation.nbytes(), 0);
    EXPECT_NE(moved.data(), nullptr);
    EXPECT_EQ(moved.nbytes(), 64);
}

TEST(CudaDeviceTest, CudaAllocationCopiesAndSynchronizesOnItsContext) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const std::vector<float> source{1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> destination(source.size());
    const auto nbytes = source.size() * sizeof(float);
    citrius::impl::CudaAllocation first(nbytes, device->execution_context());
    citrius::impl::CudaAllocation second(nbytes, device->execution_context());

    first.copy_from_host_async(source.data(), nbytes);
    second.copy_from_device_async(first, nbytes);
    second.copy_to_host_async(destination.data(), nbytes);
    second.synchronize();

    EXPECT_EQ(destination, source);
    EXPECT_THROW(first.copy_from_host_async(source.data(), nbytes + 1), std::invalid_argument);
}

TEST(CudaDeviceTest, ArgmaxMatchesCpuForTiesNegativesAndIrregularSizes) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device) GTEST_SKIP() << error;
    constexpr std::int64_t vocabulary_size = 151936;
    std::vector<float> input(static_cast<std::size_t>(2 * vocabulary_size), -10.0f);
    input[12345] = -1.0f;
    input[54321] = -1.0f;
    input[static_cast<std::size_t>(vocabulary_size + 151935)] = 4.0f;
    const auto result = citrius::argmax(
        make_cuda_tensor(*device, {2, vocabulary_size}, input), -1);
    EXPECT_EQ(result.dtype(), citrius::DType::Int64);
    EXPECT_EQ(indices(result), std::vector<std::int64_t>({12345, 151935}));
    EXPECT_EQ(indices(result), indices(citrius::argmax(citrius::Tensor(input, {2, vocabulary_size}), -1)
                                           .to(citrius::Device::cuda())));
}
TEST(CudaDeviceTest, ItemMaterializesPendingCudaScalar) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto input = make_cuda_tensor(*device, {5}, {-3, 7, 2, 7, 1});
    EXPECT_EQ(citrius::argmax(input).item<std::int64_t>(), 1);
}

TEST(CudaDeviceTest, OffsetScalarCopiesOnlyItsLogicalElementToHost) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto values = make_cuda_tensor(*device, {4}, {1, 2, 3, 4});
    const citrius::Tensor scalar(
        {}, {}, 2, citrius::DType::Float32, values.device(), values.storage());

    EXPECT_FLOAT_EQ(scalar.item<float>(), 3.0f);
}

TEST(CudaDeviceTest, SelectCreatesAViewAndMaterializesOnlyTheSelectedScalar) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto tensor = make_cuda_tensor(
        *device, {2, 3, 4},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    const auto selected = tensor.select(1, 2);

    EXPECT_EQ(selected.storage(), tensor.storage());
    EXPECT_EQ(selected.strides(), citrius::Strides({12, 1}));
    EXPECT_EQ(selected.storage_offset(), 8);
    EXPECT_FLOAT_EQ(selected.select(0, 1).select(0, 3).item<float>(), 23.0f);
}

TEST(CudaDeviceTest, ContiguousMaterializesAStridedViewOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto tensor = make_cuda_tensor(
        *device, {2, 3, 4},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    const auto packed = citrius::contiguous(tensor.select(1, 2));

    EXPECT_EQ(packed.device(), tensor.device());
    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ(values(packed), std::vector<float>({8, 9, 10, 11, 20, 21, 22, 23}));
}

TEST(CudaDeviceTest, GatherRowsStaysOnCudaAndMatchesTheCpuReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto table = make_cuda_tensor(
        *device, {4, 3}, {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32});
    const auto row_indices = citrius::from_vector(
        std::vector<std::int64_t>{2, 0, 3, 1}, {2, 2}, citrius::Device::cuda());

    const auto output = citrius::gather_rows(table, row_indices);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_EQ(output.shape(), citrius::Shape({2, 2, 3}));
    EXPECT_EQ(
        values(output),
        (std::vector<float>{20, 21, 22, 0, 1, 2, 30, 31, 32, 10, 11, 12}));
}

TEST(CudaDeviceTest, GatherRowsRejectsOutOfRangeCudaIndices) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto table = make_cuda_tensor(*device, {2, 2}, {0, 1, 2, 3});
    const auto row_indices = citrius::from_vector(
        std::vector<std::int64_t>{0, 2}, citrius::Device::cuda());

    EXPECT_THROW(citrius::gather_rows(table, row_indices), std::out_of_range);
}

TEST(CudaDeviceTest, RecognizesRowAndColumnMajorCudaMatrixViews) {
    const citrius::Tensor matrix({2, 3}, citrius::DType::Float32);
    const auto row_layout = citrius::impl::cuda_matrix_layout(matrix);
    const auto column_layout = citrius::impl::cuda_matrix_layout(
        citrius::transpose(matrix, 0, 1));

    EXPECT_EQ(row_layout.type, citrius::impl::CudaMatrixLayoutType::RowMajor);
    EXPECT_EQ(row_layout.leading_dimension, 3);
    EXPECT_EQ(column_layout.type, citrius::impl::CudaMatrixLayoutType::ColumnMajor);
    EXPECT_EQ(column_layout.leading_dimension, 3);
}

TEST(CudaDeviceTest, CublasMatmulConsumesTransposedWeightView) {
    std::string error;
    auto baseline = make_cuda_device(&error);
    if (!baseline)
        GTEST_SKIP() << error;
    citrius::impl::CublasCudaDeviceImpl cublas;
    const auto input = make_cuda_tensor(*baseline, {2, 3}, {1, 2, 3, 4, 5, 6});
    const auto weight = make_cuda_tensor(*baseline, {2, 3}, {1, 0, 1, 0, 1, 1});

    const auto output = cublas.matmul(input, citrius::transpose(weight, 0, 1));

    EXPECT_EQ(values(output), std::vector<float>({4, 5, 10, 11}));
}

TEST(CudaDeviceTest, CutlassMatmulConsumesTransposedWeightView) {
    std::string error;
    auto baseline = make_cuda_device(&error);
    if (!baseline)
        GTEST_SKIP() << error;
    citrius::impl::CutlassCudaDeviceImpl cutlass;
    const auto input = make_cuda_tensor(*baseline, {2, 3}, {1, 2, 3, 4, 5, 6});
    const auto weight = make_cuda_tensor(*baseline, {2, 3}, {1, 0, 1, 0, 1, 1});

    const auto output = cutlass.matmul(input, citrius::transpose(weight, 0, 1));

    EXPECT_EQ(values(output), std::vector<float>({4, 5, 10, 11}));
}

TEST(CudaDeviceTest, QwenAttentionHandlesPermutedViewsAndGroupedQueryHeads) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    citrius::models::Qwen3Config config;
    config.vocab_size = 32;
    config.hidden_size = 8;
    config.intermediate_size = 16;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 2;
    config.max_position_embeddings = 16;
    config.device = citrius::Device::cuda();
    citrius::models::Qwen3Attention attention(config);
    const auto input = make_cuda_tensor(
        *device, {1, 3, 8},
        {1, 2, 3, 4, 5, 6, 7, 8,
         2, 3, 4, 5, 6, 7, 8, 9,
         3, 4, 5, 6, 7, 8, 9, 10});

    const auto output = attention(input);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_EQ(output.shape(), citrius::Shape({1, 3, 8}));
    for (const float value : values(output)) EXPECT_TRUE(std::isfinite(value));
}

TEST(CudaDeviceTest, ConcatMaterializesSplitViewsNativelyOnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const auto input = make_cuda_tensor(*device, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    const auto parts = citrius::split(input, 2, 1);

    const auto output = citrius::concat({parts[1], parts[0]}, 1);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_EQ(output.shape(), citrius::Shape({2, 4}));
    EXPECT_EQ(values(output), std::vector<float>({3, 4, 1, 2, 7, 8, 5, 6}));
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

TEST(CudaDeviceTest, CublasTensorCoreMatmulSupportsFloat16AndBFloat16) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    citrius::impl::CublasCudaDeviceImpl cublas;
    const std::vector<float> left_values{1, 2, 3, 4, 5, 6};
    const std::vector<float> right_values{1, 2, 3, 4, 5, 6};
    for (const auto dtype : {citrius::DType::Float16, citrius::DType::BFloat16}) {
        const auto left = citrius::from_vector(
            left_values, {2, 3}, dtype, citrius::Device::cuda());
        const auto right = citrius::from_vector(
            right_values, {3, 2}, dtype, citrius::Device::cuda());
        const auto output = cublas.matmul(left, right);
        EXPECT_EQ(output.dtype(), dtype);
        const auto actual = floating_values(output);
        const std::vector<float> expected{22, 28, 49, 64};
        for (std::size_t index = 0; index < actual.size(); ++index)
            EXPECT_NEAR(actual[index], expected[index], dtype == citrius::DType::Float16 ? 0.05f : 0.25f);
    }
}

TEST(CudaDeviceTest, CastsFloat16AndBFloat16OnCuda) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor input(
        std::vector<float>{1.0f, -2.5f, 0.333333f}, citrius::Device::cuda());
    for (const auto dtype : {citrius::DType::Float16, citrius::DType::BFloat16}) {
        const auto restored = citrius::cast(
            citrius::cast(input, dtype), citrius::DType::Float32);
        const auto actual = values(restored);
        EXPECT_NEAR(actual[0], 1.0f, 1e-3f);
        EXPECT_NEAR(actual[1], -2.5f, 1e-3f);
        EXPECT_NEAR(actual[2], 0.333333f, 2e-3f);
    }
}

TEST(CudaDeviceTest, FusedRmsNormMatchesReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;

    constexpr std::int64_t row_size = 128;
    std::vector<float> input_values(static_cast<std::size_t>(3 * row_size));
    std::vector<float> weight_values(static_cast<std::size_t>(row_size));
    for (std::int64_t index = 0; index < 3 * row_size; ++index)
        input_values[static_cast<std::size_t>(index)] =
            static_cast<float>((index * 13) % 37 - 18) / 9.0f;
    for (std::int64_t index = 0; index < row_size; ++index)
        weight_values[static_cast<std::size_t>(index)] = 0.5f + static_cast<float>(index % 7) / 8.0f;

    const citrius::Tensor input(input_values, {3, row_size}, citrius::Device::cuda());
    const citrius::Tensor weight(weight_values, {row_size}, citrius::Device::cuda());
    const auto optimized = device->try_rms_norm(input, weight, 1e-6f);
    ASSERT_TRUE(optimized.has_value());
    const auto actual = values(*optimized);

    for (std::int64_t row = 0; row < 3; ++row) {
        float sum_squares = 0.0f;
        for (std::int64_t column = 0; column < row_size; ++column) {
            const float value = input_values[static_cast<std::size_t>(row * row_size + column)];
            sum_squares += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(sum_squares / row_size + 1e-6f);
        for (std::int64_t column = 0; column < row_size; ++column) {
            const auto index = static_cast<std::size_t>(row * row_size + column);
            EXPECT_NEAR(actual[index], input_values[index] * inverse_rms *
                                           weight_values[static_cast<std::size_t>(column)],
                        2e-6f);
        }
    }
}

TEST(CudaDeviceTest, FusedSwiGluMatchesReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;

    std::vector<float> gate_values(1027);
    std::vector<float> up_values(1027);
    for (std::size_t index = 0; index < gate_values.size(); ++index) {
        gate_values[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 4.0f;
        up_values[index] = static_cast<float>(index % 17) / 3.0f - 2.0f;
    }
    const citrius::Tensor gate(gate_values, citrius::Device::cuda());
    const citrius::Tensor up(up_values, citrius::Device::cuda());
    const auto optimized = device->try_swiglu(gate, up);
    ASSERT_TRUE(optimized.has_value());
    const auto actual = values(*optimized);
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const float expected = gate_values[index] /
            (1.0f + std::exp(-gate_values[index])) * up_values[index];
        EXPECT_NEAR(actual[index], expected, 2e-6f);
    }
}

TEST(CudaDeviceTest, FusedRmsNormRopeMatchesPortableReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;

    std::vector<float> input_values(2 * 3 * 2 * 8);
    for (std::size_t index = 0; index < input_values.size(); ++index)
        input_values[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 5.0f;
    const std::vector<float> weight_values{0.5f, 0.75f, 1.0f, 1.25f,
                                           1.5f, 1.75f, 2.0f, 2.25f};
    const citrius::Tensor cpu_input(input_values, {2, 3, 2, 8});
    const citrius::Tensor cpu_weight(weight_values, {8});
    const auto expected_tensor = citrius::rms_norm_rope(
        cpu_input, cpu_weight, 1e-6f, 10000.0f).to(citrius::Device::cuda());
    const auto expected = values(expected_tensor);

    const citrius::Tensor input(input_values, {2, 3, 2, 8}, citrius::Device::cuda());
    const citrius::Tensor weight(weight_values, {8}, citrius::Device::cuda());
    const auto optimized = device->try_rms_norm_rope(input, weight, 1e-6f, 10000.0f);
    ASSERT_TRUE(optimized.has_value());
    EXPECT_EQ(optimized->shape(), (citrius::Shape{2, 2, 3, 8}));
    const auto actual = values(*optimized);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], 3e-6f);
}

TEST(CudaDeviceTest, FusedAddRmsNormMatchesPortableReference) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    std::vector<float> left_values(3 * 128);
    std::vector<float> right_values(3 * 128);
    std::vector<float> weight_values(128);
    for (std::size_t index = 0; index < left_values.size(); ++index) {
        left_values[index] = static_cast<float>(static_cast<int>(index % 19) - 9) / 4.0f;
        right_values[index] = static_cast<float>(static_cast<int>(index % 13) - 6) / 5.0f;
    }
    for (std::size_t index = 0; index < weight_values.size(); ++index)
        weight_values[index] = 0.5f + static_cast<float>(index % 9) / 10.0f;
    const citrius::Tensor left(left_values, {3, 128}, citrius::Device::cuda());
    const citrius::Tensor right(right_values, {3, 128}, citrius::Device::cuda());
    const citrius::Tensor weight(weight_values, {128}, citrius::Device::cuda());
    const auto optimized = device->try_add_rms_norm(left, right, weight, 1e-6f);
    ASSERT_TRUE(optimized.has_value());
    const auto residual = values(optimized->first);
    const auto normalized = values(optimized->second);
    for (std::int64_t row = 0; row < 3; ++row) {
        float sum_squares = 0.0f;
        for (std::int64_t column = 0; column < 128; ++column) {
            const auto index = static_cast<std::size_t>(row * 128 + column);
            const float expected_residual = left_values[index] + right_values[index];
            EXPECT_FLOAT_EQ(residual[index], expected_residual);
            sum_squares += expected_residual * expected_residual;
        }
        const float inverse_rms = 1.0f / std::sqrt(sum_squares / 128.0f + 1e-6f);
        for (std::int64_t column = 0; column < 128; ++column) {
            const auto index = static_cast<std::size_t>(row * 128 + column);
            EXPECT_NEAR(normalized[index], residual[index] * inverse_rms *
                                               weight_values[static_cast<std::size_t>(column)],
                        3e-6f);
        }
    }
}

TEST(CudaDeviceTest, LastDimensionReductionsHandleIrregularQwenSizedRows) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    for (const std::int64_t row_size : {37LL, 128LL, 1024LL, 151936LL}) {
        std::vector<float> input_values(static_cast<std::size_t>(2 * row_size));
        for (std::int64_t index = 0; index < 2 * row_size; ++index)
            input_values[static_cast<std::size_t>(index)] =
                static_cast<float>((index * 17) % 101 - 50) / 7.0f;
        const citrius::Tensor input(
            input_values, {2, row_size}, citrius::Device::cuda());
        std::vector<float> expected_sums(2, 0.0f);
        std::vector<float> expected_maxima(2, -std::numeric_limits<float>::infinity());
        for (std::int64_t row = 0; row < 2; ++row) {
            for (std::int64_t column = 0; column < row_size; ++column) {
                const float value = input_values[static_cast<std::size_t>(row * row_size + column)];
                expected_sums[static_cast<std::size_t>(row)] += value;
                expected_maxima[static_cast<std::size_t>(row)] =
                    std::max(expected_maxima[static_cast<std::size_t>(row)], value);
            }
        }

        const auto sums = values(citrius::sum(input, -1));
        const auto means = values(citrius::mean(input, -1));
        const auto maxima = values(citrius::max(input, -1));

        for (std::size_t row = 0; row < 2; ++row) {
            EXPECT_NEAR(sums[row], expected_sums[row], 2e-3f) << "row size " << row_size;
            EXPECT_NEAR(means[row], expected_sums[row] / row_size, 2e-5f)
                << "row size " << row_size;
            EXPECT_FLOAT_EQ(maxima[row], expected_maxima[row]) << "row size " << row_size;
        }
    }
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

TEST(CudaDeviceTest, SoftmaxComposesNativeCudaOperations) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 10001, 10002, 10003},
        {2, 3}, citrius::Device::cuda());

    const auto output = citrius::nn::functional::softmax(input, -1);
    const auto output_values = values(output);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_EQ(output.shape(), input.shape());
    for (std::size_t index = 0; index < 3; ++index)
        EXPECT_NEAR(output_values[index], output_values[index + 3], 1e-6f);
    EXPECT_NEAR(output_values[0] + output_values[1] + output_values[2], 1.0f, 1e-6f);
    EXPECT_NEAR(output_values[3] + output_values[4] + output_values[5], 1.0f, 1e-6f);
}

TEST(CudaDeviceTest, LayerNormComposesNativeCudaOperations) {
    std::string error;
    auto device = make_cuda_device(&error);
    if (!device)
        GTEST_SKIP() << error;
    citrius::nn::LayerNorm layer_norm(3, 1e-5f, true, citrius::Device::cuda());
    layer_norm.weight() = citrius::Tensor(
        std::vector<float>{1.0f, 2.0f, 3.0f}, citrius::Device::cuda());
    layer_norm.bias() = citrius::Tensor(
        std::vector<float>{0.5f, 0.0f, -0.5f}, citrius::Device::cuda());
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, citrius::Device::cuda());

    const auto output = layer_norm(input);
    const auto output_values = values(output);

    EXPECT_EQ(output.device(), citrius::Device::cuda());
    EXPECT_NEAR(output_values[0], -0.724736f, 1e-5f);
    EXPECT_NEAR(output_values[1], 0.0f, 1e-6f);
    EXPECT_NEAR(output_values[2], 3.174207f, 1e-5f);
    EXPECT_NEAR(output_values[3], output_values[0], 1e-5f);
    EXPECT_NEAR(output_values[4], output_values[1], 1e-6f);
    EXPECT_NEAR(output_values[5], output_values[2], 1e-5f);
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
