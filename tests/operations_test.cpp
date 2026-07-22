#include "exceptions.h"
#include "operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <memory>
#include <cmath>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(OperationsTest, AddsUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3});
    const citrius::Tensor right(std::vector<float>{10, 20, 30});

    EXPECT_EQ(values(citrius::add(left, right)), std::vector<float>({11, 22, 33}));
}

TEST(OperationsTest, SubtractsUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{10, 20, 30});
    const citrius::Tensor right(std::vector<float>{1, 2, 3});

    EXPECT_EQ(values(citrius::sub(left, right)), std::vector<float>({9, 18, 27}));
}

TEST(OperationsTest, RmsNormUsesPortableDeviceFallback) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor weight(std::vector<float>{1.5f, 0.5f});

    const auto result = values(citrius::rms_norm(input, weight, 1e-6f));
    EXPECT_NEAR(result[0], 0.9486832f, 1e-6f);
    EXPECT_NEAR(result[1], 0.6324555f, 1e-6f);
    EXPECT_NEAR(result[2], 1.2727922f, 1e-6f);
    EXPECT_NEAR(result[3], 0.5656854f, 1e-6f);
}

TEST(OperationsTest, RmsNormValidatesArguments) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    EXPECT_THROW(
        citrius::rms_norm(input, citrius::Tensor(std::vector<float>{1}), 1e-6f),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::rms_norm(input, citrius::Tensor(std::vector<float>{1, 1}), 0.0f),
        std::invalid_argument);
}

TEST(OperationsTest, SwiGluUsesPortableDeviceFallback) {
    const citrius::Tensor gate(std::vector<float>{-1, 0, 1, 2});
    const citrius::Tensor up(std::vector<float>{1, 2, 3, 4});
    const auto result = values(citrius::swiglu(gate, up));
    EXPECT_NEAR(result[0], -0.2689414f, 1e-6f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_NEAR(result[2], 2.1931758f, 1e-6f);
    EXPECT_NEAR(result[3], 7.0463766f, 1e-6f);
}

TEST(OperationsTest, RmsNormRopeUsesPortableDeviceFallback) {
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 1, 4});
    const citrius::Tensor weight(std::vector<float>{1, 1, 1, 1});
    const auto result = citrius::rms_norm_rope(input, weight, 1e-6f, 10000.0f);
    EXPECT_EQ(result.shape(), (citrius::Shape{1, 1, 2, 4}));
    const auto actual = values(result);
    const float inverse_first = 1.0f / std::sqrt(7.5f + 1e-6f);
    EXPECT_NEAR(actual[0], inverse_first, 1e-6f);
    EXPECT_NEAR(actual[3], 4.0f * inverse_first, 1e-6f);
}

TEST(OperationsTest, AddRmsNormReturnsResidualAndNormalizedValues) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor right(std::vector<float>{2, 2, 1, 0}, {2, 2});
    const citrius::Tensor weight(std::vector<float>{1.5f, 0.5f});
    const auto result = citrius::add_rms_norm(left, right, weight, 1e-6f);
    EXPECT_EQ(values(result.residual), std::vector<float>({3, 4, 4, 4}));
    const auto normalized = values(result.normalized);
    EXPECT_NEAR(normalized[0], 1.2727922f, 1e-6f);
    EXPECT_NEAR(normalized[1], 0.5656854f, 1e-6f);
    EXPECT_NEAR(normalized[2], 1.5f, 1e-6f);
    EXPECT_NEAR(normalized[3], 0.5f, 1e-6f);
}

TEST(OperationsTest, MultipliesUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const citrius::Tensor right(
        std::vector<float>{7, 8, 9, 10, 11, 12},
        {3, 2});

    EXPECT_EQ(
        values(citrius::matmul(left, right)),
        std::vector<float>({58, 64, 139, 154}));
}

TEST(OperationsTest, ThrowsDeviceMismatchException) {
    const citrius::Tensor cpu(std::vector<float>{1, 2});
    const citrius::Tensor other(
        cpu.shape(),
        cpu.dtype(),
        citrius::Device::cuda(),
        cpu.storage());

    EXPECT_THROW(citrius::add(cpu, other), citrius::DeviceMismatchException);
}

TEST(OperationsTest, OperatorsDelegateToTopLevelOperations) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor right(std::vector<float>{5, 6, 7, 8}, {2, 2});

    EXPECT_EQ(values(left + right), std::vector<float>({6, 8, 10, 12}));
    EXPECT_EQ(values(left - right), std::vector<float>({-4, -4, -4, -4}));
    EXPECT_EQ(values(left * right), std::vector<float>({19, 22, 43, 50}));
}

TEST(OperationsTest, BroadcastsTrailingDimensions) {
    const citrius::Tensor matrix(
        std::vector<float>{1, 2, 3, 4, 5, 6},
        {2, 3});
    const citrius::Tensor row(std::vector<float>{10, 20, 30});

    const auto result = citrius::add(matrix, row);

    EXPECT_EQ(result.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(values(result), std::vector<float>({11, 22, 33, 14, 25, 36}));
    EXPECT_EQ(
        values(citrius::mul(matrix, row)),
        std::vector<float>({10, 40, 90, 40, 100, 180}));
    EXPECT_EQ(
        values(citrius::div(matrix, row)),
        std::vector<float>({0.1f, 0.1f, 0.1f, 0.4f, 0.25f, 0.2f}));
}

TEST(OperationsTest, SupportsScalarAndMaximumOperations) {
    const citrius::Tensor input(std::vector<float>{1, 2, 4});

    EXPECT_EQ(values(input + 2.0f), std::vector<float>({3, 4, 6}));
    EXPECT_EQ(values(10.0f - input), std::vector<float>({9, 8, 6}));
    EXPECT_EQ(values(input * 3.0f), std::vector<float>({3, 6, 12}));
    EXPECT_EQ(values(8.0f / input), std::vector<float>({8, 4, 2}));
    EXPECT_EQ(
        values(citrius::maximum(input, citrius::from_vector(std::vector<float>{2.0f}))),
        std::vector<float>({2, 2, 4}));
}

TEST(OperationsTest, ComputesUnaryMath) {
    const citrius::Tensor input(std::vector<float>{0, 1, 4});

    const auto exponentials = values(citrius::exp(input));
    EXPECT_FLOAT_EQ(exponentials[0], 1.0f);
    EXPECT_NEAR(exponentials[1], std::exp(1.0f), 1e-6f);
    EXPECT_EQ(values(citrius::sqrt(input)), std::vector<float>({0, 1, 2}));
    EXPECT_EQ(values(citrius::pow(input, 2.0f)), std::vector<float>({0, 1, 16}));
}

TEST(OperationsTest, BroadcastsBooleanMask) {
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6},
        {2, 3});
    const citrius::Tensor mask = citrius::from_vector(
        std::vector<bool>{false, true, false},
        {1, 3});

    const auto result = citrius::masked_fill(input, mask, -100.0f);

    EXPECT_EQ(values(result), std::vector<float>({1, -100, 3, 4, -100, 6}));
}

TEST(OperationsTest, RejectsIncompatibleBroadcastShapes) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor right(std::vector<float>{1, 2, 3});

    EXPECT_THROW(citrius::add(left, right), std::invalid_argument);
    EXPECT_THROW(
        citrius::masked_fill(left, citrius::from_vector(std::vector<bool>{true, false, true}), 0),
        std::invalid_argument);
}

TEST(OperationsTest, MultipliesBatchedMatrices) {
    const citrius::Tensor left(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
        {2, 2, 2});
    const citrius::Tensor right(
        std::vector<float>{1, 0, 0, 1, 2, 0, 0, 2},
        {2, 2, 2});

    const auto result = citrius::matmul(left, right);

    EXPECT_EQ(result.shape(), citrius::Shape({2, 2, 2}));
    EXPECT_EQ(values(result), std::vector<float>({1, 2, 3, 4, 10, 12, 14, 16}));
}

TEST(OperationsTest, BroadcastsMatmulBatchDimensions) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor right(
        std::vector<float>{1, 1, 2, 3, 4, 5},
        {3, 2, 1});

    const auto result = citrius::matmul(left, right);

    EXPECT_EQ(result.shape(), citrius::Shape({3, 2, 1}));
    EXPECT_EQ(values(result), std::vector<float>({3, 7, 8, 18, 14, 32}));
}

TEST(OperationsTest, SupportsAttentionMatmulShapes) {
    const citrius::Tensor query(
        std::vector<float>{1, 0, 0, 1, 1, 1, 2, 1},
        {1, 2, 2, 2});
    const auto key_transposed = citrius::transpose(query, -2, -1);

    const auto scores = citrius::matmul(query, key_transposed);
    const auto context = citrius::matmul(scores, query);

    EXPECT_EQ(scores.shape(), citrius::Shape({1, 2, 2, 2}));
    EXPECT_EQ(context.shape(), citrius::Shape({1, 2, 2, 2}));
}

TEST(OperationsTest, RejectsInvalidBatchedMatmulShapes) {
    const citrius::Tensor matrix(std::vector<float>{1, 2, 3, 4}, {2, 2});
    EXPECT_THROW(
        citrius::matmul(
            citrius::unsqueeze(matrix, 0),
            citrius::Tensor(std::vector<float>{1, 2, 3}, {3, 1})),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::matmul(
            citrius::Tensor(std::vector<float>(2 * 2 * 2), {2, 2, 2}),
            citrius::Tensor(std::vector<float>(3 * 2 * 2), {3, 2, 2})),
        std::invalid_argument);
    EXPECT_THROW(citrius::matmul(citrius::Tensor(std::vector<float>{1, 2}), matrix),
                 std::invalid_argument);
}
