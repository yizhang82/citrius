#include "nn/functional.h"

#include "exceptions.h"
#include "impl/cpu_storage.h"
#include "reduction_operations.h"
#include "tensor_factory.h"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(FunctionalTest, ReluClampsNegativeValuesToZero) {
    const citrius::Tensor input(std::vector<float>{-2.0f, -0.0f, 0.0f, 1.5f});

    const auto result = citrius::nn::functional::relu(input);

    EXPECT_EQ(result.shape(), input.shape());
    EXPECT_EQ(result.device(), input.device());
    EXPECT_EQ(values(result), (std::vector<float>{0.0f, 0.0f, 0.0f, 1.5f}));
}

TEST(FunctionalTest, GeluMatchesTanhApproximationReference) {
    const citrius::Tensor input(
        std::vector<float>{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f},
        {1, 5});

    const auto result = values(citrius::nn::functional::gelu(input));
    const std::vector<float> expected{-0.003637f, -0.158808f, 0.0f, 0.841192f, 2.996363f};

    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(result[index], expected[index], 1e-5f);
    }
}

TEST(FunctionalTest, GeluRemainsFiniteForLargeMagnitudes) {
    const citrius::Tensor input(std::vector<float>{-100.0f, 100.0f});

    const auto result = values(citrius::nn::functional::gelu(input));

    EXPECT_TRUE(std::isfinite(result[0]));
    EXPECT_TRUE(std::isfinite(result[1]));
    EXPECT_NEAR(result[0], 0.0f, 1e-6f);
    EXPECT_NEAR(result[1], 100.0f, 1e-6f);
}

TEST(FunctionalTest, SoftmaxNormalizesSelectedDimension) {
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 1, 2, 3},
        {2, 3});

    const auto result = citrius::nn::functional::softmax(input, -1);
    const auto totals = values(citrius::sum(result, -1));

    EXPECT_EQ(result.shape(), input.shape());
    EXPECT_NEAR(totals[0], 1.0f, 1e-6f);
    EXPECT_NEAR(totals[1], 1.0f, 1e-6f);
}

TEST(FunctionalTest, SoftmaxIsStableForLargeValues) {
    const citrius::Tensor input(std::vector<float>{10000, 10001, 10002});

    const auto result = values(citrius::nn::functional::softmax(input, 0));

    for (const float value : result) EXPECT_TRUE(std::isfinite(value));
    EXPECT_NEAR(result[0] + result[1] + result[2], 1.0f, 1e-6f);
    EXPECT_LT(result[0], result[1]);
    EXPECT_LT(result[1], result[2]);
}

TEST(FunctionalTest, SoftmaxMakesMaskedValuesNegligible) {
    const citrius::Tensor input(std::vector<float>{1, -10000, 2});

    const auto result = values(citrius::nn::functional::softmax(input, -1));

    EXPECT_FLOAT_EQ(result[1], 0.0f);
}

TEST(FunctionalTest, ScaledDotProductAttentionComputesWeightedValues) {
    const citrius::Tensor query(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor key(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor value(std::vector<float>{1, 0, 0, 1}, {2, 2});

    const auto result = citrius::nn::functional::scaled_dot_product_attention(
        query,
        key,
        value);
    const auto actual = values(result);

    EXPECT_EQ(result.shape(), (citrius::Shape{2, 2}));
    EXPECT_NEAR(actual[0], 0.669762f, 1e-5f);
    EXPECT_NEAR(actual[1], 0.330238f, 1e-5f);
    EXPECT_NEAR(actual[2], 0.330238f, 1e-5f);
    EXPECT_NEAR(actual[3], 0.669762f, 1e-5f);
}

TEST(FunctionalTest, ScaledDotProductAttentionSupportsBatchAndHeadDimensions) {
    const std::vector<float> identity_heads{1, 0, 0, 1, 1, 0, 0, 1};
    const citrius::Tensor query(identity_heads, {1, 2, 2, 2});
    const citrius::Tensor key(identity_heads, {1, 2, 2, 2});
    const citrius::Tensor value(identity_heads, {1, 2, 2, 2});

    const auto result = citrius::nn::functional::scaled_dot_product_attention(
        query,
        key,
        value);
    const auto actual = values(result);

    EXPECT_EQ(result.shape(), (citrius::Shape{1, 2, 2, 2}));
    ASSERT_EQ(actual.size(), identity_heads.size());
    EXPECT_NEAR(actual[0], 0.669762f, 1e-5f);
    EXPECT_NEAR(actual[4], 0.669762f, 1e-5f);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsMismatchedHeadDimensions) {
    const citrius::Tensor query(std::vector<float>(6), {2, 3});
    const citrius::Tensor key(std::vector<float>(8), {2, 4});
    const citrius::Tensor value(std::vector<float>(4), {2, 2});

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(query, key, value),
        std::invalid_argument);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsInputsWithRankBelowTwo) {
    const citrius::Tensor vector(std::vector<float>{1, 2});
    const citrius::Tensor matrix(std::vector<float>{1, 0, 0, 1}, {2, 2});

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(vector, matrix, matrix),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(matrix, vector, matrix),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(matrix, matrix, vector),
        std::invalid_argument);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsDeviceMismatch) {
    const citrius::Tensor cpu(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor other(
        cpu.shape(),
        cpu.dtype(),
        citrius::Device::cuda(),
        cpu.storage());

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(cpu, other, cpu),
        citrius::DeviceMismatchException);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(cpu, cpu, other),
        citrius::DeviceMismatchException);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsUndefinedInputs) {
    const citrius::Tensor undefined;
    const citrius::Tensor matrix(std::vector<float>{1, 0, 0, 1}, {2, 2});

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(undefined, matrix, matrix),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(matrix, undefined, matrix),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(matrix, matrix, undefined),
        std::invalid_argument);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsMismatchedKeyAndValueLengths) {
    const citrius::Tensor query(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor key(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor value(std::vector<float>{1, 0, 0, 1, 1, 1}, {3, 2});

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(query, key, value),
        std::invalid_argument);
}

TEST(FunctionalTest, ScaledDotProductAttentionExcludesMaskedKeys) {
    const citrius::Tensor query(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor key(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor value(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor causal_mask = citrius::from_vector(
        std::vector<bool>{false, true, false, false},
        {2, 2});

    const auto result = values(citrius::nn::functional::scaled_dot_product_attention(
        query,
        key,
        value,
        causal_mask));

    EXPECT_NEAR(result[0], 1.0f, 1e-6f);
    EXPECT_NEAR(result[1], 0.0f, 1e-6f);
    EXPECT_NEAR(result[2], 0.330238f, 1e-5f);
    EXPECT_NEAR(result[3], 0.669762f, 1e-5f);
}

TEST(FunctionalTest, ScaledDotProductAttentionRejectsInvalidMask) {
    const citrius::Tensor input(std::vector<float>{1, 0, 0, 1}, {2, 2});
    const citrius::Tensor float_mask(std::vector<float>{0, 1, 0, 0}, {2, 2});
    const citrius::Tensor wrong_shape =
        citrius::from_vector(std::vector<bool>{false, true, false}, {3});

    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(
            input,
            input,
            input,
            float_mask),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::nn::functional::scaled_dot_product_attention(
            input,
            input,
            input,
            wrong_shape),
        std::invalid_argument);
}
