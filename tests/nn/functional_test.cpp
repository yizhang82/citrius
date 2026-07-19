#include "nn/functional.h"

#include "impl/cpu_storage.h"
#include "reduction_operations.h"

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
