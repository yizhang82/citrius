#include "nn/layer_norm.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(LayerNormTest, InitializesAndRegistersAffineParameters) {
    citrius::nn::LayerNorm layer_norm({2, 3});

    EXPECT_EQ(layer_norm.normalized_shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(layer_norm.weight().shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(layer_norm.bias().shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(layer_norm.named_parameters().size(), 2u);
    EXPECT_EQ(values(layer_norm.weight()), std::vector<float>(6, 1.0f));
    EXPECT_EQ(values(layer_norm.bias()), std::vector<float>(6, 0.0f));
}

TEST(LayerNormTest, AppliesLearnableScaleAndBias) {
    citrius::nn::LayerNorm layer_norm(2);
    layer_norm.weight() = citrius::Tensor(std::vector<float>{2.0f, 3.0f});
    layer_norm.bias() = citrius::Tensor(std::vector<float>{0.5f, -0.5f});
    const citrius::Tensor input(std::vector<float>{1.0f, 3.0f, 2.0f, 4.0f}, {2, 2});

    const auto result = values(layer_norm(input));

    EXPECT_NEAR(result[0], -1.49999f, 1e-5f);
    EXPECT_NEAR(result[1], 2.49999f, 1e-5f);
    EXPECT_NEAR(result[2], -1.49999f, 1e-5f);
    EXPECT_NEAR(result[3], 2.49999f, 1e-5f);
}

TEST(LayerNormTest, SupportsNormalizationWithoutAffineParameters) {
    citrius::nn::LayerNorm layer_norm(2, 1e-5f, false);
    const citrius::Tensor input(std::vector<float>{1.0f, 3.0f}, {1, 2});

    const auto result = values(layer_norm(input));

    EXPECT_FALSE(layer_norm.weight().defined());
    EXPECT_FALSE(layer_norm.bias().defined());
    EXPECT_TRUE(layer_norm.named_parameters().empty());
    EXPECT_NEAR(result[0], -0.999995f, 1e-6f);
    EXPECT_NEAR(result[1], 0.999995f, 1e-6f);
}

TEST(LayerNormTest, RejectsInvalidConfigurationAndInputShape) {
    EXPECT_THROW(citrius::nn::LayerNorm(citrius::Shape{}), std::invalid_argument);
    EXPECT_THROW(citrius::nn::LayerNorm(0), std::invalid_argument);
    EXPECT_THROW(citrius::nn::LayerNorm(2, 0.0f), std::invalid_argument);

    citrius::nn::LayerNorm layer_norm({2, 3});
    EXPECT_THROW(
        layer_norm(citrius::Tensor(std::vector<float>(8), {2, 2, 2})),
        std::invalid_argument);
}
