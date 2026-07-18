#include "nn/linear.h"

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

TEST(LinearTest, RegistersParametersWithTorchShapes) {
    citrius::nn::Linear linear(3, 2);

    EXPECT_EQ(linear.weight().shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(linear.bias().shape(), citrius::Shape({2}));
    EXPECT_EQ(linear.named_parameters().size(), 2u);
    EXPECT_EQ(linear.in_features(), 3);
    EXPECT_EQ(linear.out_features(), 2);
    EXPECT_TRUE(linear.has_bias());
}

TEST(LinearTest, ComputesForwardWithBias) {
    citrius::nn::Linear linear(3, 2);
    linear.weight() = citrius::Tensor(
        std::vector<float>{1, 2, 3, 4, 5, 6},
        {2, 3});
    linear.bias() = citrius::Tensor(std::vector<float>{10, 20});
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6},
        {2, 3});

    const auto output = linear(input);

    EXPECT_EQ(output.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(values(output), std::vector<float>({24, 52, 42, 97}));
}

TEST(LinearTest, PreservesLeadingDimensions) {
    citrius::nn::Linear linear(2, 1, false);
    linear.weight() = citrius::Tensor(std::vector<float>{2, 3}, {1, 2});
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
        {2, 2, 2});

    const auto output = linear(input);

    EXPECT_EQ(output.shape(), citrius::Shape({2, 2, 1}));
    EXPECT_EQ(values(output), std::vector<float>({8, 18, 28, 38}));
    EXPECT_FALSE(linear.bias().defined());
    EXPECT_EQ(linear.named_parameters().size(), 1u);
}

TEST(LinearTest, RejectsInvalidShapes) {
    EXPECT_THROW(citrius::nn::Linear(0, 2), std::invalid_argument);

    citrius::nn::Linear linear(3, 2);
    EXPECT_THROW(
        linear(citrius::Tensor(std::vector<float>{1, 2}, {1, 2})),
        std::invalid_argument);
}
