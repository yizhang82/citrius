#include "nn/feed_forward.h"

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

void set_identity(citrius::nn::Linear& linear) {
    const auto size = linear.in_features();
    std::vector<float> weight(static_cast<std::size_t>(size * size), 0.0f);
    for (std::int64_t index = 0; index < size; ++index) {
        weight[static_cast<std::size_t>(index * size + index)] = 1.0f;
    }
    linear.weight() = citrius::Tensor(weight, {size, size});
    if (linear.has_bias()) {
        linear.bias() = citrius::Tensor(std::vector<float>(static_cast<std::size_t>(size), 0.0f));
    }
}

} // namespace

TEST(FeedForwardTest, RegistersExpansionAndContractionLayers) {
    citrius::nn::FeedForward feed_forward(4, 16);

    EXPECT_EQ(feed_forward.embed_dim(), 4);
    EXPECT_EQ(feed_forward.hidden_dim(), 16);
    EXPECT_EQ(feed_forward.named_children().size(), 2u);
    EXPECT_EQ(feed_forward.named_parameters().size(), 4u);
    EXPECT_EQ(
        feed_forward.input_projection().weight().shape(),
        (citrius::Shape{16, 4}));
    EXPECT_EQ(
        feed_forward.output_projection().weight().shape(),
        (citrius::Shape{4, 16}));
}

TEST(FeedForwardTest, RejectsInvalidDimensions) {
    EXPECT_THROW(citrius::nn::FeedForward(0, 4), std::invalid_argument);
    EXPECT_THROW(citrius::nn::FeedForward(4, 0), std::invalid_argument);
}

TEST(FeedForwardTest, AppliesLinearGeluLinearToEveryToken) {
    citrius::nn::FeedForward feed_forward(2, 2);
    set_identity(feed_forward.input_projection());
    set_identity(feed_forward.output_projection());
    const citrius::Tensor input(
        std::vector<float>{-1.0f, 0.0f, 1.0f, 2.0f},
        {1, 2, 2});

    const auto output = feed_forward(input);
    const auto actual = values(output);
    const std::vector<float> expected{-0.158808f, 0.0f, 0.841192f, 1.954598f};

    EXPECT_EQ(output.shape(), input.shape());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(FeedForwardTest, SupportsHiddenDimensionDifferentFromEmbeddingDimension) {
    citrius::nn::FeedForward feed_forward(2, 4);
    const citrius::Tensor input(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
        {1, 2, 2});

    const auto output = feed_forward(input);

    EXPECT_TRUE(output.defined());
    EXPECT_EQ(output.shape(), input.shape());
}

TEST(FeedForwardTest, RejectsInvalidInputShape) {
    citrius::nn::FeedForward feed_forward(4, 8);

    EXPECT_THROW(feed_forward(citrius::Tensor()), std::invalid_argument);
    EXPECT_THROW(
        feed_forward(citrius::Tensor(std::vector<float>{1, 2, 3, 4}, {1, 4})),
        std::invalid_argument);
    EXPECT_THROW(
        feed_forward(citrius::Tensor(std::vector<float>{1, 2, 3}, {1, 1, 3})),
        std::invalid_argument);
}
