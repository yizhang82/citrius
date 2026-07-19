#include "nn/transformer_block.h"

#include "impl/cpu_storage.h"
#include "tensor_factory.h"

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

void zero_linear(citrius::nn::Linear& linear) {
    linear.weight() = citrius::Tensor(
        std::vector<float>(
            static_cast<std::size_t>(linear.in_features() * linear.out_features()),
            0.0f),
        {linear.out_features(), linear.in_features()});
    if (linear.has_bias()) {
        linear.bias() = citrius::Tensor(
            std::vector<float>(static_cast<std::size_t>(linear.out_features()), 0.0f));
    }
}

void set_identity(citrius::nn::Linear& linear) {
    const auto size = linear.in_features();
    std::vector<float> weight(static_cast<std::size_t>(size * size), 0.0f);
    for (std::int64_t index = 0; index < size; ++index) {
        weight[static_cast<std::size_t>(index * size + index)] = 1.0f;
    }
    linear.weight() = citrius::Tensor(weight, {size, size});
    if (linear.has_bias()) {
        linear.bias() = citrius::Tensor(
            std::vector<float>(static_cast<std::size_t>(size), 0.0f));
    }
}

void zero_feed_forward(citrius::nn::FeedForward& feed_forward) {
    zero_linear(feed_forward.input_projection());
    zero_linear(feed_forward.output_projection());
}

} // namespace

TEST(TransformerBlockTest, RegistersPreNormComponents) {
    citrius::nn::TransformerBlock block(4, 2, 16);

    EXPECT_EQ(block.embed_dim(), 4);
    EXPECT_EQ(block.num_heads(), 2);
    EXPECT_EQ(block.hidden_dim(), 16);
    EXPECT_EQ(block.named_children().size(), 4u);
    EXPECT_EQ(block.named_parameters().size(), 16u);
    EXPECT_EQ(block.attention_norm().normalized_shape(), (citrius::Shape{4}));
    EXPECT_EQ(block.feed_forward_norm().normalized_shape(), (citrius::Shape{4}));
}

TEST(TransformerBlockTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(citrius::nn::TransformerBlock(0, 2, 8), std::invalid_argument);
    EXPECT_THROW(citrius::nn::TransformerBlock(4, 0, 8), std::invalid_argument);
    EXPECT_THROW(citrius::nn::TransformerBlock(4, 2, 0), std::invalid_argument);
    EXPECT_THROW(citrius::nn::TransformerBlock(5, 2, 8), std::invalid_argument);
    EXPECT_THROW(citrius::nn::TransformerBlock(4, 2, 8, 0.0f), std::invalid_argument);
}

TEST(TransformerBlockTest, ResidualConnectionsPreserveInputWhenBranchesAreZero) {
    citrius::nn::TransformerBlock block(4, 2, 8);
    zero_linear(block.attention().query_projection());
    zero_linear(block.attention().key_projection());
    zero_linear(block.attention().value_projection());
    zero_linear(block.attention().output_projection());
    zero_feed_forward(block.feed_forward());
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
        {1, 2, 4});

    const auto output = block(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(values(output), values(input));
}

TEST(TransformerBlockTest, ForwardsAttentionMask) {
    citrius::nn::TransformerBlock block(4, 2, 8);
    set_identity(block.attention().query_projection());
    set_identity(block.attention().key_projection());
    set_identity(block.attention().value_projection());
    set_identity(block.attention().output_projection());
    zero_feed_forward(block.feed_forward());
    const citrius::Tensor input(
        std::vector<float>{1, 0, 0, 1, 0, 1, 1, 0},
        {1, 2, 4});
    const citrius::Tensor causal_mask = citrius::from_vector(
        std::vector<bool>{false, true, false, false},
        {2, 2});

    const auto unmasked = values(block(input));
    const auto masked = values(block.forward(input, causal_mask));

    ASSERT_EQ(masked.size(), unmasked.size());
    EXPECT_NE(masked[0], unmasked[0]);
}

TEST(TransformerBlockTest, RejectsInvalidInputShape) {
    citrius::nn::TransformerBlock block(4, 2, 8);

    EXPECT_THROW(block(citrius::Tensor()), std::invalid_argument);
    EXPECT_THROW(
        block(citrius::Tensor(std::vector<float>{1, 2, 3, 4}, {1, 4})),
        std::invalid_argument);
    EXPECT_THROW(
        block(citrius::Tensor(std::vector<float>{1, 2, 3}, {1, 1, 3})),
        std::invalid_argument);
}
