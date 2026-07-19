#include "nn/multi_head_attention.h"

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

TEST(MultiHeadAttentionTest, RegistersProjectionParameters) {
    citrius::nn::MultiHeadAttention attention(4, 2);

    EXPECT_EQ(attention.embed_dim(), 4);
    EXPECT_EQ(attention.num_heads(), 2);
    EXPECT_EQ(attention.head_dim(), 2);
    EXPECT_EQ(attention.named_children().size(), 4u);
    EXPECT_EQ(attention.named_parameters().size(), 8u);
    EXPECT_EQ(attention.query_projection().weight().shape(), (citrius::Shape{4, 4}));
    EXPECT_EQ(attention.output_projection().bias().shape(), (citrius::Shape{4}));
}

TEST(MultiHeadAttentionTest, RejectsInvalidDimensions) {
    EXPECT_THROW(citrius::nn::MultiHeadAttention(0, 2), std::invalid_argument);
    EXPECT_THROW(citrius::nn::MultiHeadAttention(4, 0), std::invalid_argument);
    EXPECT_THROW(citrius::nn::MultiHeadAttention(5, 2), std::invalid_argument);
}

TEST(MultiHeadAttentionTest, ComputesIndependentAttentionHeads) {
    citrius::nn::MultiHeadAttention attention(4, 2);
    set_identity(attention.query_projection());
    set_identity(attention.key_projection());
    set_identity(attention.value_projection());
    set_identity(attention.output_projection());
    const citrius::Tensor input(
        std::vector<float>{1, 0, 0, 1, 0, 1, 1, 0},
        {1, 2, 4});

    const auto output = attention(input);
    const auto actual = values(output);
    const std::vector<float> expected{
        0.669762f,
        0.330238f,
        0.330238f,
        0.669762f,
        0.330238f,
        0.669762f,
        0.669762f,
        0.330238f};

    EXPECT_EQ(output.shape(), input.shape());
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-5f);
    }
}

TEST(MultiHeadAttentionTest, RejectsInvalidInputShape) {
    citrius::nn::MultiHeadAttention attention(4, 2);

    EXPECT_THROW(attention(citrius::Tensor()), std::invalid_argument);
    EXPECT_THROW(
        attention(citrius::Tensor(std::vector<float>{1, 2, 3, 4}, {1, 4})),
        std::invalid_argument);
    EXPECT_THROW(
        attention(citrius::Tensor(std::vector<float>{1, 2, 3}, {1, 1, 3})),
        std::invalid_argument);
}
