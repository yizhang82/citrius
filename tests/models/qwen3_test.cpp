#include "models/qwen3.h"

#include "impl/cpu_storage.h"
#include "tensor_factory.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

citrius::models::Qwen3Config tiny_config() {
    citrius::models::Qwen3Config config;
    config.vocab_size = 8;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 2;
    config.max_position_embeddings = 16;
    config.rope_theta = 10000.0f;
    return config;
}

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(Qwen3Test, RmsNormUsesRootMeanSquareWithoutCentering) {
    citrius::models::Qwen3RMSNorm norm(2, 1e-6f);
    const citrius::Tensor input(std::vector<float>{3.0f, 4.0f}, {1, 2});

    const auto result = values(norm(input));

    EXPECT_NEAR(result[0], 0.848528f, 1e-5f);
    EXPECT_NEAR(result[1], 1.131371f, 1e-5f);
}

TEST(Qwen3Test, ConstructsOfficialSmallCheckpointShapeConfiguration) {
    citrius::models::Qwen3Config config;
    EXPECT_NO_THROW(config.validate());
    EXPECT_EQ(config.hidden_size, 1024);
    EXPECT_EQ(config.num_attention_heads, 16);
    EXPECT_EQ(config.num_key_value_heads, 8);
    EXPECT_EQ(config.head_dim, 128);
}

TEST(Qwen3Test, ProducesVocabularyLogits) {
    citrius::models::Qwen3ForCausalLM model(tiny_config());
    const auto input_ids = citrius::from_vector(std::vector<std::int64_t>{1, 2}, {1, 2});

    const auto logits = model(input_ids);

    EXPECT_EQ(logits.shape(), (citrius::Shape{1, 2, 8}));
    EXPECT_EQ(model.model().token_embedding().weight().shape(), (citrius::Shape{8, 4}));
}

TEST(Qwen3Test, CausalMaskPreventsFutureTokensAffectingEarlierLogits) {
    citrius::models::Qwen3ForCausalLM model(tiny_config());
    const auto first = model(citrius::from_vector(std::vector<std::int64_t>{1, 2}, {1, 2}));
    const auto second = model(citrius::from_vector(std::vector<std::int64_t>{1, 3}, {1, 2}));
    const auto first_values = values(first);
    const auto second_values = values(second);

    for (std::size_t index = 0; index < 8; ++index) {
        EXPECT_NEAR(first_values[index], second_values[index], 1e-5f);
    }
}
