#include "models/qwen3.h"

#include "impl/cpu_storage.h"
#include "safetensors.h"
#include "tensor_factory.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
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

std::filesystem::path write_safetensors(
    const std::string& filename,
    const std::string& header,
    const std::vector<unsigned char>& data) {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t length = header.size();
    for (int index = 0; index < 8; ++index) {
        stream.put(static_cast<char>((length >> (8 * index)) & 0xff));
    }
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    stream.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return path;
}

void append_f32(std::vector<unsigned char>& data, float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (int index = 0; index < 4; ++index) {
        data.push_back(static_cast<unsigned char>((bits >> (8 * index)) & 0xff));
    }
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

TEST(Qwen3Test, LastTokenForwardMatchesFullLogits) {
    citrius::models::Qwen3ForCausalLM model(tiny_config());
    const auto input_ids =
        citrius::from_vector(std::vector<std::int64_t>{1, 2, 3}, {1, 3});
    const auto full = model(input_ids);
    const auto last = model.forward_last_token(input_ids);
    EXPECT_EQ(last.shape(), (citrius::Shape{1, 1, 8}));
    const auto full_values = values(full);
    const auto last_values = values(last);
    ASSERT_EQ(last_values.size(), 8u);
    for (std::size_t index = 0; index < last_values.size(); ++index)
        EXPECT_NEAR(last_values[index], full_values[16 + index], 1e-6f);
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

TEST(Qwen3Test, LoadsFloat32AndBfloat16Safetensors) {
    const std::string header =
        R"({"float":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("bfloat":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]},)"
        R"("half":{"dtype":"F16","shape":[2],"data_offsets":[12,16]}})";
    std::vector<unsigned char> data;
    append_f32(data, 1.5f);
    append_f32(data, -2.0f);
    const std::uint16_t one = 0x3f80;
    const std::uint16_t minus_half = 0xbf00;
    data.push_back(static_cast<unsigned char>(one & 0xff));
    data.push_back(static_cast<unsigned char>(one >> 8));
    data.push_back(static_cast<unsigned char>(minus_half & 0xff));
    data.push_back(static_cast<unsigned char>(minus_half >> 8));
    const std::uint16_t half_one = 0x3c00;
    const std::uint16_t half_minus_half = 0xb800;
    data.push_back(static_cast<unsigned char>(half_one & 0xff));
    data.push_back(static_cast<unsigned char>(half_one >> 8));
    data.push_back(static_cast<unsigned char>(half_minus_half & 0xff));
    data.push_back(static_cast<unsigned char>(half_minus_half >> 8));
    const auto path = write_safetensors("citrius_dtype_test.safetensors", header, data);

    const auto tensors = citrius::load_safetensors(path);
    std::filesystem::remove(path);

    EXPECT_EQ(values(tensors.at("float")), (std::vector<float>{1.5f, -2.0f}));
    EXPECT_EQ(values(tensors.at("bfloat")), (std::vector<float>{1.0f, -0.5f}));
    EXPECT_EQ(values(tensors.at("half")), (std::vector<float>{1.0f, -0.5f}));
}

TEST(Qwen3Test, MapsHuggingFaceEmbeddingWeight) {
    auto config = tiny_config();
    citrius::models::Qwen3ForCausalLM model(config);
    std::vector<unsigned char> data;
    for (int index = 0; index < 32; ++index) append_f32(data, static_cast<float>(index));
    const std::string header =
        R"({"model.embed_tokens.weight":{"dtype":"F32","shape":[8,4],"data_offsets":[0,128]}})";
    const auto path = write_safetensors("citrius_qwen_mapping_test.safetensors", header, data);

    citrius::models::load_qwen3_weights(model, path, false);
    std::filesystem::remove(path);

    const auto loaded = values(model.model().token_embedding().weight());
    ASSERT_EQ(loaded.size(), 32u);
    EXPECT_FLOAT_EQ(loaded.front(), 0.0f);
    EXPECT_FLOAT_EQ(loaded.back(), 31.0f);
}
