#include "models/qwen3.h"
#include "nn/embedding.h"
#include "nn/feed_forward.h"
#include "nn/layer_norm.h"
#include "nn/linear.h"
#include "nn/multi_head_attention.h"
#include "nn/transformer_block.h"
#include "shape_operations.h"
#include "tensor_factory.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

struct DeviceCase {
    const char* name;
    citrius::Device device;
};

std::vector<DeviceCase> device_cases() {
    std::vector<DeviceCase> devices{{"CPU", citrius::Device::cpu()}};
#ifdef CITRIUS_TEST_HAS_CUDA
    devices.push_back({"CUDA", citrius::Device::cuda()});
#endif
#ifdef CITRIUS_TEST_HAS_METAL
    devices.push_back({"Metal", citrius::Device::metal()});
#endif
    return devices;
}

std::vector<float> values(const citrius::Tensor& tensor) {
    const citrius::Tensor cpu = citrius::contiguous(tensor).to(citrius::Device::cpu());
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(cpu.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + cpu.numel());
}

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "element " << index;
}

void set_linear(citrius::nn::Linear& linear, std::vector<float> weight, std::vector<float> bias,
                citrius::Device device) {
    linear.weight() =
        citrius::Tensor(std::move(weight), {linear.out_features(), linear.in_features()}, device);
    if (linear.has_bias())
        linear.bias() = citrius::Tensor(std::move(bias), {linear.out_features()}, device);
}

void set_identity(citrius::nn::Linear& linear, citrius::Device device) {
    ASSERT_EQ(linear.in_features(), linear.out_features());
    const auto size = linear.in_features();
    std::vector<float> weight(static_cast<std::size_t>(size * size), 0.0f);
    for (std::int64_t index = 0; index < size; ++index)
        weight[static_cast<std::size_t>(index * size + index)] = 1.0f;
    set_linear(linear, std::move(weight), std::vector<float>(static_cast<std::size_t>(size)),
               device);
}

void set_zero(citrius::nn::Linear& linear, citrius::Device device) {
    set_linear(linear,
               std::vector<float>(
                   static_cast<std::size_t>(linear.in_features() * linear.out_features()), 0.0f),
               std::vector<float>(static_cast<std::size_t>(linear.out_features()), 0.0f), device);
}

class DeviceNnConformanceTest : public testing::TestWithParam<DeviceCase> {
  protected:
    citrius::Device device() const {
        return GetParam().device;
    }

    citrius::Tensor tensor(std::vector<float> data, citrius::Shape shape) const {
        return citrius::Tensor(std::move(data), std::move(shape), device());
    }
};

TEST_P(DeviceNnConformanceTest, EmbeddingMatchesTorchIndexSelection) {
    citrius::nn::Embedding embedding(4, 3, device());
    embedding.weight() = tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {4, 3});
    const auto indices =
        citrius::from_vector(std::vector<std::int64_t>{3, 1, 0, 3}, {2, 2}, device());

    const auto output = embedding(indices);
    EXPECT_EQ(output.shape(), (citrius::Shape{2, 2, 3}));
    EXPECT_EQ(values(output), (std::vector<float>{10, 11, 12, 4, 5, 6, 1, 2, 3, 10, 11, 12}));
}

TEST_P(DeviceNnConformanceTest, LinearMatchesTorchAffineProjection) {
    citrius::nn::Linear linear(3, 2, true, device());
    set_linear(linear, {1, 2, 3, -1, 0.5f, 2}, {0.25f, -0.5f}, device());
    const auto output = linear(tensor({1, 2, 3, 4, 5, 6}, {1, 2, 3}));

    EXPECT_EQ(output.shape(), (citrius::Shape{1, 2, 2}));
    expect_near(values(output), {14.25f, 5.5f, 32.25f, 10.0f});
}

TEST_P(DeviceNnConformanceTest, LayerNormMatchesTorchBiasedVariance) {
    citrius::nn::LayerNorm layer_norm(3, 1e-5f, true, device());
    layer_norm.weight() = tensor({1, 2, 3}, {3});
    layer_norm.bias() = tensor({0.5f, 0, -0.5f}, {3});

    expect_near(values(layer_norm(tensor({1, 2, 3, 4, 5, 6}, {2, 3}))),
                {-0.724736f, 0.0f, 3.174207f, -0.724736f, 0.0f, 3.174207f});
}

TEST_P(DeviceNnConformanceTest, FeedForwardMatchesTorchGeluComposition) {
    citrius::nn::FeedForward feed_forward(2, 2, true, device());
    set_identity(feed_forward.input_projection(), device());
    set_identity(feed_forward.output_projection(), device());

    expect_near(values(feed_forward(tensor({-1, 0, 1, 2}, {1, 2, 2}))),
                {-0.158808f, 0.0f, 0.841192f, 1.954598f});
}

TEST_P(DeviceNnConformanceTest, MultiHeadAttentionMatchesTorchAcrossHeadsAndMask) {
    citrius::nn::MultiHeadAttention attention(4, 2, true, device());
    set_identity(attention.query_projection(), device());
    set_identity(attention.key_projection(), device());
    set_identity(attention.value_projection(), device());
    set_identity(attention.output_projection(), device());
    const auto input = tensor({1, 0, 0, 1, 0, 1, 1, 0}, {1, 2, 4});
    const auto mask =
        citrius::from_vector(std::vector<bool>{false, true, false, false}, {2, 2}, device());

    expect_near(values(attention.forward(input, mask)),
                {1.0f, 0.0f, 0.0f, 1.0f, 0.330238f, 0.669762f, 0.669762f, 0.330238f});
}

TEST_P(DeviceNnConformanceTest, TransformerBlockMatchesTorchPreNormResidualComposition) {
    citrius::nn::TransformerBlock block(4, 2, 4, 1e-5f, true, device());
    set_identity(block.attention().query_projection(), device());
    set_identity(block.attention().key_projection(), device());
    set_identity(block.attention().value_projection(), device());
    set_identity(block.attention().output_projection(), device());
    set_zero(block.feed_forward().input_projection(), device());
    set_zero(block.feed_forward().output_projection(), device());
    const auto input = tensor({1, 0, 0, 1, 0, 1, 1, 0}, {1, 2, 4});
    const auto mask =
        citrius::from_vector(std::vector<bool>{false, true, false, false}, {2, 2}, device());

    expect_near(values(block.forward(input, mask)),
                {1.99997997f, -0.99997997f, -0.99997997f, 1.99997997f, -0.88835585f, 1.88835585f,
                 1.88835585f, -0.88835585f},
                3e-5f);
}

citrius::models::Qwen3Config tiny_qwen_config(citrius::Device device) {
    citrius::models::Qwen3Config config;
    config.vocab_size = 8;
    config.hidden_size = 2;
    config.intermediate_size = 2;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = 2;
    config.max_position_embeddings = 8;
    config.device = device;
    return config;
}

TEST_P(DeviceNnConformanceTest, QwenRmsNormAndMlpMatchTorchComposition) {
    citrius::models::Qwen3RMSNorm norm(2, 1e-6f, device());
    norm.weight() = tensor({1.5f, 0.5f}, {2});
    expect_near(values(norm(tensor({1, 2, 3, 4}, {1, 2, 2}))),
                {0.9486832f, 0.6324555f, 1.2727922f, 0.5656854f}, 2e-5f);

    citrius::models::Qwen3MLP mlp(tiny_qwen_config(device()));
    set_identity(mlp.gate_projection(), device());
    set_identity(mlp.up_projection(), device());
    set_identity(mlp.down_projection(), device());
    expect_near(values(mlp(tensor({-1, 0, 1, 2}, {1, 2, 2}))),
                {0.2689414f, 0.0f, 0.7310586f, 3.5231883f}, 2e-5f);
}

INSTANTIATE_TEST_SUITE_P(AllCompiledDevices, DeviceNnConformanceTest,
                         testing::ValuesIn(device_cases()),
                         [](const testing::TestParamInfo<DeviceCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace
