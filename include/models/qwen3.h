#pragma once

#include "nn/embedding.h"
#include "nn/linear.h"
#include "nn/module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace citrius::models {

struct Qwen3Config {
    std::int64_t vocab_size = 151936;
    std::int64_t hidden_size = 1024;
    std::int64_t intermediate_size = 3072;
    std::int64_t num_hidden_layers = 28;
    std::int64_t num_attention_heads = 16;
    std::int64_t num_key_value_heads = 8;
    std::int64_t head_dim = 128;
    std::int64_t max_position_embeddings = 40960;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    Device device = Device::cpu();

    void validate() const;
};

class Qwen3RMSNorm final : public nn::Module {
public:
    Qwen3RMSNorm(
        std::int64_t hidden_size,
        float eps = 1e-6f,
        Device device = Device::cpu());
    Tensor forward(const Tensor& input) override;
    Tensor& weight();
    const Tensor& weight() const;

private:
    std::int64_t hidden_size_;
    float eps_;
};

class Qwen3MLP final : public nn::Module {
public:
    explicit Qwen3MLP(const Qwen3Config& config);
    Tensor forward(const Tensor& input) override;
    nn::Linear& gate_projection();
    nn::Linear& up_projection();
    nn::Linear& down_projection();

private:
    std::shared_ptr<nn::Linear> gate_projection_;
    std::shared_ptr<nn::Linear> up_projection_;
    std::shared_ptr<nn::Linear> down_projection_;
};

class Qwen3Attention final : public nn::Module {
public:
    explicit Qwen3Attention(const Qwen3Config& config);
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor& attn_mask);

    nn::Linear& query_projection();
    nn::Linear& key_projection();
    nn::Linear& value_projection();
    nn::Linear& output_projection();
    Qwen3RMSNorm& query_norm();
    Qwen3RMSNorm& key_norm();

private:
    Qwen3Config config_;
    std::shared_ptr<nn::Linear> query_projection_;
    std::shared_ptr<nn::Linear> key_projection_;
    std::shared_ptr<nn::Linear> value_projection_;
    std::shared_ptr<nn::Linear> output_projection_;
    std::shared_ptr<Qwen3RMSNorm> query_norm_;
    std::shared_ptr<Qwen3RMSNorm> key_norm_;
};

class Qwen3DecoderLayer final : public nn::Module {
public:
    explicit Qwen3DecoderLayer(const Qwen3Config& config);
    Tensor forward(const Tensor& input) override;
    Tensor forward(const Tensor& input, const Tensor& attn_mask);
    Qwen3Attention& attention();
    Qwen3MLP& mlp();
    Qwen3RMSNorm& input_norm();
    Qwen3RMSNorm& post_attention_norm();

private:
    std::shared_ptr<Qwen3Attention> attention_;
    std::shared_ptr<Qwen3MLP> mlp_;
    std::shared_ptr<Qwen3RMSNorm> input_norm_;
    std::shared_ptr<Qwen3RMSNorm> post_attention_norm_;
};

class Qwen3Model final : public nn::Module {
public:
    explicit Qwen3Model(Qwen3Config config);
    Tensor forward(const Tensor& input_ids) override;
    nn::Embedding& token_embedding();
    Qwen3DecoderLayer& layer(std::size_t index);
    Qwen3RMSNorm& norm();
    const Qwen3Config& config() const;

private:
    Qwen3Config config_;
    std::shared_ptr<nn::Embedding> token_embedding_;
    std::vector<std::shared_ptr<Qwen3DecoderLayer>> layers_;
    std::shared_ptr<Qwen3RMSNorm> norm_;
};

class Qwen3ForCausalLM final : public nn::Module {
public:
    explicit Qwen3ForCausalLM(Qwen3Config config = {});
    Tensor forward(const Tensor& input_ids) override;
    Qwen3Model& model();
    const Qwen3Config& config() const;

private:
    std::shared_ptr<Qwen3Model> model_;
};

} // namespace citrius::models
