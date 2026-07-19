#include "models/qwen3.h"

#include "nn/functional.h"
#include "operations.h"
#include "reduction_operations.h"
#include "safetensors.h"
#include "shape_operations.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citrius::models {
namespace {

Tensor silu(const Tensor& input) {
    return div(input, add(exp(mul(input, -1.0f)), 1.0f));
}

Tensor causal_mask(std::int64_t sequence_length, Device device) {
    std::vector<bool> values(static_cast<std::size_t>(sequence_length * sequence_length));
    for (std::int64_t row = 0; row < sequence_length; ++row) {
        for (std::int64_t column = row + 1; column < sequence_length; ++column) {
            values[static_cast<std::size_t>(row * sequence_length + column)] = true;
        }
    }
    return from_vector(values, {sequence_length, sequence_length}, device);
}

Tensor repeat_key_value_heads(const Tensor& tensor, std::int64_t repetitions) {
    if (repetitions == 1) return tensor;
    const auto heads = split(tensor, 1, 1);
    std::vector<Tensor> repeated;
    repeated.reserve(heads.size() * static_cast<std::size_t>(repetitions));
    for (const Tensor& head : heads) {
        for (std::int64_t index = 0; index < repetitions; ++index) repeated.push_back(head);
    }
    return concat(repeated, 1);
}

Tensor apply_rope(const Tensor& tensor, float theta) {
    const auto& shape = tensor.shape();
    const std::int64_t sequence_length = shape[2];
    const std::int64_t head_dim = shape[3];
    if (head_dim % 2 != 0) throw std::invalid_argument("Qwen3 RoPE head_dim must be even");

    std::vector<float> cosine(static_cast<std::size_t>(sequence_length * head_dim));
    std::vector<float> sine(cosine.size());
    const std::int64_t half = head_dim / 2;
    for (std::int64_t position = 0; position < sequence_length; ++position) {
        for (std::int64_t index = 0; index < half; ++index) {
            const float frequency = 1.0f / std::pow(theta, static_cast<float>(2 * index) / static_cast<float>(head_dim));
            const float angle = static_cast<float>(position) * frequency;
            const float cos_value = std::cos(angle);
            const float sin_value = std::sin(angle);
            cosine[static_cast<std::size_t>(position * head_dim + index)] = cos_value;
            cosine[static_cast<std::size_t>(position * head_dim + half + index)] = cos_value;
            sine[static_cast<std::size_t>(position * head_dim + index)] = sin_value;
            sine[static_cast<std::size_t>(position * head_dim + half + index)] = sin_value;
        }
    }

    const Tensor cos_tensor = from_vector(cosine, {1, 1, sequence_length, head_dim}, tensor.device());
    const Tensor sin_tensor = from_vector(sine, {1, 1, sequence_length, head_dim}, tensor.device());
    const auto halves = split(tensor, half, -1);
    const Tensor rotated = concat({mul(halves[1], -1.0f), halves[0]}, -1);
    return add(mul(tensor, cos_tensor), mul(rotated, sin_tensor));
}

} // namespace

void Qwen3Config::validate() const {
    if (vocab_size <= 0 || hidden_size <= 0 || intermediate_size <= 0 ||
        num_hidden_layers <= 0 || num_attention_heads <= 0 ||
        num_key_value_heads <= 0 || head_dim <= 0 || max_position_embeddings <= 0) {
        throw std::invalid_argument("Qwen3 dimensions must be positive");
    }
    if (num_attention_heads % num_key_value_heads != 0) {
        throw std::invalid_argument("Qwen3 query heads must be divisible by key/value heads");
    }
    if (head_dim % 2 != 0) throw std::invalid_argument("Qwen3 head_dim must be even");
    if (!(rms_norm_eps > 0.0f) || !(rope_theta > 0.0f)) {
        throw std::invalid_argument("Qwen3 normalization epsilon and RoPE theta must be positive");
    }
}

Qwen3RMSNorm::Qwen3RMSNorm(std::int64_t hidden_size, float eps, Device device)
    : hidden_size_(hidden_size), eps_(eps) {
    if (hidden_size <= 0 || !(eps > 0.0f)) {
        throw std::invalid_argument("Qwen3RMSNorm size and epsilon must be positive");
    }
    register_parameter(
        "weight",
        Tensor(std::vector<float>(static_cast<std::size_t>(hidden_size), 1.0f), {hidden_size}, device));
}

Tensor Qwen3RMSNorm::forward(const Tensor& input) {
    CITRIUS_ENSURE_TENSOR_DEFINED(input);
    if (input.dtype() != DType::Float32 || input.ndim() == 0 || input.shape().back() != hidden_size_) {
        throw std::invalid_argument("Qwen3RMSNorm requires Float32 input ending in hidden_size");
    }
    const Tensor variance = mean(pow(input, 2.0f), -1, true);
    return mul(div(input, sqrt(add(variance, eps_))), weight());
}

Tensor& Qwen3RMSNorm::weight() { return parameter("weight"); }
const Tensor& Qwen3RMSNorm::weight() const { return parameter("weight"); }

Qwen3MLP::Qwen3MLP(const Qwen3Config& config) {
    config.validate();
    gate_projection_ = register_module("gate_proj", std::make_shared<nn::Linear>(config.hidden_size, config.intermediate_size, false, config.device));
    up_projection_ = register_module("up_proj", std::make_shared<nn::Linear>(config.hidden_size, config.intermediate_size, false, config.device));
    down_projection_ = register_module("down_proj", std::make_shared<nn::Linear>(config.intermediate_size, config.hidden_size, false, config.device));
}

Tensor Qwen3MLP::forward(const Tensor& input) {
    return (*down_projection_)(mul(silu((*gate_projection_)(input)), (*up_projection_)(input)));
}

nn::Linear& Qwen3MLP::gate_projection() { return *gate_projection_; }
nn::Linear& Qwen3MLP::up_projection() { return *up_projection_; }
nn::Linear& Qwen3MLP::down_projection() { return *down_projection_; }

Qwen3Attention::Qwen3Attention(const Qwen3Config& config) : config_(config) {
    config_.validate();
    query_projection_ = register_module("q_proj", std::make_shared<nn::Linear>(config.hidden_size, config.num_attention_heads * config.head_dim, false, config.device));
    key_projection_ = register_module("k_proj", std::make_shared<nn::Linear>(config.hidden_size, config.num_key_value_heads * config.head_dim, false, config.device));
    value_projection_ = register_module("v_proj", std::make_shared<nn::Linear>(config.hidden_size, config.num_key_value_heads * config.head_dim, false, config.device));
    output_projection_ = register_module("o_proj", std::make_shared<nn::Linear>(config.num_attention_heads * config.head_dim, config.hidden_size, false, config.device));
    query_norm_ = register_module("q_norm", std::make_shared<Qwen3RMSNorm>(config.head_dim, config.rms_norm_eps, config.device));
    key_norm_ = register_module("k_norm", std::make_shared<Qwen3RMSNorm>(config.head_dim, config.rms_norm_eps, config.device));
}

Tensor Qwen3Attention::forward(const Tensor& input) {
    return forward(input, Tensor());
}

Tensor Qwen3Attention::forward(const Tensor& input, const Tensor& attn_mask) {
    CITRIUS_ENSURE_TENSOR_DEFINED(input);
    CITRIUS_ENSURE_TENSOR_DIM(input, 3);
    if (input.shape().back() != config_.hidden_size) {
        throw std::invalid_argument("Qwen3Attention input must end in hidden_size");
    }
    const std::int64_t batch = input.shape()[0];
    const std::int64_t sequence = input.shape()[1];
    Tensor query = reshape((*query_projection_)(input), {batch, sequence, config_.num_attention_heads, config_.head_dim});
    Tensor key = reshape((*key_projection_)(input), {batch, sequence, config_.num_key_value_heads, config_.head_dim});
    Tensor value = reshape((*value_projection_)(input), {batch, sequence, config_.num_key_value_heads, config_.head_dim});
    query = (*query_norm_)(query);
    key = (*key_norm_)(key);
    query = apply_rope(permute(query, {0, 2, 1, 3}), config_.rope_theta);
    key = apply_rope(permute(key, {0, 2, 1, 3}), config_.rope_theta);
    value = permute(value, {0, 2, 1, 3});
    const std::int64_t groups = config_.num_attention_heads / config_.num_key_value_heads;
    key = repeat_key_value_heads(key, groups);
    value = repeat_key_value_heads(value, groups);
    Tensor output = nn::functional::scaled_dot_product_attention(query, key, value, attn_mask);
    output = reshape(permute(output, {0, 2, 1, 3}), {batch, sequence, config_.num_attention_heads * config_.head_dim});
    return (*output_projection_)(output);
}

nn::Linear& Qwen3Attention::query_projection() { return *query_projection_; }
nn::Linear& Qwen3Attention::key_projection() { return *key_projection_; }
nn::Linear& Qwen3Attention::value_projection() { return *value_projection_; }
nn::Linear& Qwen3Attention::output_projection() { return *output_projection_; }
Qwen3RMSNorm& Qwen3Attention::query_norm() { return *query_norm_; }
Qwen3RMSNorm& Qwen3Attention::key_norm() { return *key_norm_; }

Qwen3DecoderLayer::Qwen3DecoderLayer(const Qwen3Config& config) {
    config.validate();
    attention_ = register_module("self_attn", std::make_shared<Qwen3Attention>(config));
    mlp_ = register_module("mlp", std::make_shared<Qwen3MLP>(config));
    input_norm_ = register_module("input_layernorm", std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps, config.device));
    post_attention_norm_ = register_module("post_attention_layernorm", std::make_shared<Qwen3RMSNorm>(config.hidden_size, config.rms_norm_eps, config.device));
}

Tensor Qwen3DecoderLayer::forward(const Tensor& input) { return forward(input, Tensor()); }
Tensor Qwen3DecoderLayer::forward(const Tensor& input, const Tensor& attn_mask) {
    const Tensor attended = attention_->forward((*input_norm_)(input), attn_mask);
    const Tensor attention_residual = add(input, attended);
    return add(attention_residual, (*mlp_)((*post_attention_norm_)(attention_residual)));
}
Qwen3Attention& Qwen3DecoderLayer::attention() { return *attention_; }
Qwen3MLP& Qwen3DecoderLayer::mlp() { return *mlp_; }
Qwen3RMSNorm& Qwen3DecoderLayer::input_norm() { return *input_norm_; }
Qwen3RMSNorm& Qwen3DecoderLayer::post_attention_norm() { return *post_attention_norm_; }

Qwen3Model::Qwen3Model(Qwen3Config config) : config_(std::move(config)) {
    config_.validate();
    token_embedding_ = register_module("embed_tokens", std::make_shared<nn::Embedding>(config_.vocab_size, config_.hidden_size, config_.device));
    layers_.reserve(static_cast<std::size_t>(config_.num_hidden_layers));
    for (std::int64_t index = 0; index < config_.num_hidden_layers; ++index) {
        layers_.push_back(register_module("layers_" + std::to_string(index), std::make_shared<Qwen3DecoderLayer>(config_)));
    }
    norm_ = register_module("norm", std::make_shared<Qwen3RMSNorm>(config_.hidden_size, config_.rms_norm_eps, config_.device));
}

Tensor Qwen3Model::forward(const Tensor& input_ids) {
    CITRIUS_ENSURE_TENSOR_DEFINED(input_ids);
    CITRIUS_ENSURE_TENSOR_DIM(input_ids, 2);
    if (input_ids.dtype() != DType::Int64) throw std::invalid_argument("Qwen3 input_ids must be Int64");
    if (input_ids.shape()[1] > config_.max_position_embeddings) {
        throw std::invalid_argument("Qwen3 sequence exceeds max_position_embeddings");
    }
    Tensor hidden = (*token_embedding_)(input_ids);
    const Tensor mask = causal_mask(input_ids.shape()[1], input_ids.device());
    for (const auto& layer : layers_) hidden = layer->forward(hidden, mask);
    return (*norm_)(hidden);
}

nn::Embedding& Qwen3Model::token_embedding() { return *token_embedding_; }
Qwen3DecoderLayer& Qwen3Model::layer(std::size_t index) { return *layers_.at(index); }
Qwen3RMSNorm& Qwen3Model::norm() { return *norm_; }
const Qwen3Config& Qwen3Model::config() const { return config_; }

Qwen3ForCausalLM::Qwen3ForCausalLM(Qwen3Config config) {
    model_ = register_module("model", std::make_shared<Qwen3Model>(std::move(config)));
}

Tensor Qwen3ForCausalLM::forward(const Tensor& input_ids) {
    const Tensor hidden = (*model_)(input_ids);
    return matmul(hidden, transpose(model_->token_embedding().weight(), 0, 1));
}

Qwen3Model& Qwen3ForCausalLM::model() { return *model_; }
const Qwen3Config& Qwen3ForCausalLM::config() const { return model_->config(); }

void load_qwen3_weights(
    Qwen3ForCausalLM& model,
    const std::filesystem::path& path,
    bool strict) {
    TensorMap tensors = load_safetensors(path, model.config().device);
    const auto assign = [&](const std::string& name, Tensor& destination) {
        const auto found = tensors.find(name);
        if (found == tensors.end()) {
            if (strict) throw std::runtime_error("missing Qwen3 checkpoint tensor: " + name);
            return;
        }
        if (found->second.shape() != destination.shape()) {
            throw std::runtime_error("Qwen3 checkpoint shape mismatch: " + name);
        }
        destination = found->second;
        tensors.erase(found);
    };

    assign("model.embed_tokens.weight", model.model().token_embedding().weight());
    for (std::int64_t index = 0; index < model.config().num_hidden_layers; ++index) {
        Qwen3DecoderLayer& layer = model.model().layer(static_cast<std::size_t>(index));
        const std::string prefix = "model.layers." + std::to_string(index) + ".";
        assign(prefix + "self_attn.q_proj.weight", layer.attention().query_projection().weight());
        assign(prefix + "self_attn.k_proj.weight", layer.attention().key_projection().weight());
        assign(prefix + "self_attn.v_proj.weight", layer.attention().value_projection().weight());
        assign(prefix + "self_attn.o_proj.weight", layer.attention().output_projection().weight());
        assign(prefix + "self_attn.q_norm.weight", layer.attention().query_norm().weight());
        assign(prefix + "self_attn.k_norm.weight", layer.attention().key_norm().weight());
        assign(prefix + "mlp.gate_proj.weight", layer.mlp().gate_projection().weight());
        assign(prefix + "mlp.up_proj.weight", layer.mlp().up_projection().weight());
        assign(prefix + "mlp.down_proj.weight", layer.mlp().down_projection().weight());
        assign(prefix + "input_layernorm.weight", layer.input_norm().weight());
        assign(prefix + "post_attention_layernorm.weight", layer.post_attention_norm().weight());
    }
    assign("model.norm.weight", model.model().norm().weight());
    tensors.erase("lm_head.weight");
    if (strict && !tensors.empty()) {
        throw std::runtime_error("unexpected Qwen3 checkpoint tensor: " + tensors.begin()->first);
    }
}

} // namespace citrius::models
