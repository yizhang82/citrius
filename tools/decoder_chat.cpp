#include "citrius.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string model_type = "qwen3";
    std::filesystem::path checkpoint;
    std::filesystem::path tokenizer;
    std::string prompt = "Hello";
    std::int64_t max_new_tokens = 64;
    citrius::Device device = citrius::Device::cpu();
};

class DecoderBackend {
  public:
    virtual ~DecoderBackend() = default;
    virtual citrius::Tensor logits(const citrius::Tensor& input_ids) = 0;
    virtual std::int64_t eos_token_id() const = 0;
    virtual std::int64_t max_sequence_length() const = 0;
    virtual citrius::Device device() const = 0;
};

class Qwen3Backend final : public DecoderBackend {
  public:
    explicit Qwen3Backend(const Options& options) {
        citrius::models::Qwen3Config config;
        config.device = options.device;
        model_ = std::make_unique<citrius::models::Qwen3ForCausalLM>(config);
        citrius::models::load_qwen3_weights(*model_, options.checkpoint);
    }

    citrius::Tensor logits(const citrius::Tensor& input_ids) override {
        return model_->forward_last_token(input_ids);
    }

    std::int64_t eos_token_id() const override {
        return 151645;
    }
    std::int64_t max_sequence_length() const override {
        return model_->config().max_position_embeddings;
    }
    citrius::Device device() const override {
        return model_->config().device;
    }

  private:
    std::unique_ptr<citrius::models::Qwen3ForCausalLM> model_;
};

std::int64_t last_token_argmax(const citrius::Tensor& logits) {
    const auto final_logits = logits.index(
        {citrius::indexing::Ellipsis, -1, citrius::indexing::Slice()});
    return citrius::argmax(final_logits, -1).item<std::int64_t>();
}

std::vector<std::int64_t> generate(DecoderBackend& model, std::vector<std::int64_t>& context,
                                   std::int64_t max_new_tokens,
                                   const citrius::models::Qwen3Tokenizer& tokenizer) {
    std::vector<std::int64_t> generated;
    for (std::int64_t step = 0; step < max_new_tokens; ++step) {
        if (static_cast<std::int64_t>(context.size()) >= model.max_sequence_length())
            break;
        const citrius::Tensor input_ids = citrius::from_vector(
            context, {1, static_cast<std::int64_t>(context.size())}, model.device());
        const std::int64_t next_token = last_token_argmax(model.logits(input_ids));
        context.push_back(next_token);
        generated.push_back(next_token);
        std::cout << next_token << " (" << tokenizer.token_display_name(next_token) << ')'
                  << std::flush;
        if (next_token == model.eos_token_id())
            break;
        std::cout << ' ' << std::flush;
    }
    std::cout << '\n';
    return generated;
}

void usage() {
    std::cout << "Usage: decoder_chat --checkpoint FILE [options]\n"
              << "  --model-type qwen3  Decoder adapter (default: qwen3)\n"
              << "  --tokenizer DIR     Directory containing vocab.json and merges.txt\n"
              << "  --prompt TEXT       User prompt (default: Hello)\n"
              << "  --max-new-tokens N  Tokens to generate (default: 64)\n"
              << "  --device cpu|cuda   Execution device (default: cpu)\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto value = [&]() -> std::string {
            if (++index >= argc)
                throw std::invalid_argument("missing value for " + argument);
            return argv[index];
        };
        if (argument == "--checkpoint")
            options.checkpoint = value();
        else if (argument == "--model-type")
            options.model_type = value();
        else if (argument == "--tokenizer")
            options.tokenizer = value();
        else if (argument == "--prompt")
            options.prompt = value();
        else if (argument == "--max-new-tokens")
            options.max_new_tokens = std::stoll(value());
        else if (argument == "--device") {
            const std::string device = value();
            if (device == "cpu")
                options.device = citrius::Device::cpu();
            else if (device == "cuda")
                options.device = citrius::Device::cuda();
            else
                throw std::invalid_argument("device must be cpu or cuda");
        } else if (argument == "--help" || argument == "-h") {
            usage();
            std::exit(0);
        } else
            throw std::invalid_argument("unknown option: " + argument);
    }
    if (options.checkpoint.empty())
        throw std::invalid_argument("--checkpoint is required");
    if (options.tokenizer.empty())
        options.tokenizer = options.checkpoint.parent_path();
    if (options.max_new_tokens <= 0)
        throw std::invalid_argument("max-new-tokens must be positive");
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.model_type != "qwen3") {
            throw std::invalid_argument("unsupported model type: " + options.model_type);
        }
        Qwen3Backend model(options);
        const citrius::models::Qwen3Tokenizer tokenizer(options.tokenizer);
        std::vector<std::int64_t> context = tokenizer.encode_chat_prompt(options.prompt);

        std::cout << "prompt token IDs: ";
        for (const auto token : context)
            std::cout << token << " (" << tokenizer.token_display_name(token) << ") ";
        std::cout << "\ngenerated token IDs: ";
        const auto generated = generate(model, context, options.max_new_tokens, tokenizer);
        std::cout << "response: " << tokenizer.decode(generated, true) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "decoder_chat: " << error.what() << '\n';
        return 1;
    }
}
