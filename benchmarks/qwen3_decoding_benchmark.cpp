#include "models/qwen3.h"

#include "indexing_operations.h"
#include "reduction_operations.h"
#include "tensor_factory.h"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::int64_t default_generated_token_count = 500;

struct Options {
    citrius::Device device;
    citrius::DType dtype = citrius::DType::Float32;
    std::int64_t generated_token_count = default_generated_token_count;
};

const char* dtype_name(citrius::DType dtype) {
    switch (dtype) {
    case citrius::DType::Float32: return "float32";
    case citrius::DType::Float16: return "float16";
    case citrius::DType::BFloat16: return "bfloat16";
    default: return "unknown";
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool has_device = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--cpu") {
            if (has_device) throw std::invalid_argument("device may only be specified once");
            options.device = citrius::Device::cpu();
            has_device = true;
        } else if (argument == "--cuda") {
            if (has_device) throw std::invalid_argument("device may only be specified once");
#ifdef CITRIUS_HAS_CUDA
            options.device = citrius::Device::cuda();
            has_device = true;
#else
            throw std::invalid_argument(
                "CUDA support was not enabled; configure with build.bat --cuda");
#endif
        } else if (argument == "--metal") {
            if (has_device) throw std::invalid_argument("device may only be specified once");
#ifdef CITRIUS_HAS_METAL
            options.device = citrius::Device::metal();
            has_device = true;
#else
            throw std::invalid_argument(
                "Metal support was not enabled; configure with ./build.sh --metal");
#endif
        } else if (argument == "--tokens") {
            if (++index == argc)
                throw std::invalid_argument("--tokens requires a positive integer");
            const std::string value = argv[index];
            const auto [end, error] = std::from_chars(
                value.data(), value.data() + value.size(), options.generated_token_count);
            if (error != std::errc{} || end != value.data() + value.size() ||
                options.generated_token_count <= 0) {
                throw std::invalid_argument("--tokens requires a positive integer");
            }
        } else if (argument == "--dtype") {
            if (++index == argc)
                throw std::invalid_argument("--dtype requires float32, float16, or bfloat16");
            const std::string value = argv[index];
            if (value == "float32") options.dtype = citrius::DType::Float32;
            else if (value == "float16") options.dtype = citrius::DType::Float16;
            else if (value == "bfloat16") options.dtype = citrius::DType::BFloat16;
            else throw std::invalid_argument(
                "--dtype requires float32, float16, or bfloat16");
        } else {
            throw std::invalid_argument(
                "usage: qwen3_decoding_benchmark --cpu|--cuda|--metal [--tokens N] "
                "[--dtype float32|float16|bfloat16]");
        }
    }
    if (!has_device)
        throw std::invalid_argument(
            "usage: qwen3_decoding_benchmark --cpu|--cuda|--metal [--tokens N] "
            "[--dtype float32|float16|bfloat16]");
    if (options.device.type == citrius::DeviceType::Metal &&
        options.dtype != citrius::DType::Float32) {
        throw std::invalid_argument("Metal Qwen3 decoding currently requires float32");
    }
    return options;
}

std::int64_t last_token_argmax(const citrius::Tensor& logits) {
    const auto final_logits = logits.index(
        {citrius::indexing::Ellipsis, -1, citrius::indexing::Slice()});
    return citrius::argmax(final_logits, -1).item<std::int64_t>();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const citrius::Device device = options.device;
        const std::int64_t generated_token_count = options.generated_token_count;
        citrius::models::Qwen3Config config;
        config.device = device;
        config.dtype = options.dtype;

        const char* device_name = device.type == citrius::DeviceType::CUDA ? "CUDA" :
            device.type == citrius::DeviceType::Metal ? "Metal" : "CPU";
        std::cout << "Qwen3-0.6B checkpoint-free greedy decoding baseline\n"
                  << "Device: " << device_name
                  << "\nParameter dtype: " << dtype_name(options.dtype)
                  << "\nGenerated tokens: " << generated_token_count << "\nPrompt tokens: 9\n"
                  << "Constructing initialized model (excluded from timings)...\n";
        citrius::models::Qwen3ForCausalLM model(config);
        std::vector<std::int64_t> context{151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198};

        std::vector<double> token_seconds;
        token_seconds.reserve(generated_token_count);
        std::vector<std::int64_t> generated;
        generated.reserve(generated_token_count);
        const auto benchmark_start = Clock::now();
        for (std::int64_t index = 0; index < generated_token_count; ++index) {
            const auto token_start = Clock::now();
            const auto input_ids = citrius::from_vector(
                context, {1, static_cast<std::int64_t>(context.size())}, device);
            const std::int64_t token = last_token_argmax(model.forward_last_token(input_ids));
            const auto token_end = Clock::now();
            token_seconds.push_back(std::chrono::duration<double>(token_end - token_start).count());
            generated.push_back(token);
            context.push_back(token);
            std::cout << "Token " << index + 1 << "/" << generated_token_count << ": " << token
                      << " in " << std::fixed << std::setprecision(3)
                      << token_seconds.back() * 1000.0 << " ms" << std::endl;
        }
        const double total_seconds =
            std::chrono::duration<double>(Clock::now() - benchmark_start).count();
        double decode_seconds = 0.0;
        for (std::size_t index = 1; index < token_seconds.size(); ++index)
            decode_seconds += token_seconds[index];

        std::cout << "\nTTFT: " << token_seconds.front() * 1000.0 << " ms\n"
                  << "End-to-end throughput: " << generated_token_count / total_seconds
                  << " tokens/s\n";
        if (generated_token_count > 1)
            std::cout << "Post-first-token throughput: "
                      << (generated_token_count - 1) / decode_seconds << " tokens/s\n";
        else
            std::cout << "Post-first-token throughput: n/a\n";
        std::cout << "Total generation time: " << total_seconds * 1000.0 << " ms\n"
                  << "Generated token IDs:";
        for (const auto token : generated)
            std::cout << ' ' << token;
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qwen3_decoding_benchmark: " << error.what() << '\n';
        return 1;
    }
}
