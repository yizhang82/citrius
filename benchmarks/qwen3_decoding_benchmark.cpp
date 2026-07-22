#include "models/qwen3.h"

#include "indexing_operations.h"
#include "reduction_operations.h"
#include "tensor_factory.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::int64_t generated_token_count = 10;

citrius::Device parse_device(int argc, char** argv) {
    if (argc != 2) {
        throw std::invalid_argument("usage: qwen3_decoding_benchmark --cpu|--cuda");
    }
    const std::string argument = argv[1];
    if (argument == "--cpu")
        return citrius::Device::cpu();
    if (argument == "--cuda") {
#ifdef CITRIUS_HAS_CUDA
        return citrius::Device::cuda();
#else
        throw std::invalid_argument(
            "CUDA support was not enabled; configure with build.bat --cuda");
#endif
    }
    throw std::invalid_argument("usage: qwen3_decoding_benchmark --cpu|--cuda");
}

std::int64_t last_token_argmax(const citrius::Tensor& logits) {
    const auto final_logits = logits.index(
        {citrius::indexing::Ellipsis, -1, citrius::indexing::Slice()});
    return citrius::argmax(final_logits, -1).item<std::int64_t>();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const citrius::Device device = parse_device(argc, argv);
        citrius::models::Qwen3Config config;
        config.device = device;

        std::cout << "Qwen3-0.6B checkpoint-free greedy decoding baseline\n"
                  << "Device: " << (device.type == citrius::DeviceType::CUDA ? "CUDA" : "CPU")
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
                  << " tokens/s\n"
                  << "Post-first-token throughput: " << (generated_token_count - 1) / decode_seconds
                  << " tokens/s\n"
                  << "Total generation time: " << total_seconds * 1000.0 << " ms\n"
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
