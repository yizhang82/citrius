#include "cpu_storage.h"
#include "operations.h"
#include "tensor_factory.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Result {
    double minimum_ms;
    double average_ms;
    double gflops;
    float checksum;
};

std::vector<float> input_values(std::int64_t count, int seed) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<float>(((i + seed) % 17) - 8) / 8.0f;
    }
    return values;
}

float cpu_checksum(const citrius::Tensor& tensor) {
    const auto cpu_tensor = tensor.to(citrius::Device::cpu());
    const auto storage =
        std::static_pointer_cast<citrius::CpuMemTensorStorageImpl>(cpu_tensor.storage());
    const float* values = storage->data_as<float>();
    float checksum = 0.0f;
    for (std::int64_t i = 0; i < cpu_tensor.numel(); ++i) checksum += values[i];
    return checksum;
}

template <typename Operation>
Result measure(std::int64_t size, int iterations, Operation operation) {
    operation();
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(iterations));
    float checksum = 0.0f;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto start = Clock::now();
        checksum = operation();
        const auto end = Clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    const double minimum = *std::min_element(timings.begin(), timings.end());
    double total = 0.0;
    for (double timing : timings) total += timing;
    const double average = total / iterations;
    const double operations = 2.0 * static_cast<double>(size) * size * size;
    return {minimum, average, operations / (minimum * 1.0e6), checksum};
}

Result benchmark_cpu(std::int64_t size, int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    return measure(size, iterations, [&] {
        citrius::Tensor a(a_values, {size, size});
        auto b = citrius::TensorFactory::from_vector(b_values, {size, size});
        return cpu_checksum(citrius::matmul(a, b));
    });
}

#ifdef CITRIUS_HAS_CUDA
Result benchmark_cuda(std::int64_t size, int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    return measure(size, iterations, [&] {
        citrius::Tensor a(a_values, {size, size}, citrius::Device::cuda());
        auto b = citrius::TensorFactory::from_vector(
            b_values, {size, size}, citrius::Device::cuda());
        auto output = citrius::matmul(a, b);
        return cpu_checksum(output);
    });
}
#endif

void print_result(const char* backend, std::int64_t size, int iterations, const Result& result) {
    std::cout << std::left << std::setw(7) << backend << std::right << std::setw(7) << size
              << std::setw(8) << iterations << std::fixed << std::setprecision(3)
              << std::setw(13) << result.minimum_ms << std::setw(13) << result.average_ms
              << std::setw(13) << result.gflops << std::setw(15) << result.checksum << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "--cpu" && std::string(argv[1]) != "--cuda")) {
        std::cerr << "Usage: matmul_benchmark --cpu|--cuda\n";
        return 1;
    }

    const bool use_cuda = std::string(argv[1]) == "--cuda";
#ifndef CITRIUS_HAS_CUDA
    if (use_cuda) {
        std::cerr << "CUDA support was not enabled for this build.\n";
        return 1;
    }
#endif

    std::cout << "End-to-end square Float32 matmul (one warmup; input setup and output read included)\n";
    std::cout << std::left << std::setw(7) << "Backend" << std::right << std::setw(7) << "N"
              << std::setw(8) << "Runs" << std::setw(13) << "Best ms"
              << std::setw(13) << "Avg ms" << std::setw(13) << "GFLOP/s"
              << std::setw(15) << "Checksum" << '\n';

    try {
        for (const auto [size, iterations] :
             std::vector<std::pair<std::int64_t, int>>{{128, 50}, {256, 50}, {512, 50}}) {
#ifdef CITRIUS_HAS_CUDA
            const auto result = use_cuda ? benchmark_cuda(size, iterations)
                                         : benchmark_cpu(size, iterations);
#else
            const auto result = benchmark_cpu(size, iterations);
#endif
            print_result(use_cuda ? "CUDA" : "CPU", size, iterations, result);
        }
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
