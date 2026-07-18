#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr int threads_per_block = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct Add {
    __device__ float operator()(float a, float b) const { return a + b; }
    static constexpr const char* name = "add";
};

struct Sub {
    __device__ float operator()(float a, float b) const { return a - b; }
    static constexpr const char* name = "sub";
};

template <typename Operation>
__global__ void grid_stride_kernel(
    const float* a,
    const float* b,
    float* out,
    std::int64_t count) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (auto i = first; i < count; i += stride) out[i] = Operation{}(a[i], b[i]);
}

template <int ElementsPerThread, typename Operation>
__global__ void fixed_work_kernel(
    const float* a,
    const float* b,
    float* out,
    std::int64_t count) {
    const auto first = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
#pragma unroll
    for (int pass = 0; pass < ElementsPerThread; ++pass) {
        const auto i = first + static_cast<std::int64_t>(pass) * stride;
        if (i < count) out[i] = Operation{}(a[i], b[i]);
    }
}

enum class Strategy { GridStride, FixedWork };

struct Configuration {
    std::string label;
    Strategy strategy;
    int value;
};

struct Result {
    double best_us;
    double average_us;
    double bandwidth_gbs;
    unsigned blocks;
    std::int64_t threads;
    double average_elements_per_thread;
};

unsigned divide_up(std::int64_t numerator, std::int64_t denominator) {
    return static_cast<unsigned>((numerator + denominator - 1) / denominator);
}

template <typename Operation>
void launch_fixed(
    int elements_per_thread,
    unsigned blocks,
    const float* a,
    const float* b,
    float* out,
    std::int64_t count) {
    switch (elements_per_thread) {
        case 2: fixed_work_kernel<2, Operation><<<blocks, threads_per_block>>>(a, b, out, count); break;
        case 4: fixed_work_kernel<4, Operation><<<blocks, threads_per_block>>>(a, b, out, count); break;
        case 8: fixed_work_kernel<8, Operation><<<blocks, threads_per_block>>>(a, b, out, count); break;
        case 16: fixed_work_kernel<16, Operation><<<blocks, threads_per_block>>>(a, b, out, count); break;
        case 32: fixed_work_kernel<32, Operation><<<blocks, threads_per_block>>>(a, b, out, count); break;
        default: throw std::logic_error("unsupported elements-per-thread configuration");
    }
}

template <typename Operation>
Result benchmark_configuration(
    const Configuration& configuration,
    int multiprocessors,
    const float* a,
    const float* b,
    float* out,
    std::int64_t count,
    int iterations,
    int samples) {
    unsigned blocks = 0;
    if (configuration.strategy == Strategy::GridStride) {
        const unsigned required = divide_up(count, threads_per_block);
        blocks = configuration.value == 0
            ? required
            : std::min(required, static_cast<unsigned>(multiprocessors * configuration.value));
    } else {
        blocks = divide_up(count, static_cast<std::int64_t>(threads_per_block) * configuration.value);
    }
    blocks = std::max(1u, blocks);

    const auto launch = [&] {
        if (configuration.strategy == Strategy::GridStride) {
            grid_stride_kernel<Operation><<<blocks, threads_per_block>>>(a, b, out, count);
        } else {
            launch_fixed<Operation>(configuration.value, blocks, a, b, out, count);
        }
    };

    launch();
    check_cuda(cudaGetLastError(), "failed to launch warmup kernel");
    check_cuda(cudaDeviceSynchronize(), "warmup kernel failed");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "failed to create start event");
    check_cuda(cudaEventCreate(&stop), "failed to create stop event");

    double best_us = std::numeric_limits<double>::infinity();
    double total_us = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
        check_cuda(cudaEventRecord(start), "failed to record start event");
        for (int iteration = 0; iteration < iterations; ++iteration) launch();
        check_cuda(cudaEventRecord(stop), "failed to record stop event");
        check_cuda(cudaEventSynchronize(stop), "benchmark kernels failed");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "failed to measure kernels");
        const double per_launch_us = elapsed_ms * 1000.0 / iterations;
        best_us = std::min(best_us, per_launch_us);
        total_us += per_launch_us;
    }
    check_cuda(cudaEventDestroy(start), "failed to destroy start event");
    check_cuda(cudaEventDestroy(stop), "failed to destroy stop event");

    const auto logical_threads = static_cast<std::int64_t>(blocks) * threads_per_block;
    const double bytes = static_cast<double>(count) * 3.0 * sizeof(float);
    return {
        best_us,
        total_us / samples,
        bytes / (best_us * 1.0e3),
        blocks,
        logical_threads,
        static_cast<double>(count) / logical_threads,
    };
}

template <typename Operation>
void validate(
    const Configuration& configuration,
    int multiprocessors,
    const float* device_a,
    const float* device_b,
    float* device_out,
    const std::vector<float>& a,
    const std::vector<float>& b) {
    const auto count = static_cast<std::int64_t>(a.size());
    unsigned blocks = configuration.strategy == Strategy::GridStride
        ? (configuration.value == 0
               ? divide_up(count, threads_per_block)
               : std::min(divide_up(count, threads_per_block),
                          static_cast<unsigned>(multiprocessors * configuration.value)))
        : divide_up(count, static_cast<std::int64_t>(threads_per_block) * configuration.value);
    blocks = std::max(1u, blocks);
    if (configuration.strategy == Strategy::GridStride) {
        grid_stride_kernel<Operation><<<blocks, threads_per_block>>>(device_a, device_b, device_out, count);
    } else {
        launch_fixed<Operation>(configuration.value, blocks, device_a, device_b, device_out, count);
    }
    check_cuda(cudaGetLastError(), "failed to launch validation kernel");
    std::vector<float> actual(a.size());
    check_cuda(cudaMemcpy(actual.data(), device_out, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "failed to copy validation output");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float expected = std::is_same_v<Operation, Add> ? a[i] + b[i] : a[i] - b[i];
        if (actual[i] != expected) {
            throw std::runtime_error(configuration.label + " produced an incorrect value at element " +
                                     std::to_string(i));
        }
    }
}

void print_header() {
    std::cout << std::left << std::setw(8) << "Op" << std::setw(22) << "Configuration"
              << std::right << std::setw(9) << "Blocks" << std::setw(13) << "Threads"
              << std::setw(11) << "Elem/thr" << std::setw(12) << "Best us"
              << std::setw(12) << "Avg us" << std::setw(13) << "GB/s" << '\n';
}

template <typename Operation>
void run_operation(
    const std::vector<Configuration>& configurations,
    int multiprocessors,
    const float* device_a,
    const float* device_b,
    float* device_out,
    const std::vector<float>& a,
    const std::vector<float>& b,
    int iterations,
    int samples) {
    for (const auto& configuration : configurations) {
        validate<Operation>(configuration, multiprocessors, device_a, device_b, device_out, a, b);
        const auto result = benchmark_configuration<Operation>(
            configuration, multiprocessors, device_a, device_b, device_out,
            static_cast<std::int64_t>(a.size()), iterations, samples);
        std::cout << std::left << std::setw(8) << Operation::name << std::setw(22) << configuration.label
                  << std::right << std::setw(9) << result.blocks << std::setw(13) << result.threads
                  << std::fixed << std::setprecision(2) << std::setw(11)
                  << result.average_elements_per_thread << std::setw(12) << result.best_us
                  << std::setw(12) << result.average_us << std::setw(13)
                  << result.bandwidth_gbs << '\n';
    }
}

int parse_positive(const char* value, const char* option) {
    const long parsed = std::strtol(value, nullptr, 10);
    if (parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return static_cast<int>(parsed);
}

} // namespace

int main(int argc, char** argv) {
    int size = 1024;
    int iterations = 1000;
    int samples = 7;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option(argv[argument]);
        if ((option == "--size" || option == "--iterations" || option == "--samples") &&
            argument + 1 >= argc) {
            std::cerr << option << " requires a value\n";
            return 1;
        }
        if (option == "--size") size = parse_positive(argv[++argument], "--size");
        else if (option == "--iterations") iterations = parse_positive(argv[++argument], "--iterations");
        else if (option == "--samples") samples = parse_positive(argv[++argument], "--samples");
        else {
            std::cerr << "Usage: cuda_elementwise_benchmark [--size N] [--iterations N] [--samples N]\n";
            return 1;
        }
    }

    try {
        int device = 0;
        check_cuda(cudaGetDevice(&device), "failed to get CUDA device");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "failed to query CUDA device");

        const auto count = static_cast<std::int64_t>(size) * size;
        if (count > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            throw std::overflow_error("tensor size is too large");
        }
        std::vector<float> a(static_cast<std::size_t>(count));
        std::vector<float> b(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            a[static_cast<std::size_t>(i)] = static_cast<float>((i % 31) - 15) / 16.0f;
            b[static_cast<std::size_t>(i)] = static_cast<float>((i % 17) - 8) / 8.0f;
        }

        float* device_a = nullptr;
        float* device_b = nullptr;
        float* device_out = nullptr;
        const auto bytes = static_cast<std::size_t>(count) * sizeof(float);
        check_cuda(cudaMalloc(&device_a, bytes), "failed to allocate input A");
        check_cuda(cudaMalloc(&device_b, bytes), "failed to allocate input B");
        check_cuda(cudaMalloc(&device_out, bytes), "failed to allocate output");
        check_cuda(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice), "failed to copy input A");
        check_cuda(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice), "failed to copy input B");

        const std::vector<Configuration> configurations = {
            {"exact grid", Strategy::GridStride, 0},
            {"grid-stride 1xSM", Strategy::GridStride, 1},
            {"grid-stride 2xSM", Strategy::GridStride, 2},
            {"grid-stride 4xSM", Strategy::GridStride, 4},
            {"grid-stride 8xSM", Strategy::GridStride, 8},
            {"grid-stride 16xSM", Strategy::GridStride, 16},
            {"grid-stride 32xSM", Strategy::GridStride, 32},
            {"fixed 2 elem/thread", Strategy::FixedWork, 2},
            {"fixed 4 elem/thread", Strategy::FixedWork, 4},
            {"fixed 8 elem/thread", Strategy::FixedWork, 8},
            {"fixed 16 elem/thread", Strategy::FixedWork, 16},
            {"fixed 32 elem/thread", Strategy::FixedWork, 32},
        };

        std::cout << "CUDA elementwise kernel benchmark\n"
                  << "Device: " << properties.name << " (" << properties.multiProcessorCount << " SMs)\n"
                  << "Tensor: " << size << " x " << size << " (" << count << " Float32 elements)\n"
                  << "Timing: " << samples << " samples x " << iterations
                  << " launches; allocations, copies, and synchronization excluded\n"
                  << "Bandwidth counts two reads plus one write (12 bytes/element).\n\n";
        print_header();
        run_operation<Add>(configurations, properties.multiProcessorCount, device_a, device_b,
                           device_out, a, b, iterations, samples);
        run_operation<Sub>(configurations, properties.multiProcessorCount, device_a, device_b,
                           device_out, a, b, iterations, samples);

        check_cuda(cudaFree(device_a), "failed to free input A");
        check_cuda(cudaFree(device_b), "failed to free input B");
        check_cuda(cudaFree(device_out), "failed to free output");
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
