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
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <int BlockX, int BlockY>
__global__ void naive_matmul(
    const float* a,
    const float* b,
    float* out,
    std::int64_t size) {
    const auto row = static_cast<std::int64_t>(blockIdx.y) * BlockY + threadIdx.y;
    const auto col = static_cast<std::int64_t>(blockIdx.x) * BlockX + threadIdx.x;
    if (row >= size || col >= size) return;

    float total = 0.0f;
    for (std::int64_t inner = 0; inner < size; ++inner) {
        total += a[row * size + inner] * b[inner * size + col];
    }
    out[row * size + col] = total;
}

template <int Tile>
__global__ void tiled_matmul(
    const float* a,
    const float* b,
    float* out,
    std::int64_t size) {
    __shared__ float a_tile[Tile][Tile];
    __shared__ float b_tile[Tile][Tile];

    const auto row = static_cast<std::int64_t>(blockIdx.y) * Tile + threadIdx.y;
    const auto col = static_cast<std::int64_t>(blockIdx.x) * Tile + threadIdx.x;
    float total = 0.0f;

    for (std::int64_t start = 0; start < size; start += Tile) {
        const auto a_col = start + threadIdx.x;
        const auto b_row = start + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] = row < size && a_col < size
            ? a[row * size + a_col]
            : 0.0f;
        b_tile[threadIdx.y][threadIdx.x] = b_row < size && col < size
            ? b[b_row * size + col]
            : 0.0f;
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < Tile; ++inner) {
            total += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < size && col < size) out[row * size + col] = total;
}

enum class Strategy { Naive, Tiled };

struct Configuration {
    const char* label;
    Strategy strategy;
    int x;
    int y;
};

struct Result {
    double best_ms;
    double average_ms;
    double tflops;
    std::int64_t blocks;
    int threads_per_block;
};

unsigned divide_up(std::int64_t value, int divisor) {
    return static_cast<unsigned>((value + divisor - 1) / divisor);
}

void launch(
    const Configuration& configuration,
    const float* a,
    const float* b,
    float* out,
    std::int64_t size) {
    if (configuration.strategy == Strategy::Naive) {
        const dim3 blocks(divide_up(size, configuration.x), divide_up(size, configuration.y));
        if (configuration.x == 8 && configuration.y == 8) {
            naive_matmul<8, 8><<<blocks, dim3(8, 8)>>>(a, b, out, size);
        } else if (configuration.x == 16 && configuration.y == 16) {
            naive_matmul<16, 16><<<blocks, dim3(16, 16)>>>(a, b, out, size);
        } else if (configuration.x == 32 && configuration.y == 8) {
            naive_matmul<32, 8><<<blocks, dim3(32, 8)>>>(a, b, out, size);
        } else if (configuration.x == 8 && configuration.y == 32) {
            naive_matmul<8, 32><<<blocks, dim3(8, 32)>>>(a, b, out, size);
        } else if (configuration.x == 32 && configuration.y == 16) {
            naive_matmul<32, 16><<<blocks, dim3(32, 16)>>>(a, b, out, size);
        } else if (configuration.x == 16 && configuration.y == 32) {
            naive_matmul<16, 32><<<blocks, dim3(16, 32)>>>(a, b, out, size);
        } else {
            throw std::logic_error("unsupported naive configuration");
        }
        return;
    }

    const dim3 blocks(divide_up(size, configuration.x), divide_up(size, configuration.y));
    if (configuration.x == 8) {
        tiled_matmul<8><<<blocks, dim3(8, 8)>>>(a, b, out, size);
    } else if (configuration.x == 16) {
        tiled_matmul<16><<<blocks, dim3(16, 16)>>>(a, b, out, size);
    } else if (configuration.x == 32) {
        tiled_matmul<32><<<blocks, dim3(32, 32)>>>(a, b, out, size);
    } else {
        throw std::logic_error("unsupported tiled configuration");
    }
}

void validate(
    const Configuration& configuration,
    const float* device_a,
    const float* device_b,
    float* device_out,
    std::int64_t size,
    const std::vector<float>& expected) {
    launch(configuration, device_a, device_b, device_out, size);
    check_cuda(cudaGetLastError(), "failed to launch validation kernel");
    std::vector<float> actual(expected.size());
    check_cuda(cudaMemcpy(actual.data(), device_out, actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
               "failed to copy validation output");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-4f * std::max(1.0f, std::abs(expected[i]));
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error(std::string(configuration.label) +
                                     " produced an incorrect value at element " + std::to_string(i));
        }
    }
}

Result benchmark(
    const Configuration& configuration,
    const float* device_a,
    const float* device_b,
    float* device_out,
    std::int64_t size,
    int iterations,
    int samples) {
    launch(configuration, device_a, device_b, device_out, size);
    check_cuda(cudaGetLastError(), "failed to launch warmup kernel");
    check_cuda(cudaDeviceSynchronize(), "warmup kernel failed");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "failed to create start event");
    check_cuda(cudaEventCreate(&stop), "failed to create stop event");
    double best_ms = std::numeric_limits<double>::infinity();
    double total_ms = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
        check_cuda(cudaEventRecord(start), "failed to record start event");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            launch(configuration, device_a, device_b, device_out, size);
        }
        check_cuda(cudaEventRecord(stop), "failed to record stop event");
        check_cuda(cudaEventSynchronize(stop), "benchmark kernels failed");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "failed to measure kernels");
        const double per_launch_ms = elapsed_ms / iterations;
        best_ms = std::min(best_ms, per_launch_ms);
        total_ms += per_launch_ms;
    }
    check_cuda(cudaEventDestroy(start), "failed to destroy start event");
    check_cuda(cudaEventDestroy(stop), "failed to destroy stop event");

    const double operations = 2.0 * static_cast<double>(size) * size * size;
    const auto block_columns = divide_up(size, configuration.x);
    const auto block_rows = divide_up(size, configuration.y);
    return {
        best_ms,
        total_ms / samples,
        operations / (best_ms * 1.0e9),
        static_cast<std::int64_t>(block_columns) * block_rows,
        configuration.x * configuration.y,
    };
}

int parse_positive(const char* value, const char* option) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (*value == '\0' || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer");
    }
    return static_cast<int>(parsed);
}

void validate_reference_samples(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& expected,
    std::int64_t size) {
    const std::vector<std::int64_t> coordinates = {
        0, size / 7, size / 3, size / 2, (size * 5) / 7, size - 1,
    };
    for (const auto row : coordinates) {
        for (const auto col : coordinates) {
            float cpu_value = 0.0f;
            for (std::int64_t inner = 0; inner < size; ++inner) {
                cpu_value += a[static_cast<std::size_t>(row * size + inner)] *
                             b[static_cast<std::size_t>(inner * size + col)];
            }
            const float gpu_value = expected[static_cast<std::size_t>(row * size + col)];
            const float tolerance = 1.0e-4f * std::max(1.0f, std::abs(cpu_value));
            if (std::abs(gpu_value - cpu_value) > tolerance) {
                throw std::runtime_error("reference kernel failed independent CPU validation at [" +
                                         std::to_string(row) + ", " + std::to_string(col) + "]");
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    int size = 1024;
    int iterations = 20;
    int samples = 7;
    try {
        for (int argument = 1; argument < argc; ++argument) {
            const std::string option(argv[argument]);
            if ((option == "--size" || option == "--iterations" || option == "--samples") &&
                argument + 1 >= argc) {
                throw std::invalid_argument(option + " requires a value");
            }
            if (option == "--size") size = parse_positive(argv[++argument], "--size");
            else if (option == "--iterations") iterations = parse_positive(argv[++argument], "--iterations");
            else if (option == "--samples") samples = parse_positive(argv[++argument], "--samples");
            else throw std::invalid_argument(
                "usage: cuda_matmul_benchmark [--size N] [--iterations N] [--samples N]");
        }

        int device = 0;
        check_cuda(cudaGetDevice(&device), "failed to get CUDA device");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "failed to query CUDA device");

        const auto count = static_cast<std::int64_t>(size) * size;
        if (count > static_cast<std::int64_t>(std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            throw std::overflow_error("matrix size is too large");
        }
        std::vector<float> a(static_cast<std::size_t>(count));
        std::vector<float> b(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            a[static_cast<std::size_t>(i)] = static_cast<float>((i % 13) - 6) / 8.0f;
            b[static_cast<std::size_t>(i)] = static_cast<float>((i % 11) - 5) / 8.0f;
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
            {"naive 8x8", Strategy::Naive, 8, 8},
            {"naive 16x16 (current)", Strategy::Naive, 16, 16},
            {"naive 32x8", Strategy::Naive, 32, 8},
            {"naive 8x32", Strategy::Naive, 8, 32},
            {"naive 32x16", Strategy::Naive, 32, 16},
            {"naive 16x32", Strategy::Naive, 16, 32},
            {"tiled 8x8", Strategy::Tiled, 8, 8},
            {"tiled 16x16", Strategy::Tiled, 16, 16},
            {"tiled 32x32", Strategy::Tiled, 32, 32},
        };

        // The current 16x16 kernel is the full-output reference. Its accumulation
        // order matches the candidates; a relative tolerance still permits minor
        // compiler differences while detecting indexing and edge-tile failures.
        const Configuration reference{"reference", Strategy::Naive, 16, 16};
        launch(reference, device_a, device_b, device_out, size);
        check_cuda(cudaGetLastError(), "failed to launch reference kernel");
        std::vector<float> expected(static_cast<std::size_t>(count));
        check_cuda(cudaMemcpy(expected.data(), device_out, bytes, cudaMemcpyDeviceToHost),
                   "failed to copy reference output");
        validate_reference_samples(a, b, expected, size);

        std::cout << "CUDA square Float32 matmul kernel benchmark\n"
                  << "Device: " << properties.name << " (" << properties.multiProcessorCount << " SMs)\n"
                  << "Matrices: " << size << " x " << size << '\n'
                  << "Timing: " << samples << " samples x " << iterations
                  << " launches; allocations, copies, and synchronization excluded\n\n"
                  << std::left << std::setw(24) << "Configuration" << std::right
                  << std::setw(12) << "Blocks" << std::setw(12) << "Thr/block"
                  << std::setw(13) << "Best ms" << std::setw(13) << "Avg ms"
                  << std::setw(13) << "TFLOP/s" << '\n';

        for (const auto& configuration : configurations) {
            validate(configuration, device_a, device_b, device_out, size, expected);
            const auto result = benchmark(configuration, device_a, device_b, device_out,
                                          size, iterations, samples);
            std::cout << std::left << std::setw(24) << configuration.label << std::right
                      << std::setw(12) << result.blocks << std::setw(12) << result.threads_per_block
                      << std::fixed << std::setprecision(3) << std::setw(13) << result.best_ms
                      << std::setw(13) << result.average_ms << std::setw(13) << result.tflops << '\n';
        }

        check_cuda(cudaFree(device_a), "failed to free input A");
        check_cuda(cudaFree(device_b), "failed to free input B");
        check_cuda(cudaFree(device_out), "failed to free output");
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
