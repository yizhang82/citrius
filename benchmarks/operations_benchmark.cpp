#include "impl/cpu_storage.h"
#include "operations.h"
#include "tensor_factory.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cublas_cuda_device.h"
#include "impl/cuda_device.h"
#include "impl/cutlass_cuda_device.h"
#endif

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
using citrius::impl::CpuMemTensorStorageImpl;

struct Result {
    double minimum_ms;
    double average_ms;
    double total_ms;
    double gflops;
    float checksum;
};

enum class Operation {
    Add,
    Sub,
    Matmul,
};

const char* operation_name(Operation operation) {
    switch (operation) {
        case Operation::Add: return "add";
        case Operation::Sub: return "sub";
        case Operation::Matmul: return "matmul";
    }
    return "unknown";
}

double operation_count(Operation operation, std::int64_t size) {
    if (operation == Operation::Matmul) {
        return 2.0 * static_cast<double>(size) * size * size;
    }
    return static_cast<double>(size) * size;
}

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
        std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu_tensor.storage());
    const float* values = storage->data_as<float>();
    float checksum = 0.0f;
    for (std::int64_t i = 0; i < cpu_tensor.numel(); ++i) checksum += values[i];
    return checksum;
}

template <typename Operation>
Result measure(double operations, int iterations, Operation operation) {
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
    return {minimum, average, total, operations / (minimum * 1.0e6), checksum};
}

Result benchmark_cpu(Operation operation, std::int64_t size, int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    return measure(operation_count(operation, size), iterations, [&] {
        citrius::Tensor a(a_values, {size, size});
        auto b = citrius::TensorFactory::from_vector(b_values, {size, size});
        switch (operation) {
            case Operation::Add: return cpu_checksum(citrius::add(a, b));
            case Operation::Sub: return cpu_checksum(citrius::sub(a, b));
            case Operation::Matmul: return cpu_checksum(citrius::matmul(a, b));
        }
        throw std::logic_error("unknown benchmark operation");
    });
}

#ifdef CITRIUS_HAS_CUDA
template <typename CudaDevice>
Result benchmark_cuda(
    const CudaDevice& device,
    Operation operation,
    std::int64_t size,
    int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    return measure(operation_count(operation, size), iterations, [&] {
        citrius::Tensor a(a_values, {size, size}, citrius::Device::cuda());
        auto b = citrius::TensorFactory::from_vector(
            b_values, {size, size}, citrius::Device::cuda());
        switch (operation) {
            case Operation::Add: return cpu_checksum(device.add(a, b));
            case Operation::Sub: return cpu_checksum(device.sub(a, b));
            case Operation::Matmul: return cpu_checksum(device.matmul(a, b));
        }
        throw std::logic_error("unknown benchmark operation");
    });
}
#endif

void print_result(
    Operation operation,
    std::int64_t size,
    int iterations,
    const Result& result) {
    std::cout << std::left << std::setw(11) << operation_name(operation)
              << std::right << std::setw(7) << size
              << std::setw(8) << iterations << std::fixed << std::setprecision(3)
              << std::setw(13) << result.minimum_ms << std::setw(13) << result.average_ms
              << std::setw(13) << result.total_ms
              << std::setw(13) << result.gflops << std::setw(15) << result.checksum << '\n';
}

void print_section_header(const char* backend) {
    std::cout << "\n" << backend << "\n";
    std::cout << std::left << std::setw(11) << "Operation"
              << std::right << std::setw(7) << "N"
              << std::setw(8) << "Runs" << std::setw(13) << "Best ms"
              << std::setw(13) << "Avg ms" << std::setw(13) << "Total ms"
              << std::setw(13) << "GFLOP/s"
              << std::setw(15) << "Checksum" << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2 || (std::string(argv[1]) != "--cpu" &&
                      std::string(argv[1]) != "--cuda" &&
                      std::string(argv[1]) != "--all")) {
        std::cerr << "Usage: operations_benchmark --cpu|--cuda|--all\n";
        return 1;
    }

    const std::string selection(argv[1]);
    const bool use_cpu = selection == "--cpu" || selection == "--all";
    const bool use_cuda = selection == "--cuda" || selection == "--all";
#ifndef CITRIUS_HAS_CUDA
    if (use_cuda) {
        std::cerr << "CUDA support was not enabled for this build.\n";
        return 1;
    }
#endif

    std::cout << "End-to-end square Float32 operations (one warmup; input setup and output read included)\n";

    try {
        if (use_cpu) {
            print_section_header("CPU");
            for (const auto [size, iterations] :
                 std::vector<std::pair<std::int64_t, int>>{
                     {128, 50}, {256, 50}, {512, 50}, {1024, 50}}) {
                for (Operation operation : {Operation::Add, Operation::Sub, Operation::Matmul}) {
                    print_result(operation, size, iterations, benchmark_cpu(operation, size, iterations));
                }
            }
        }
#ifdef CITRIUS_HAS_CUDA
        if (use_cuda) {
            // Construct each implementation once so backend setup (notably the
            // cuBLAS handle) is not repeatedly charged to individual iterations.
            citrius::impl::CudaDeviceImpl reference;
            citrius::impl::CublasCudaDeviceImpl cublas;
            citrius::impl::CutlassCudaDeviceImpl cutlass;
            const auto run_cuda = [&](const char* name, const auto& device) {
                print_section_header(name);
                for (const auto [size, iterations] :
                     std::vector<std::pair<std::int64_t, int>>{
                         {128, 50}, {256, 50}, {512, 50}, {1024, 50}}) {
                    for (Operation operation : {Operation::Add, Operation::Sub, Operation::Matmul}) {
                        print_result(operation, size, iterations,
                                     benchmark_cuda(device, operation, size, iterations));
                    }
                }
            };
            run_cuda("CUDA (reference)", reference);
            run_cuda("CUDA (cuBLAS)", cublas);
            run_cuda("CUDA (CUTLASS)", cutlass);
        }
#endif
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
