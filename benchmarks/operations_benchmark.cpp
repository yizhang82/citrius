#include "impl/cpu_device.h"
#include "impl/multi_thread_cpu_device.h"
#include "impl/cpu_storage.h"
#include "operations.h"
#include "tensor_factory.h"

#ifdef CITRIUS_HAS_CUDA
#include "impl/cublas_cuda_device.h"
#include "impl/cuda_device.h"
#include "impl/cutlass_cuda_device.h"
#include <cuda_runtime.h>
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

Result benchmark_cpu(
    const citrius::impl::CpuDeviceImpl& device,
    Operation operation,
    std::int64_t size,
    int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    citrius::Tensor a(a_values, {size, size});
    auto b = citrius::from_vector(b_values, {size, size});
    auto out = device.empty({size, size}, citrius::DType::Float32);
    auto result = measure(operation_count(operation, size), iterations, [&] {
        switch (operation) {
            case Operation::Add: device.add_out(a, b, out); break;
            case Operation::Sub: device.sub_out(a, b, out); break;
            case Operation::Matmul: device.matmul_out(a, b, out); break;
        }
        return 0.0f;
    });
    result.checksum = cpu_checksum(out);
    return result;
}

#ifdef CITRIUS_HAS_CUDA
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename CudaDevice>
Result benchmark_cuda(
    const CudaDevice& device,
    Operation operation,
    std::int64_t size,
    int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    citrius::Tensor a(a_values, {size, size}, citrius::Device::cuda());
    auto b = citrius::from_vector(
        b_values, {size, size}, citrius::Device::cuda());
    auto out = device.empty({size, size}, citrius::DType::Float32);
    const auto run = [&] {
        switch (operation) {
            case Operation::Add: device.add_out(a, b, out); return;
            case Operation::Sub: device.sub_out(a, b, out); return;
            case Operation::Matmul: device.matmul_out(a, b, out); return;
        }
        throw std::logic_error("unknown benchmark operation");
    };

    run();
    cudaEvent_t start = nullptr, stop = nullptr;
    check_cuda(cudaEventCreate(&start), "failed to create CUDA start event");
    check_cuda(cudaEventCreate(&stop), "failed to create CUDA stop event");
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(iterations));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        check_cuda(cudaEventRecord(start), "failed to record CUDA start event");
        run();
        check_cuda(cudaEventRecord(stop), "failed to record CUDA stop event");
        check_cuda(cudaEventSynchronize(stop), "failed to synchronize CUDA stop event");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "failed to measure CUDA operation");
        timings.push_back(elapsed_ms);
    }
    check_cuda(cudaEventDestroy(start), "failed to destroy CUDA start event");
    check_cuda(cudaEventDestroy(stop), "failed to destroy CUDA stop event");

    const double minimum = *std::min_element(timings.begin(), timings.end());
    double total = 0.0;
    for (double timing : timings) total += timing;
    const double average = total / iterations;
    return {minimum, average, total,
            operation_count(operation, size) / (minimum * 1.0e6), cpu_checksum(out)};
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

    std::cout << "Preallocated square Float32 operations (one warmup; CUDA timed with device events)\n";

    try {
        if (use_cpu) {
            const citrius::impl::CpuDeviceImpl reference;
            const citrius::impl::MultiThreadCpuDeviceImpl multi_thread;
            const auto run_cpu = [&](const char* name, const auto& device) {
                print_section_header(name);
                for (const auto [size, iterations] :
                     std::vector<std::pair<std::int64_t, int>>{
                         {128, 50}, {256, 50}, {512, 50}, {1024, 5}}) {
                    for (Operation operation : {Operation::Add, Operation::Sub, Operation::Matmul}) {
                        print_result(operation, size, iterations,
                                     benchmark_cpu(device, operation, size, iterations));
                    }
                }
            };
            run_cpu("CPU (reference)", reference);
            const std::string label = "CPU (multi-thread, " +
                                      std::to_string(multi_thread.thread_count()) + " threads)";
            run_cpu(label.c_str(), multi_thread);
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
