#include "impl/cpu_device.h"
#include "impl/cpu_storage.h"
#include "impl/multi_thread_cpu_device.h"
#include "operations.h"
#include "reduction_operations.h"
#include "tensor_factory.h"
#include "nn/functional.h"

#ifdef CITRIUS_HAS_METAL
#include "impl/metal_device.h"
#endif

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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using citrius::impl::CpuMemTensorStorageImpl;

class TeeBuffer final : public std::streambuf {
public:
    TeeBuffer(std::streambuf* first, std::streambuf* second)
        : first_(first), second_(second) {}

protected:
    int overflow(int character) override {
        if (character == traits_type::eof()) return traits_type::not_eof(character);
        const auto value = static_cast<char>(character);
        return first_->sputc(value) == traits_type::eof() ||
                second_->sputc(value) == traits_type::eof()
            ? traits_type::eof()
            : character;
    }

    int sync() override {
        return first_->pubsync() == 0 && second_->pubsync() == 0 ? 0 : -1;
    }

private:
    std::streambuf* first_;
    std::streambuf* second_;
};

std::string html_escape(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text) {
        switch (character) {
        case '&': escaped += "&amp;"; break;
        case '<': escaped += "&lt;"; break;
        case '>': escaped += "&gt;"; break;
        case '"': escaped += "&quot;"; break;
        default: escaped += character; break;
        }
    }
    return escaped;
}

struct BenchmarkRecord {
    std::string backend;
    std::string operation;
    std::int64_t size;
    int iterations;
    double best_ms;
    double average_ms;
    double throughput;
};

std::vector<BenchmarkRecord> benchmark_records;
std::string current_backend;

void write_html_report(
    const std::string& path,
    const std::string& output,
    const std::vector<BenchmarkRecord>& records) {
    std::ofstream report(path, std::ios::binary);
    if (!report) throw std::runtime_error("failed to open HTML report: " + path);
    report << "<!doctype html>\n"
           << "<html lang=\"en\"><head><meta charset=\"utf-8\">\n"
           << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
           << "<title>Citrius Operations Benchmark</title>\n"
           << "<style>body{margin:0;background:#0b1020;color:#e5e7eb;font-family:system-ui,sans-serif}"
              "main{max-width:1500px;margin:auto;padding:2rem}h1{margin-bottom:.25rem}"
              ".subtitle{color:#9ca3af;margin-top:0}.chart{background:#111827;border:1px solid #374151;"
              "border-radius:.75rem;padding:1rem;margin:1.5rem 0;overflow-x:auto}"
              "svg{display:block;min-width:100%}.grid{stroke:#374151;stroke-width:1}.axis{fill:#9ca3af;"
              "font-size:12px}.label{fill:#d1d5db;font-size:12px}.title{fill:#f9fafb;font-size:18px;"
              "font-weight:600}.legend{fill:#d1d5db;font-size:13px}details{margin-top:2rem}"
              "summary{cursor:pointer;font-weight:600}pre{overflow:auto;background:#030712;"
              "border:1px solid #374151;border-radius:.5rem;padding:1.25rem;line-height:1.45;"
              "color:#d1fae5}</style></head><body><main>\n"
           << "<h1>Citrius Operations Benchmark</h1>"
           << "<p class=\"subtitle\">Best latency by operation and backend; logarithmic milliseconds."
              " Lower is better.</p>\n";

    const std::vector<std::string> colors{
        "#60a5fa", "#34d399", "#f59e0b", "#f472b6", "#a78bfa", "#fb7185"};
    std::set<std::int64_t> sizes;
    for (const auto& record : records) sizes.insert(record.size);
    for (const std::int64_t size : sizes) {
        std::vector<std::string> operations;
        std::vector<std::string> backends;
        double minimum = std::numeric_limits<double>::infinity();
        double maximum = 0.0;
        for (const auto& record : records) {
            if (record.size != size) continue;
            if (std::find(operations.begin(), operations.end(), record.operation) == operations.end())
                operations.push_back(record.operation);
            if (std::find(backends.begin(), backends.end(), record.backend) == backends.end())
                backends.push_back(record.backend);
            if (record.best_ms > 0.0) {
                minimum = std::min(minimum, record.best_ms);
                maximum = std::max(maximum, record.best_ms);
            }
        }
        if (operations.empty() || !std::isfinite(minimum)) continue;
        const int width = std::max(1100, static_cast<int>(operations.size()) * 105);
        constexpr int height = 620;
        constexpr int left = 85;
        constexpr int right = 30;
        constexpr int top = 75;
        constexpr int bottom = 150;
        const double plot_width = width - left - right;
        const double plot_height = height - top - bottom;
        double log_min = std::floor(std::log10(minimum));
        double log_max = std::ceil(std::log10(maximum));
        if (log_max <= log_min) log_max = log_min + 1.0;
        const auto y_for = [&](double value) {
            return top + (log_max - std::log10(value)) / (log_max - log_min) * plot_height;
        };

        report << "<section class=\"chart\"><svg viewBox=\"0 0 " << width << ' ' << height
               << "\" role=\"img\" aria-label=\"Performance comparison for size " << size
               << "\">\n<text class=\"title\" x=\"" << left << "\" y=\"30\">N = "
               << size << "</text>\n";
        for (int tick = 0; tick <= 5; ++tick) {
            const double exponent = log_min + (log_max - log_min) * tick / 5.0;
            const double value = std::pow(10.0, exponent);
            const double y = y_for(value);
            report << "<line class=\"grid\" x1=\"" << left << "\" x2=\""
                   << width - right << "\" y1=\"" << y << "\" y2=\"" << y << "\"/>"
                   << "<text class=\"axis\" text-anchor=\"end\" x=\"" << left - 8
                   << "\" y=\"" << y + 4 << "\">" << std::setprecision(3) << value
                   << " ms</text>\n";
        }
        const double group_width = plot_width / operations.size();
        const double bar_width = std::max(3.0, group_width * 0.72 / backends.size());
        for (std::size_t operation_index = 0; operation_index < operations.size(); ++operation_index) {
            const double group_start = left + operation_index * group_width + group_width * 0.14;
            for (std::size_t backend_index = 0; backend_index < backends.size(); ++backend_index) {
                const auto found = std::find_if(records.begin(), records.end(), [&](const auto& record) {
                    return record.size == size && record.operation == operations[operation_index] &&
                           record.backend == backends[backend_index];
                });
                if (found == records.end() || found->best_ms <= 0.0) continue;
                const double y = y_for(found->best_ms);
                const double x = group_start + backend_index * bar_width;
                report << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\""
                       << std::max(2.0, bar_width - 1.5) << "\" height=\""
                       << top + plot_height - y << "\" fill=\""
                       << colors[backend_index % colors.size()] << "\"><title>"
                       << html_escape(backends[backend_index]) << " · "
                       << html_escape(operations[operation_index]) << ": "
                       << std::setprecision(6) << found->best_ms << " ms</title></rect>\n";
            }
            const double label_x = left + (operation_index + 0.5) * group_width;
            report << "<text class=\"label\" text-anchor=\"end\" transform=\"translate("
                   << label_x << ',' << height - bottom + 18 << ") rotate(-45)\">"
                   << html_escape(operations[operation_index]) << "</text>\n";
        }
        double legend_x = left;
        for (std::size_t index = 0; index < backends.size(); ++index) {
            report << "<rect x=\"" << legend_x << "\" y=\"48\" width=\"12\" height=\"12\" fill=\""
                   << colors[index % colors.size()] << "\"/><text class=\"legend\" x=\""
                   << legend_x + 18 << "\" y=\"59\">" << html_escape(backends[index])
                   << "</text>\n";
            legend_x += 28 + backends[index].size() * 8;
        }
        report << "</svg></section>\n";
    }
    report << "<details><summary>Raw benchmark output</summary><pre>"
           << html_escape(output)
           << "</pre></details></main></body></html>\n";
    if (!report) throw std::runtime_error("failed to write HTML report: " + path);
}

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
    BroadcastAdd,
    BroadcastMul,
    ScalarMul,
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceVariance,
    Exp,
    Sqrt,
    Power,
    MaskedFill,
    Softmax,
    LayerNorm,
    Matmul,
    BatchedMatmul,
};

constexpr std::int64_t batched_matmul_batch_size = 8;

const char* operation_name(Operation operation) {
    switch (operation) {
    case Operation::Add:
        return "add";
    case Operation::Sub:
        return "sub";
    case Operation::BroadcastAdd:
        return "broadcast-add";
    case Operation::BroadcastMul:
        return "broadcast-mul";
    case Operation::ScalarMul:
        return "scalar-mul";
    case Operation::ReduceSum:
        return "reduce-sum";
    case Operation::ReduceMean:
        return "reduce-mean";
    case Operation::ReduceMax:
        return "reduce-max";
    case Operation::ReduceVariance:
        return "reduce-var";
    case Operation::Exp:
        return "exp";
    case Operation::Sqrt:
        return "sqrt";
    case Operation::Power:
        return "pow";
    case Operation::MaskedFill:
        return "masked-fill";
    case Operation::Softmax:
        return "softmax";
    case Operation::LayerNorm:
        return "layer-norm";
    case Operation::Matmul:
        return "matmul";
    case Operation::BatchedMatmul:
        return "batch-matmul";
    }
    return "unknown";
}

double operation_count(Operation operation, std::int64_t size) {
    if (operation == Operation::Matmul) {
        return 2.0 * static_cast<double>(size) * size * size;
    }
    if (operation == Operation::BatchedMatmul) {
        return batched_matmul_batch_size * 2.0 * static_cast<double>(size) * size * size;
    }
    return static_cast<double>(size) * size;
}

int benchmark_iterations(Operation operation, std::int64_t size, int default_iterations) {
    if (operation != Operation::BatchedMatmul)
        return default_iterations;
    switch (size) {
    case 128:
        return 10;
    case 256:
        return 5;
    case 512:
        return 1;
    case 1024:
        return 1;
    default:
        return 0;
    }
}

std::vector<float> input_values(std::int64_t count, int seed) {
    std::vector<float> values(static_cast<std::size_t>(count));
    for (std::int64_t i = 0; i < count; ++i) {
        values[static_cast<std::size_t>(i)] = static_cast<float>(((i + seed) % 17) - 8) / 8.0f;
    }
    return values;
}

float cpu_checksum(const citrius::Tensor& tensor) {
    const auto cpu_tensor = tensor.to(citrius::Device::cpu());
    const auto storage = std::static_pointer_cast<CpuMemTensorStorageImpl>(cpu_tensor.storage());
    const float* values = storage->data_as<float>();
    float checksum = 0.0f;
    for (std::int64_t i = 0; i < cpu_tensor.numel(); ++i)
        checksum += values[i];
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
    for (double timing : timings)
        total += timing;
    const double average = total / iterations;
    return {minimum, average, total, operations / (minimum * 1.0e6), checksum};
}

Result benchmark_cpu(const citrius::impl::CpuDeviceImpl& device, Operation operation,
                     std::int64_t size, int iterations) {
    if (operation == Operation::BatchedMatmul) {
        const auto input = input_values(batched_matmul_batch_size * size * size, 3);
        const auto other = input_values(batched_matmul_batch_size * size * size, 7);
        const citrius::Tensor a(input, {batched_matmul_batch_size, size, size});
        const citrius::Tensor b(other, {batched_matmul_batch_size, size, size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.batched_matmul(a, b);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::BroadcastAdd || operation == Operation::BroadcastMul) {
        const citrius::Tensor a(input_values(size * size, 3), {size, size});
        const citrius::Tensor b(input_values(size, 7), {size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = operation == Operation::BroadcastAdd
                ? citrius::add(a, b)
                : citrius::mul(a, b);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::ScalarMul) {
        const citrius::Tensor input(input_values(size * size, 3), {size, size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = citrius::mul(input, 1.5f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::ReduceSum || operation == Operation::ReduceMean ||
        operation == Operation::ReduceMax || operation == Operation::ReduceVariance) {
        const citrius::Tensor input(input_values(size * size, 3), {size, size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            switch (operation) {
            case Operation::ReduceSum:
                output = citrius::sum(input, 1);
                break;
            case Operation::ReduceMean:
                output = citrius::mean(input, 1);
                break;
            case Operation::ReduceMax:
                output = citrius::max(input, 1);
                break;
            case Operation::ReduceVariance:
                output = citrius::variance(input, 1);
                break;
            default:
                break;
            }
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::Exp || operation == Operation::Sqrt ||
        operation == Operation::Power) {
        auto values = input_values(size * size, 3);
        if (operation == Operation::Sqrt) {
            for (float& value : values) value = std::abs(value) + 0.01f;
        }
        const citrius::Tensor input(values, {size, size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = operation == Operation::Exp
                ? citrius::exp(input)
                : operation == Operation::Sqrt
                    ? citrius::sqrt(input)
                    : citrius::pow(input, 2.0f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::MaskedFill) {
        const citrius::Tensor input(input_values(size * size, 3), {size, size});
        std::vector<bool> mask_values(static_cast<std::size_t>(size));
        for (std::int64_t index = 0; index < size; ++index)
            mask_values[static_cast<std::size_t>(index)] = index % 3 == 0;
        const citrius::Tensor mask = citrius::from_vector(mask_values, {size});
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = citrius::masked_fill(input, mask, -100.0f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::Softmax || operation == Operation::LayerNorm) {
        const citrius::Tensor input(input_values(size * size, 3), {size, size});
        const citrius::Tensor weight(
            std::vector<float>(static_cast<std::size_t>(size), 1.0f));
        const citrius::Tensor bias(
            std::vector<float>(static_cast<std::size_t>(size), 0.0f));
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = operation == Operation::Softmax
                ? citrius::nn::functional::softmax(input, -1)
                : citrius::nn::functional::layer_norm(
                      input, {size}, weight, bias, 1e-5f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    citrius::Tensor a(a_values, {size, size});
    auto b = citrius::from_vector(b_values, {size, size});
    auto out = device.empty({size, size}, citrius::DType::Float32);
    auto result = measure(operation_count(operation, size), iterations, [&] {
        switch (operation) {
        case Operation::Add:
            device.add_out(a, b, out);
            break;
        case Operation::Sub:
            device.sub_out(a, b, out);
            break;
        case Operation::Matmul:
            device.matmul_out(a, b, out);
            break;
        case Operation::BroadcastAdd:
        case Operation::BroadcastMul:
        case Operation::ScalarMul:
        case Operation::ReduceSum:
        case Operation::ReduceMean:
        case Operation::ReduceMax:
        case Operation::ReduceVariance:
        case Operation::Exp:
        case Operation::Sqrt:
        case Operation::Power:
        case Operation::MaskedFill:
        case Operation::Softmax:
        case Operation::LayerNorm:
        case Operation::BatchedMatmul:
            break;
        }
        return 0.0f;
    });
    result.checksum = cpu_checksum(out);
    return result;
}

#ifdef CITRIUS_HAS_METAL
Result benchmark_metal(const citrius::impl::MetalDeviceImpl& device, Operation operation,
                       std::int64_t size, int iterations) {
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    const citrius::Tensor a(a_values, {size, size}, citrius::Device::metal());
    const citrius::Tensor b(b_values, {size, size}, citrius::Device::metal());
    citrius::Tensor output;
    auto result = measure(operation_count(operation, size), iterations, [&] {
        switch (operation) {
        case Operation::Add:
            output = device.add(a, b);
            break;
        case Operation::Sub:
            output = device.sub(a, b);
            break;
        case Operation::Matmul:
            output = device.matmul(a, b);
            break;
        default:
            throw std::logic_error("unsupported Metal benchmark operation");
        }
        return 0.0f;
    });
    result.checksum = cpu_checksum(output);
    return result;
}
#endif

#ifdef CITRIUS_HAS_CUDA
void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename CudaDevice>
Result benchmark_cuda(const CudaDevice& device, Operation operation, std::int64_t size,
                      int iterations) {
    if (operation == Operation::BatchedMatmul) {
        const auto count = batched_matmul_batch_size * size * size;
        const auto a_values = input_values(count, 3);
        const auto b_values = input_values(count, 7);
        const auto shape = citrius::Shape{batched_matmul_batch_size, size, size};
        const auto a = citrius::from_vector(a_values, shape, citrius::Device::cuda());
        const auto b = citrius::from_vector(b_values, shape, citrius::Device::cuda());
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.batched_matmul(a, b);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::BroadcastAdd || operation == Operation::BroadcastMul) {
        const auto a_values = input_values(size * size, 3);
        const auto b_values = input_values(size, 7);
        const citrius::Tensor a(a_values, {size, size}, citrius::Device::cuda());
        const citrius::Tensor b(b_values, {size}, citrius::Device::cuda());
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.broadcast_elementwise(
                a, b,
                operation == Operation::BroadcastAdd
                    ? citrius::impl::CudaElementwiseOperation::Add
                    : citrius::impl::CudaElementwiseOperation::Multiply);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::ScalarMul) {
        const auto values = input_values(size * size, 3);
        const citrius::Tensor input(values, {size, size}, citrius::Device::cuda());
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.scalar_elementwise(
                input, 1.5f, citrius::impl::CudaElementwiseOperation::Multiply);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::ReduceSum || operation == Operation::ReduceMean ||
        operation == Operation::ReduceMax || operation == Operation::ReduceVariance) {
        const auto values = input_values(size * size, 3);
        const citrius::Tensor input(values, {size, size}, citrius::Device::cuda());
        citrius::impl::CudaReductionOperation reduction;
        switch (operation) {
        case Operation::ReduceSum:
            reduction = citrius::impl::CudaReductionOperation::Sum;
            break;
        case Operation::ReduceMean:
            reduction = citrius::impl::CudaReductionOperation::Mean;
            break;
        case Operation::ReduceMax:
            reduction = citrius::impl::CudaReductionOperation::Maximum;
            break;
        case Operation::ReduceVariance:
            reduction = citrius::impl::CudaReductionOperation::Variance;
            break;
        default:
            throw std::logic_error("unknown reduction benchmark operation");
        }
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.reduce(input, {1}, false, reduction);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::Exp || operation == Operation::Sqrt ||
        operation == Operation::Power) {
        auto values = input_values(size * size, 3);
        if (operation == Operation::Sqrt) {
            for (float& value : values) value = std::abs(value) + 0.01f;
        }
        const citrius::Tensor input(values, {size, size}, citrius::Device::cuda());
        const auto unary_operation = operation == Operation::Exp
            ? citrius::impl::CudaUnaryOperation::Exp
            : operation == Operation::Sqrt
                ? citrius::impl::CudaUnaryOperation::Sqrt
                : citrius::impl::CudaUnaryOperation::Power;
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.unary(input, unary_operation, 2.0f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::MaskedFill) {
        const auto values = input_values(size * size, 3);
        std::vector<bool> mask_values(static_cast<std::size_t>(size));
        for (std::int64_t index = 0; index < size; ++index)
            mask_values[static_cast<std::size_t>(index)] = index % 3 == 0;
        const citrius::Tensor input(values, {size, size}, citrius::Device::cuda());
        const citrius::Tensor mask =
            citrius::from_vector(mask_values, {size}, citrius::Device::cuda());
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = device.masked_fill(input, mask, -100.0f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    if (operation == Operation::Softmax || operation == Operation::LayerNorm) {
        const auto values = input_values(size * size, 3);
        const citrius::Tensor input(values, {size, size}, citrius::Device::cuda());
        citrius::Tensor weight;
        citrius::Tensor bias;
        if (operation == Operation::LayerNorm) {
            weight = citrius::from_vector(
                std::vector<float>(static_cast<std::size_t>(size), 1.0f),
                {size}, citrius::Device::cuda());
            bias = citrius::from_vector(
                std::vector<float>(static_cast<std::size_t>(size), 0.0f),
                {size}, citrius::Device::cuda());
        }
        citrius::Tensor output;
        auto result = measure(operation_count(operation, size), iterations, [&] {
            output = operation == Operation::Softmax
                ? citrius::nn::functional::softmax(input, -1)
                : citrius::nn::functional::layer_norm(
                      input, {size}, weight, bias, 1e-5f);
            return 0.0f;
        });
        result.checksum = cpu_checksum(output);
        return result;
    }
    const auto a_values = input_values(size * size, 3);
    const auto b_values = input_values(size * size, 7);
    citrius::Tensor a(a_values, {size, size}, citrius::Device::cuda());
    auto b = citrius::from_vector(b_values, {size, size}, citrius::Device::cuda());
    auto out = device.empty({size, size}, citrius::DType::Float32);
    const auto run = [&] {
        switch (operation) {
        case Operation::Add:
            device.add_out(a, b, out);
            return;
        case Operation::Sub:
            device.sub_out(a, b, out);
            return;
        case Operation::Matmul:
            device.matmul_out(a, b, out);
            return;
        case Operation::BroadcastAdd:
        case Operation::BroadcastMul:
        case Operation::ScalarMul:
        case Operation::ReduceSum:
        case Operation::ReduceMean:
        case Operation::ReduceMax:
        case Operation::ReduceVariance:
        case Operation::Exp:
        case Operation::Sqrt:
        case Operation::Power:
        case Operation::MaskedFill:
        case Operation::Softmax:
        case Operation::LayerNorm:
        case Operation::BatchedMatmul:
            return;
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
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop),
                   "failed to measure CUDA operation");
        timings.push_back(elapsed_ms);
    }
    check_cuda(cudaEventDestroy(start), "failed to destroy CUDA start event");
    check_cuda(cudaEventDestroy(stop), "failed to destroy CUDA stop event");

    const double minimum = *std::min_element(timings.begin(), timings.end());
    double total = 0.0;
    for (double timing : timings)
        total += timing;
    const double average = total / iterations;
    return {minimum, average, total, operation_count(operation, size) / (minimum * 1.0e6),
            cpu_checksum(out)};
}
#endif

void print_result(Operation operation, std::int64_t size, int iterations, const Result& result) {
    benchmark_records.push_back(
        {current_backend, operation_name(operation), size, iterations,
         result.minimum_ms, result.average_ms, result.gflops});
    std::cout << std::left << std::setw(14) << operation_name(operation) << std::right
              << std::setw(7) << size << std::setw(8) << iterations << std::fixed
              << std::setprecision(3) << std::setw(13) << result.minimum_ms << std::setw(13)
              << result.average_ms << std::setw(13) << result.total_ms << std::setw(13)
              << result.gflops << std::setw(15) << result.checksum << '\n';
}

void print_section_header(const char* backend) {
    current_backend = backend;
    std::cout << "\n" << backend << "\n";
    std::cout << std::left << std::setw(14) << "Operation" << std::right << std::setw(7) << "N"
              << std::setw(8) << "Runs" << std::setw(13) << "Best ms" << std::setw(13) << "Avg ms"
              << std::setw(13) << "Total ms" << std::setw(13) << "GFLOP/s" << std::setw(15)
              << "Checksum" << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || (std::string(argv[1]) != "--cpu" && std::string(argv[1]) != "--metal" &&
                     std::string(argv[1]) != "--cuda" && std::string(argv[1]) != "--all")) {
        std::cerr << "Usage: operations_benchmark --cpu|--metal|--cuda|--all "
                     "[--html [FILE]]\n";
        return 1;
    }

    const std::string selection(argv[1]);
    std::string html_path;
    for (int argument = 2; argument < argc; ++argument) {
        if (std::string(argv[argument]) != "--html" || !html_path.empty()) {
            std::cerr << "Usage: operations_benchmark --cpu|--metal|--cuda|--all "
                         "[--html [FILE]]\n";
            return 1;
        }
        html_path = "operations_benchmark.html";
        if (argument + 1 < argc && std::string(argv[argument + 1]).rfind("--", 0) != 0)
            html_path = argv[++argument];
    }
    const bool use_cpu = selection == "--cpu" || selection == "--all";
    bool use_metal = selection == "--metal";
#ifdef CITRIUS_HAS_METAL
    use_metal = use_metal || selection == "--all";
#endif
    const bool use_cuda = selection == "--cuda" || selection == "--all";
#ifndef CITRIUS_HAS_METAL
    if (use_metal) {
        std::cerr << "Metal support was not enabled for this build.\n";
        return 1;
    }
#endif
#ifndef CITRIUS_HAS_CUDA
    if (use_cuda) {
        std::cerr << "CUDA support was not enabled for this build.\n";
        return 1;
    }
#endif

    std::ostringstream captured_output;
    std::streambuf* original_output = std::cout.rdbuf();
    TeeBuffer tee_output(original_output, captured_output.rdbuf());
    if (!html_path.empty()) std::cout.rdbuf(&tee_output);

    std::cout << "Square Float32 operations\n"
              << "Single-thread CPU reference: one run; batched N=1024 skipped\n"
              << "Multi-thread CPU batched runs={128:10, 256:5, 512:1, 1024:1}\n"
              << "CPU functional workloads use current top-level paths; only operations with "
                 "multi-thread dispatch run in parallel\n"
              << "Metal: 50 runs per supported workload\n"
              << "CUDA: 50 runs per workload, including batched matmul\n";

    try {
        if (use_cpu) {
            const citrius::impl::CpuDeviceImpl reference;
            const citrius::impl::MultiThreadCpuDeviceImpl multi_thread;
            const auto run_cpu = [&](const char* name, const auto& device, bool reference_only) {
                print_section_header(name);
                for (const auto [size, iterations] : std::vector<std::pair<std::int64_t, int>>{
                         {128, 50}, {256, 50}, {512, 50}, {1024, 5}}) {
                    std::vector<Operation> operations{Operation::Add, Operation::Sub};
                    if (!reference_only) {
                        operations.insert(
                            operations.end(),
                            {Operation::BroadcastAdd, Operation::BroadcastMul,
                             Operation::ScalarMul, Operation::ReduceSum,
                             Operation::ReduceMean, Operation::ReduceMax,
                             Operation::ReduceVariance, Operation::Exp,
                             Operation::Sqrt, Operation::Power,
                             Operation::MaskedFill, Operation::Softmax,
                             Operation::LayerNorm});
                    }
                    operations.insert(
                        operations.end(), {Operation::Matmul, Operation::BatchedMatmul});
                    for (Operation operation : operations) {
                        const int runs =
                            reference_only ? 1 : benchmark_iterations(operation, size, iterations);
                        if (runs == 0)
                            continue;
                        if (reference_only && operation == Operation::BatchedMatmul && size == 1024)
                            continue;
                        print_result(operation, size, runs,
                                     benchmark_cpu(device, operation, size, runs));
                    }
                }
            };
            run_cpu("CPU (reference)", reference, true);
            const std::string label =
                "CPU (multi-thread, " + std::to_string(multi_thread.thread_count()) + " threads)";
            run_cpu(label.c_str(), multi_thread, false);
        }
#ifdef CITRIUS_HAS_METAL
        if (use_metal) {
            const citrius::impl::MetalDeviceImpl metal;
            print_section_header("Metal");
            for (const auto [size, iterations] : std::vector<std::pair<std::int64_t, int>>{
                     {128, 50}, {256, 50}, {512, 50}, {1024, 50}}) {
                for (Operation operation :
                     {Operation::Add, Operation::Sub, Operation::Matmul}) {
                    print_result(operation, size, iterations,
                                 benchmark_metal(metal, operation, size, iterations));
                }
            }
        }
#endif
#ifdef CITRIUS_HAS_CUDA
        if (use_cuda) {
            // Construct each implementation once so backend setup (notably the
            // cuBLAS handle) is not repeatedly charged to individual iterations.
            citrius::impl::CudaDeviceImpl reference;
            citrius::impl::CublasCudaDeviceImpl cublas;
            citrius::impl::CutlassCudaDeviceImpl cutlass;
            const auto run_cuda = [&](const char* name, const auto& device, bool elementwise) {
                print_section_header(name);
                for (const auto [size, iterations] : std::vector<std::pair<std::int64_t, int>>{
                         {128, 50}, {256, 50}, {512, 50}, {1024, 50}}) {
                    std::vector<Operation> operations{Operation::Add, Operation::Sub};
                    if (elementwise) {
                        operations.insert(
                            operations.end(),
                            {Operation::BroadcastAdd, Operation::BroadcastMul,
                             Operation::ScalarMul, Operation::ReduceSum,
                             Operation::ReduceMean, Operation::ReduceMax,
                             Operation::ReduceVariance, Operation::Exp,
                             Operation::Sqrt, Operation::Power,
                             Operation::MaskedFill, Operation::Softmax,
                             Operation::LayerNorm});
                    }
                    operations.insert(
                        operations.end(), {Operation::Matmul, Operation::BatchedMatmul});
                    for (Operation operation : operations) {
                        print_result(operation, size, iterations,
                                     benchmark_cuda(device, operation, size, iterations));
                    }
                }
            };
            // Elementwise implementations are inherited unchanged by the
            // cuBLAS and CUTLASS variants, so report them once.
            run_cuda("CUDA (reference)", reference, true);
            run_cuda("CUDA (cuBLAS)", cublas, false);
            run_cuda("CUDA (CUTLASS)", cutlass, false);
        }
#endif
    } catch (const std::exception& error) {
        if (!html_path.empty()) std::cout.rdbuf(original_output);
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }

    if (!html_path.empty()) {
        std::cout.rdbuf(original_output);
        try {
            write_html_report(html_path, captured_output.str(), benchmark_records);
            std::cout << "HTML report: " << html_path << '\n';
        } catch (const std::exception& error) {
            std::cerr << "Benchmark failed: " << error.what() << '\n';
            return 1;
        }
    }
}
