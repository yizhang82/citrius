#include "impl/multi_thread_cpu_device.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace citrius::impl {
namespace {

std::size_t configured_thread_count(std::size_t requested) {
    if (requested != 0) return requested;
    if (const char* value = std::getenv("CITRIUS_CPU_THREADS")) {
        std::size_t parsed = 0;
        const char* end = value + std::char_traits<char>::length(value);
        const auto result = std::from_chars(value, end, parsed);
        if (result.ec != std::errc() || result.ptr != end || parsed == 0) {
            throw std::invalid_argument("CITRIUS_CPU_THREADS must be a positive integer");
        }
        return parsed;
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

void require_inputs(const Tensor& a, const Tensor& b) {
    if (!a.defined()) throw std::invalid_argument("left tensor is undefined");
    if (!b.defined()) throw std::invalid_argument("right tensor is undefined");
    if (a.dtype() != DType::Float32) throw std::invalid_argument("left tensor must be Float32");
    if (b.dtype() != DType::Float32) throw std::invalid_argument("right tensor must be Float32");
}

CpuMemTensorStorageImpl& output_storage(Tensor& output) {
    if (!output.defined()) throw std::invalid_argument("output tensor is undefined");
    if (output.dtype() != DType::Float32) throw std::invalid_argument("output tensor must be Float32");
    if (output.storage()->type() != TensorStorageType::CpuMemory) {
        throw std::invalid_argument("output tensor must use CpuMemory storage");
    }
    return static_cast<CpuMemTensorStorageImpl&>(*output.storage());
}

template <typename Function>
void parallel_for(
    std::int64_t begin,
    std::int64_t end,
    std::size_t maximum_threads,
    std::int64_t minimum_items_per_thread,
    Function function) {
    const auto item_count = end - begin;
    if (item_count <= 0) return;
    const auto useful_threads = static_cast<std::size_t>(
        (item_count + minimum_items_per_thread - 1) / minimum_items_per_thread);
    const auto workers = std::min(maximum_threads, useful_threads);
    if (workers <= 1) {
        function(begin, end);
        return;
    }

    const auto chunk = (item_count + static_cast<std::int64_t>(workers) - 1) /
                       static_cast<std::int64_t>(workers);
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        const auto chunk_begin = begin + static_cast<std::int64_t>(worker) * chunk;
        const auto chunk_end = std::min(end, chunk_begin + chunk);
        threads.emplace_back([=, &function] { function(chunk_begin, chunk_end); });
    }
    function(begin, std::min(end, begin + chunk));
    for (auto& thread : threads) thread.join();
}

} // namespace

MultiThreadCpuDeviceImpl::MultiThreadCpuDeviceImpl(std::size_t thread_count)
    : thread_count_(configured_thread_count(thread_count)) {}

std::size_t MultiThreadCpuDeviceImpl::thread_count() const { return thread_count_; }

void MultiThreadCpuDeviceImpl::add_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_inputs(a, b);
    if (a.shape() != b.shape() || a.shape() != output.shape()) throw std::invalid_argument("tensor shapes must match");
    const auto& as = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(a.storage()));
    const auto& bs = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(b.storage()));
    auto& os = output_storage(output);
    const float* ap = as.data_as<float>(); const float* bp = bs.data_as<float>(); float* op = os.data_as<float>();
    parallel_for(0, a.numel(), thread_count_, 65'536, [&](std::int64_t begin, std::int64_t end) {
        for (auto i = begin; i < end; ++i) op[i] = ap[i] + bp[i];
    });
}

void MultiThreadCpuDeviceImpl::sub_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_inputs(a, b);
    if (a.shape() != b.shape() || a.shape() != output.shape()) throw std::invalid_argument("tensor shapes must match");
    const auto& as = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(a.storage()));
    const auto& bs = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(b.storage()));
    auto& os = output_storage(output);
    const float* ap = as.data_as<float>(); const float* bp = bs.data_as<float>(); float* op = os.data_as<float>();
    parallel_for(0, a.numel(), thread_count_, 65'536, [&](std::int64_t begin, std::int64_t end) {
        for (auto i = begin; i < end; ++i) op[i] = ap[i] - bp[i];
    });
}

void MultiThreadCpuDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_inputs(a, b);
    if (a.ndim() != 2 || b.ndim() != 2) throw std::invalid_argument("matmul expects 2D tensors");
    if (a.shape()[1] != b.shape()[0]) throw std::invalid_argument("matmul inner dimensions must match");
    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    if (output.shape() != Shape({m, n})) throw std::invalid_argument("matmul output shape must be [m, n]");
    const auto& as = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(a.storage()));
    const auto& bs = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(b.storage()));
    auto& os = output_storage(output);
    const float* ap = as.data_as<float>(); const float* bp = bs.data_as<float>(); float* op = os.data_as<float>();
    parallel_for(0, m, thread_count_, 1, [&](std::int64_t begin, std::int64_t end) {
        for (auto row = begin; row < end; ++row) {
            for (std::int64_t col = 0; col < n; ++col) {
                float total = 0.0f;
                for (std::int64_t inner = 0; inner < k; ++inner) total += ap[row * k + inner] * bp[inner * n + col];
                op[row * n + col] = total;
            }
        }
    });
}

void MultiThreadCpuDeviceImpl::batched_matmul_out(
    const Tensor& a,
    const Tensor& b,
    Tensor& output) const {
    require_inputs(a, b);
    if (a.ndim() < 2 || b.ndim() < 2 || (a.ndim() == 2 && b.ndim() == 2)) {
        throw std::invalid_argument("batched_matmul expects at least one batched input");
    }
    const auto m = a.shape()[a.ndim() - 2];
    const auto k = a.shape().back();
    const auto n = b.shape().back();
    if (k != b.shape()[b.ndim() - 2]) {
        throw std::invalid_argument("batched_matmul inner dimensions must match");
    }
    const Shape output_batch(output.shape().begin(), output.shape().end() - 2);
    const Shape a_batch(a.shape().begin(), a.shape().end() - 2);
    const Shape b_batch(b.shape().begin(), b.shape().end() - 2);
    auto make_strides = [](const Shape& shape) {
        Strides result(shape.size(), 1);
        for (std::size_t index = shape.size(); index-- > 1;) {
            result[index - 1] = result[index] * shape[index];
        }
        return result;
    };
    const auto output_strides = make_strides(output_batch);
    const auto a_strides = make_strides(a_batch);
    const auto b_strides = make_strides(b_batch);
    auto mapped_batch = [&](std::int64_t batch, const Shape& shape, const Strides& shape_strides) {
        const std::size_t padding = output_batch.size() - shape.size();
        std::int64_t mapped = 0;
        for (std::size_t axis = padding; axis < output_batch.size(); ++axis) {
            const auto coordinate = (batch / output_strides[axis]) % output_batch[axis];
            if (shape[axis - padding] != 1) mapped += coordinate * shape_strides[axis - padding];
        }
        return mapped;
    };
    const auto& as = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(a.storage()));
    const auto& bs = static_cast<const CpuMemTensorStorageImpl&>(*ensure_storage(b.storage()));
    auto& os = output_storage(output);
    const float* ap = as.data_as<float>();
    const float* bp = bs.data_as<float>();
    float* op = os.data_as<float>();
    const std::int64_t batch_count = output.numel() / (m * n);
    parallel_for(0, batch_count * m, thread_count_, 1, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t item = begin; item < end; ++item) {
            const auto batch = item / m;
            const auto row = item % m;
            const float* am = ap + mapped_batch(batch, a_batch, a_strides) * m * k;
            const float* bm = bp + mapped_batch(batch, b_batch, b_strides) * k * n;
            float* om = op + batch * m * n;
            for (std::int64_t col = 0; col < n; ++col) {
                float total = 0.0f;
                for (std::int64_t inner = 0; inner < k; ++inner) {
                    total += am[row * k + inner] * bm[inner * n + col];
                }
                om[row * n + col] = total;
            }
        }
    });
}

} // namespace citrius::impl
