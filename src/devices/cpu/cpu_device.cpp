#include "impl/cpu_device.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace citrius::impl {

namespace {

void require_defined(const Tensor& tensor, const char* name) {
    if (!tensor.defined()) {
        throw std::invalid_argument(std::string(name) + " tensor is undefined");
    }
}

void require_float32(const Tensor& tensor, const char* name) {
    if (tensor.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(name) + " tensor must be Float32");
    }
}

void require_same_shape(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("tensor shapes must match");
    }
}

void require_2d_matmul_shapes(const Tensor& a, const Tensor& b) {
    if (a.ndim() != 2 || b.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }
}

Shape broadcast_batch_shape(const Shape& left, const Shape& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Shape output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto a = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto b = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (a != b && a != 1 && b != 1) {
            throw std::invalid_argument("batched_matmul batch dimensions are not broadcastable");
        }
        output[rank - 1 - offset] = std::max(a, b);
    }
    return output;
}

Strides strides(const Shape& shape) {
    Strides result(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        result[index - 1] = result[index] * shape[index];
    }
    return result;
}

std::int64_t batch_index(
    std::int64_t output_index,
    const Shape& output_shape,
    const Shape& input_shape) {
    const auto output_strides = strides(output_shape);
    const auto input_strides = strides(input_shape);
    const std::size_t padding = output_shape.size() - input_shape.size();
    std::int64_t input_index = 0;
    for (std::size_t axis = padding; axis < output_shape.size(); ++axis) {
        const auto coordinate =
            (output_index / output_strides[axis]) % output_shape[axis];
        if (input_shape[axis - padding] != 1) {
            input_index += coordinate * input_strides[axis - padding];
        }
    }
    return input_index;
}

Shape batched_output_shape(const Tensor& a, const Tensor& b) {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    if (a.ndim() < 2 || b.ndim() < 2 || (a.ndim() == 2 && b.ndim() == 2)) {
        throw std::invalid_argument("batched_matmul expects at least one batched input");
    }
    const auto k = a.shape().back();
    if (k != b.shape()[b.ndim() - 2]) {
        throw std::invalid_argument("batched_matmul inner dimensions must match");
    }
    const Shape a_batch(a.shape().begin(), a.shape().end() - 2);
    const Shape b_batch(b.shape().begin(), b.shape().end() - 2);
    Shape output = broadcast_batch_shape(a_batch, b_batch);
    output.push_back(a.shape()[a.ndim() - 2]);
    output.push_back(b.shape().back());
    return output;
}

} // namespace

DeviceType CpuDeviceImpl::type() const {
    return DeviceType::CPU;
}


Tensor CpuDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<CpuMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype),
        dtype);
    return Tensor(std::move(shape), dtype, Device::cpu(), std::move(storage));
}

Tensor CpuDeviceImpl::add(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    add_out(a, b, output);
    return output;
}

void CpuDeviceImpl::add_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    require_defined(output, "output");
    require_float32(output, "output");
    require_same_shape(a, output);
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t i = 0; i < a.numel(); ++i) output_data[i] = a_data[i] + b_data[i];

}

Tensor CpuDeviceImpl::sub(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    sub_out(a, b, output);
    return output;
}

void CpuDeviceImpl::sub_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);
    require_defined(output, "output");
    require_float32(output, "output");
    require_same_shape(a, output);
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t i = 0; i < a.numel(); ++i) output_data[i] = a_data[i] - b_data[i];

}

Tensor CpuDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);

    const std::int64_t m = a.shape()[0];
    const std::int64_t k = a.shape()[1];
    const std::int64_t n = b.shape()[1];

    auto output = empty({m, n}, a.dtype());
    matmul_out(a, b, output);
    return output;
}

Tensor CpuDeviceImpl::batched_matmul(const Tensor& a, const Tensor& b) const {
    auto output = empty(batched_output_shape(a, b), a.dtype());
    batched_matmul_out(a, b, output);
    return output;
}

void CpuDeviceImpl::matmul_out(const Tensor& a, const Tensor& b, Tensor& output) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);

    const std::int64_t m = a.shape()[0];
    const std::int64_t k = a.shape()[1];
    const std::int64_t n = b.shape()[1];
    require_defined(output, "output");
    require_float32(output, "output");
    if (output.shape() != Shape({m, n})) throw std::invalid_argument("matmul output shape must be [m, n]");
    const auto& a_storage = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& b_storage = require_cpu_storage(*ensure_storage(b.storage()));
    auto& output_storage = require_cpu_storage(*output.storage());

    const float* a_data = a_storage.data_as<float>();
    const float* b_data = b_storage.data_as<float>();
    float* output_data = output_storage.data_as<float>();

    for (std::int64_t row = 0; row < m; ++row) {
        for (std::int64_t col = 0; col < n; ++col) {
            float total = 0.0f;
            for (std::int64_t inner = 0; inner < k; ++inner) {
                total += a_data[row * k + inner] * b_data[inner * n + col];
            }
            output_data[row * n + col] = total;
        }
    }

}

void CpuDeviceImpl::batched_matmul_out(
    const Tensor& a,
    const Tensor& b,
    Tensor& output) const {
    const Shape expected = batched_output_shape(a, b);
    if (output.shape() != expected) {
        throw std::invalid_argument("batched_matmul output has an incorrect shape");
    }
    const auto m = a.shape()[a.ndim() - 2];
    const auto k = a.shape().back();
    const auto n = b.shape().back();
    const Shape output_batch(expected.begin(), expected.end() - 2);
    const Shape a_batch(a.shape().begin(), a.shape().end() - 2);
    const Shape b_batch(b.shape().begin(), b.shape().end() - 2);
    const auto& as = require_cpu_storage(*ensure_storage(a.storage()));
    const auto& bs = require_cpu_storage(*ensure_storage(b.storage()));
    auto& os = require_cpu_storage(*output.storage());
    const float* ap = as.data_as<float>();
    const float* bp = bs.data_as<float>();
    float* op = os.data_as<float>();
    const std::int64_t batch_count =
        std::accumulate(output_batch.begin(), output_batch.end(), std::int64_t{1}, std::multiplies<>());
    for (std::int64_t batch = 0; batch < batch_count; ++batch) {
        const float* am = ap + batch_index(batch, output_batch, a_batch) * m * k;
        const float* bm = bp + batch_index(batch, output_batch, b_batch) * k * n;
        float* om = op + batch * m * n;
        for (std::int64_t row = 0; row < m; ++row) {
            for (std::int64_t col = 0; col < n; ++col) {
                float total = 0.0f;
                for (std::int64_t inner = 0; inner < k; ++inner) {
                    total += am[row * k + inner] * bm[inner * n + col];
                }
                om[row * n + col] = total;
            }
        }
    }
}

TensorStoragePtr CpuDeviceImpl::ensure_storage(
    const TensorStoragePtr& storage,
    ConversionPolicy policy) const {
    if (!storage) {
        throw std::invalid_argument("tensor has no storage");
    }

    if (storage->type() == TensorStorageType::CpuMemory) {
        return storage;
    }

    if (policy == ConversionPolicy::CopyToDevice) {
        throw std::invalid_argument("copying non-CPU storage to CPU is not implemented");
    }

    throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
}

const CpuMemTensorStorageImpl& CpuDeviceImpl::require_cpu_storage(
    const ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CpuMemory) {
        throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
    }

    return static_cast<const CpuMemTensorStorageImpl&>(storage);
}

CpuMemTensorStorageImpl& CpuDeviceImpl::require_cpu_storage(ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::CpuMemory) {
        throw std::invalid_argument("CpuDeviceImpl requires CpuMemory storage");
    }

    return static_cast<CpuMemTensorStorageImpl&>(storage);
}

} // namespace citrius::impl
