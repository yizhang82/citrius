#include "shape_operations.h"

#include "impl/cpu_storage.h"
#include "tensor_factory.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace citrius {
namespace {

std::int64_t product(const Shape& shape, std::size_t begin, std::size_t end) {
    return std::accumulate(
        shape.begin() + static_cast<std::ptrdiff_t>(begin),
        shape.begin() + static_cast<std::ptrdiff_t>(end),
        std::int64_t{1},
        std::multiplies<>());
}

std::int64_t normalize_dim(std::int64_t dim, std::size_t ndim, bool allow_end = false) {
    const std::int64_t count = static_cast<std::int64_t>(ndim) + (allow_end ? 1 : 0);
    if (dim < 0) dim += count;
    if (dim < 0 || dim >= count) throw std::out_of_range("tensor dimension is out of range");
    return dim;
}

Shape resolved_shape(const Tensor& tensor, Shape shape) {
    std::int64_t inferred = -1;
    std::int64_t known = 1;
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (shape[index] == -1) {
            if (inferred != -1) throw std::invalid_argument("only one reshape dimension may be inferred");
            inferred = static_cast<std::int64_t>(index);
        } else {
            if (shape[index] < 0) throw std::invalid_argument("reshape dimensions must be non-negative or -1");
            known *= shape[index];
        }
    }
    if (inferred != -1) {
        if (known == 0 || tensor.numel() % known != 0) {
            throw std::invalid_argument("cannot infer reshape dimension");
        }
        shape[static_cast<std::size_t>(inferred)] = tensor.numel() / known;
    }
    if (product(shape, 0, shape.size()) != tensor.numel()) {
        throw std::invalid_argument("reshape must preserve the number of elements");
    }
    return shape;
}

Strides contiguous_strides(const Shape& shape) {
    Strides strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

const std::byte* cpu_bytes(const Tensor& tensor) {
    return static_cast<const std::byte*>(
        std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(tensor.storage())->data());
}

std::byte* cpu_bytes(Tensor& tensor) {
    return static_cast<std::byte*>(
        std::static_pointer_cast<impl::CpuMemTensorStorageImpl>(tensor.storage())->data());
}

Tensor materialize_permutation(const Tensor& tensor, const std::vector<std::int64_t>& dimensions) {
    const Tensor source = tensor.to(Device::cpu());
    Shape output_shape(dimensions.size());
    for (std::size_t index = 0; index < dimensions.size(); ++index) {
        output_shape[index] = tensor.shape()[static_cast<std::size_t>(dimensions[index])];
    }
    Tensor output = empty(output_shape, tensor.dtype(), Device::cpu());
    const Strides input_strides = contiguous_strides(tensor.shape());
    const Strides output_strides = contiguous_strides(output_shape);
    const std::size_t element_size = dtype_size(tensor.dtype());

    for (std::int64_t output_index = 0; output_index < output.numel(); ++output_index) {
        std::int64_t remainder = output_index;
        std::int64_t input_index = 0;
        for (std::size_t axis = 0; axis < output_shape.size(); ++axis) {
            const std::int64_t coordinate = remainder / output_strides[axis];
            remainder %= output_strides[axis];
            input_index += coordinate * input_strides[static_cast<std::size_t>(dimensions[axis])];
        }
        std::memcpy(
            cpu_bytes(output) + output_index * element_size,
            cpu_bytes(source) + input_index * element_size,
            element_size);
    }
    return output.to(tensor.device());
}

} // namespace

Tensor select(const Tensor& tensor, std::int64_t dim, std::int64_t index) {
    ENSURE_TENSOR_DEFINED(tensor);
    dim = normalize_dim(dim, tensor.ndim());
    const auto axis = static_cast<std::size_t>(dim);
    const auto size = tensor.shape()[axis];
    if (index < 0) index += size;
    if (index < 0 || index >= size) throw std::out_of_range("tensor index is out of range");

    Shape shape = tensor.shape();
    Strides strides = tensor.strides();
    const auto offset = tensor.storage_offset() + index * strides[axis];
    shape.erase(shape.begin() + static_cast<std::ptrdiff_t>(axis));
    strides.erase(strides.begin() + static_cast<std::ptrdiff_t>(axis));
    return Tensor(std::move(shape), std::move(strides), offset, tensor.dtype(),
                  tensor.device(), tensor.storage());
}

Tensor slice(
    const Tensor& tensor,
    std::int64_t dim,
    std::int64_t start,
    std::int64_t end,
    std::int64_t step) {
    ENSURE_TENSOR_DEFINED(tensor);
    if (step <= 0) throw std::invalid_argument("slice step must be positive");
    dim = normalize_dim(dim, tensor.ndim());
    const auto axis = static_cast<std::size_t>(dim);
    const auto size = tensor.shape()[axis];
    if (start < 0) start += size;
    if (end < 0) end += size;
    start = std::clamp<std::int64_t>(start, 0, size);
    end = std::clamp<std::int64_t>(end, 0, size);
    const auto length = end <= start ? 0 : 1 + (end - start - 1) / step;

    Shape shape = tensor.shape();
    Strides strides = tensor.strides();
    const auto offset = tensor.storage_offset() + start * strides[axis];
    shape[axis] = length;
    strides[axis] *= step;
    return Tensor(std::move(shape), std::move(strides), offset, tensor.dtype(),
                  tensor.device(), tensor.storage());
}

Tensor reshape(const Tensor& tensor, Shape shape) {
    ENSURE_TENSOR_DEFINED(tensor);
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument("reshaping a non-contiguous tensor requires contiguous()");
    }
    shape = resolved_shape(tensor, std::move(shape));
    auto strides = contiguous_strides(shape);
    return Tensor(std::move(shape), std::move(strides), tensor.storage_offset(),
                  tensor.dtype(), tensor.device(), tensor.storage());
}

Tensor view(const Tensor& tensor, Shape shape) {
    return reshape(tensor, std::move(shape));
}

Tensor flatten(const Tensor& tensor, std::int64_t start_dim, std::int64_t end_dim) {
    ENSURE_TENSOR_DEFINED(tensor);
    start_dim = normalize_dim(start_dim, tensor.ndim());
    end_dim = normalize_dim(end_dim, tensor.ndim());
    if (start_dim > end_dim) throw std::invalid_argument("flatten start_dim must not exceed end_dim");
    Shape shape;
    shape.insert(shape.end(), tensor.shape().begin(), tensor.shape().begin() + start_dim);
    shape.push_back(product(tensor.shape(), static_cast<std::size_t>(start_dim), static_cast<std::size_t>(end_dim + 1)));
    shape.insert(shape.end(), tensor.shape().begin() + end_dim + 1, tensor.shape().end());
    return reshape(tensor, std::move(shape));
}

Tensor unsqueeze(const Tensor& tensor, std::int64_t dim) {
    ENSURE_TENSOR_DEFINED(tensor);
    dim = normalize_dim(dim, tensor.ndim(), true);
    Shape shape = tensor.shape();
    shape.insert(shape.begin() + dim, 1);
    return reshape(tensor, std::move(shape));
}

Tensor squeeze(const Tensor& tensor) {
    ENSURE_TENSOR_DEFINED(tensor);
    Shape shape;
    for (const auto dimension : tensor.shape()) {
        if (dimension != 1) shape.push_back(dimension);
    }
    return reshape(tensor, std::move(shape));
}

Tensor squeeze(const Tensor& tensor, std::int64_t dim) {
    ENSURE_TENSOR_DEFINED(tensor);
    dim = normalize_dim(dim, tensor.ndim());
    if (tensor.shape()[static_cast<std::size_t>(dim)] != 1) return tensor;
    Shape shape = tensor.shape();
    shape.erase(shape.begin() + dim);
    return reshape(tensor, std::move(shape));
}

Tensor permute(const Tensor& tensor, std::vector<std::int64_t> dimensions) {
    ENSURE_TENSOR_DEFINED(tensor);
    if (dimensions.size() != tensor.ndim()) throw std::invalid_argument("permute dimensions must match tensor rank");
    std::vector<bool> seen(tensor.ndim(), false);
    for (auto& dimension : dimensions) {
        dimension = normalize_dim(dimension, tensor.ndim());
        if (seen[static_cast<std::size_t>(dimension)]) throw std::invalid_argument("permute dimensions must be unique");
        seen[static_cast<std::size_t>(dimension)] = true;
    }
    return materialize_permutation(tensor, dimensions);
}

Tensor transpose(const Tensor& tensor, std::int64_t dim0, std::int64_t dim1) {
    ENSURE_TENSOR_DEFINED(tensor);
    dim0 = normalize_dim(dim0, tensor.ndim());
    dim1 = normalize_dim(dim1, tensor.ndim());
    std::vector<std::int64_t> dimensions(tensor.ndim());
    std::iota(dimensions.begin(), dimensions.end(), std::int64_t{0});
    std::swap(dimensions[static_cast<std::size_t>(dim0)], dimensions[static_cast<std::size_t>(dim1)]);
    return permute(tensor, std::move(dimensions));
}

Tensor contiguous(const Tensor& tensor) {
    ENSURE_TENSOR_DEFINED(tensor);
    return tensor;
}

std::vector<Tensor> split(const Tensor& tensor, std::int64_t split_size, std::int64_t dim) {
    ENSURE_TENSOR_DEFINED(tensor);
    if (split_size <= 0) throw std::invalid_argument("split_size must be positive");
    dim = normalize_dim(dim, tensor.ndim());
    std::vector<Tensor> result;
    for (std::int64_t offset = 0; offset < tensor.shape()[dim]; offset += split_size) {
        const std::int64_t size = std::min(split_size, tensor.shape()[dim] - offset);
        Shape part_shape = tensor.shape();
        part_shape[dim] = size;
        Tensor part = empty(part_shape, tensor.dtype(), Device::cpu());
        const Tensor source = tensor.to(Device::cpu());
        const std::int64_t inner = product(tensor.shape(), static_cast<std::size_t>(dim + 1), tensor.ndim());
        const std::int64_t outer = product(tensor.shape(), 0, static_cast<std::size_t>(dim));
        const std::size_t element_size = dtype_size(tensor.dtype());
        for (std::int64_t outer_index = 0; outer_index < outer; ++outer_index) {
            const auto source_index = (outer_index * tensor.shape()[dim] + offset) * inner;
            const auto destination_index = outer_index * size * inner;
            std::memcpy(cpu_bytes(part) + destination_index * element_size,
                        cpu_bytes(source) + source_index * element_size,
                        static_cast<std::size_t>(size * inner) * element_size);
        }
        result.push_back(part.to(tensor.device()));
    }
    return result;
}

std::vector<Tensor> chunk(const Tensor& tensor, std::int64_t chunks, std::int64_t dim) {
    if (chunks <= 0) throw std::invalid_argument("chunks must be positive");
    dim = normalize_dim(dim, tensor.ndim());
    const std::int64_t size = (tensor.shape()[dim] + chunks - 1) / chunks;
    return split(tensor, std::max<std::int64_t>(size, 1), dim);
}

Tensor concat(const std::vector<Tensor>& tensors, std::int64_t dim) {
    if (tensors.empty()) throw std::invalid_argument("concat expects at least one tensor");
    const Tensor& first = tensors.front();
    if (!first.defined()) throw std::invalid_argument("concat tensors must be defined");
    dim = normalize_dim(dim, first.ndim());
    Shape output_shape = first.shape();
    output_shape[dim] = 0;
    for (const Tensor& tensor : tensors) {
        if (!tensor.defined() || tensor.ndim() != first.ndim() || tensor.dtype() != first.dtype() || tensor.device() != first.device()) {
            throw std::invalid_argument("concat tensors must have matching rank, dtype, and device");
        }
        for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
            if (axis != static_cast<std::size_t>(dim) && tensor.shape()[axis] != first.shape()[axis]) {
                throw std::invalid_argument("concat tensor shapes must match outside the concatenation dimension");
            }
        }
        output_shape[dim] += tensor.shape()[dim];
    }
    Tensor output = empty(output_shape, first.dtype(), Device::cpu());
    const std::int64_t inner = product(first.shape(), static_cast<std::size_t>(dim + 1), first.ndim());
    const std::int64_t outer = product(first.shape(), 0, static_cast<std::size_t>(dim));
    const std::size_t element_size = dtype_size(first.dtype());
    std::int64_t dim_offset = 0;
    for (const Tensor& tensor : tensors) {
        const Tensor source = tensor.to(Device::cpu());
        for (std::int64_t outer_index = 0; outer_index < outer; ++outer_index) {
            const auto source_index = outer_index * tensor.shape()[dim] * inner;
            const auto destination_index = (outer_index * output_shape[dim] + dim_offset) * inner;
            std::memcpy(cpu_bytes(output) + destination_index * element_size,
                        cpu_bytes(source) + source_index * element_size,
                        static_cast<std::size_t>(tensor.shape()[dim] * inner) * element_size);
        }
        dim_offset += tensor.shape()[dim];
    }
    return output.to(first.device());
}

} // namespace citrius
