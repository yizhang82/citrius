#include "metal_device.h"

#include "cpu_storage.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <climits>
#include <stdexcept>
#include <string>
#include <utility>

namespace citrius {

namespace {

const char* kMetalKernels = R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        out[id] = a[id] + b[id];
    }
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) {
        out[id] = a[id] - b[id];
    }
}

kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]) {
    const uint row = id.y;
    const uint col = id.x;
    const uint m = dims.x;
    const uint k = dims.y;
    const uint n = dims.z;

    if (row >= m || col >= n) {
        return;
    }

    float total = 0.0f;
    for (uint inner = 0; inner < k; ++inner) {
        total += a[row * k + inner] * b[inner * n + col];
    }
    out[row * n + col] = total;
}
)";

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

id<MTLBuffer> buffer_from_storage(const MetalMemTensorStorageImpl& storage) {
    return (__bridge id<MTLBuffer>)storage.handle().ptr;
}

} // namespace

class MetalDeviceImpl::Impl {
public:
    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            throw std::runtime_error("Metal device is not available");
        }

        queue = [device newCommandQueue];
        if (queue == nil) {
            throw std::runtime_error("failed to create Metal command queue");
        }

        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kMetalKernels];
        library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            throw std::runtime_error(
                std::string("failed to compile Metal kernels: ") +
                (error ? [[error localizedDescription] UTF8String] : "unknown error"));
        }

        add_pipeline = pipeline("add_f32");
        sub_pipeline = pipeline("sub_f32");
        matmul_pipeline = pipeline("matmul_f32");
    }

    id<MTLComputePipelineState> pipeline(const char* name) {
        NSString* function_name = [NSString stringWithUTF8String:name];
        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            throw std::runtime_error(std::string("missing Metal function: ") + name);
        }

        NSError* error = nil;
        id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
        if (state == nil) {
            throw std::runtime_error(
                std::string("failed to create Metal pipeline: ") +
                (error ? [[error localizedDescription] UTF8String] : "unknown error"));
        }

        return state;
    }

    void run_elementwise(
        id<MTLComputePipelineState> pipeline,
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> out,
        std::int64_t count) const {
        // The Metal kernels use a 32-bit element count, so reject tensors that
        // cannot be addressed by the current simple launch path.
        if (count < 0 || count > static_cast<std::int64_t>(UINT32_MAX)) {
            throw std::invalid_argument("Metal elementwise count is too large");
        }

        uint32_t kernel_count = static_cast<uint32_t>(count);

        // A command buffer records one batch of GPU work; the compute encoder
        // records the specific kernel launch and its bound inputs.
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        // These binding slots must match the [[buffer(N)]] annotations in the
        // add_f32/sub_f32 Metal kernels.
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&kernel_count length:sizeof(kernel_count) atIndex:3];

        // Launch one logical GPU thread per tensor element. Metal partitions
        // the grid into threadgroups of up to 256 threads for this simple path.
        const NSUInteger threads = std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
        MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(kernel_count), 1, 1);
        MTLSize group_size = MTLSizeMake(threads, 1, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        // Keep execution synchronous for now so returned tensors are ready to
        // read immediately in tests and early user code.
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }

    void run_matmul(
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> out,
        std::int64_t m,
        std::int64_t k,
        std::int64_t n) const {
        // The first matmul kernel uses uint dimensions in Metal. This keeps
        // validation explicit until larger shape handling is designed.
        if (m < 0 || k < 0 || n < 0 || m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX) {
            throw std::invalid_argument("Metal matmul dimensions are too large");
        }

        // dims is interpreted by the Metal kernel as {m, k, n}, where the
        // operation is [m x k] multiplied by [k x n] to produce [m x n].
        std::array<uint32_t, 3> dims = {
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(k),
            static_cast<uint32_t>(n),
        };

        // Record a single compute dispatch for the whole output matrix.
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];

        // Binding slots mirror the matmul_f32 kernel signature.
        [encoder setComputePipelineState:matmul_pipeline];
        [encoder setBuffer:a offset:0 atIndex:0];
        [encoder setBuffer:b offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:dims.data() length:sizeof(uint32_t) * dims.size() atIndex:3];

        // The grid is shaped like the output matrix. Each thread computes one
        // output cell, with id.x selecting the column and id.y selecting the row.
        MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(n), static_cast<NSUInteger>(m), 1);
        MTLSize group_size = MTLSizeMake(8, 8, 1);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:group_size];
        [encoder endEncoding];

        // Synchronous completion keeps the public operation semantics eager.
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }

    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> sub_pipeline = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
};

MetalDeviceImpl::MetalDeviceImpl()
    : impl_(std::make_unique<Impl>()) {}

MetalDeviceImpl::~MetalDeviceImpl() = default;

DeviceType MetalDeviceImpl::type() const {
    return DeviceType::Metal;
}

Tensor MetalDeviceImpl::empty(Shape shape, DType dtype) const {
    const Tensor metadata(shape, dtype, Device::cpu());
    auto storage = std::make_shared<MetalMemTensorStorageImpl>(
        static_cast<std::size_t>(metadata.numel()) * dtype_size(dtype),
        dtype);
    return Tensor(std::move(shape), dtype, Device::metal(), std::move(storage));
}

Tensor MetalDeviceImpl::add(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    auto a_storage_ptr = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto b_storage_ptr = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto& a_storage = require_metal_storage(*a_storage_ptr);
    const auto& b_storage = require_metal_storage(*b_storage_ptr);
    auto& output_storage = require_metal_storage(*output.storage());

    impl_->run_elementwise(
        impl_->add_pipeline,
        buffer_from_storage(a_storage),
        buffer_from_storage(b_storage),
        buffer_from_storage(output_storage),
        a.numel());

    return output;
}

Tensor MetalDeviceImpl::sub(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_same_shape(a, b);

    auto output = empty(a.shape(), a.dtype());
    auto a_storage_ptr = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto b_storage_ptr = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto& a_storage = require_metal_storage(*a_storage_ptr);
    const auto& b_storage = require_metal_storage(*b_storage_ptr);
    auto& output_storage = require_metal_storage(*output.storage());

    impl_->run_elementwise(
        impl_->sub_pipeline,
        buffer_from_storage(a_storage),
        buffer_from_storage(b_storage),
        buffer_from_storage(output_storage),
        a.numel());

    return output;
}

Tensor MetalDeviceImpl::matmul(const Tensor& a, const Tensor& b) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    require_2d_matmul_shapes(a, b);

    const std::int64_t m = a.shape()[0];
    const std::int64_t k = a.shape()[1];
    const std::int64_t n = b.shape()[1];

    auto output = empty({m, n}, a.dtype());
    auto a_storage_ptr = ensure_storage(a.storage(), ConversionPolicy::CopyToDevice);
    auto b_storage_ptr = ensure_storage(b.storage(), ConversionPolicy::CopyToDevice);
    const auto& a_storage = require_metal_storage(*a_storage_ptr);
    const auto& b_storage = require_metal_storage(*b_storage_ptr);
    auto& output_storage = require_metal_storage(*output.storage());

    impl_->run_matmul(
        buffer_from_storage(a_storage),
        buffer_from_storage(b_storage),
        buffer_from_storage(output_storage),
        m,
        k,
        n);

    return output;
}

TensorStoragePtr MetalDeviceImpl::ensure_storage(
    const TensorStoragePtr& storage,
    ConversionPolicy policy) const {
    if (!storage) {
        throw std::invalid_argument("tensor has no storage");
    }

    if (storage->type() == TensorStorageType::MetalMemory) {
        return storage;
    }

    if (storage->type() == TensorStorageType::CpuMemory && policy == ConversionPolicy::CopyToDevice) {
        const auto& cpu_storage = static_cast<const CpuMemTensorStorageImpl&>(*storage);
        auto metal_storage = std::make_shared<MetalMemTensorStorageImpl>(
            cpu_storage.nbytes(),
            cpu_storage.dtype());
        metal_storage->copy_from_host(cpu_storage.data(), cpu_storage.nbytes());
        return metal_storage;
    }

    throw std::invalid_argument("MetalDeviceImpl requires MetalMemory storage");
}

const MetalMemTensorStorageImpl& MetalDeviceImpl::require_metal_storage(
    const ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::MetalMemory) {
        throw std::invalid_argument("MetalDeviceImpl requires MetalMemory storage");
    }

    return static_cast<const MetalMemTensorStorageImpl&>(storage);
}

MetalMemTensorStorageImpl& MetalDeviceImpl::require_metal_storage(ITensorStorage& storage) const {
    if (storage.type() != TensorStorageType::MetalMemory) {
        throw std::invalid_argument("MetalDeviceImpl requires MetalMemory storage");
    }

    return static_cast<MetalMemTensorStorageImpl&>(storage);
}

} // namespace citrius
