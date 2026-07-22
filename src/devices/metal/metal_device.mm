#include "impl/metal_device.h"

#include "impl/cpu_storage.h"
#include "impl/batched_matmul_layout.h"
#include "impl/metal_context.h"
#include "tensor_utils.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <climits>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace citrius::impl {

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

inline float apply_binary(float a, float b, uint operation) {
    switch (operation) {
        case 0: return a + b;
        case 1: return a - b;
        case 2: return a * b;
        case 3: return a / b;
        default: return max(a, b);
    }
}

kernel void broadcast_elementwise_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const long* metadata [[buffer(3)]],
    constant uint4& config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = config.x;
    const uint rank = config.y;
    const uint a_rank = config.z;
    const uint b_rank = config.w;
    if (id >= count) return;
    device const long* shape = metadata;
    device const long* output_strides = shape + rank;
    device const long* a_shape = output_strides + rank;
    device const long* a_strides = a_shape + a_rank;
    device const long* b_shape = a_strides + a_rank;
    device const long* b_strides = b_shape + b_rank;
    device const long* operation = b_strides + b_rank;
    long a_index = 0;
    long b_index = 0;
    for (uint axis = 0; axis < rank; ++axis) {
        const long coordinate = (long(id) / output_strides[axis]) % shape[axis];
        const uint a_padding = rank - a_rank;
        const uint b_padding = rank - b_rank;
        if (axis >= a_padding && a_shape[axis - a_padding] != 1)
            a_index += coordinate * a_strides[axis - a_padding];
        if (axis >= b_padding && b_shape[axis - b_padding] != 1)
            b_index += coordinate * b_strides[axis - b_padding];
    }
    out[id] = apply_binary(a[a_index], b[b_index], uint(*operation));
}

kernel void scalar_elementwise_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scalar [[buffer(3)]],
    constant uint2& config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float a = config.y ? scalar : input[id];
    const float b = config.y ? input[id] : scalar;
    out[id] = apply_binary(a, b, config.x);
}

kernel void unary_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& operation [[buffer(3)]],
    constant float& argument [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    switch (operation) {
        case 0: out[id] = exp(input[id]); break;
        case 1: out[id] = sqrt(input[id]); break;
        default: out[id] = pow(input[id], argument); break;
    }
}

kernel void masked_fill_f32(
    device const float* input [[buffer(0)]],
    device const uchar* mask [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const long* metadata [[buffer(3)]],
    constant uint3& config [[buffer(4)]],
    constant float& value [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = config.x;
    const uint rank = config.y;
    const uint mask_rank = config.z;
    if (id >= count) return;
    device const long* shape = metadata;
    device const long* output_strides = shape + rank;
    device const long* mask_shape = output_strides + rank;
    device const long* mask_strides = mask_shape + mask_rank;
    long mask_index = 0;
    const uint padding = rank - mask_rank;
    for (uint axis = 0; axis < rank; ++axis) {
        const long coordinate = (long(id) / output_strides[axis]) % shape[axis];
        if (axis >= padding && mask_shape[axis - padding] != 1)
            mask_index += coordinate * mask_strides[axis - padding];
    }
    out[id] = mask[mask_index] ? value : input[id];
}

kernel void contiguous_copy(
    device const uchar* input [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    device const long* metadata [[buffer(2)]],
    constant uint3& config [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = config.x;
    const uint rank = config.y;
    const uint element_size = config.z;
    if (id >= count) return;
    device const long* shape = metadata;
    device const long* output_strides = shape + rank;
    device const long* input_strides = output_strides + rank;
    long input_index = 0;
    for (uint axis = 0; axis < rank; ++axis) {
        const long coordinate = (long(id) / output_strides[axis]) % shape[axis];
        input_index += coordinate * input_strides[axis];
    }
    for (uint byte = 0; byte < element_size; ++byte)
        out[id * element_size + byte] = input[input_index * element_size + byte];
}

kernel void concat_copy(
    device const uchar* input [[buffer(0)]],
    device uchar* out [[buffer(1)]],
    device const long* metadata [[buffer(2)]],
    constant uint4& config [[buffer(3)]],
    constant uint3& concat_config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = config.x;
    const uint rank = config.y;
    const uint dimension = config.z;
    const uint element_size = config.w;
    if (id >= count) return;
    device const long* shape = metadata;
    device const long* input_contiguous_strides = shape + rank;
    device const long* input_strides = input_contiguous_strides + rank;
    device const long* output_strides = input_strides + rank;
    long input_index = 0;
    long output_index = 0;
    for (uint axis = 0; axis < rank; ++axis) {
        long coordinate = (long(id) / input_contiguous_strides[axis]) % shape[axis];
        input_index += coordinate * input_strides[axis];
        if (axis == dimension) coordinate += concat_config.x;
        output_index += coordinate * output_strides[axis];
    }
    for (uint byte = 0; byte < element_size; ++byte)
        out[output_index * element_size + byte] = input[input_index * element_size + byte];
}

kernel void gather_rows_kernel(
    device const uchar* table [[buffer(0)]],
    device const long* indices [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    device atomic_int* invalid [[buffer(3)]],
    constant uint4& config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    const uint index_count = config.x;
    const uint row_size = config.y;
    const uint element_size = config.z;
    const uint table_rows = config.w;
    const uint count = index_count * row_size;
    if (id >= count) return;
    const uint position = id / row_size;
    const uint column = id % row_size;
    const long row = indices[position];
    if (row < 0 || row >= long(table_rows)) {
        atomic_store_explicit(invalid, 1, memory_order_relaxed);
        return;
    }
    const long source = row * row_size + column;
    for (uint byte = 0; byte < element_size; ++byte)
        out[id * element_size + byte] = table[source * element_size + byte];
}

kernel void reduce_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const long* metadata [[buffer(2)]],
    constant uint4& config [[buffer(3)]],
    constant uint2& output_config [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    const uint input_count = config.x;
    const uint output_count = config.y;
    const uint rank = config.z;
    const uint operation = config.w;
    const uint output_rank = output_config.x;
    const uint reduced_count = output_config.y;
    if (id >= output_count) return;
    device const long* shape = metadata;
    device const long* contiguous = shape + rank;
    device const long* input_strides = contiguous + rank;
    device const long* reduced = input_strides + rank;
    device const long* output_strides = reduced + rank;
    float total = operation == 2 ? -INFINITY : 0.0f;
    float mean = 0.0f;
    if (operation == 3) {
        for (uint input_id = 0; input_id < input_count; ++input_id) {
            uint output_id = 0;
            uint output_axis = 0;
            long input_index = 0;
            for (uint axis = 0; axis < rank; ++axis) {
                const long coordinate = (long(input_id) / contiguous[axis]) % shape[axis];
                input_index += coordinate * input_strides[axis];
                if (!reduced[axis]) output_id += uint(coordinate * output_strides[output_axis++]);
            }
            if (output_id == id) mean += input[input_index];
        }
        mean /= float(reduced_count);
    }
    for (uint input_id = 0; input_id < input_count; ++input_id) {
        uint output_id = 0;
        uint output_axis = 0;
        long input_index = 0;
        for (uint axis = 0; axis < rank; ++axis) {
            const long coordinate = (long(input_id) / contiguous[axis]) % shape[axis];
            input_index += coordinate * input_strides[axis];
            if (!reduced[axis]) output_id += uint(coordinate * output_strides[output_axis++]);
        }
        if (output_id != id) continue;
        const float value = input[input_index];
        if (operation == 2) total = max(total, value);
        else if (operation == 3) total += (value - mean) * (value - mean);
        else total += value;
    }
    if (operation == 1 || operation == 3) total /= float(reduced_count);
    out[id] = total;
}

kernel void argmax_f32(
    device const float* input [[buffer(0)]],
    device long* out [[buffer(1)]],
    device const long* metadata [[buffer(2)]],
    constant uint3& config [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    const uint output_count = config.x;
    const uint rank = config.y;
    const uint axis = config.z;
    if (id >= output_count) return;
    device const long* shape = metadata;
    device const long* input_strides = shape + rank;
    device const long* output_shape = input_strides + rank;
    device const long* output_strides = output_shape + rank - 1;
    long base = 0;
    uint output_axis = 0;
    for (uint input_axis = 0; input_axis < rank; ++input_axis) {
        if (input_axis == axis) continue;
        const long coordinate = (long(id) / output_strides[output_axis]) % output_shape[output_axis];
        base += coordinate * input_strides[input_axis];
        ++output_axis;
    }
    float best = input[base];
    long best_index = 0;
    for (long index = 1; index < shape[axis]; ++index) {
        const float value = input[base + index * input_strides[axis]];
        if (value > best) { best = value; best_index = index; }
    }
    out[id] = best_index;
}

kernel void softmax_last_dimension_f32(
    device const float* input [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    if (row >= dims.x) return;
    const uint width = dims.y;
    const uint base = row * width;
    threadgroup float scratch[256];
    float local_max = -INFINITY;
    for (uint col = lane; col < width; col += lanes)
        local_max = max(local_max, input[base + col]);
    scratch[lane] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = scratch[0];
    float local_sum = 0.0f;
    for (uint col = lane; col < width; col += lanes) {
        const float value = exp(input[base + col] - row_max);
        out[base + col] = value;
        local_sum += value;
    }
    scratch[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lanes / 2; stride > 0; stride /= 2) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_sum = 1.0f / scratch[0];
    for (uint col = lane; col < width; col += lanes)
        out[base + col] *= inverse_sum;
}

kernel void scaled_dot_product_attention_f32(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const uchar* mask [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    constant uint3& options [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    const uint heads = dims.x;
    const uint key_value_heads = dims.y;
    const uint query_length = dims.z;
    const uint key_length = dims.w;
    const uint head_dim = options.x;
    const uint row_count = options.y;
    if (row >= row_count) return;
    const uint query_position = row % query_length;
    const uint head = (row / query_length) % heads;
    const uint batch = row / (heads * query_length);
    const uint query_base = row * head_dim;
    threadgroup float accumulator[256];
    threadgroup float reduction[256];
    threadgroup float state[4];
    for (uint col = lane; col < head_dim; col += lanes) accumulator[col] = 0.0f;
    if (lane == 0) { state[0] = -INFINITY; state[1] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint key_value_head = head / (heads / key_value_heads);
    for (uint key_position = 0; key_position < key_length; ++key_position) {
        const uint key_base =
            ((batch * key_value_heads + key_value_head) * key_length + key_position) * head_dim;
        float score = 0.0f;
        for (uint col = lane; col < head_dim; col += lanes)
            score += query[query_base + col] * key[key_base + col];
        reduction[lane] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = lanes / 2; stride > 0; stride /= 2) {
            if (lane < stride) reduction[lane] += reduction[lane + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) {
            const bool excluded = (options.z & 1u && key_position > query_position) ||
                (options.z & 2u && mask[query_position * key_length + key_position]);
            score = excluded ? -INFINITY : reduction[0] * rsqrt(float(head_dim));
            const float previous_maximum = state[0];
            const float next_maximum = max(previous_maximum, score);
            const float previous_scale = isinf(previous_maximum) ? 0.0f
                : exp(previous_maximum - next_maximum);
            const float value_scale = isinf(score) ? 0.0f : exp(score - next_maximum);
            state[0] = next_maximum;
            state[1] = state[1] * previous_scale + value_scale;
            state[2] = previous_scale;
            state[3] = value_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint col = lane; col < head_dim; col += lanes)
            accumulator[col] = accumulator[col] * state[2] + value[key_base + col] * state[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint col = lane; col < head_dim; col += lanes)
        output[query_base + col] = accumulator[col] / state[1];
}

kernel void matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 thread_id [[thread_position_in_threadgroup]]) {
    constexpr uint tile_size = 16;
    threadgroup float a_tile[tile_size][tile_size];
    threadgroup float b_tile[tile_size][tile_size];
    const uint row = group_id.y * tile_size + thread_id.y;
    const uint col = group_id.x * tile_size + thread_id.x;
    const uint m = dims.x;
    const uint k = dims.y;
    const uint n = dims.z;

    float total = 0.0f;
    for (uint tile = 0; tile < (k + tile_size - 1) / tile_size; ++tile) {
        const uint a_col = tile * tile_size + thread_id.x;
        const uint b_row = tile * tile_size + thread_id.y;
        a_tile[thread_id.y][thread_id.x] = row < m && a_col < k
            ? a[row * k + a_col] : 0.0f;
        b_tile[thread_id.y][thread_id.x] = b_row < k && col < n
            ? b[b_row * n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < tile_size; ++inner)
            total += a_tile[thread_id.y][inner] * b_tile[inner][thread_id.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) out[row * n + col] = total;
}

kernel void batched_matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const long* a_offsets [[buffer(3)]],
    device const long* b_offsets [[buffer(4)]],
    constant uint4& dims [[buffer(5)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]]) {
    constexpr uint tile_size = 16;
    threadgroup float a_tile[tile_size][tile_size];
    threadgroup float b_tile[tile_size][tile_size];
    const uint col = group_id.x * tile_size + thread_id.x;
    const uint row = group_id.y * tile_size + thread_id.y;
    const uint batch = group_id.z;
    const uint m = dims.x;
    const uint k = dims.y;
    const uint n = dims.z;
    const uint batches = dims.w;
    if (batch >= batches) return;
    float total = 0.0f;
    const long a_base = a_offsets[batch];
    const long b_base = b_offsets[batch];
    for (uint tile = 0; tile < (k + tile_size - 1) / tile_size; ++tile) {
        const uint a_col = tile * tile_size + thread_id.x;
        const uint b_row = tile * tile_size + thread_id.y;
        a_tile[thread_id.y][thread_id.x] = row < m && a_col < k
            ? a[a_base + row * k + a_col] : 0.0f;
        b_tile[thread_id.y][thread_id.x] = b_row < k && col < n
            ? b[b_base + b_row * n + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < tile_size; ++inner)
            total += a_tile[thread_id.y][inner] * b_tile[inner][thread_id.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < m && col < n) out[(batch * m + row) * n + col] = total;
}

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint2& dims [[buffer(3)]],
    constant float& epsilon [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= dims.x) return;
    const uint width = dims.y;
    const uint base = row * width;
    float squares = 0.0f;
    for (uint col = 0; col < width; ++col) squares += input[base + col] * input[base + col];
    const float inverse = rsqrt(squares / float(width) + epsilon);
    for (uint col = 0; col < width; ++col)
        out[base + col] = input[base + col] * inverse * weight[col];
}

kernel void swiglu_f32(
    device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) out[id] = (gate[id] / (1.0f + exp(-gate[id]))) * up[id];
}

kernel void add_rms_norm_f32(
    device const float* left [[buffer(0)]], device const float* right [[buffer(1)]],
    device const float* weight [[buffer(2)]], device float* residual [[buffer(3)]],
    device float* normalized [[buffer(4)]], constant uint2& dims [[buffer(5)]],
    constant float& epsilon [[buffer(6)]], uint row [[thread_position_in_grid]]) {
    if (row >= dims.x) return;
    const uint width = dims.y;
    const uint base = row * width;
    float squares = 0.0f;
    for (uint col = 0; col < width; ++col) {
        const float value = left[base + col] + right[base + col];
        residual[base + col] = value;
        squares += value * value;
    }
    const float inverse = rsqrt(squares / float(width) + epsilon);
    for (uint col = 0; col < width; ++col)
        normalized[base + col] = residual[base + col] * inverse * weight[col];
}

kernel void rms_norm_rope_f32(
    device const float* input [[buffer(0)]], device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint4& dims [[buffer(3)]],
    constant float2& parameters [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    const uint batch = dims.x;
    const uint sequence = dims.y;
    const uint heads = dims.z;
    const uint width = dims.w;
    if (row >= batch * sequence * heads) return;
    const uint head = row % heads;
    const uint position = (row / heads) % sequence;
    const uint batch_index = row / (heads * sequence);
    const uint input_base = row * width;
    const uint output_base = ((batch_index * heads + head) * sequence + position) * width;
    float squares = 0.0f;
    for (uint col = 0; col < width; ++col) squares += input[input_base + col] * input[input_base + col];
    const float inverse = rsqrt(squares / float(width) + parameters.x);
    const uint half_width = width / 2;
    for (uint col = 0; col < width; ++col) {
        const uint frequency_index = col % half_width;
        const float angle = float(position) * pow(parameters.y, -2.0f * float(frequency_index) / float(width));
        const uint other = col < half_width ? col + half_width : col - half_width;
        const float current = input[input_base + col] * inverse * weight[col];
        float rotated = input[input_base + other] * inverse * weight[other];
        if (col < half_width) rotated = -rotated;
        out[output_base + col] = current * cos(angle) + rotated * sin(angle);
    }
}
)";

void require_defined(const Tensor& tensor, const char* name) {
    (void)name;
    ENSURE_TENSOR_DEFINED(tensor);
}

void require_float32(const Tensor& tensor, const char* name) {
    (void)name;
    ENSURE_TENSOR_DTYPE(tensor, DType::Float32);
}

void require_same_shape(const Tensor& a, const Tensor& b) {
    ENSURE_TENSOR_SHAPE(b, a.shape());
}

void require_2d_matmul_shapes(const Tensor& a, const Tensor& b) {
    ENSURE_TENSOR_DIM(a, 2);
    ENSURE_TENSOR_DIM(b, 2);

    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul inner dimensions must match");
    }
}

Shape broadcast_shape(const Shape& left, const Shape& right) {
    const std::size_t rank = std::max(left.size(), right.size());
    Shape output(rank, 1);
    for (std::size_t offset = 0; offset < rank; ++offset) {
        const auto left_dim = offset < left.size() ? left[left.size() - 1 - offset] : 1;
        const auto right_dim = offset < right.size() ? right[right.size() - 1 - offset] : 1;
        if (left_dim != right_dim && left_dim != 1 && right_dim != 1)
            throw std::invalid_argument("tensor shapes are not broadcastable");
        output[rank - 1 - offset] = std::max(left_dim, right_dim);
    }
    return output;
}

Strides row_major_strides(const Shape& shape) {
    Strides result(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;)
        result[index - 1] = result[index] * shape[index];
    return result;
}

id<MTLBuffer> buffer_from_storage(const MetalMemTensorStorageImpl& storage) {
    return (__bridge id<MTLBuffer>)storage.handle().ptr;
}

} // namespace

class MetalDeviceImpl::Impl {
public:
    Impl() {
        context = metal_execution_context();
        device = (__bridge id<MTLDevice>)context->device();
        queue = (__bridge id<MTLCommandQueue>)context->command_queue();

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
        batched_matmul_pipeline = pipeline("batched_matmul_f32");
        rms_norm_pipeline = pipeline("rms_norm_f32");
        swiglu_pipeline = pipeline("swiglu_f32");
        add_rms_norm_pipeline = pipeline("add_rms_norm_f32");
        rms_norm_rope_pipeline = pipeline("rms_norm_rope_f32");
        broadcast_pipeline = pipeline("broadcast_elementwise_f32");
        scalar_pipeline = pipeline("scalar_elementwise_f32");
        unary_pipeline = pipeline("unary_f32");
        masked_fill_pipeline = pipeline("masked_fill_f32");
        contiguous_pipeline = pipeline("contiguous_copy");
        concat_pipeline = pipeline("concat_copy");
        gather_pipeline = pipeline("gather_rows_kernel");
        reduce_pipeline = pipeline("reduce_f32");
        argmax_pipeline = pipeline("argmax_f32");
        softmax_pipeline = pipeline("softmax_last_dimension_f32");
        attention_pipeline = pipeline("scaled_dot_product_attention_f32");
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
        id<MTLCommandBuffer> command_buffer = this->command_buffer();
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

        submit(command_buffer);
    }

    void run_matmul(
        id<MTLBuffer> a,
        id<MTLBuffer> b,
        id<MTLBuffer> out,
        std::int64_t m,
        std::int64_t k,
        std::int64_t n,
        NSUInteger a_offset,
        NSUInteger b_offset,
        bool transpose_b,
        NSUInteger b_row_bytes) const {
        // The first matmul kernel uses uint dimensions in Metal. This keeps
        // validation explicit until larger shape handling is designed.
        if (m < 0 || k < 0 || n < 0 || m > UINT32_MAX || k > UINT32_MAX || n > UINT32_MAX) {
            throw std::invalid_argument("Metal matmul dimensions are too large");
        }

        id<MTLCommandBuffer> command_buffer = this->command_buffer();
        MPSMatrixDescriptor* a_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(m)
            columns:static_cast<NSUInteger>(k)
            rowBytes:static_cast<NSUInteger>(k * sizeof(float))
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* output_descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(m)
            columns:static_cast<NSUInteger>(n)
            rowBytes:static_cast<NSUInteger>(n * sizeof(float))
            dataType:MPSDataTypeFloat32];
        MPSMatrix* left = [[MPSMatrix alloc] initWithBuffer:a
            offset:a_offset descriptor:a_descriptor];
        MPSMatrix* result = [[MPSMatrix alloc] initWithBuffer:out
            offset:0 descriptor:output_descriptor];
        const std::string multiplication_key = std::to_string(m) + ':' +
            std::to_string(k) + ':' + std::to_string(n) + ':' +
            (transpose_b ? "t" : "n");
        const std::string weight_key = std::to_string(
            reinterpret_cast<std::uintptr_t>((__bridge void*)b)) + ':' +
            std::to_string(b_offset) + ':' + std::to_string(transpose_b ? n : k) + ':' +
            std::to_string(transpose_b ? k : n) + ':' + std::to_string(b_row_bytes);
        MPSMatrixMultiplication* multiplication = nil;
        MPSMatrix* right = nil;
        {
            std::lock_guard lock(mps_cache_mutex);
            if (const auto found = multiplication_cache.find(multiplication_key);
                found != multiplication_cache.end()) {
                multiplication = found->second;
            } else {
                multiplication = [[MPSMatrixMultiplication alloc]
                    initWithDevice:device transposeLeft:NO transposeRight:transpose_b
                    resultRows:static_cast<NSUInteger>(m)
                    resultColumns:static_cast<NSUInteger>(n)
                    interiorColumns:static_cast<NSUInteger>(k)
                    alpha:1.0 beta:0.0];
                multiplication_cache.emplace(multiplication_key, multiplication);
            }
            if (const auto found = weight_matrix_cache.find(weight_key);
                found != weight_matrix_cache.end()) {
                right = found->second;
            } else {
                MPSMatrixDescriptor* b_descriptor = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:static_cast<NSUInteger>(transpose_b ? n : k)
                    columns:static_cast<NSUInteger>(transpose_b ? k : n)
                    rowBytes:b_row_bytes dataType:MPSDataTypeFloat32];
                right = [[MPSMatrix alloc] initWithBuffer:b
                    offset:b_offset descriptor:b_descriptor];
                weight_matrix_cache.emplace(weight_key, right);
            }
        }
        [multiplication encodeToCommandBuffer:command_buffer
            leftMatrix:left rightMatrix:right resultMatrix:result];

        submit(command_buffer);
    }

    void run_1d(id<MTLComputePipelineState> state, NSUInteger count,
                const std::function<void(id<MTLComputeCommandEncoder>)>& bind) const {
        if (count > UINT32_MAX) throw std::invalid_argument("Metal element count is too large");
        if (count == 0) return;
        id<MTLCommandBuffer> command_buffer = this->command_buffer();
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:state];
        bind(encoder);
        const NSUInteger threads = std::min<NSUInteger>(state.maxTotalThreadsPerThreadgroup, 256);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        submit(command_buffer);
    }

    void submit(id<MTLCommandBuffer> command_buffer) const {
        context->submit((__bridge void*)command_buffer);
    }

    id<MTLCommandBuffer> command_buffer() const {
        return (__bridge id<MTLCommandBuffer>)context->command_buffer();
    }

    void synchronize() const {
        context->synchronize();
    }

    std::shared_ptr<MetalExecutionContext> context;
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> add_pipeline = nil;
    id<MTLComputePipelineState> sub_pipeline = nil;
    id<MTLComputePipelineState> matmul_pipeline = nil;
    id<MTLComputePipelineState> batched_matmul_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_pipeline = nil;
    id<MTLComputePipelineState> swiglu_pipeline = nil;
    id<MTLComputePipelineState> add_rms_norm_pipeline = nil;
    id<MTLComputePipelineState> rms_norm_rope_pipeline = nil;
    id<MTLComputePipelineState> broadcast_pipeline = nil;
    id<MTLComputePipelineState> scalar_pipeline = nil;
    id<MTLComputePipelineState> unary_pipeline = nil;
    id<MTLComputePipelineState> masked_fill_pipeline = nil;
    id<MTLComputePipelineState> contiguous_pipeline = nil;
    id<MTLComputePipelineState> concat_pipeline = nil;
    id<MTLComputePipelineState> gather_pipeline = nil;
    id<MTLComputePipelineState> reduce_pipeline = nil;
    id<MTLComputePipelineState> argmax_pipeline = nil;
    id<MTLComputePipelineState> softmax_pipeline = nil;
    id<MTLComputePipelineState> attention_pipeline = nil;
    mutable std::mutex mps_cache_mutex;
    mutable std::unordered_map<std::string, MPSMatrixMultiplication*>
        multiplication_cache;
    mutable std::unordered_map<std::string, MPSMatrix*> weight_matrix_cache;
};

MetalDeviceImpl::MetalDeviceImpl()
    : impl_([] {
        static std::shared_ptr<Impl> shared = std::make_shared<Impl>();
        return shared;
    }()) {}

MetalDeviceImpl::~MetalDeviceImpl() = default;

DeviceType MetalDeviceImpl::type() const {
    return DeviceType::Metal;
}

Tensor MetalDeviceImpl::empty(Shape shape, DType dtype) const {
    std::int64_t count = 1;
    for (const auto dimension : shape) {
        if (dimension < 0) throw std::invalid_argument("tensor dimensions cannot be negative");
        if (dimension == 0) {
            count = 0;
        } else if (count != 0) {
            if (count > std::numeric_limits<std::int64_t>::max() / dimension)
                throw std::invalid_argument("tensor shape is too large");
            count *= dimension;
        }
    }
    const auto element_size = dtype_size(dtype);
    if (static_cast<std::uint64_t>(count) >
        std::numeric_limits<std::size_t>::max() / element_size)
        throw std::invalid_argument("tensor allocation is too large");
    auto storage = std::make_shared<MetalMemTensorStorageImpl>(
        static_cast<std::size_t>(count) * element_size,
        dtype,
        impl_->context);
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

Tensor MetalDeviceImpl::broadcast_elementwise(
    const Tensor& a, const Tensor& b, MetalElementwiseOperation operation) const {
    require_defined(a, "left");
    require_defined(b, "right");
    require_float32(a, "left");
    require_float32(b, "right");
    ENSURE_TENSOR_DEVICE_MATCH_2(a, b);
    const Shape output_shape = broadcast_shape(a.shape(), b.shape());
    Tensor output = empty(output_shape, DType::Float32);
    const auto& a_storage = require_metal_storage(*a.storage());
    const auto& b_storage = require_metal_storage(*b.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides output_strides = row_major_strides(output_shape);
    std::vector<std::int64_t> metadata;
    metadata.insert(metadata.end(), output_shape.begin(), output_shape.end());
    metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
    metadata.insert(metadata.end(), a.shape().begin(), a.shape().end());
    metadata.insert(metadata.end(), a.strides().begin(), a.strides().end());
    metadata.insert(metadata.end(), b.shape().begin(), b.shape().end());
    metadata.insert(metadata.end(), b.strides().begin(), b.strides().end());
    metadata.push_back(static_cast<std::int64_t>(operation));
    id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
        length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    std::array<uint32_t, 4> config = {static_cast<uint32_t>(output.numel()),
        static_cast<uint32_t>(output.ndim()), static_cast<uint32_t>(a.ndim()),
        static_cast<uint32_t>(b.ndim())};
    impl_->run_1d(impl_->broadcast_pipeline, static_cast<NSUInteger>(output.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(a_storage)
                offset:static_cast<NSUInteger>(a.storage_offset() * sizeof(float)) atIndex:0];
            [encoder setBuffer:buffer_from_storage(b_storage)
                offset:static_cast<NSUInteger>(b.storage_offset() * sizeof(float)) atIndex:1];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
            [encoder setBuffer:metadata_buffer offset:0 atIndex:3];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:4];
        });
    return output;
}

Tensor MetalDeviceImpl::scalar_elementwise(
    const Tensor& tensor, float scalar, MetalElementwiseOperation operation,
    bool scalar_is_left) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    if (!tensor.is_contiguous())
        throw std::invalid_argument("Metal scalar elementwise requires contiguous input");
    Tensor output = empty(tensor.shape(), tensor.dtype());
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const uint32_t count = static_cast<uint32_t>(tensor.numel());
    std::array<uint32_t, 2> config = {static_cast<uint32_t>(operation), scalar_is_left ? 1u : 0u};
    impl_->run_1d(impl_->scalar_pipeline, count, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:buffer_from_storage(input_storage)
            offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
        [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&scalar length:sizeof(scalar) atIndex:3];
        [encoder setBytes:config.data() length:sizeof(config) atIndex:4];
    });
    return output;
}

Tensor MetalDeviceImpl::unary(
    const Tensor& tensor, MetalUnaryOperation operation, float argument) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    if (!tensor.is_contiguous()) throw std::invalid_argument("Metal unary requires contiguous input");
    Tensor output = empty(tensor.shape(), tensor.dtype());
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const uint32_t count = static_cast<uint32_t>(tensor.numel());
    const uint32_t op = static_cast<uint32_t>(operation);
    impl_->run_1d(impl_->unary_pipeline, count, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:buffer_from_storage(input_storage)
            offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
        [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&op length:sizeof(op) atIndex:3];
        [encoder setBytes:&argument length:sizeof(argument) atIndex:4];
    });
    return output;
}

Tensor MetalDeviceImpl::masked_fill(const Tensor& tensor, const Tensor& mask, float value) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    ENSURE_TENSOR_DEFINED(mask);
    ENSURE_TENSOR_DTYPE(mask, DType::Bool);
    ENSURE_TENSOR_DEVICE_MATCH_2(tensor, mask);
    if (!tensor.is_contiguous()) throw std::invalid_argument("Metal masked_fill requires contiguous input");
    if (broadcast_shape(tensor.shape(), mask.shape()) != tensor.shape())
        throw std::invalid_argument("masked_fill mask must broadcast to the input shape");
    Tensor output = empty(tensor.shape(), tensor.dtype());
    const auto& input_storage = require_metal_storage(*tensor.storage());
    const auto& mask_storage = require_metal_storage(*mask.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides output_strides = row_major_strides(tensor.shape());
    std::vector<std::int64_t> metadata;
    metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
    metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
    metadata.insert(metadata.end(), mask.shape().begin(), mask.shape().end());
    metadata.insert(metadata.end(), mask.strides().begin(), mask.strides().end());
    id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
        length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    std::array<uint32_t, 3> config = {static_cast<uint32_t>(tensor.numel()),
        static_cast<uint32_t>(tensor.ndim()), static_cast<uint32_t>(mask.ndim())};
    impl_->run_1d(impl_->masked_fill_pipeline, static_cast<NSUInteger>(tensor.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(input_storage)
                offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
            [encoder setBuffer:buffer_from_storage(mask_storage)
                offset:static_cast<NSUInteger>(mask.storage_offset()) atIndex:1];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
            [encoder setBuffer:metadata_buffer offset:0 atIndex:3];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:4];
            [encoder setBytes:&value length:sizeof(value) atIndex:5];
        });
    return output;
}

Tensor MetalDeviceImpl::contiguous(const Tensor& tensor) const {
    require_defined(tensor, "input");
    if (tensor.device().type != DeviceType::Metal)
        throw std::invalid_argument("Metal contiguous requires a Metal tensor");
    if (tensor.is_contiguous()) return tensor;
    Tensor output = empty(tensor.shape(), tensor.dtype());
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides output_strides = row_major_strides(tensor.shape());
    std::vector<std::int64_t> metadata;
    metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
    metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
    metadata.insert(metadata.end(), tensor.strides().begin(), tensor.strides().end());
    id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
        length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    std::array<uint32_t, 3> config = {static_cast<uint32_t>(tensor.numel()),
        static_cast<uint32_t>(tensor.ndim()), static_cast<uint32_t>(dtype_size(tensor.dtype()))};
    impl_->run_1d(impl_->contiguous_pipeline, static_cast<NSUInteger>(tensor.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(input_storage)
                offset:static_cast<NSUInteger>(tensor.storage_offset() * dtype_size(tensor.dtype()))
                atIndex:0];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
            [encoder setBuffer:metadata_buffer offset:0 atIndex:2];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:3];
        });
    return output;
}

Tensor MetalDeviceImpl::concat(
    const std::vector<Tensor>& tensors, std::int64_t dimension) const {
    if (tensors.empty()) throw std::invalid_argument("concat expects at least one tensor");
    Shape output_shape = tensors.front().shape();
    output_shape[static_cast<std::size_t>(dimension)] = 0;
    for (const Tensor& tensor : tensors)
        output_shape[static_cast<std::size_t>(dimension)] +=
            tensor.shape()[static_cast<std::size_t>(dimension)];
    Tensor output = empty(output_shape, tensors.front().dtype());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides output_strides = row_major_strides(output_shape);
    std::int64_t dimension_offset = 0;
    for (const Tensor& tensor : tensors) {
        const auto& input_storage = require_metal_storage(*tensor.storage());
        const Strides input_contiguous_strides = row_major_strides(tensor.shape());
        std::vector<std::int64_t> metadata;
        metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
        metadata.insert(metadata.end(), input_contiguous_strides.begin(), input_contiguous_strides.end());
        metadata.insert(metadata.end(), tensor.strides().begin(), tensor.strides().end());
        metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
        id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
            length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
        std::array<uint32_t, 4> config = {static_cast<uint32_t>(tensor.numel()),
            static_cast<uint32_t>(tensor.ndim()), static_cast<uint32_t>(dimension),
            static_cast<uint32_t>(dtype_size(tensor.dtype()))};
        std::array<uint32_t, 3> concat_config = {
            static_cast<uint32_t>(dimension_offset), 0, 0};
        impl_->run_1d(impl_->concat_pipeline, static_cast<NSUInteger>(tensor.numel()),
            [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:buffer_from_storage(input_storage)
                    offset:static_cast<NSUInteger>(tensor.storage_offset() * dtype_size(tensor.dtype()))
                    atIndex:0];
                [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
                [encoder setBuffer:metadata_buffer offset:0 atIndex:2];
                [encoder setBytes:config.data() length:sizeof(config) atIndex:3];
                [encoder setBytes:concat_config.data() length:sizeof(concat_config) atIndex:4];
            });
        dimension_offset += tensor.shape()[static_cast<std::size_t>(dimension)];
    }
    return output;
}

Tensor MetalDeviceImpl::gather_rows(const Tensor& table, const Tensor& indices) const {
    require_defined(table, "table");
    require_defined(indices, "indices");
    ENSURE_TENSOR_DIM(table, 2);
    ENSURE_TENSOR_DTYPE(indices, DType::Int64);
    ENSURE_TENSOR_DEVICE_MATCH_2(table, indices);
    if (!table.is_contiguous() || !indices.is_contiguous())
        throw std::invalid_argument("Metal gather_rows requires contiguous inputs");
    Shape output_shape = indices.shape();
    output_shape.push_back(table.shape()[1]);
    Tensor output = empty(output_shape, table.dtype());
    const auto& table_storage = require_metal_storage(*table.storage());
    const auto& indices_storage = require_metal_storage(*indices.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    int zero = 0;
    id<MTLBuffer> invalid = [impl_->device newBufferWithBytes:&zero length:sizeof(zero)
        options:MTLResourceStorageModeShared];
    std::array<uint32_t, 4> config = {static_cast<uint32_t>(indices.numel()),
        static_cast<uint32_t>(table.shape()[1]), static_cast<uint32_t>(dtype_size(table.dtype())),
        static_cast<uint32_t>(table.shape()[0])};
    impl_->run_1d(impl_->gather_pipeline, static_cast<NSUInteger>(output.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(table_storage)
                offset:static_cast<NSUInteger>(table.storage_offset() * dtype_size(table.dtype())) atIndex:0];
            [encoder setBuffer:buffer_from_storage(indices_storage)
                offset:static_cast<NSUInteger>(indices.storage_offset() * sizeof(std::int64_t)) atIndex:1];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
            [encoder setBuffer:invalid offset:0 atIndex:3];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:4];
        });
    impl_->synchronize();
    if (*static_cast<const int*>(invalid.contents) != 0)
        throw std::out_of_range("gather_rows index is out of range");
    return output;
}

Tensor MetalDeviceImpl::reduce(
    const Tensor& tensor, const std::vector<std::int64_t>& dimensions,
    bool keepdim, MetalReductionOperation operation) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    std::vector<bool> is_reduced(tensor.ndim(), false);
    for (const auto dimension : dimensions)
        is_reduced[static_cast<std::size_t>(dimension)] = true;
    Shape output_shape;
    Shape compact_output_shape;
    std::int64_t reduced_count = 1;
    for (std::size_t axis = 0; axis < tensor.ndim(); ++axis) {
        if (is_reduced[axis]) {
            reduced_count *= tensor.shape()[axis];
            if (keepdim) output_shape.push_back(1);
        } else {
            output_shape.push_back(tensor.shape()[axis]);
            compact_output_shape.push_back(tensor.shape()[axis]);
        }
    }
    Tensor output = empty(output_shape, DType::Float32);
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides input_contiguous = row_major_strides(tensor.shape());
    const Strides output_strides = row_major_strides(compact_output_shape);
    std::vector<std::int64_t> metadata;
    metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
    metadata.insert(metadata.end(), input_contiguous.begin(), input_contiguous.end());
    metadata.insert(metadata.end(), tensor.strides().begin(), tensor.strides().end());
    for (const bool reduced : is_reduced) metadata.push_back(reduced ? 1 : 0);
    metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
    id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
        length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    std::array<uint32_t, 4> config = {static_cast<uint32_t>(tensor.numel()),
        static_cast<uint32_t>(output.numel()), static_cast<uint32_t>(tensor.ndim()),
        static_cast<uint32_t>(operation)};
    std::array<uint32_t, 2> output_config = {
        static_cast<uint32_t>(compact_output_shape.size()), static_cast<uint32_t>(reduced_count)};
    impl_->run_1d(impl_->reduce_pipeline, static_cast<NSUInteger>(output.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(input_storage)
                offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
            [encoder setBuffer:metadata_buffer offset:0 atIndex:2];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:3];
            [encoder setBytes:output_config.data() length:sizeof(output_config) atIndex:4];
        });
    return output;
}

Tensor MetalDeviceImpl::argmax(
    const Tensor& tensor, std::int64_t dimension, bool keepdim) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    const auto axis = static_cast<std::size_t>(dimension);
    if (tensor.shape()[axis] == 0)
        throw std::invalid_argument("argmax cannot reduce an empty dimension");
    Shape compact_output_shape = tensor.shape();
    compact_output_shape.erase(compact_output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    Shape output_shape = tensor.shape();
    if (keepdim) output_shape[axis] = 1;
    else output_shape = compact_output_shape;
    Tensor output = empty(output_shape, DType::Int64);
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const Strides output_strides = row_major_strides(compact_output_shape);
    std::vector<std::int64_t> metadata;
    metadata.insert(metadata.end(), tensor.shape().begin(), tensor.shape().end());
    metadata.insert(metadata.end(), tensor.strides().begin(), tensor.strides().end());
    metadata.insert(metadata.end(), compact_output_shape.begin(), compact_output_shape.end());
    metadata.insert(metadata.end(), output_strides.begin(), output_strides.end());
    id<MTLBuffer> metadata_buffer = [impl_->device newBufferWithBytes:metadata.data()
        length:metadata.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    std::array<uint32_t, 3> config = {static_cast<uint32_t>(output.numel()),
        static_cast<uint32_t>(tensor.ndim()), static_cast<uint32_t>(axis)};
    impl_->run_1d(impl_->argmax_pipeline, static_cast<NSUInteger>(output.numel()),
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(input_storage)
                offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
            [encoder setBuffer:metadata_buffer offset:0 atIndex:2];
            [encoder setBytes:config.data() length:sizeof(config) atIndex:3];
        });
    return output;
}

Tensor MetalDeviceImpl::argmax(const Tensor& tensor) const {
    const Tensor packed = contiguous(tensor);
    Tensor flattened({packed.numel()}, {1}, packed.storage_offset(), packed.dtype(),
                     packed.device(), packed.storage());
    return argmax(flattened, 0, false);
}

Tensor MetalDeviceImpl::softmax_last_dimension(const Tensor& tensor) const {
    require_defined(tensor, "input");
    require_float32(tensor, "input");
    if (!tensor.is_contiguous() || tensor.ndim() == 0)
        throw std::invalid_argument("Metal softmax requires contiguous non-scalar input");
    const auto width = tensor.shape().back();
    if (width <= 0) throw std::invalid_argument("softmax dimension cannot be empty");
    const auto rows = tensor.numel() / width;
    if (rows > UINT32_MAX || width > UINT32_MAX)
        throw std::invalid_argument("Metal softmax dimensions are too large");
    Tensor output = empty(tensor.shape(), DType::Float32);
    const auto& input_storage = require_metal_storage(*tensor.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    std::array<uint32_t, 2> dims = {
        static_cast<uint32_t>(rows), static_cast<uint32_t>(width)};
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->softmax_pipeline];
    [encoder setBuffer:buffer_from_storage(input_storage)
        offset:static_cast<NSUInteger>(tensor.storage_offset() * sizeof(float)) atIndex:0];
    [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:1];
    [encoder setBytes:dims.data() length:sizeof(dims) atIndex:2];
    const NSUInteger lanes = std::min<NSUInteger>(
        256, impl_->softmax_pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
    [encoder endEncoding];
    impl_->submit(command_buffer);
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

    const Tensor packed_a = a.is_contiguous() ? a : contiguous(a);
    const bool transposed_b = !b.is_contiguous() && b.strides()[0] == 1 &&
        b.strides()[1] >= b.shape()[0];
    const Tensor packed_b = b.is_contiguous() || transposed_b ? b : contiguous(b);
    auto output = empty({m, n}, a.dtype());
    auto a_storage_ptr = ensure_storage(packed_a.storage(), ConversionPolicy::CopyToDevice);
    auto b_storage_ptr = ensure_storage(packed_b.storage(), ConversionPolicy::CopyToDevice);
    const auto& a_storage = require_metal_storage(*a_storage_ptr);
    const auto& b_storage = require_metal_storage(*b_storage_ptr);
    auto& output_storage = require_metal_storage(*output.storage());

    impl_->run_matmul(
        buffer_from_storage(a_storage),
        buffer_from_storage(b_storage),
        buffer_from_storage(output_storage),
        m,
        k,
        n,
        static_cast<NSUInteger>(packed_a.storage_offset() * sizeof(float)),
        static_cast<NSUInteger>(packed_b.storage_offset() * sizeof(float)),
        transposed_b,
        static_cast<NSUInteger>((transposed_b ? packed_b.strides()[1] : packed_b.strides()[0]) *
            sizeof(float)));

    return output;
}

Tensor MetalDeviceImpl::batched_matmul(const Tensor& a, const Tensor& b) const {
    const Tensor packed_a = a.is_contiguous() ? a : contiguous(a);
    const Tensor packed_b = b.is_contiguous() ? b : contiguous(b);
    const BatchedLayout layout = make_batched_layout(packed_a, packed_b);
    Tensor output = empty(layout.output, DType::Float32);
    if (output.numel() == 0) return output;
    const auto& a_storage = require_metal_storage(*packed_a.storage());
    const auto& b_storage = require_metal_storage(*packed_b.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    id<MTLBuffer> a_offsets = [impl_->device newBufferWithBytes:layout.a_offsets.data()
        length:layout.a_offsets.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> b_offsets = [impl_->device newBufferWithBytes:layout.b_offsets.data()
        length:layout.b_offsets.size() * sizeof(std::int64_t) options:MTLResourceStorageModeShared];
    const auto m = packed_a.shape()[packed_a.ndim() - 2];
    const auto k = packed_a.shape().back();
    const auto n = packed_b.shape().back();
    std::array<uint32_t, 4> dims = {static_cast<uint32_t>(m), static_cast<uint32_t>(k),
        static_cast<uint32_t>(n), static_cast<uint32_t>(layout.a_offsets.size())};
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->batched_matmul_pipeline];
    [encoder setBuffer:buffer_from_storage(a_storage)
        offset:static_cast<NSUInteger>(packed_a.storage_offset() * sizeof(float)) atIndex:0];
    [encoder setBuffer:buffer_from_storage(b_storage)
        offset:static_cast<NSUInteger>(packed_b.storage_offset() * sizeof(float)) atIndex:1];
    [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
    [encoder setBuffer:a_offsets offset:0 atIndex:3];
    [encoder setBuffer:b_offsets offset:0 atIndex:4];
    [encoder setBytes:dims.data() length:sizeof(dims) atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(
        static_cast<NSUInteger>((n + 15) / 16), static_cast<NSUInteger>((m + 15) / 16),
        layout.a_offsets.size()) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    impl_->submit(command_buffer);
    return output;
}

std::optional<Tensor> MetalDeviceImpl::try_rms_norm(
    const Tensor& input, const Tensor& weight, float epsilon) const {
    if (!input.is_contiguous() || !weight.is_contiguous()) return std::nullopt;
    Tensor output = empty(input.shape(), DType::Float32);
    const auto& input_storage = require_metal_storage(*input.storage());
    const auto& weight_storage = require_metal_storage(*weight.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const uint32_t width = static_cast<uint32_t>(input.shape().back());
    const uint32_t rows = static_cast<uint32_t>(input.numel() / width);
    std::array<uint32_t, 2> dims = {rows, width};
    impl_->run_1d(impl_->rms_norm_pipeline, rows, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:buffer_from_storage(input_storage)
            offset:static_cast<NSUInteger>(input.storage_offset() * sizeof(float)) atIndex:0];
        [encoder setBuffer:buffer_from_storage(weight_storage)
            offset:static_cast<NSUInteger>(weight.storage_offset() * sizeof(float)) atIndex:1];
        [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
        [encoder setBytes:dims.data() length:sizeof(dims) atIndex:3];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
    });
    return output;
}

std::optional<Tensor> MetalDeviceImpl::try_swiglu(
    const Tensor& gate, const Tensor& up) const {
    if (!gate.is_contiguous() || !up.is_contiguous()) return std::nullopt;
    Tensor output = empty(gate.shape(), DType::Float32);
    const auto& gate_storage = require_metal_storage(*gate.storage());
    const auto& up_storage = require_metal_storage(*up.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    const uint32_t count = static_cast<uint32_t>(gate.numel());
    impl_->run_1d(impl_->swiglu_pipeline, count, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:buffer_from_storage(gate_storage)
            offset:static_cast<NSUInteger>(gate.storage_offset() * sizeof(float)) atIndex:0];
        [encoder setBuffer:buffer_from_storage(up_storage)
            offset:static_cast<NSUInteger>(up.storage_offset() * sizeof(float)) atIndex:1];
        [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
        [encoder setBytes:&count length:sizeof(count) atIndex:3];
    });
    return output;
}

std::optional<std::pair<Tensor, Tensor>> MetalDeviceImpl::try_add_rms_norm(
    const Tensor& left, const Tensor& right, const Tensor& weight, float epsilon) const {
    if (!left.is_contiguous() || !right.is_contiguous() || !weight.is_contiguous())
        return std::nullopt;
    Tensor residual = empty(left.shape(), DType::Float32);
    Tensor normalized = empty(left.shape(), DType::Float32);
    const auto& left_storage = require_metal_storage(*left.storage());
    const auto& right_storage = require_metal_storage(*right.storage());
    const auto& weight_storage = require_metal_storage(*weight.storage());
    auto& residual_storage = require_metal_storage(*residual.storage());
    auto& normalized_storage = require_metal_storage(*normalized.storage());
    const uint32_t width = static_cast<uint32_t>(left.shape().back());
    const uint32_t rows = static_cast<uint32_t>(left.numel() / width);
    std::array<uint32_t, 2> dims = {rows, width};
    impl_->run_1d(impl_->add_rms_norm_pipeline, rows, [&](id<MTLComputeCommandEncoder> encoder) {
        [encoder setBuffer:buffer_from_storage(left_storage)
            offset:static_cast<NSUInteger>(left.storage_offset() * sizeof(float)) atIndex:0];
        [encoder setBuffer:buffer_from_storage(right_storage)
            offset:static_cast<NSUInteger>(right.storage_offset() * sizeof(float)) atIndex:1];
        [encoder setBuffer:buffer_from_storage(weight_storage)
            offset:static_cast<NSUInteger>(weight.storage_offset() * sizeof(float)) atIndex:2];
        [encoder setBuffer:buffer_from_storage(residual_storage) offset:0 atIndex:3];
        [encoder setBuffer:buffer_from_storage(normalized_storage) offset:0 atIndex:4];
        [encoder setBytes:dims.data() length:sizeof(dims) atIndex:5];
        [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:6];
    });
    return std::make_pair(std::move(residual), std::move(normalized));
}

std::optional<Tensor> MetalDeviceImpl::try_rms_norm_rope(
    const Tensor& input, const Tensor& weight, float epsilon, float theta) const {
    if (!input.is_contiguous() || !weight.is_contiguous()) return std::nullopt;
    Shape output_shape = {input.shape()[0], input.shape()[2], input.shape()[1], input.shape()[3]};
    Tensor output = empty(output_shape, DType::Float32);
    const auto& input_storage = require_metal_storage(*input.storage());
    const auto& weight_storage = require_metal_storage(*weight.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    std::array<uint32_t, 4> dims = {static_cast<uint32_t>(input.shape()[0]),
        static_cast<uint32_t>(input.shape()[1]), static_cast<uint32_t>(input.shape()[2]),
        static_cast<uint32_t>(input.shape()[3])};
    std::array<float, 2> parameters = {epsilon, theta};
    const NSUInteger rows = static_cast<NSUInteger>(input.numel() / input.shape().back());
    impl_->run_1d(impl_->rms_norm_rope_pipeline, rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:buffer_from_storage(input_storage)
                offset:static_cast<NSUInteger>(input.storage_offset() * sizeof(float)) atIndex:0];
            [encoder setBuffer:buffer_from_storage(weight_storage)
                offset:static_cast<NSUInteger>(weight.storage_offset() * sizeof(float)) atIndex:1];
            [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:2];
            [encoder setBytes:dims.data() length:sizeof(dims) atIndex:3];
            [encoder setBytes:parameters.data() length:sizeof(parameters) atIndex:4];
        });
    return output;
}

std::optional<Tensor> MetalDeviceImpl::try_scaled_dot_product_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& mask, bool is_causal) const {
    if (query.dtype() != DType::Float32 || key.dtype() != DType::Float32 ||
        value.dtype() != DType::Float32 || query.ndim() != 4 || key.ndim() != 4 ||
        value.ndim() != 4 || query.device().type != DeviceType::Metal ||
        key.device() != query.device() || value.device() != query.device() ||
        !query.is_contiguous() || !key.is_contiguous() || !value.is_contiguous() ||
        query.shape()[0] != key.shape()[0] || query.shape()[1] % key.shape()[1] != 0 ||
        key.shape() != value.shape() || query.shape()[3] != key.shape()[3] ||
        query.shape()[3] > 256) return std::nullopt;
    const auto query_length = query.shape()[2];
    const auto key_length = key.shape()[2];
    if (mask.defined() && (mask.dtype() != DType::Bool || mask.device() != query.device() ||
        mask.shape() != Shape{query_length, key_length} || !mask.is_contiguous()))
        return std::nullopt;
    Tensor output = empty(query.shape(), DType::Float32);
    if (output.numel() == 0) return output;
    const auto& query_storage = require_metal_storage(*query.storage());
    const auto& key_storage = require_metal_storage(*key.storage());
    const auto& value_storage = require_metal_storage(*value.storage());
    auto& output_storage = require_metal_storage(*output.storage());
    id<MTLBuffer> mask_buffer = nil;
    NSUInteger mask_offset = 0;
    if (mask.defined()) {
        mask_buffer = buffer_from_storage(require_metal_storage(*mask.storage()));
        mask_offset = static_cast<NSUInteger>(mask.storage_offset());
    }
    const auto rows = query.shape()[0] * query.shape()[1] * query_length;
    std::array<uint32_t, 4> dims = {static_cast<uint32_t>(query.shape()[1]),
        static_cast<uint32_t>(key.shape()[1]), static_cast<uint32_t>(query_length),
        static_cast<uint32_t>(key_length)};
    std::array<uint32_t, 3> options = {static_cast<uint32_t>(query.shape()[3]),
        static_cast<uint32_t>(rows), static_cast<uint32_t>((is_causal ? 1 : 0) |
            (mask.defined() ? 2 : 0))};
    id<MTLCommandBuffer> command_buffer = impl_->command_buffer();
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:impl_->attention_pipeline];
    [encoder setBuffer:buffer_from_storage(query_storage)
        offset:static_cast<NSUInteger>(query.storage_offset() * sizeof(float)) atIndex:0];
    [encoder setBuffer:buffer_from_storage(key_storage)
        offset:static_cast<NSUInteger>(key.storage_offset() * sizeof(float)) atIndex:1];
    [encoder setBuffer:buffer_from_storage(value_storage)
        offset:static_cast<NSUInteger>(value.storage_offset() * sizeof(float)) atIndex:2];
    [encoder setBuffer:mask_buffer offset:mask_offset atIndex:3];
    [encoder setBuffer:buffer_from_storage(output_storage) offset:0 atIndex:4];
    [encoder setBytes:dims.data() length:sizeof(dims) atIndex:5];
    [encoder setBytes:options.data() length:sizeof(options) atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(rows), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    impl_->submit(command_buffer);
    return output;
}

void MetalDeviceImpl::synchronize() const {
    impl_->synchronize();
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
            cpu_storage.dtype(),
            impl_->context);
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

} // namespace citrius::impl
