# Automatic Fusion Design

## Status

This document describes the proposed automatic kernel-fusion architecture for
Citrius. It is a design target, not a description of functionality that is
already implemented.

Citrius currently uses eager execution and several explicit fused device
operations, including RMSNorm, SwiGLU, RMSNorm plus RoPE, scaled dot-product
attention, and residual addition plus RMSNorm. Automatic fusion should retain
these kernels as optimized implementations and extend fusion to operation
combinations that have not been written by hand.

## Goals

The fusion system should:

- Preserve the existing eager `Tensor` and `Module` APIs.
- Capture existing model code without requiring a separate graph DSL.
- Discover compatible elementwise and reduction regions automatically.
- Select handwritten kernels when they are the best implementation.
- Generate kernels for supported regions without handwritten equivalents.
- Optimize GEMM boundaries through library epilogues and mixed-dtype execution.
- Provide strict diagnostics proving which implementation executed.
- Support differential testing against eager and handwritten execution.
- Keep a path open for CPU and Metal without requiring all backends initially.

The initial implementation is not intended to generate general GEMM kernels,
support data-dependent control flow, or transparently delay every eager tensor
operation.

## Overall architecture

```text
Tensor and Module APIs
        |
        v
Explicit graph capture
        |
        v
High-level tensor IR
        |
        +-- shape, dtype, layout, and alias analysis
        +-- decomposition of compound operations
        +-- graph rewrites and fusion grouping
        |
        v
Implementation selection
        |
        +-- handwritten fused kernel
        +-- backend library call or GEMM epilogue
        +-- generated kernel
        +-- eager fallback
        |
        v
Kernel-level loop and reduction IR
        |
        v
CUDA C++ source generation and NVRTC compilation
```

Two IR levels are intentional. The tensor IR expresses model semantics and
logical layouts. The kernel IR expresses loops, indexing, reductions, memory
placement, synchronization, and launch strategy.

## Execution modes

Fusion behavior should be explicit and testable:

```cpp
enum class FusionMode {
    Handwritten,
    Generated,
    Disabled,
};
```

- `Handwritten` uses the existing `try_*` device hooks and other specialized
  implementations.
- `Generated` decomposes compound operations, captures their primitives, forms
  fusion groups, and compiles generated kernels.
- `Disabled` executes canonical primitive decompositions eagerly.

The process default may be selected with `CITRIUS_FUSION`, but tests and library
callers should use a scoped thread-local override rather than mutate the process
environment:

```cpp
{
    ScopedFusionMode mode(FusionMode::Generated);
    Tensor output = model(input);
}
```

Generated-mode tests need a strict option that rejects silent fallback:

```cpp
struct FusionOptions {
    FusionMode mode = FusionMode::Handwritten;
    bool require_fusion = false;
};
```

When `require_fusion` is true, an unsupported region, compilation failure, or
eager fallback is an error.

## Capture boundary

The first public capture mechanism should be explicit:

```cpp
auto compiled = citrius::compile(
    model,
    example_inputs,
    FusionOptions{.mode = FusionMode::Generated});
```

During capture, tensor operations return symbolic tensors instead of allocating
storage and executing immediately. Top-level operations conceptually follow:

```cpp
if (CaptureContext* capture = current_capture_context()) {
    return capture->record(OpCode::Add, {left, right});
}
return eager_add(left, right);
```

Static C++ control flow is resolved during tracing. Operations that inspect
tensor data, such as `item()` used in a branch condition, should initially be
rejected during capture.

## High-level tensor IR

The tensor IR describes values, operations, logical views, and symbolic shapes:

```cpp
struct Value {
    ValueId id;
    ShapeExpr shape;
    DType dtype;
    Device device;
    Layout layout;
};

struct Operation {
    OpCode opcode;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    Attributes attributes;
};
```

`ShapeExpr` may include constants, symbols such as batch and sequence length,
and simple arithmetic expressions. The compiler should specialize stable model
dimensions such as hidden size and head dimension while keeping batch and
sequence dimensions dynamic where practical.

Views remain metadata in this IR whenever possible:

```text
%weight_t = transpose_view %weight, [1, 0]
%output   = matmul %input, %weight_t
```

Each value carries a logical layout:

```cpp
struct Layout {
    ShapeExpr shape;
    StrideExpr strides;
    OffsetExpr offset;
};
```

Layout and alias analysis determine whether a consumer can use a view directly
or must insert a materialization boundary.

## Canonical decompositions

Every compound fused operation needs a backend-independent primitive
decomposition. Generated and disabled modes use these decompositions instead of
calling the corresponding `try_*` hook.

Examples include:

```text
swiglu(gate, up)
    -> multiply(silu(gate), up)

add_rms_norm(left, right, weight, epsilon)
    -> residual = add(left, right)
    -> squares = multiply(residual, residual)
    -> mean_square = mean(squares, axis=-1, keepdim=true)
    -> inverse_rms = rsqrt(add(mean_square, epsilon))
    -> normalized = multiply(multiply(residual, inverse_rms), weight)
```

Decomposition must use an internal guard so primitive recording or execution
cannot recursively redispatch to the compound operation.

## Fusion grouping

Operations are classified as:

- Elementwise
- Reduction
- View
- Backend library call
- Opaque or side-effecting

A conservative initial producer-consumer rule may fuse producer `P` into
consumer `C` when:

- Both run on the same device.
- Both have supported dtypes and layouts.
- `P` is elementwise and `C` is elementwise or a supported reduction.
- Their iteration domains are compatible.
- `P` has one external consumer, or recomputation is explicitly profitable.
- No unsafe read/write alias is introduced.
- Register and shared-memory estimates remain within configured limits.

Views may be folded into consumer index expressions. Fusion stops or inserts a
copy for unsupported strides, dangerous aliases, backend-library layout
requirements, or reductions whose dependencies cross the selected execution
scope.

The compiler should retain a cost model. A legal fusion is not necessarily
profitable if it increases register pressure, repeats expensive computation,
or prevents use of an optimized library operation.

## Implementation selection

Each fusion group is matched in this order:

1. A known handwritten fused kernel.
2. A supported backend-library operation or epilogue.
3. A generated kernel.
4. Eager execution, unless strict generated mode forbids fallback.

Handwritten kernels remain important as tuned implementations, correctness
oracles, and fallbacks. Automatic fusion should not generate general matrix
multiplication initially. GEMMs should use cuBLAS, cuBLASLt, or CUTLASS, with
supported work fused into epilogues.

Examples of graph rewrites around GEMMs include:

```text
down_projection(x) + residual
    -> GEMM with residual-add epilogue

gate_projection(x), up_projection(x)
    -> packed gate/up GEMM followed by SwiGLU

q_projection(x), k_projection(x), v_projection(x)
    -> packed QKV GEMM followed by split views

BF16 GEMM followed by FP32 cast
    -> BF16-input, FP32-accumulating, FP32-output GEMM
```

## Kernel IR

Generated fusion groups lower to an execution-oriented kernel IR:

```cpp
struct Kernel {
    std::string name;
    std::vector<KernelParameter> parameters;
    std::vector<Statement> body;
    LaunchConfiguration launch;
};

using Statement = std::variant<
    ParallelFor,
    SerialFor,
    Assign,
    Store,
    BlockReduction,
    Barrier,
    If>;

using Expression = std::variant<
    Constant,
    Variable,
    Load,
    BinaryExpression,
    UnaryExpression,
    IndexExpression>;
```

The first kernel IR needs to express:

- Grid-stride elementwise loops
- Per-thread scalar expressions
- Loads and stores with broadcast indexing
- Serial and parallel loops
- Block reductions
- Register and shared-memory values
- Barriers and broadcast of reduction results
- Float16, BFloat16, and Float32 conversions

For a fused residual addition and RMSNorm, lowering may select one block per row
and 256 threads per block:

```text
parallel block row in [0, rows)
  thread_local sum_squares = 0

  parallel thread column in [0, columns)
    residual = left[row, column] + right[row, column]
    store residual_output[row, column] = residual
    sum_squares += residual * residual

  block_reduce_add sum_squares
  inverse_rms = rsqrt(sum_squares / columns + epsilon)

  parallel thread column in [0, columns)
    normalized = residual_output[row, column] * inverse_rms * weight[column]
    store normalized_output[row, column] = normalized
```

The initial reduction implementation may use shared memory. Later tuning can
introduce warp shuffles, vectorized loads, specialized hidden sizes, and
multi-warp schedules without changing the tensor IR.

## CUDA source generation

The first generated backend should emit self-contained CUDA C++ and compile it
with NVRTC. Generated source should not include Citrius project headers.

A CUDA emitter walks the kernel IR:

```cpp
class CudaEmitter {
public:
    std::string emit(const Kernel& kernel);

private:
    void emit_statement(const Statement& statement);
    std::string emit_expression(const Expression& expression);
};
```

A simple generated elementwise kernel may look like:

```cpp
extern "C" __global__
void fused_8f92c3(
    const float* input0,
    const float* input1,
    float* output0,
    long long count)
{
    const long long first =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long stride =
        static_cast<long long>(blockDim.x) * gridDim.x;

    for (long long index = first; index < count; index += stride) {
        output0[index] = fmaxf(input0[index] + input1[index], 0.0f);
    }
}
```

NVRTC produces PTX, which is loaded with the CUDA Driver API. The resulting
module and function are owned by a compiled-kernel object and launched on the
same `CudaExecutionContext` stream used by eager kernels and cuBLAS.

Compilation diagnostics must always retain the NVRTC log on failure. Debug
configuration should also make the generated source and kernel IR available in
error messages or diagnostic dumps.

## Kernel arguments and launch

Generated kernels receive raw storage pointers plus only the dynamic metadata
needed by the specialization. Static dimensions and layout choices should be
embedded in generated source when they are part of the cache key.

Driver API arguments are addresses of host-side argument storage:

```cpp
CUdeviceptr input = ...;
CUdeviceptr output = ...;
long long count = ...;

void* arguments[] = {&input, &output, &count};
```

Launch scheduling is part of lowering rather than graph capture. Initial
schedules should be intentionally simple:

- Elementwise: 256-thread blocks and a grid-stride loop.
- Row reduction: one block per row and shared-memory reduction.
- Softmax: maximum reduction, exponentiation, sum reduction, normalization.

## Compilation cache

Runtime compilation must not occur repeatedly during decoding. An in-memory
cache key should include:

```text
canonical kernel IR hash
backend and device architecture
input and output dtypes
rank and layout class
specialized dimensions
compiler and fast-math options
kernel IR version
```

The IR hash is based on canonical structure rather than temporary value names.
The in-memory cache maps the key to a loaded module and function. A later disk
cache may store PTX or cubin, but needs explicit versioning and invalidation.

Fusion statistics should report at least:

```cpp
struct FusionStats {
    std::size_t captured_operations = 0;
    std::size_t fusion_groups = 0;
    std::size_t generated_kernels = 0;
    std::size_t handwritten_kernels = 0;
    std::size_t eager_operations = 0;
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
};
```

## Qwen3 integration and testing

Qwen3 is the primary integration workload because it already has handwritten
fused kernels that serve as differential-test oracles. Tests should compare
three executions with identical deterministic parameters and inputs:

```text
disabled primitive execution
handwritten fused execution
generated fused execution
```

Testing occurs at four levels:

1. Generated kernel versus CPU or eager primitive reference.
2. Generated kernel versus the corresponding handwritten CUDA kernel.
3. Structural capture assertions verifying the primitive graph and fusion
   group that were formed.
4. Tiny deterministic and checkpoint-backed Qwen3 model comparisons.

Generated tests use `require_fusion = true` and assert statistics such as:

```cpp
EXPECT_EQ(stats.generated_kernels, 1);
EXPECT_EQ(stats.handwritten_kernels, 0);
EXPECT_EQ(stats.eager_operations, 0);
```

Numerical tests should cover contiguous and supported strided layouts, all
supported dtypes, empty cases where legal, and sizes around warp and block
boundaries such as 31, 32, 33, 127, 128, 129, and the Qwen hidden dimensions.
GPU reductions should use documented tolerances rather than exact equality.

The initial generated Qwen target should be SwiGLU because it is a pure
elementwise expression with an existing handwritten reference:

```text
silu(gate) * up
```

The second target should be residual addition plus RMSNorm, which adds multiple
outputs, a block reduction, shared memory, and synchronization. RMSNorm plus
RoPE and attention follow only after these mechanisms are reliable.

For Qwen performance, compiler-directed graph rewrites around GEMMs, memory
planning, layout propagation, and KV caching are likely more valuable than
generic elementwise fusion alone. The fusion compiler nevertheless provides a
systematic way to select and test those implementations.

## Development sequence

1. Add `FusionMode`, scoped overrides, strict generated mode, and statistics.
2. Add canonical primitive decompositions for the existing compound operations.
3. Add explicit module or callable capture and a high-level tensor graph.
4. Replay captured graphs eagerly to validate capture and shape inference.
5. Implement elementwise fusion grouping and scalar-expression kernel IR.
6. Emit grid-stride CUDA kernels and compile them through NVRTC.
7. Add canonical cache keys, an in-memory compiled-kernel cache, and diagnostics.
8. Differential-test generated SwiGLU in isolation and through tiny Qwen3.
9. Add block-reduction IR and generated residual-add plus RMSNorm.
10. Add GEMM implementation selection, packed projections, and epilogues.
11. Add lifetime analysis and reusable intermediate-buffer planning.
12. Extend the stable kernel IR to Metal and CPU implementations as justified.

This sequence provides a complete capture-to-execution vertical slice before
adding complex reductions or adopting a heavyweight compiler framework. An
internal IR plus NVRTC is the preferred starting point. LLVM or MLIR can be
considered later if the stabilized IR, optimization requirements, or multi-
backend needs justify the dependency and integration cost.
