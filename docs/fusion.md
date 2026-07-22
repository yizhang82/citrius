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
- Keep symbolic execution orthogonal to autograd state.
- Allow the same tensor IR to represent compiled forward and backward graphs.

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

## Tensor representation

Symbolic execution should not use a `SymbolicTensor` subclass or virtual
`Tensor` methods. Citrius passes and returns tensors by value, so subclassing
would introduce slicing and force pointer-based polymorphism through the public
API. Instead, `Tensor` remains a small value handle around shared tensor
identity:

```cpp
class Tensor {
public:
    bool defined() const;
    bool is_eager() const;
    bool is_symbolic() const;

private:
    std::shared_ptr<impl::TensorImpl> impl_;
};
```

`TensorImpl` separates the execution payload from optional differentiation
state:

```cpp
struct TensorImpl {
    Shape shape;
    Strides strides;
    std::int64_t storage_offset = 0;
    DType dtype = DType::Float32;
    Device device;

    TensorPayload payload;
    std::shared_ptr<AutogradMeta> autograd;
    std::shared_ptr<VersionCounter> version;
};

using TensorPayload = std::variant<
    EagerTensorPayload,
    SymbolicTensorPayload>;

struct EagerTensorPayload {
    TensorStoragePtr storage;
};

struct SymbolicTensorPayload {
    GraphValue value;
};
```

An undefined tensor has a null `impl_`. Most inference tensors leave
`autograd == nullptr`, avoiding unnecessary differentiation metadata.

A copied tensor handle shares its `TensorImpl`. A newly created view receives a
new `TensorImpl`, shares eager storage when applicable, and records a view
relationship. An explicit deep copy receives new storage and new identity.
`detach()` shares storage and versioning but creates a new identity without a
gradient edge.

Symbolic tensors have metadata and a graph value but no storage. Operations
that require host data or a memory handle, such as `item()` and direct storage
access, reject symbolic values. View operations transform symbolic layout
metadata and record view nodes rather than materializing memory.

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

## Autograd integration

Symbolic representation and differentiation state are independent axes. An
eager or symbolic tensor may require gradients, and neither concern should
produce another tensor subclass or payload alternative.

### Eager autograd metadata

Autograd metadata should contain only differentiation-specific state:

```cpp
struct AutogradMeta {
    bool requires_grad = false;
    bool retain_grad = false;
    bool is_leaf = false;

    Tensor gradient;
    Edge gradient_edge;

    std::weak_ptr<TensorImpl> view_base;
    std::shared_ptr<ViewInfo> view_info;
    std::vector<BackwardHook> hooks;
};

struct Edge {
    std::shared_ptr<BackwardNode> function;
    std::size_t input_index = 0;
};
```

An eager differentiable operation executes normally and attaches a backward
node only when gradient mode is enabled and at least one input requires a
gradient. Leaf tensors use gradient accumulators rather than producer nodes.

Thread-local `NoGradGuard` and `InferenceModeGuard` should be distinct.
`NoGradGuard` disables recording while retaining ordinary tensor and version
behavior. `InferenceModeGuard` may omit autograd metadata and version tracking
where safe.

### Operation definitions

Operation semantics should be registered once and reused by capture, fusion,
and graph differentiation:

```cpp
struct OperationDefinition {
    InferFunction infer;
    DecomposeFunction decompose;
    DifferentiateFunction differentiate;
    FusionClass fusion_class;
    AliasBehavior alias_behavior;
};
```

The differentiation callback builds ordinary tensor IR. For multiplication:

```text
%grad_left_raw  = multiply %grad_output, %right
%grad_right_raw = multiply %grad_output, %left
%grad_left      = sum_to_shape %grad_left_raw, shape(%left)
%grad_right     = sum_to_shape %grad_right_raw, shape(%right)
```

This lets forward and backward graphs use the same inference, canonicalization,
fusion, lowering, and code-generation machinery.

### Ahead-of-time autograd

Training compilation should differentiate the captured forward graph before
lowering:

```text
captured forward graph
        |
        +-- determine required gradients
        +-- determine values required by backward
        v
differentiate tensor IR
        |
        +-- optimized forward graph
        +-- optimized backward graph
        +-- saved-value and recomputation plan
        v
joint fusion, lowering, and memory planning
```

A compiled training artifact contains:

```cpp
struct CompiledTrainingPlan {
    ExecutablePlan forward;
    ExecutablePlan backward;
    SavedValuePlan saved_values;
};
```

Calling a compiled forward plan creates an invocation context containing the
saved values and parameter bindings required by its matching backward plan.
Forward-only compilation omits this machinery.

### Saved values and recomputation

Derivative rules declare whether they need inputs, outputs, or metadata:

```cpp
enum class SaveKind {
    Input,
    Output,
    Metadata,
};
```

The compiler may save a value, recompute it, compress it, or derive it from
another retained value. Initial behavior should always save required values;
recomputation and activation checkpointing are later optimizations.

Backward lifetimes participate in memory planning. A saved forward value is
released after its final backward use, and gradient buffers may reuse slots
when their lifetimes do not overlap.

### Views, mutation, and versioning

Views and their bases share a version counter. Any in-place write increments
it. A saved tensor records the expected version and backward rejects execution
if the value was modified after being saved:

```cpp
struct SavedTensor {
    Tensor tensor;
    std::uint64_t expected_version = 0;
};
```

The first autograd and compiler implementations should remain functional and
support little or no mutation. If general in-place APIs are added, capture must
functionalize them into new SSA values before optimization.

View derivative rules apply inverse layout transformations where possible:

```text
reshape     -> reshape gradient to the input shape
transpose   -> apply the inverse transpose
expand      -> sum gradient to the input shape
slice       -> scatter gradient into an input-shaped zero tensor
contiguous  -> restore the logical input layout
```

The alias analysis used for fusion must be shared with view autograd and
functionalization.

### Backward fusion

Backward graphs often contain larger elementwise regions than inference graphs.
For SwiGLU, one generated backward kernel can read `gate`, `up`, and
`grad_output`, then write both `grad_gate` and `grad_up` without materializing
sigmoid, SiLU, or derivative intermediates.

Multiple gradient contributions initially form explicit additions in the IR.
The compiler may fuse these additions with their producers or plan a dedicated
accumulation buffer. The initial scheduler should execute one backward graph on
one device stream and avoid concurrent accumulation and atomics.

### Numeric policy

Autograd and fusion must distinguish storage, computation, accumulation,
output, and gradient-accumulation dtypes:

```cpp
struct NumericPolicy {
    DType input_dtype;
    DType compute_dtype;
    DType accumulation_dtype;
    DType output_dtype;
    DType gradient_accumulation_dtype;
};
```

For mixed-precision training, BF16 activations and weights may use FP32 GEMM and
gradient accumulation, while losses, master weights, and optimizer state remain
FP32. These choices must be explicit in IR and cache keys rather than inferred
from a single tensor dtype.

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

## Work items

The checkboxes below track implementation rather than design completion. Items
are ordered so each milestone leaves a testable vertical slice.

### Tensor identity, views, and modes

- [ ] Introduce shared `TensorImpl` while preserving existing public value
  semantics and shallow-copy behavior.
- [ ] Represent eager and symbolic execution with a tagged `TensorPayload`.
- [ ] Define exact copy, deep-copy, view, and detach semantics in tests.
- [ ] Add shared version counters for storage aliases and views.
- [ ] Add `AutogradMeta` as an optional allocation.
- [ ] Add thread-local gradient, inference, capture, and fusion modes with scoped
  guards.
- [ ] Reject storage access, `item()`, and data-dependent control flow for
  symbolic tensors with actionable errors.

### Operation semantics and primitive execution

- [ ] Add an operation-definition registry containing inference,
  decomposition, differentiation, fusion, and alias behavior.
- [ ] Add canonical primitive decompositions for SwiGLU, RMSNorm,
  residual-add plus RMSNorm, RMSNorm plus RoPE, and attention.
- [ ] Prevent recursive redispatch while executing or recording a
  decomposition.
- [ ] Implement `FusionMode::Disabled` and verify it executes only primitives.
- [ ] Implement `FusionMode::Handwritten` through the existing `try_*` hooks.
- [ ] Add fusion statistics and strict no-fallback enforcement.

### Graph capture and tensor IR

- [ ] Add graph, value, operation, parameter, and symbolic-shape types.
- [ ] Import eager module parameters into a capture as named graph parameters.
- [ ] Route top-level operations and view operations through the active capture
  context.
- [ ] Add explicit `compile(callable, examples, options)` and
  `compile(module, examples, options)` entry points.
- [ ] Validate graph ownership and reject mixing unrelated symbolic graphs.
- [ ] Implement shape, dtype, device, layout, and alias inference.
- [ ] Replay captured graphs eagerly as the first executable-plan backend.
- [ ] Add graph dumps and structural capture tests.

### Fusion and kernel IR

- [ ] Classify elementwise, reduction, view, library, and opaque operations.
- [ ] Implement conservative producer-consumer elementwise fusion.
- [ ] Fold views into consumer indexing where layouts permit.
- [ ] Add legality checks for aliases, external uses, layouts, and devices.
- [ ] Add a scalar-expression kernel IR with grid-stride scheduling.
- [ ] Lower broadcasted Float32 elementwise groups to kernel IR.
- [ ] Add Float16 and BFloat16 loads, stores, and conversions.
- [ ] Add multiple-output generated kernels.
- [ ] Add block-reduction, shared-memory, barrier, and broadcast IR.

### CUDA runtime compilation

- [ ] Add an optional NVRTC build dependency and runtime capability check.
- [ ] Emit self-contained CUDA C++ from elementwise kernel IR.
- [ ] Compile generated source to PTX and preserve source and compiler logs on
  failure.
- [ ] Load PTX through the CUDA Driver API and own module lifetimes safely.
- [ ] Launch generated kernels on the shared `CudaExecutionContext` stream.
- [ ] Create canonical IR hashes and architecture-aware cache keys.
- [ ] Add a thread-safe in-memory compiled-kernel cache.
- [ ] Test cache hits, concurrent compilation, compilation failures, and module
  cleanup.

### Qwen3 fusion validation

- [ ] Generate SwiGLU from primitive `silu` and multiply operations.
- [ ] Compare disabled, handwritten, and generated SwiGLU over boundary sizes
  and supported dtypes.
- [ ] Require generated execution and assert fusion statistics in tests.
- [ ] Run a deterministic tiny Qwen3 MLP in all three fusion modes.
- [ ] Generate residual-add plus RMSNorm with both required outputs.
- [ ] Compare generated reductions using documented absolute and relative
  tolerances.
- [ ] Run the full deterministic tiny Qwen3 model in all fusion modes and
  compare logits and selected tokens.
- [ ] Add a short checkpoint-backed greedy decoding comparison.

### GEMM, layout, and memory optimization

- [ ] Add backend selection for mixed-input and mixed-output GEMMs.
- [ ] Evaluate cuBLASLt or CUTLASS epilogues for residual addition, activation,
  and output conversion.
- [ ] Recognize and pack QKV projections.
- [ ] Recognize and pack gate/up projections.
- [ ] Propagate layouts through reshape, transpose, permute, and GEMM consumers.
- [ ] Add value lifetime analysis and reusable intermediate-buffer slots.
- [ ] Report materializations, allocated bytes, and peak temporary memory in
  compilation diagnostics.
- [ ] Add performance gates comparing compiled execution with current
  handwritten Qwen3 execution.

### Eager autograd foundation

- [ ] Add `requires_grad`, leaf state, gradient edges, and leaf accumulation.
- [ ] Implement `NoGradGuard` and `InferenceModeGuard` semantics.
- [ ] Implement a backward engine with dependency counting and one-stream CUDA
  scheduling.
- [ ] Add eager backward rules for add, multiply, matmul, sum, mean, unary
  operations, reshape, transpose, and broadcast.
- [ ] Implement `sum_to_shape` and the initial view inverse operations.
- [ ] Add saved tensors with version validation.
- [ ] Test repeated backward, retained graphs, detached tensors, views, and
  illegal mutation.

### Compiled autograd

- [ ] Build backward tensor IR using registered differentiation rules.
- [ ] Determine forward values and metadata required by backward.
- [ ] Produce paired forward and backward executable plans.
- [ ] Bind each forward invocation to its saved-value backward context.
- [ ] Fuse elementwise backward regions and generate multiple gradient outputs.
- [ ] Plan saved-value and gradient-buffer lifetimes jointly.
- [ ] Add compiled gradient accumulation for shared graph inputs and parameters.
- [ ] Add training state, gradient requirements, and numeric policy to
  specialization and cache keys.
- [ ] Differential-test eager and compiled forward values and gradients.
- [ ] Add gradcheck-style finite-difference tests for generated operations.

### Later work

- [ ] Add save-versus-recompute analysis and activation checkpointing.
- [ ] Functionalize supported in-place operations during capture.
- [ ] Add higher-order gradient support deliberately rather than implicitly.
- [ ] Add mixed-precision training, FP32 master weights, and loss scaling.
- [ ] Extend the stable kernel IR to Metal.
- [ ] Select an interpreted, template-based, or JIT CPU kernel-IR backend.
- [ ] Reevaluate LLVM or MLIR only after the internal IR requirements stabilize.

## Milestone gates

The first generated-inference milestone is complete when a captured SwiGLU
region compiles through NVRTC, executes without fallback, hits the compilation
cache on its second invocation, and matches eager and handwritten results.

The first generated-reduction milestone is complete when residual-add plus
RMSNorm produces both outputs correctly across boundary sizes and runs inside a
tiny Qwen3 model.

The first compiled-autograd milestone is complete when a captured elementwise
function produces forward values and first-order gradients matching eager
autograd, with the backward elementwise region executed as a generated kernel.

The Qwen compiler milestone is complete when a deterministic tiny Qwen3 model
runs in disabled, handwritten, and generated modes with matching selected
tokens, no unintended eager fallbacks, bounded numerical error, and recorded
compile-time, memory, and steady-state performance results.

This sequence provides a complete capture-to-execution vertical slice before
adding complex reductions or adopting a heavyweight compiler framework. An
internal IR plus NVRTC is the preferred starting point. LLVM or MLIR can be
considered later if the stabilized IR, optimization requirements, or multi-
backend needs justify the dependency and integration cost.
