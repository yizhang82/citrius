# Transformer Implementation Plan

This document tracks the work required to build a forward-only Transformer on
Citrius. The first target is correct Float32 inference on CPU. CUDA and Metal
should use the same public APIs and will be enabled phase by phase as their
required kernels become available.

## Current foundation

Citrius currently provides:

- contiguous tensors and CPU, CUDA, and Metal storage;
- Float32 elementwise add and subtract;
- two-dimensional matrix multiplication;
- a forward-only `nn::Module` with parameters and child modules;
- `nn::Linear` with optional bias and leading-dimension preservation.

The current `Linear` implementation stages its weight transpose and expanded
bias through CPU memory on every forward call. This is functionally correct,
but it must be replaced with native tensor views, transpose, and broadcasting
before building larger models.

## Scope

The initial Transformer milestone includes:

- forward inference only;
- Float32 tensors;
- batched input shaped `[batch, sequence, embedding]`;
- causal self-attention and optional padding masks;
- token and positional embeddings;
- multi-head attention, feed-forward layers, residual connections, and layer
  normalization;
- deterministic CPU reference behavior and backend parity tests.

The initial milestone does not include:

- autograd or backward passes;
- optimizers or training;
- mixed precision or quantization;
- distributed execution;
- fused attention kernels;
- model checkpoint compatibility with PyTorch;
- key/value caching for autoregressive generation.

## Phase 1: Tensor shape and view operations

Status: CPU reference implementation complete. Reshape, view, flatten, squeeze,
and unsqueeze share storage. Because tensors do not yet carry stride metadata,
transpose, permute, split, chunk, and concat currently materialize contiguous
results. CUDA and Metal use host staging for these layout-changing operations
until native backend kernels or stride-aware views are added.

Implement the metadata and data-layout operations needed to form attention
heads:

- `reshape` and `view`;
- `flatten`;
- `unsqueeze` and `squeeze`;
- `transpose` for two dimensions;
- general `permute`;
- `contiguous`;
- `split` or `chunk`;
- `concat`.

Views share storage whenever the current contiguous layout permits it.
Operations that require a different physical layout currently return a
materialized contiguous tensor.
Shape, stride, device, dtype, and invalid-dimension behavior need unit tests.

Acceptance criteria:

- `[B, S, E]` can be transformed into `[B, H, S, D]` and back, where
  `E == H * D`;
- transposed and permuted tensors preserve values;
- invalid reshapes and dimension indices produce clear exceptions;
- `Linear` no longer copies its weight to CPU merely to transpose it.

## Phase 2: Broadcasting and elementwise math

Status: CPU reference implementation complete. Broadcasted add, subtract,
multiply, divide, maximum, scalar overloads, exponential, square root, power,
Bool tensor construction, and masked fill are available. Operations without a
native backend kernel currently stage through CPU while preserving the requested
output device.

Extend elementwise dispatch with NumPy/PyTorch-style trailing-dimension
broadcasting. Add:

- broadcasted `add`, `sub`, `mul`, and `div`;
- tensor-scalar overloads;
- `exp`, `sqrt`, and `pow`;
- `maximum` or an equivalent primitive;
- `masked_fill` for boolean masks.

Acceptance criteria:

- bias shaped `[E]` broadcasts over `[B, S, E]`;
- attention masks broadcast over batch and head dimensions;
- incompatible shapes fail before backend execution;
- CPU, CUDA, and Metal produce equivalent results where enabled;
- `Linear` uses native bias broadcasting rather than materializing repeated
  bias values.

## Phase 3: Reductions and softmax

Status: CPU reference implementation complete. Sum, mean, maximum, and
population variance support full, single-dimension, and multi-dimension
reductions with negative dimensions and optional `keepdim`. Numerically stable
softmax is available through `nn::functional::softmax`.

Implement dimension-aware reductions:

- `sum`;
- `mean`;
- `max` with values and, when needed, indices;
- variance or the primitives required to compute it;
- optional `keepdim` behavior;
- negative dimension indices.

Build numerically stable `nn::functional::softmax` using max subtraction before
exponentiation.

Acceptance criteria:

- softmax rows sum to approximately one;
- large input values do not overflow in ordinary Float32 use;
- masked attention positions receive zero probability;
- reduction tests cover leading, middle, trailing, and negative dimensions.

## Phase 4: Batched matrix multiplication

Status: CPU implementation complete. The public `matmul` entry point routes 2D
inputs to device `matmul` and higher-rank inputs to device `batched_matmul`.
Batched execution broadcasts leading dimensions and writes directly into CPU
tensor storage. CUDA supports a tiled 16x16 reference kernel, cuBLAS pointer-array
batched GEMM, and CUTLASS row-major GEMM launches. Metal remains deferred and
throws an explicit error without host fallback.

Generalize `matmul` beyond two dimensions:

```text
[B, M, K] x [B, K, N] -> [B, M, N]
```

Batch dimensions should eventually broadcast, allowing attention operands such
as `[B, H, S, D]`. The CPU implementation remains the correctness reference;
backend-specific optimized implementations can follow without changing the
public API.

Acceptance criteria:

- query-key multiplication produces `[B, H, S, S]`;
- attention-value multiplication produces `[B, H, S, D]`;
- batch broadcasting is tested explicitly;
- dimension and inner-size mismatches produce clear exceptions.

## Phase 5: Transformer building blocks

Status: In progress. ReLU, tanh-approximate GELU, functional layer normalization,
`nn::LayerNorm`, inference-only `nn::Dropout`, and registered sequential
`nn::ModuleList` are complete and covered by CPU reference tests. Integer tensor
construction, indexed gather, and `nn::Embedding` remain. SiLU is deferred until
a model requires it.

Add the following functional operations and modules:

- `nn::functional::relu`;
- `nn::functional::gelu`;
- `nn::functional::silu` when needed;
- `nn::functional::layer_norm`;
- `nn::LayerNorm` with learnable scale and bias;
- `nn::Embedding` with integer token lookup;
- `nn::Dropout`, implemented as an identity operation for inference initially;
- `nn::ModuleList` or an equivalent registered child-module container.

Embedding requires integer input tensors and indexed gather support. This work
must define how vectors of integer data are constructed, transferred, and
printed across devices.

Acceptance criteria:

- LayerNorm matches a trusted reference within a documented Float32 tolerance;
- Embedding returns `[B, S, E]` for token IDs shaped `[B, S]`;
- GELU matches the selected exact or approximate formula consistently;
- all learnable tensors appear in recursive `named_parameters()` traversal.

## Phase 6: Multi-head self-attention

Implement `nn::MultiHeadAttention` from existing primitives:

1. project input through query, key, and value `Linear` modules;
2. reshape and permute projections into attention heads;
3. compute `Q @ transpose(K) / sqrt(head_dimension)`;
4. apply causal and optional padding masks;
5. apply softmax over the key dimension;
6. compute attention weights times `V`;
7. merge heads and apply the output projection.

Validate that `embedding_dimension % number_of_heads == 0` during construction.
The initial implementation should favor clarity and composability over fusion.

Acceptance criteria:

- output shape equals input shape;
- causal masking prevents a position from observing future tokens;
- one-head and multi-head results match a trusted reference;
- child projections and their parameters are registered with stable names;
- CPU results are deterministic for fixed parameters and inputs.

## Phase 7: Transformer block

Implement a pre-normalization Transformer block:

```text
x = x + attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

The feed-forward network is:

```text
Linear(E, F) -> GELU -> Linear(F, E)
```

Dropout locations should be represented in the module structure even while
dropout is an inference-only identity operation.

Acceptance criteria:

- the block preserves `[B, S, E]`;
- residual paths and normalization order are covered by deterministic tests;
- all nested parameters have predictable qualified names;
- multiple blocks can be stored and executed in order.

## Phase 8: End-to-end Transformer

Build a small decoder-only Transformer containing:

- token embeddings;
- learned positional embeddings initially;
- a sequence of Transformer blocks;
- final layer normalization;
- output projection to vocabulary logits;
- causal-mask construction.

Add an example that accepts a fixed token sequence and prints logits or greedy
next-token IDs. Keep model dimensions small enough for quick CPU execution.

Acceptance criteria:

- input `[B, S]` token IDs produce `[B, S, V]` logits;
- a tiny model matches a trusted reference with imported fixed parameters;
- the example runs without arguments on CPU and supports the existing backend
  flags;
- invalid token IDs, sequence lengths, and model dimensions fail clearly.

## Phase 9: Backend parity and optimization

After the reference model is correct:

- remove accidental host-device transfers from forward paths;
- implement missing CUDA and Metal kernels;
- benchmark attention, projection, normalization, and end-to-end inference;
- cache reusable causal masks;
- consider fused QKV projection;
- consider fused scaled dot-product attention;
- add lower-precision dtypes only with numerical tests;
- add key/value caching for token-by-token generation.

Optimizations must preserve the composable reference implementation so that
backend kernels can always be checked against it.

## Testing strategy

Each phase should include:

- focused unit tests for shapes, values, errors, and parameter registration;
- deterministic reference inputs and parameters;
- CPU reference tests required in every build;
- conditional CUDA and Metal parity tests;
- tolerances documented per operation rather than applied globally;
- at least one integration test combining the newly completed phases.

End-to-end reference data should be generated once from a small, explicitly
configured model and stored as test fixtures. Tests should not require PyTorch
at build or runtime.

## Definition of done

The initial Transformer milestone is complete when a decoder-only model can run
a deterministic Float32 forward pass on CPU, produce logits matching a trusted
reference, expose all nested parameters through `named_parameters()`, and run
through a documented example. CUDA and Metal parity may follow incrementally,
but unsupported backend paths must fail explicitly rather than silently copy or
fall back to CPU.
