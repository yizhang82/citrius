# Citrius Development Plan

This document tracks Citrius through four major milestones:

1. **Transformer prototype — complete.** A correct forward-only Transformer,
   Qwen3-0.6B model, checkpoint loader, native tokenizer, and greedy decoder.
2. **Qwen3 inference performance — current.** Make batch-one interactive Qwen3
   inference efficient on CUDA without losing the composable reference path.
3. **Training.** Add automatic differentiation, optimizers, mixed-precision
   training, checkpointing, and distributed execution.
4. **Production serving.** Add continuous batching, paged caches, scheduling,
   observability, fault handling, and stable serving APIs.

CUDA and Metal use the same public tensor APIs and are enabled incrementally as
their required kernels become available.

## Phase 1: Transformer prototype

Status: **Complete.**

The sections below preserve the implementation sequence and acceptance criteria
used to reach the first working decoder prototype.

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
- Qwen3-0.6B architecture and Hugging Face safetensors compatibility;
- native Qwen3 byte-level BPE tokenization and chat formatting;
- checkpoint-backed greedy decoding;
- deterministic CPU reference behavior and backend parity tests.

The initial milestone does not include:

- autograd or backward passes;
- optimizers or training;
- mixed precision or quantization;
- distributed execution;
- fused attention kernels;
- key/value caching for autoregressive generation.

### 1.1 Tensor shape and view operations

Status: CPU reference implementation complete. Reshape, view, flatten, squeeze,
and unsqueeze share storage. Because tensors do not yet carry stride metadata,
transpose, permute, split, chunk, and concat currently materialize contiguous
results. CUDA and Metal use host staging for these layout-changing operations
until native backend kernels or stride-aware views are added.

Implement the metadata and data-layout operations needed to form attention
heads:

- per-dimension strides, a storage offset, contiguity state, and shared
  allocation ownership;
- `reshape` and `view`;
- `flatten`;
- `unsqueeze` and `squeeze`;
- `transpose` for two dimensions;
- general `permute`;
- `contiguous`, backed by a native strided-to-contiguous CUDA kernel rather than
  host staging;
- `split` or `chunk`;
- `concat`.

Views share storage whenever their layout can be represented by shape, strides,
and a storage offset. Operations requiring a different physical layout return
an explicitly materialized contiguous tensor on the tensor's current device.
Shape, stride, device, dtype, and invalid-dimension behavior need unit tests.

Acceptance criteria:

- `[B, S, E]` can be transformed into `[B, H, S, D]` and back, where
  `E == H * D`;
- transposed and permuted tensors preserve values;
- invalid reshapes and dimension indices produce clear exceptions;
- `Linear` no longer copies its weight to CPU merely to transpose it.

### 1.2 Broadcasting and elementwise math

Status: CPU reference implementation complete. Broadcasted add, subtract,
multiply, divide, maximum, scalar overloads, exponential, square root, power,
Bool tensor construction, and masked fill are available. CUDA has native
broadcasted add, subtract, multiply, divide, maximum, scalar arithmetic,
exponential, square root, scalar-exponent power, and broadcasted masked fill.
Other operations without a native backend kernel currently stage through CPU
while preserving the requested output device.

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

### 1.3 Reductions and softmax

Status: CPU reference implementation complete. Sum, mean, maximum, and
population variance support full, single-dimension, and multi-dimension
reductions with negative dimensions and optional `keepdim`. CUDA has a native
general reference kernel for the same reduction variants. Numerically stable
softmax is available through `nn::functional::softmax`. CUDA softmax composes
entirely from native CUDA primitives and has device parity coverage; a fused
implementation remains a later optimization.

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

### 1.4 Batched matrix multiplication

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

### 1.5 Transformer building blocks

Status: CPU reference implementation complete. ReLU, tanh-approximate GELU,
functional layer normalization, `nn::LayerNorm`, inference-only `nn::Dropout`,
registered sequential `nn::ModuleList`, Int64 tensor construction, indexed row
gather, and `nn::Embedding` are covered by CPU reference tests. Indexed gather
uses a native CUDA kernel with device parity and bounds-error coverage; other
non-CPU backends still stage through CPU. SiLU is deferred until a model requires
it. LayerNorm composes entirely from native CUDA primitives and has device parity
coverage; a fused implementation remains a later optimization.

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

### 1.6 Multi-head self-attention

Readiness: The CPU reference primitives and modules required by this phase are
complete. CUDA is functionally capable of executing the same composition, but
some consumers still materialize transpose and permute views. Indexed Embedding
lookup remains on CUDA. Those performance gaps do not block implementation of
the reference Transformer.

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

### 1.7 Transformer block

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

### 1.8 End-to-end Transformer

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

### 1.9 Backend parity and prototype optimization

After the reference model is correct:

- remove accidental host-device transfers from forward paths;
- implement missing CUDA and Metal kernels;
- benchmark attention, projection, normalization, and end-to-end inference;
- cache reusable causal masks;
- consider fused QKV projection;
- consider fused scaled dot-product attention;
- add lower-precision dtypes only with numerical tests;
- add key/value caching for token-by-token generation.

Current CUDA optimization order:

1. add stride/storage-offset views plus native CUDA `contiguous()` so transpose,
   permute, split, slice, and final-position selection never require host
   staging;
2. implement native indexed gather to remove Embedding host staging;
3. make CUDA GEMM consume transpose/layout metadata or persistent prepacked
   weights so `Linear` never materializes a transposed weight per forward;
4. specialize contiguous last-dimension sum, mean, and maximum using one block
   per output row with warp shuffles and shared-memory reduction;
5. add a numerically stable cooperative variance kernel, retaining the current
   general arbitrary-dimension reduction kernel as a fallback;
6. fuse softmax's maximum, exponential, sum, and division sequence;
7. fuse LayerNorm's mean, variance, normalization, and affine sequence.

Each optimization step must add or extend `operations_benchmark` CPU/CUDA
comparisons and retain parity tests against the composable CPU reference.

Optimizations must preserve the composable reference implementation so that
backend kernels can always be checked against it.

## Phase 2: Qwen3 inference performance

Status: Qwen3-0.6B model construction, Hugging Face safetensors loading, native
byte-level BPE tokenization, chat prompt formatting, and greedy decoding are
implemented. The checkpoint-free `qwen3_decoding_benchmark` generates exactly
10 tokens and reports TTFT, end-to-end tokens per second, and post-first-token
tokens per second.

The initial nine-token-prompt baseline is:

| Backend | TTFT | End-to-end throughput | Total time for 10 tokens |
| --- | ---: | ---: | ---: |
| CPU | 17,081.783 ms | 0.059 tokens/s | 170,325.068 ms |
| CUDA | 18,411.104 ms | 0.054 tokens/s | 185,485.003 ms |

These measurements use initialized weights without loading a checkpoint. The
current decoder uses Float32 tensors, synchronous CUDA kernels, repeated
allocation, full-sequence recomputation, CPU argmax, and no KV cache. The CUDA
result is therefore an execution-overhead baseline rather than an optimized GPU
inference result.

Implement the following work items in order. Each step must retain a reference
path and extend the decoding benchmark where applicable.

### 2.1 GPU argmax and final-position logits

- add public dimension-aware `argmax` returning Int64 indices;
- implement the CPU correctness reference;
- implement CUDA reduction for non-power-of-two vocabulary sizes;
- preserve first-index tie behavior;
- select or produce only the final sequence position's logits;
- copy only the resulting token scalar to the host;
- benchmark the Qwen3 vocabulary size of 151,936 explicitly.

Acceptance criteria:

- CUDA results exactly match the CPU reference;
- generated token IDs are unchanged;
- greedy decoding no longer transfers `[sequence, vocabulary]` logits to CPU;
- ties, negative values, and irregular reduction sizes have unit coverage.

### 2.2 Stream-ordered execution and tensor materialization

- complete the stride/storage-offset view foundation required by inference,
  including metadata-only transpose, permute, split, slice, and last-position
  selection;
- implement native CUDA strided-to-contiguous materialization for consumers that
  cannot operate on a view directly;
- define a CUDA execution stream owned by each device;
- make operations enqueue work and return potentially pending tensors;
- define `to(cpu)`, scalar `item`, printing, and host pointer access as explicit
  materialization boundaries;
- use stream events for cross-stream and transfer dependencies;
- guarantee temporary storage lifetime through the final queued consumer;
- remove per-operation `cudaDeviceSynchronize()` calls after the contract is in
  place;
- synchronize only when host code observes a result.

Acceptance criteria:

- Qwen3 layout operations and final-position selection do not stage through CPU;
- CUDA `contiguous()` correctly materializes representative transposed, sliced,
  and offset views;
- ordinary same-stream tensor composition does not synchronize the host;
- greedy generation has at most one required host materialization per token;
- asynchronous execution passes correctness and storage-lifetime stress tests.

### 2.3 CUDA allocation and workspace reuse

- add a stream-aware caching allocator for general tensor storage;
- reuse freed blocks without globally synchronizing the device;
- add output-buffer variants for hot operations;
- preallocate stable decoder temporaries and attention workspaces;
- record allocation count and peak device memory in the benchmark.

Acceptance criteria:

- steady-state token decoding performs no `cudaMalloc` or `cudaFree` calls;
- decode buffer addresses remain stable enough for later CUDA Graph capture.

### 2.4 Native BF16 and FP16 inference

- preserve BF16 checkpoint weights instead of expanding them to Float32;
- implement BF16 and FP16 tensor storage, transfer, and operation dispatch;
- use FP32 accumulation for reductions and other numerically sensitive paths;
- route projection GEMMs through Tensor Core-capable cuBLAS configurations;
- pass transpose/layout information directly to cuBLAS/cuBLASLt or retain
  persistent prepacked weights instead of transposing weights per forward;
- verify layouts do not introduce per-forward transposes or copies.

Acceptance criteria:

- Qwen3 BF16 weights occupy approximately their checkpoint size on-device;
- BF16/FP16 logits remain within documented tolerances of the Float32 reference;
- profiling confirms Tensor Core execution for model projections.

### 2.5 Prefill/decode split and KV cache

- introduce separate `prefill(input_ids)` and `decode(next_token, cache)` APIs;
- preallocate per-layer K/V storage with contiguous head dimensions;
- support Qwen3 grouped-query attention with 16 query heads and 8 KV heads;
- map query-head groups directly to K/V heads without materializing repeated K/V
  tensors through split, concat, or repeat operations;
- append one K/V position per generated token;
- precompute and retain RoPE sin/cos tables on CUDA, or generate and apply them
  in a native CUDA kernel, using the absolute cached position during decode;
- run the LM head only for the final hidden state;
- stop recomputing previous tokens after prompt prefill.

Acceptance criteria:

- cached and full-sequence logits agree within the selected dtype tolerance;
- steady-state inference performs no per-layer host construction or transfer of
  RoPE tables;
- grouped-query attention does not allocate physically repeated K/V heads;
- post-first-token work processes one query token;
- short-context decode latency no longer grows materially on every generated
  token.

### 2.6 Flash Attention prefill

- add a fused scaled-dot-product-attention device interface;
- implement tiled causal attention with online, numerically stable softmax;
- support BF16/FP16 inputs, FP32 accumulation, head dimension 128, and GQA;
- avoid materializing `[heads, sequence, sequence]` attention scores;
- handle non-aligned sequence-length tails;
- benchmark prompt lengths 9, 128, 512, and 2,048.

Acceptance criteria:

- the fused kernel matches the composable reference attention implementation;
- attention temporary memory grows linearly rather than quadratically with
  sequence length;
- TTFT improves for medium and long prompts.

### 2.7 Flash decoding over the KV cache

- implement a separate single-query attention kernel for `query_length == 1`;
- stream K/V cache blocks through an online softmax reduction;
- map two query heads to each Qwen3 KV head directly;
- avoid materializing attention scores;
- design the cache layout for coalesced reads and future paged allocation;
- benchmark cache lengths 128, 512, 2,048, 8,192, and 32,768.

Acceptance criteria:

- cached decode attention matches the reference implementation;
- latency and effective memory bandwidth are reported by cache length;
- the kernel remains numerically stable at the maximum supported context.

### 2.8 Decoder kernel fusion

- fuse residual addition with RMSNorm where beneficial;
- fuse Q/K normalization and RoPE application;
- fuse SiLU with gated-MLP multiplication;
- fuse K/V transformation with cache writes where practical;
- consider packed or grouped Q/K/V projection after individual GEMMs are
  profiled;
- preserve unfused implementations as correctness references.

Acceptance criteria:

- profiling shows fewer launches and intermediate allocations per layer;
- fused and unfused decoder logits agree within documented tolerances.

### 2.9 GPU sampling and device-resident generation

- generalize GPU argmax into temperature, top-k, and top-p sampling;
- maintain token and context storage on the device;
- transfer generated token IDs only when the caller requests them;
- allow batched token materialization for non-interactive generation.

Acceptance criteria:

- greedy mode remains deterministic;
- seeded sampling is reproducible;
- generation can execute multiple steps without a full tensor host transfer.

### 2.10 CUDA Graph capture

- stabilize decode shapes, storage addresses, and workspaces;
- capture the repeated one-token decode path;
- update token input, position, and logical cache length between replays;
- measure capture cost separately from steady-state replay.

Acceptance criteria:

- graph replay produces the same tokens as the ordinary CUDA path;
- steady-state profiling shows substantially reduced launch overhead.

### 2.11 Performance measurement and gates

- add NVTX ranges for prefill, each decoder layer, attention, MLP, LM head,
  argmax/sampling, allocation, and materialization;
- report cold and warm TTFT, prefill tokens/s, post-first-token tokens/s,
  median/worst token latency, transfer bytes, allocation count, and peak memory;
- test prompt lengths 1, 9, 128, 512, and 2,048 with a fixed 10-token decode;
- test Float32, BF16, and FP16 where supported;
- emit a machine-readable result alongside human-readable output;
- track performance regressions separately from functional unit tests.

Performance milestones for a desktop RTX 5070 Ti are directional until the
preceding execution and dtype work establishes trustworthy GPU measurements:

1. GPU argmax removes the full-logits host transfer;
2. streaming execution and allocator reuse remove synchronization and allocation
   collapse;
3. BF16/FP16 Tensor Core execution plus KV caching reaches useful interactive
   throughput;
4. fused decode attention and decoder kernels target 100--250 tokens/s;
5. CUDA Graphs and mature kernels target 250--450 tokens/s for short-context,
   batch-one Qwen3-0.6B decoding.

## Phase 3: Training

Status: **Planned after the inference-performance milestone.**

Training must build on the same tensor, dtype, allocator, and execution-stream
contracts established for inference. It must not introduce a second incompatible
execution model.

### 3.1 Tensor views, indexing, and data manipulation

Extend the stride-aware inference view foundation into a complete PyTorch-style
indexing model before autograd. Training data
processing needs richer tensor manipulation than inference alone: token-window
creation, shuffled gathers, shifted input/target pairs, batching, masking,
padding, and packing should not require handwritten backend-specific code or
unnecessary full-tensor copies.

The shared inference/training view model includes:

- per-dimension strides,
- a storage offset,
- explicit contiguity and view information, and
- shared ownership of the underlying allocation so views cannot outlive storage.

The public API should keep indexing tensor-valued and scalar extraction explicit:

```cpp
using namespace citrius::indexing;

Tensor row = tensor.index({2, Slice()});
Tensor tail = tensor.index({Ellipsis, Slice(1, None)});
Tensor expanded = tensor.index({None, Ellipsis});
float value = tensor.index({2, 3}).item<float>();
```

Provide `Slice`, `None`, `Ellipsis`, and `TensorIndex` in
`citrius::indexing`, alongside specialized operations such as `select`, `slice`,
`narrow`, and `contiguous`. Basic integer and slice indexing should return views
that alias the original storage. Boolean-mask and tensor-based advanced indexing
should return newly allocated tensors, matching PyTorch's basic-versus-advanced
indexing distinction. Mutation through indexed regions should be introduced
explicitly through `index_put_` rather than hidden proxy assignment semantics.

Implement this in stages:

1. Reuse and validate the stride/storage-offset metadata and view lifetime model
   established for inference.
2. Extend `select`, `slice`, and `narrow` with negative indices and general
   `Tensor::index` composition.
3. Add `None`, `Ellipsis`, and broader non-contiguous kernel handling.
4. Add `index_put_`, boolean-mask indexing, and tensor advanced indexing.
5. Use the public API in data-pipeline utilities for batching, token windows,
   masks, and shifted training targets.

Backend kernels must either consume shape, stride, and storage-offset metadata
correctly or request an explicit contiguous materialization. They must not
silently stage through the host. Bounds, invalid slices, unsupported index
combinations, and illegal writes to broadcast or overlapping views need clear
errors.

This phase is complete when:

- basic indexing produces storage-aliasing views with correct values and
  negative-index behavior;
- advanced indexing produces independent results with CPU/CUDA parity;
- data-pipeline code can construct batches, masks, token windows, and shifted
  targets without mandatory full-tensor copies or host staging; and
- scalar reads remain explicit as `tensor.index(...).item<T>()`, copying only
  the requested value when the tensor is on CUDA.

### 3.2 Automatic differentiation

- define gradient ownership, leaf tensors, and `requires_grad` semantics;
- record backward graphs for tensor operations and modules;
- implement reverse topological execution and gradient accumulation;
- detect invalid in-place mutation using tensor versioning;
- support `detach`, inference/no-grad scopes, and graph release;
- add numerical gradient checks for every differentiable operation.

### 3.3 Backward kernels and trainable modules

- implement backward operations for elementwise math, reductions, views,
  broadcasting, matmul, embeddings, normalization, and attention;
- add parameter and buffer traversal needed by training loops;
- implement training-mode dropout with reproducible device RNG state;
- validate complete Transformer-block gradients against a trusted reference.

### 3.4 Optimizers and gradient processing

- implement SGD and AdamW as initial optimizers;
- add parameter groups, weight decay, and learning-rate schedules;
- add gradient zeroing, clipping, and global norm calculation;
- define optimizer state serialization and restoration.

### 3.5 Mixed-precision training

- support BF16 and FP16 forward/backward execution with FP32 master state where
  required;
- implement automatic loss scaling for FP16;
- define autocast policies per operation;
- test overflow detection, skipped updates, and convergence against Float32.

### 3.6 Training memory efficiency

- add activation checkpointing/recomputation;
- reuse forward and backward workspaces safely;
- support gradient accumulation over microbatches;
- profile activation, gradient, optimizer, and temporary memory independently.

### 3.7 Data and checkpoint lifecycle

- define dataset, sampler, batch, and asynchronous input-pipeline interfaces;
- serialize model, optimizer, scheduler, RNG, and training-step state;
- support resumable sharded checkpoints for larger models;
- preserve Hugging Face-compatible parameter import/export where practical.

### 3.8 Distributed training

- implement data-parallel gradient synchronization;
- add process-group, rank, device, and collective abstractions;
- overlap gradient communication with backward computation;
- add tensor/sequence parallelism only after single-device training is stable;
- test deterministic restart and failure reporting across ranks.

Phase 3 acceptance criteria:

- training batches, masks, token windows, and shifted targets are constructed
  through the public tensor indexing/view API without mandatory host staging;
- a tiny Transformer trains end-to-end and reduces a known loss;
- gradients and optimizer updates match a trusted reference;
- BF16 training remains stable on a representative workload;
- interrupted training resumes without losing optimizer or RNG state;
- multi-GPU data-parallel results match single-GPU training within documented
  tolerances.

## Phase 4: Production serving

Status: **Planned after training fundamentals and a stable inference engine.**

Production serving turns the optimized model path into a multi-request system.
Latency, throughput, memory isolation, cancellation, and operational behavior
become public contracts rather than benchmark details.

### 4.1 Stable inference engine API

- separate model loading, tokenizer/chat formatting, prefill, decode, sampling,
  and request lifecycle APIs;
- define versioned configuration and error types;
- support streaming token callbacks and request cancellation;
- make model and tokenizer compatibility checks explicit at load time.

### 4.2 Continuous batching and scheduling

- dynamically combine prefill and decode work from concurrent requests;
- implement fairness, priorities, deadlines, and maximum queue limits;
- separate prefill-heavy and decode-heavy scheduling decisions;
- measure time-to-first-token and inter-token latency under concurrency.

### 4.3 Paged KV-cache management

- replace fixed per-request cache allocations with reusable pages;
- support efficient request growth, completion, and cancellation;
- add prefix caching and shared prompt pages;
- enforce memory budgets and predictable out-of-memory behavior;
- compact or reclaim pages without corrupting in-flight requests.

### 4.4 Quantized deployment

- add weight-only INT8 and INT4 formats after BF16 serving is stable;
- add FP8 paths where hardware and numerical behavior justify them;
- provide calibrated accuracy and throughput comparisons;
- retain unquantized reference execution for validation.

### 4.5 Serving protocol

- provide an HTTP/gRPC boundary with streaming responses;
- support a documented chat/completions schema;
- expose tokenizer, generation, stop-sequence, and sampling controls;
- implement request-size, context-length, and output-length limits;
- keep the core inference engine independent of a particular web framework.

### 4.6 Observability and performance controls

- publish request counts, queue time, TTFT, inter-token latency, tokens/s,
  batch size, cache occupancy, allocation failures, and cancellation metrics;
- add structured logs and distributed traces without logging prompt contents by
  default;
- expose health, readiness, and model-loading state;
- define service-level objectives and overload behavior.

### 4.7 Reliability, security, and deployment

- isolate malformed requests and failed generations;
- implement graceful shutdown and draining;
- validate model artifacts and restrict filesystem/network access;
- add reproducible container builds and GPU compatibility checks;
- test restart, cancellation, timeout, overload, and device-failure behavior;
- provide rolling-update and rollback procedures.

Phase 4 acceptance criteria:

- concurrent requests stream correct tokens without cache contamination;
- cancellation promptly releases scheduler and KV-cache resources;
- latency and throughput remain within defined targets under sustained load;
- overload produces bounded queues and explicit errors rather than process
  failure;
- deployment artifacts pass repeatable health, restart, and compatibility tests.

## Cross-phase testing strategy

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

## Milestone definitions of done

Phase 1 is complete: a decoder-only model runs a deterministic Float32 forward
pass on CPU, produces reference-tested logits, exposes nested parameters, loads
Qwen3 safetensors, tokenizes native chat prompts, and performs greedy decoding.

Phase 2 is complete when checkpoint-backed Qwen3-0.6B inference has correct
cached prefill/decode execution, BF16/FP16 Tensor Core paths, fused prefill and
decode attention, explicit materialization semantics, and stable benchmarked
interactive performance on CUDA.

Phase 3 is complete when Citrius can train, checkpoint, resume, and validate a
Transformer in mixed precision on one or more GPUs.

Phase 4 is complete when the inference engine can safely serve concurrent,
streaming requests with continuous batching, paged KV caches, bounded resource
usage, observability, cancellation, and reproducible deployment.

Unsupported backend paths must always fail explicitly rather than silently copy
or fall back to CPU.
