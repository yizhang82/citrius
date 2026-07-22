# Citrius Project Notes

Citrius is a small C++ tensor library skeleton inspired by PyTorch, currently focused on a clear tensor/storage/device split.

## Current Design

The public tensor API is in `include/tensor.h`.

Tensors can be allocated directly on an enabled backend or initialized from Float32 data:

```cpp
Tensor empty({2, 2}, DType::Float32, Device::cuda());
Tensor values(std::vector<float>{1, 2, 3, 4}, {2, 2}, Device::cuda());
auto cpu_values = values.to(Device::cpu());
```

`TensorFactory` in `include/tensor_factory.h` provides matching `empty` and
`from_vector` entry points. `Tensor::to` returns a shallow copy when the requested
device already matches and transfers through host memory otherwise.

Top-level `citrius::add`, `citrius::sub`, and `citrius::matmul` dispatch through the
input tensors' device and reject device mismatches. Tensor `+`, `-`, and `*` delegate
to those functions; `*` means matrix multiplication.

`Tensor::to_string()` and `operator<<` print tensor values and metadata, copying values
to CPU for display when necessary.

`Tensor` stores metadata directly:

- `Shape`
- `DType`
- `Device`
- `std::shared_ptr<ITensorStorage>`
- a `defined_` flag

Common types live in `include/types.h`:

- `DType`
- `DeviceType`
- `Device`
- `Shape`
- `Strides`
- `dtype_size`

Storage is abstracted through `ITensorStorage` in `include/impl/storage.h`. Concrete storage implementations advertise their backend through `ITensorStorage::type()`, then compatible devices check the type before casting to the concrete implementation.

Tensor copies are shallow with respect to storage: tensor metadata is copied by value, while the underlying `ITensorStorage` is shared through `std::shared_ptr`.

Basic indexing accepts integer indices, Python-style slices, ellipsis, and inserted
axes. It returns metadata-only views and supports negative indices and slice steps:

```cpp
using namespace citrius::indexing;
Tensor tail = tensor.index({Ellipsis, Slice(std::nullopt, std::nullopt, -1)});
Tensor row = tensor[-1];
```

Explicit deep copy is available through:

```cpp
Tensor copied = tensor.copy();
```

`Tensor::copy()` preserves metadata and calls `ITensorStorage::clone()` to allocate distinct backend storage with copied contents.

Current storage implementations:

- `impl::CpuMemTensorStorageImpl` in `include/impl/cpu_storage.h` and `src/devices/cpu/cpu_storage.cpp`
- `impl::MetalMemTensorStorageImpl` in `include/impl/metal_storage.h` and `src/devices/metal/metal_storage.mm`
- `impl::CudaMemTensorStorageImpl` in `include/impl/cuda_storage.h` and `src/devices/cuda/cuda_storage.cu`

Current device implementations:

- `impl::CpuDeviceImpl` in `include/impl/cpu_device.h` and `src/devices/cpu/cpu_device.cpp`
- `impl::MetalDeviceImpl` in `include/impl/metal_device.h` and `src/devices/metal/metal_device.mm`
- `impl::CudaDeviceImpl` in `include/impl/cuda_device.h` and `src/devices/cuda/cuda_device.cu`

The device layer owns execution. Storage owns memory only. Arithmetic should live on `IDevice` implementations, not in storage.

## Backend Behavior

`CpuDeviceImpl` supports:

- `empty`
- `add`
- `sub`
- `matmul`

`MetalDeviceImpl` supports the same operations when Metal is enabled.

`CudaDeviceImpl` supports the same operations when CUDA is enabled. It accepts a CUDA
device index in its constructor and preserves that index in tensors and storage.

Current operation constraints:

- `Float32` only
- most kernels require contiguous layout; cuBLAS and CUTLASS 2D matmul consume
  row-major and transposed column-major views directly
- add/sub require identical shapes
- matmul supports 2D tensors only
- broadcasting, strides, dtype dispatch, and autograd are not implemented yet

`CpuDeviceImpl` is the single-threaded reference backend.
`MultiThreadCpuDeviceImpl` uses `std::thread::hardware_concurrency()` workers by
default. Override the worker count with a positive integer in
`CITRIUS_CPU_THREADS`, or pass an explicit count to its constructor. Add/sub
split contiguous element ranges and matmul splits output rows across workers.

Device/storage compatibility is checked through `ITensorStorage::type()`.

CPU expects:

```cpp
TensorStorageType::CpuMemory
```

Metal expects:

```cpp
TensorStorageType::MetalMemory
```

CUDA expects `TensorStorageType::CudaMemory`. CUDA operations copy CPU inputs to the
selected CUDA device; explicit `ensure_storage` calls reject incompatible storage unless
`ConversionPolicy::CopyToDevice` is requested.

`MetalDeviceImpl::ensure_storage` can copy CPU storage to Metal storage when called with:

```cpp
ConversionPolicy::CopyToDevice
```

With the default `ConversionPolicy::Error`, incompatible storage is rejected.

## Metal Notes

Metal implementation files use `.mm` because they are Objective-C++ and call Objective-C Metal APIs such as:

```objc
id<MTLBuffer>
[encoder setBuffer:buffer offset:0 atIndex:0]
```

Public Metal headers hide Objective-C types behind private `Impl` classes so regular C++ files can include them.

Metal buffers currently use `MTLResourceStorageModeShared`, which keeps host/device transfers simple for early testing.

Metal kernels are embedded as source strings in `src/metal_device.mm` and compiled at runtime.

## Build

Default builds are CPU-only:

```bash
./build.sh
```

Clean CPU-only build:

```bash
./build.sh --clean
```

Build with Metal support:

```bash
./build.sh --metal
```

Build with CUDA Toolkit support:

```bash
./build.sh --cuda
```

CUTLASS is pinned as a Git submodule. Initialize dependencies after cloning with:

```bash
git submodule update --init --recursive
```

cuBLAS is the default implementation for top-level CUDA matmul. Build with the
reference CUDA kernel as the default instead using:

```bash
./build.sh --cuda-reference
```

The build default can be overridden at runtime without rebuilding:

```bash
CITRIUS_CUDA_BACKEND=cublas ./build/operations_benchmark --cuda
CITRIUS_CUDA_BACKEND=cutlass ./build/operations_benchmark --cuda
CITRIUS_CUDA_BACKEND=reference ./build/operations_benchmark --cuda
```

On Windows PowerShell, set `$env:CITRIUS_CUDA_BACKEND` to `cublas`, `cutlass`, or
`reference` before running the program. Explicit `CublasCudaDeviceImpl`,
`CutlassCudaDeviceImpl`, and `CudaDeviceImpl` instances select their corresponding
implementations directly.

Metal and CUDA may be enabled together with `./build.sh --metal --cuda` on a platform
that provides both toolchains.

Clean Metal build:

```bash
./build.sh --clean --metal
```

On Windows, build with MSVC using:

```bat
build.bat
build.bat --clean --config Debug
build.bat --clean --cuda
```

The default Windows configuration is `Release`. If GoogleTest is not already installed,
CMake downloads the pinned test dependency into the build directory automatically.

The scripts configure CMake with:

```bash
-DCITRIUS_ENABLE_METAL=OFF
```

or:

```bash
-DCITRIUS_ENABLE_METAL=ON
```

## Tests

Run CPU-only tests:

```bash
./test.sh
```

Run CPU-only tests from a clean build:

```bash
./test.sh --clean
```

On Windows, run the tests with:

```bat
build.bat
test.bat
test.bat --config Debug
build.bat --cuda
test.bat
```

Only `build.bat` configures CMake. `test.bat` incrementally builds the test target
from the existing configuration before running it.

Run the end-to-end Float32 matmul benchmark on Windows with:

```bat
build.bat --cuda
benchmark.bat operations --cpu
benchmark.bat operations --cuda
benchmark.bat operations --all
```

Run the isolated CUDA add/sub kernel tuning benchmark with:

```bat
benchmark.bat add-kernel
benchmark.bat add-kernel --size 4096 --iterations 100 --samples 5
```

Run the isolated CUDA matmul kernel tuning benchmark with:

```bat
benchmark.bat matmul-kernel
benchmark.bat matmul-kernel --size 2048 --iterations 20 --samples 5
```

Run the checkpoint-free Qwen3-0.6B greedy decoding benchmark, which generates
50 tokens by default, with:

```bat
benchmark.bat qwen3-decoding --cpu
benchmark.bat qwen3-decoding --cuda
```

Override the generated token count with `--tokens`:

```bat
benchmark.bat qwen3-decoding --cuda --tokens 10
```

Select reduced-precision projection and embedding weights with `--dtype`. On
CUDA, `float16` and `bfloat16` use cuBLAS tensor-core GEMMs with FP32
accumulation; normalization, attention, residuals, and logits remain FP32:

```bat
benchmark.bat qwen3-decoding --cuda --dtype bfloat16 --tokens 10
```

The benchmark excludes model construction, reports time to first token (TTFT),
end-to-end tokens per second, and post-first-token tokens per second. CUDA requires
an existing build configured with `build.bat --cuda`.

Run checkpoint-backed Qwen3-0.6B chat decoding from PowerShell with:

```powershell
$qwenSnapshot = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
build\Release\decoder_chat.exe `
  --checkpoint "$qwenSnapshot\model.safetensors" `
  --tokenizer "$qwenSnapshot" `
  --prompt "Who are you?" `
  --max-new-tokens 50 `
  --device cuda `
  --dtype bfloat16
```

Use `--dtype float32`, `float16`, or `bfloat16` to select projection and embedding
weight precision. Use `--device cpu` for CPU execution. The CUDA command requires
an existing build configured with `build.bat --cuda`.

On macOS, build with `./build.sh --metal` and pass `--device metal` to run chat
decoding on Metal.

Run the equivalent Hugging Face/PyTorch Qwen3-0.6B comparison with:

```powershell
$qwenSnapshot = "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-0.6B\snapshots\c1899de289a04d12100db370d81485cdf75e47ca"
python benchmarks/run_qwen.py `
  "Qwen/Qwen3-0.6B" `
  "$qwenSnapshot\model.safetensors" `
  "Who are you?" `
  --max-token 50
```

The runner uses CUDA when available and applies Qwen's chat template with thinking
enabled. Add `--cpu` to force CPU execution, `--greedy` to disable sampling, and
`--no-cache` to disable the KV cache for comparison with the current Citrius decoder.
Use `--max-token N` to set the maximum number of newly generated tokens.
It reports device, TTFT, end-to-end throughput, post-first-token throughput, and
total generation time.

`benchmark.bat` incrementally builds only the selected benchmark target and never
configures CMake. CUDA benchmarks require an existing build configured with
`build.bat --cuda`.

On Unix-like systems, use the matching shell script:

```bash
./build.sh --metal
./benchmark.sh operations --cpu
./benchmark.sh operations --metal
./benchmark.sh operations --all
```

`benchmark.sh` also supports the `add-kernel` and `matmul-kernel` commands and options
shown above. Metal benchmarks require an existing build configured with
`./build.sh --metal`; CUDA benchmarks require `./build.sh --cuda`.

The `examples` directory contains one program that runs add, sub, and matmul. It is built
with the library and selects a backend at runtime:

```bat
build\Release\example_operations.exe --cpu
build\Release\example_operations.exe --cuda
build\Release\example_operations.exe --metal
```

Run tests with Metal enabled:

```bash
./test.sh --metal
```

Run tests with CUDA enabled:

```bash
./test.sh --cuda
```

Run tests with Metal enabled from a clean build:

```bash
./test.sh --clean --metal
```

GoogleTest is pinned and built with the project through CMake's `FetchContent` so it
uses the same C++ runtime as Citrius:

```cmake
FetchContent_MakeAvailable(googletest)
```

In sandboxed environments, `MTLCreateSystemDefaultDevice()` may return `nil`; Metal tests are written to skip in that case. On a regular macOS terminal with Metal available, the Metal tests should run normally.

## Current Test Coverage

CPU tests cover:

- CPU storage allocation
- Float32 add
- Float32 sub
- 2D Float32 matmul
- shape mismatch rejection

Metal tests cover, when enabled and available:

- Metal storage allocation
- Float32 add
- Float32 sub
- 2D Float32 matmul
- CPU-to-Metal storage copy for ops
- rejection of CPU storage when conversion policy is `Error`

Tensor tests cover:

- default undefined tensor
- constructed tensor metadata
- shallow copy sharing storage
- explicit deep copy cloning storage
