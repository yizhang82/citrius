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

Explicit deep copy is available through:

```cpp
Tensor copied = tensor.copy();
```

`Tensor::copy()` preserves metadata and calls `ITensorStorage::clone()` to allocate distinct backend storage with copied contents.

Current storage implementations:

- `impl::CpuMemTensorStorageImpl` in `include/impl/cpu_storage.h` and `src/cpu_storage.cpp`
- `impl::MetalMemTensorStorageImpl` in `include/impl/metal_storage.h` and `src/metal_storage.mm`
- `impl::CudaMemTensorStorageImpl` in `include/impl/cuda_storage.h` and `src/cuda_storage.cu`

Current device implementations:

- `impl::CpuDeviceImpl` in `include/impl/cpu_device.h` and `src/cpu_device.cpp`
- `impl::MetalDeviceImpl` in `include/impl/metal_device.h` and `src/metal_device.mm`
- `impl::CudaDeviceImpl` in `include/impl/cuda_device.h` and `src/cuda_device.cu`

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
- contiguous layout is assumed
- add/sub require identical shapes
- matmul supports 2D tensors only
- broadcasting, strides, dtype dispatch, and autograd are not implemented yet

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
test.bat
test.bat --clean --config Debug
test.bat --clean --cuda
```

Run the end-to-end Float32 matmul benchmark on Windows with:

```bat
benchmark.bat --cpu
benchmark.bat --cuda
```

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

GoogleTest is required and discovered through:

```cmake
find_package(GTest REQUIRED)
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
