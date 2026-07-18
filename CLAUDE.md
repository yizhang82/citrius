# Citrius Project Notes

Citrius is a small C++ tensor library skeleton inspired by PyTorch, currently focused on a clear tensor/storage/device split.

## Current Design

The public tensor API is in `include/tensor.h`.

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

Storage is abstracted through `ITensorStorage` in `include/storage.h`. Concrete storage implementations advertise their backend through `ITensorStorage::type()`, then compatible devices check the type before casting to the concrete implementation.

Tensor copies are shallow with respect to storage: tensor metadata is copied by value, while the underlying `ITensorStorage` is shared through `std::shared_ptr`.

Current storage implementations:

- `CpuMemTensorStorageImpl` in `include/cpu_storage.h` and `src/cpu_storage.cpp`
- `MetalMemTensorStorageImpl` in `include/metal_storage.h` and `src/metal_storage.mm`

Current device implementations:

- `CpuDeviceImpl` in `include/cpu_device.h` and `src/cpu_device.cpp`
- `MetalDeviceImpl` in `include/metal_device.h` and `src/metal_device.mm`

The device layer owns execution. Storage owns memory only. Arithmetic should live on `IDevice` implementations, not in storage.

## Backend Behavior

`CpuDeviceImpl` supports:

- `empty`
- `add`
- `sub`
- `matmul`

`MetalDeviceImpl` supports the same operations when Metal is enabled.

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

Clean Metal build:

```bash
./build.sh --clean --metal
```

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

Run tests with Metal enabled:

```bash
./test.sh --metal
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
