# Tensor Layout and View Design

This document defines the tensor metadata and view model used by Citrius. The
immediate goal is to eliminate host-staged layout operations from Qwen3 while
providing a stable foundation for indexing and autograd.

## Representation

A tensor describes a logical multidimensional view over separately owned
storage:

```cpp
class Tensor {
    Shape shape_;
    Strides strides_;
    std::int64_t storage_offset_ = 0;
    DType dtype_;
    Device device_;
    std::shared_ptr<impl::ITensorStorage> storage_;
    bool defined_ = false;
};
```

`strides_` and `storage_offset_` are measured in elements, not bytes. Byte
conversion happens only when forming a pointer:

```cpp
byte_offset = element_offset * dtype_size(dtype);
```

Keeping layout metadata on `Tensor` allows multiple tensors to interpret and
share one allocation. Storage continues to own memory and execution
dependencies; it does not own logical shape or layout.

## Address calculation

At a high level, a stride is the number of storage elements to skip when moving
one position along a dimension. For a contiguous two-dimensional tensor shaped
`[rows, columns]`, the strides are `[columns, 1]`: moving to the next column
advances one element, while moving to the next row advances `columns` elements.

This makes “row length” a useful intuition for ordinary contiguous matrices,
but stride is more general than row length. After a transpose, selection, or
slice, a logical row may contain four elements while the next logical row begins
12 storage elements later. Strides describe that physical distance, including
any gaps in the underlying storage.

For logical coordinates `index`, the corresponding storage element is:

```text
storage_offset
    + index[0] * strides[0]
    + index[1] * strides[1]
    + ...
```

A contiguous row-major tensor shaped `[2, 3, 4]` has strides `[12, 4, 1]`.
Element `[1, 2, 3]` therefore resides at storage element 23:

```text
0 + 1*12 + 2*4 + 3*1 = 23
```

Reading these strides from right to left:

- stride `1` means moving along dimension 2 advances one element;
- stride `4` means moving along dimension 1 skips one four-element block;
- stride `12` means moving along dimension 0 skips one `3 * 4` block.

Contiguous strides are calculated from right to left:

```cpp
Strides contiguous_strides(const Shape& shape) {
    Strides result(shape.size());
    std::int64_t stride = 1;
    for (std::size_t index = shape.size(); index-- > 0;) {
        result[index] = stride;
        stride *= shape[index];
    }
    return result;
}
```

## View operations

Views share the input tensor's `std::shared_ptr<ITensorStorage>`. Destroying the
original tensor cannot invalidate a surviving view.

### Select

Selecting an integer index removes a dimension and advances the storage offset.
For an input with shape `[2, 3, 4]`, strides `[12, 4, 1]`, and offset zero:

```cpp
Tensor selected = tensor.select(1, 2);
```

The result has:

```text
shape:   [2, 4]
strides: [12, 1]
offset:  8
```

In general:

```cpp
new_offset += selected_index * old_strides[dimension];
```

The selected dimension is then removed from both shape and strides. Negative
indices are normalized before calculating the offset.

### Slice

For a positive-step slice, the selected dimension is transformed as follows:

```cpp
new_offset += start * old_strides[dimension];
new_shape[dimension] = slice_length;
new_strides[dimension] *= step;
```

For example, slicing `[2:8:2]` from a contiguous one-dimensional tensor produces
shape `[3]`, stride `[2]`, and offset `2`. Initial support should require a
positive step. Negative strides can be added after kernels and bounds validation
support them consistently.

### Transpose and permute

Transpose swaps two shape entries and their corresponding strides. General
permutation reorders shape and strides together.

A contiguous `[2, 3]` tensor has strides `[3, 1]`. Transposing it produces shape
`[3, 2]` and strides `[1, 3]` without allocating or copying data. The result is a
non-contiguous view.

### Reshape and view

`view()` never silently copies. It returns a storage-sharing tensor only when the
requested shape is compatible with the current layout; otherwise it throws.

`reshape()` may materialize when necessary. The initial conservative behavior
is:

- contiguous input: share storage and use contiguous strides for the new shape;
- non-contiguous input: call `contiguous()` and reshape the result.

More non-contiguous reshape-compatible layouts can be recognized later without
changing the public contract.

## Contiguity

A tensor is contiguous when its effective strides match row-major traversal.
Dimensions of size one do not constrain the layout because their coordinate is
always zero:

```cpp
bool Tensor::is_contiguous() const {
    std::int64_t expected = 1;
    for (std::size_t index = shape_.size(); index-- > 0;) {
        if (shape_[index] == 1) continue;
        if (strides_[index] != expected) return false;
        expected *= shape_[index];
    }
    return true;
}
```

A nonzero storage offset does not make a tensor non-contiguous. A shape `[3]`,
stride `[1]`, offset `5` tensor is contiguous within its logical region.

`contiguous()` returns a shallow tensor copy when the input is already
contiguous. Otherwise it allocates contiguous output on the same device and
copies elements in logical order. CUDA must use a native strided-to-contiguous
kernel and must never stage through CPU.

## Kernel pointer convention

Citrius should pass kernels a pointer to the first logical element:

```cpp
std::byte* logical_data(const Tensor& tensor) {
    auto* base = static_cast<std::byte*>(tensor.storage()->handle().ptr);
    return base + tensor.storage_offset() * dtype_size(tensor.dtype());
}
```

Kernels then receive shape and strides relative to that pointer. They must not
add `storage_offset` a second time.

Contiguous-only kernels can use the logical-start pointer directly after either:

- rejecting a non-contiguous input explicitly, or
- requesting an explicit `contiguous()` materialization.

General strided kernels translate a linear logical index into coordinates and
use those coordinates with the tensor strides.

## Bounds and invariants

Every defined tensor must satisfy:

- `shape.size() == strides.size()`;
- storage is non-null;
- `storage_offset >= 0`;
- every dimension is non-negative;
- every initially supported stride is non-negative;
- every reachable element fits within storage;
- tensor dtype and device agree with storage.

For positive strides and a non-empty tensor, the largest reachable storage
element is:

```cpp
std::int64_t maximum_offset = storage_offset;
for (std::size_t dimension = 0; dimension < shape.size(); ++dimension) {
    maximum_offset += (shape[dimension] - 1) * strides[dimension];
}
```

Construction is valid only when:

```text
(maximum_offset + 1) * dtype_size(dtype) <= storage.nbytes()
```

Empty tensors reference no elements and require separate bounds handling.

A scalar has empty shape and strides and contains one element. `item<T>()`
continues to require `numel() == 1`; for a scalar view it copies from the
logical-start pointer, respecting the storage offset.

## Aliasing and mutation

Views alias their source storage. Future in-place writes through a view are
observable through every alias. Autograd will therefore require storage-shared
version tracking to detect illegal mutation, but versioning is not required to
introduce read-only inference views.

Advanced indexing is different from basic indexing:

- integer selection, slicing, `None`, transpose, and permutation return views
  when representable by metadata;
- boolean-mask and tensor-index operations allocate independent results;
- indexed mutation is exposed explicitly through `index_put_`, not hidden proxy
  assignment.

## Materialization boundaries

Layout changes do not imply host materialization. A CUDA transpose or slice is a
pending CUDA tensor sharing device storage. Host synchronization occurs only
when host code observes data through an explicit boundary such as:

- `to(Device::cpu())`;
- `item<T>()`;
- `to_string()` or stream output;
- an explicit host pointer/copy API.

During Qwen3 greedy decoding, the intended boundary is one scalar token transfer
after final-position selection and CUDA argmax. Intermediate transpose,
permutation, splitting, and slicing must remain on-device and asynchronous.

## Implementation sequence

1. Add strides and storage offset to `Tensor`, including validation and public
   read-only accessors.
2. Initialize contiguous strides in every allocation and tensor factory path.
3. Preserve layout metadata through shallow copy and account for the offset in
   `item<T>()`, transfers, and storage access.
4. Implement metadata-only `select`, positive-step `slice`, transpose, and
   permutation.
5. Keep existing kernels contiguous-only with explicit validation while layout
   support is introduced.
6. Implement native CPU and CUDA strided `contiguous()` materialization.
7. Teach high-value CUDA operations, especially GEMM and reductions, to consume
   useful non-contiguous layouts directly.
8. Extend the model with the complete indexing API and autograd versioning.

Each stage requires CPU and CUDA parity tests covering nonzero offsets,
singleton dimensions, empty tensors, scalar views, invalid bounds, view
lifetime, and contiguous materialization.
