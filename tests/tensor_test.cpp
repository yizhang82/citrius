#include "indexing_operations.h"
#include "shape_operations.h"
#include "tensor.h"
#include "tensor_factory.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

std::shared_ptr<citrius::impl::CpuMemTensorStorageImpl> cpu_storage(const citrius::Tensor& tensor) {
    return std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
}

std::vector<float> cpu_tensor_values(const citrius::Tensor& tensor) {
    const float* data = cpu_storage(tensor)->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(TensorTest, DefaultConstructedTensorIsUndefined) {
    citrius::Tensor undefined;
    EXPECT_FALSE(undefined.defined());
}

TEST(TensorTest, ConstructedTensorExposesMetadata) {
    citrius::Tensor tensor({2, 3}, citrius::DType::Float32, citrius::Device::cpu());

    EXPECT_TRUE(tensor.defined());
    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(tensor.strides(), citrius::Strides({3, 1}));
    EXPECT_EQ(tensor.storage_offset(), 0);
    EXPECT_TRUE(tensor.is_contiguous());
    EXPECT_EQ(tensor.dtype(), citrius::DType::Float32);
    EXPECT_EQ(tensor.device(), citrius::Device::cpu());
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.numel(), 6);
}

std::vector<std::int64_t> cpu_int64_values(const citrius::Tensor& tensor) {
    const auto* data = cpu_storage(tensor)->data_as<std::int64_t>();
    return std::vector<std::int64_t>(data, data + tensor.numel());
}

TEST(TensorTest, ConstructsFromVectorWithInferredShape) {
    const std::vector<float> values = {1.0f, 2.0f, 3.0f};
    const citrius::Tensor tensor(values);

    EXPECT_EQ(tensor.shape(), citrius::Shape({3}));
    EXPECT_EQ(tensor.dtype(), citrius::DType::Float32);
    EXPECT_EQ(cpu_tensor_values(tensor), values);
}

TEST(TensorTest, ConstructsAndPrintsFloat16AndBFloat16Values) {
    const std::vector<float> input{1.0f, -2.5f, 0.333333f};
    const auto fp16 = citrius::from_vector(input, {3}, citrius::DType::Float16);
    const auto bf16 = citrius::from_vector(input, {3}, citrius::DType::BFloat16);
    EXPECT_EQ(fp16.dtype(), citrius::DType::Float16);
    EXPECT_EQ(bf16.dtype(), citrius::DType::BFloat16);
    EXPECT_EQ(fp16.storage()->nbytes(), 6u);
    EXPECT_EQ(bf16.storage()->nbytes(), 6u);
    EXPECT_NE(fp16.to_string().find("dtype=float16"), std::string::npos);
    EXPECT_NE(bf16.to_string().find("dtype=bfloat16"), std::string::npos);
    EXPECT_NE(fp16.to_string().find("-2.5"), std::string::npos);
    EXPECT_NE(bf16.to_string().find("-2.5"), std::string::npos);
}

TEST(TensorTest, FloatingPointBitConversionsRoundTrip) {
    for (const float value : {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f}) {
        EXPECT_NEAR(citrius::float16_to_float(citrius::float_to_float16(value)), value,
                    std::max(1e-6f, std::abs(value) * 1e-3f));
        EXPECT_NEAR(citrius::bfloat16_to_float(citrius::float_to_bfloat16(value)), value,
                    std::max(1e-6f, std::abs(value) * 4e-3f));
    }
}

TEST(TensorTest, FactoryConstructsVectorWithExplicitShape) {
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto tensor = citrius::from_vector(values, {2, 2});

    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(cpu_tensor_values(tensor), values);
}

TEST(TensorTest, FactoryConstructsInt64TokenIds) {
    const std::vector<std::int64_t> values = {3, 1, 4, 1};
    const auto tensor = citrius::from_vector(values, {2, 2});

    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(tensor.dtype(), citrius::DType::Int64);
    EXPECT_EQ(cpu_int64_values(tensor), values);
    EXPECT_NE(tensor.to_string().find("dtype=int64"), std::string::npos);
}

TEST(TensorTest, FreeFactoryFunctionsCreateAndTransferTensors) {
    const auto allocated = citrius::empty({2, 3});
    const auto values = citrius::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
        {2, 2});
    const auto transferred = citrius::to(values, citrius::Device::cpu());

    EXPECT_EQ(allocated.shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(allocated.dtype(), citrius::DType::Float32);
    EXPECT_EQ(cpu_tensor_values(values), std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(transferred.storage(), values.storage());
}

TEST(TensorTest, RejectsVectorWhoseSizeDoesNotMatchShape) {
    EXPECT_THROW(
        citrius::Tensor(std::vector<float>{1.0f, 2.0f}, {2, 2}),
        std::invalid_argument);
}

TEST(TensorTest, ToSameDeviceReturnsShallowCopy) {
    const citrius::Tensor tensor(std::vector<float>{1.0f, 2.0f});
    const auto moved = tensor.to(citrius::Device::cpu());

    EXPECT_EQ(moved.storage(), tensor.storage());
}

TEST(TensorTest, ToRejectsUndefinedTensor) {
    EXPECT_THROW(citrius::Tensor().to(citrius::Device::cpu()), std::invalid_argument);
}

TEST(TensorTest, CopiesShareStorage) {
    citrius::Tensor tensor({2, 3}, citrius::DType::Float32, citrius::Device::cpu());

    citrius::Tensor copy = tensor;

    EXPECT_TRUE(copy.defined());
    EXPECT_EQ(copy.shape(), tensor.shape());
    EXPECT_EQ(copy.dtype(), tensor.dtype());
    EXPECT_EQ(copy.device(), tensor.device());
    EXPECT_EQ(copy.storage(), tensor.storage());
}

TEST(TensorTest, CopyCreatesDeepCopy) {
    citrius::Tensor tensor({2, 2}, citrius::DType::Float32, citrius::Device::cpu());
    float* data = cpu_storage(tensor)->data_as<float>();
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    citrius::Tensor copied = tensor.copy();

    EXPECT_TRUE(copied.defined());
    EXPECT_EQ(copied.shape(), tensor.shape());
    EXPECT_EQ(copied.dtype(), tensor.dtype());
    EXPECT_EQ(copied.device(), tensor.device());
    EXPECT_NE(copied.storage(), tensor.storage());
    EXPECT_EQ(cpu_tensor_values(copied), std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));

    data[0] = 100.0f;

    EXPECT_EQ(cpu_tensor_values(tensor), std::vector<float>({100.0f, 2.0f, 3.0f, 4.0f}));
    EXPECT_EQ(cpu_tensor_values(copied), std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(TensorTest, CopyRejectsUndefinedTensor) {
    const citrius::Tensor undefined;

    EXPECT_THROW(undefined.copy(), std::invalid_argument);
}

TEST(TensorTest, ExplicitLayoutExposesStridesOffsetAndContiguity) {
    auto storage = std::make_shared<citrius::impl::CpuMemTensorStorageImpl>(
        24 * sizeof(float), citrius::DType::Float32);
    const citrius::Tensor tensor(
        {2, 4}, {12, 1}, 8, citrius::DType::Float32, citrius::Device::cpu(), storage);

    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 4}));
    EXPECT_EQ(tensor.strides(), citrius::Strides({12, 1}));
    EXPECT_EQ(tensor.storage_offset(), 8);
    EXPECT_FALSE(tensor.is_contiguous());
}

TEST(TensorTest, ScalarMaterializationAndPrintingRespectStorageOffset) {
    auto storage = std::make_shared<citrius::impl::CpuMemTensorStorageImpl>(
        4 * sizeof(float), citrius::DType::Float32);
    auto* values = storage->data_as<float>();
    values[0] = 1.0f;
    values[1] = 2.0f;
    values[2] = 3.0f;
    values[3] = 4.0f;
    const citrius::Tensor scalar(
        {}, {}, 2, citrius::DType::Float32, citrius::Device::cpu(), storage);

    EXPECT_FLOAT_EQ(scalar.item<float>(), 3.0f);
    EXPECT_EQ(scalar.to_string(), "tensor([3], shape=[], dtype=float32, device=cpu)");
}

TEST(TensorTest, SelectReturnsAnAliasingMetadataOnlyView) {
    const citrius::Tensor tensor(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        {2, 3, 4});
    const auto selected = tensor.select(1, 2);

    EXPECT_EQ(selected.shape(), citrius::Shape({2, 4}));
    EXPECT_EQ(selected.strides(), citrius::Strides({12, 1}));
    EXPECT_EQ(selected.storage_offset(), 8);
    EXPECT_EQ(selected.storage(), tensor.storage());
    EXPECT_FALSE(selected.is_contiguous());
    EXPECT_FLOAT_EQ(selected.select(0, -1).select(0, 3).item<float>(), 23.0f);
}

TEST(TensorTest, SliceReturnsAnAliasingPositiveStepView) {
    const citrius::Tensor tensor(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    const auto sliced = tensor.slice(0, 2, 8, 2);

    EXPECT_EQ(sliced.shape(), citrius::Shape({3}));
    EXPECT_EQ(sliced.strides(), citrius::Strides({2}));
    EXPECT_EQ(sliced.storage_offset(), 2);
    EXPECT_EQ(sliced.storage(), tensor.storage());
    EXPECT_FALSE(sliced.is_contiguous());
    EXPECT_FLOAT_EQ(sliced.select(0, 1).item<float>(), 4.0f);
    EXPECT_THROW(tensor.slice(0, 0, 2, 0), std::invalid_argument);
    EXPECT_THROW(tensor.select(0, 10), std::out_of_range);
}

TEST(TensorTest, BasicIndexCombinesIntegersAndSlicesIntoOneAliasingView) {
    const citrius::Tensor tensor(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        {2, 3, 4});
    const auto indexed = tensor[{1, citrius::indexing::Slice(0, 3, 2), -1}];

    EXPECT_EQ(indexed.shape(), citrius::Shape({2}));
    EXPECT_EQ(indexed.strides(), citrius::Strides({8}));
    EXPECT_EQ(indexed.storage_offset(), 15);
    EXPECT_EQ(indexed.storage(), tensor.storage());
    EXPECT_EQ(cpu_tensor_values(citrius::contiguous(indexed)),
              (std::vector<float>{15, 23}));
    EXPECT_FLOAT_EQ(tensor[-1].select(0, 0).select(0, 0).item<float>(), 12.0f);
}

TEST(TensorTest, BasicIndexSupportsEllipsisNewAxisAndReverseSlices) {
    using citrius::indexing::Ellipsis;
    using citrius::indexing::None;
    using citrius::indexing::Slice;

    const citrius::Tensor tensor(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 2, 3});
    const auto indexed = tensor.index(
        {None, Ellipsis, Slice(std::nullopt, std::nullopt, -1)});

    EXPECT_EQ(indexed.shape(), (citrius::Shape{1, 2, 2, 3}));
    EXPECT_EQ(indexed.strides(), (citrius::Strides{0, 6, 3, -1}));
    EXPECT_EQ(indexed.storage_offset(), 2);
    EXPECT_EQ(indexed.storage(), tensor.storage());
    EXPECT_EQ(cpu_tensor_values(citrius::contiguous(indexed)),
              (std::vector<float>{2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9}));
}

TEST(TensorTest, BasicIndexValidatesComponentsAndHandlesEmptySlices) {
    using citrius::indexing::Ellipsis;
    using citrius::indexing::Slice;

    const citrius::Tensor tensor(std::vector<float>{0, 1, 2, 3, 4, 5}, {2, 3});
    const auto empty = tensor.index({Slice(4, 1), Slice()});

    EXPECT_EQ(empty.shape(), (citrius::Shape{0, 3}));
    EXPECT_EQ(empty.numel(), 0);
    EXPECT_THROW(tensor.index({0, 0, 0}), std::out_of_range);
    EXPECT_THROW(tensor.index({Ellipsis, Ellipsis}), std::invalid_argument);
    EXPECT_THROW(tensor.index({Slice(std::nullopt, std::nullopt, 0)}),
                 std::invalid_argument);
}

TEST(TensorTest, ExplicitLayoutValidatesMetadataAndStorageBounds) {
    auto storage = std::make_shared<citrius::impl::CpuMemTensorStorageImpl>(
        6 * sizeof(float), citrius::DType::Float32);

    EXPECT_THROW(
        citrius::Tensor({2, 3}, {3}, 0, citrius::DType::Float32,
                        citrius::Device::cpu(), storage),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::Tensor({2, 3}, {3, 1}, -1, citrius::DType::Float32,
                        citrius::Device::cpu(), storage),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::Tensor({2, 3}, {4, 1}, 0, citrius::DType::Float32,
                        citrius::Device::cpu(), storage),
        std::invalid_argument);
}

TEST(TensorTest, ItemMaterializesScalarAndValidatesTypeAndShape) {
    EXPECT_EQ(citrius::from_vector(std::vector<std::int64_t>{42}, citrius::Shape{}).item<std::int64_t>(), 42);
    EXPECT_FLOAT_EQ(citrius::from_vector(std::vector<float>{3.5f}, citrius::Shape{}).item<float>(), 3.5f);
    EXPECT_THROW(citrius::Tensor().item<float>(), std::invalid_argument);
    EXPECT_THROW(citrius::Tensor(std::vector<float>{1, 2}).item<float>(), std::invalid_argument);
    EXPECT_THROW(citrius::Tensor(std::vector<float>{1}).item<std::int64_t>(), std::invalid_argument);
}

TEST(TensorTest, ToStringIncludesValuesAndMetadata) {
    const citrius::Tensor tensor(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f},
        {2, 2});

    EXPECT_EQ(
        tensor.to_string(),
        "tensor([1, 2, 3, 4], shape=[2, 2], dtype=float32, device=cpu)");
}

TEST(TensorTest, StreamOperatorUsesTensorString) {
    const citrius::Tensor tensor(std::vector<float>{1.0f, 2.0f});
    std::ostringstream stream;

    stream << tensor;

    EXPECT_EQ(stream.str(), tensor.to_string());
}

TEST(TensorTest, UndefinedTensorHasStringRepresentation) {
    EXPECT_EQ(citrius::Tensor().to_string(), "tensor(undefined)");
}
