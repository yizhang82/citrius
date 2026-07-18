#include "tensor.h"
#include "tensor_factory.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

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
    EXPECT_EQ(tensor.dtype(), citrius::DType::Float32);
    EXPECT_EQ(tensor.device(), citrius::Device::cpu());
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.numel(), 6);
}

TEST(TensorTest, ConstructsFromVectorWithInferredShape) {
    const std::vector<float> values = {1.0f, 2.0f, 3.0f};
    const citrius::Tensor tensor(values);

    EXPECT_EQ(tensor.shape(), citrius::Shape({3}));
    EXPECT_EQ(tensor.dtype(), citrius::DType::Float32);
    EXPECT_EQ(cpu_tensor_values(tensor), values);
}

TEST(TensorTest, FactoryConstructsVectorWithExplicitShape) {
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto tensor = citrius::from_vector(values, {2, 2});

    EXPECT_EQ(tensor.shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(cpu_tensor_values(tensor), values);
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
