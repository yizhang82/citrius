#include "tensor.h"

#include <gtest/gtest.h>

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

TEST(TensorTest, CopiesShareStorage) {
    citrius::Tensor tensor({2, 3}, citrius::DType::Float32, citrius::Device::cpu());

    citrius::Tensor copy = tensor;

    EXPECT_TRUE(copy.defined());
    EXPECT_EQ(copy.shape(), tensor.shape());
    EXPECT_EQ(copy.dtype(), tensor.dtype());
    EXPECT_EQ(copy.device(), tensor.device());
    EXPECT_EQ(copy.storage(), tensor.storage());
}
