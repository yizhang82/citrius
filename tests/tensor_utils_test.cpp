#include "tensor_utils.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(TensorUtilsTest, EnsureTensorDefinedAcceptsDefinedTensor) {
    const citrius::Tensor input(std::vector<float>{1.0f});

    EXPECT_NO_THROW(ENSURE_TENSOR_DEFINED(input));
}

TEST(TensorUtilsTest, EnsureTensorDefinedRejectsUndefinedTensor) {
    const citrius::Tensor input;

    EXPECT_THROW(ENSURE_TENSOR_DEFINED(input), std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTensorShapeAcceptsMatchingShape) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Shape expected{2, 2};

    EXPECT_NO_THROW(ENSURE_TENSOR_SHAPE(input, expected));
}

TEST(TensorUtilsTest, EnsureTensorShapeRejectsMismatchedShape) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Shape expected{4};

    EXPECT_THROW(
        ENSURE_TENSOR_SHAPE(input, expected),
        std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTensorDimAcceptsMatchingRank) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {1, 2, 2});

    EXPECT_NO_THROW(ENSURE_TENSOR_DIM(input, 3));
}

TEST(TensorUtilsTest, EnsureTensorDimRejectsMismatchedRank) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(ENSURE_TENSOR_DIM(input, 3), std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTensorDtypeAcceptsMatchingType) {
    const citrius::Tensor input({2, 3}, citrius::DType::Float32);

    EXPECT_NO_THROW(ENSURE_TENSOR_DTYPE(input, citrius::DType::Float32));
}

TEST(TensorUtilsTest, EnsureTensorDtypeRejectsMismatchedType) {
    const citrius::Tensor input({2, 3}, citrius::DType::Float32);

    EXPECT_THROW(
        ENSURE_TENSOR_DTYPE(input, citrius::DType::Int64),
        std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTwoTensorDevicesMatch) {
    const citrius::Tensor first(std::vector<float>{1.0f});
    const citrius::Tensor second(std::vector<float>{2.0f});
    const citrius::Tensor other_device(
        second.shape(), second.dtype(), {citrius::DeviceType::CPU, 1}, second.storage());

    EXPECT_NO_THROW(ENSURE_TENSOR_DEVICE_MATCH_2(first, second));
    EXPECT_THROW(
        ENSURE_TENSOR_DEVICE_MATCH_2(first, other_device),
        citrius::DeviceMismatchException);
}

TEST(TensorUtilsTest, EnsureThreeTensorDevicesMatch) {
    const citrius::Tensor first(std::vector<float>{1.0f});
    const citrius::Tensor second(std::vector<float>{2.0f});
    const citrius::Tensor third(std::vector<float>{3.0f});
    const citrius::Tensor other_device(
        third.shape(), third.dtype(), {citrius::DeviceType::CPU, 1}, third.storage());

    EXPECT_NO_THROW(ENSURE_TENSOR_DEVICE_MATCH_3(first, second, third));
    EXPECT_THROW(
        ENSURE_TENSOR_DEVICE_MATCH_3(first, second, other_device),
        citrius::DeviceMismatchException);
}
