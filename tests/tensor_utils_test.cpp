#include "tensor_utils.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(TensorUtilsTest, EnsureTensorDefinedAcceptsDefinedTensor) {
    const citrius::Tensor input(std::vector<float>{1.0f});

    EXPECT_NO_THROW(CITRIUS_ENSURE_TENSOR_DEFINED(input));
}

TEST(TensorUtilsTest, EnsureTensorDefinedRejectsUndefinedTensor) {
    const citrius::Tensor input;

    EXPECT_THROW(CITRIUS_ENSURE_TENSOR_DEFINED(input), std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTensorShapeAcceptsMatchingShape) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Shape expected{2, 2};

    EXPECT_NO_THROW(CITRIUS_ENSURE_TENSOR_SHAPE(input, expected));
}

TEST(TensorUtilsTest, EnsureTensorShapeRejectsMismatchedShape) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Shape expected{4};

    EXPECT_THROW(
        CITRIUS_ENSURE_TENSOR_SHAPE(input, expected),
        std::invalid_argument);
}

TEST(TensorUtilsTest, EnsureTensorDimAcceptsMatchingRank) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {1, 2, 2});

    EXPECT_NO_THROW(CITRIUS_ENSURE_TENSOR_DIM(input, 3));
}

TEST(TensorUtilsTest, EnsureTensorDimRejectsMismatchedRank) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4}, {2, 2});

    EXPECT_THROW(CITRIUS_ENSURE_TENSOR_DIM(input, 3), std::invalid_argument);
}
