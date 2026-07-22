#include "indexing_operations.h"
#include "reduction_operations.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

std::vector<std::int64_t> indices(const citrius::Tensor& tensor) {
    const auto storage = std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const auto* data = storage->data_as<std::int64_t>();
    return std::vector<std::int64_t>(data, data + tensor.numel());
}

citrius::Tensor matrix() {
    return citrius::Tensor(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
}

} // namespace

TEST(ReductionOperationsTest, ReducesAllElements) {
    EXPECT_EQ(values(citrius::sum(matrix())), std::vector<float>({21}));
    EXPECT_EQ(values(citrius::mean(matrix())), std::vector<float>({3.5f}));
    EXPECT_EQ(values(citrius::max(matrix())), std::vector<float>({6}));
    EXPECT_EQ(citrius::sum(matrix()).shape(), citrius::Shape({}));
    EXPECT_EQ(
        citrius::sum(matrix(), std::vector<std::int64_t>{0, 1}, true).shape(),
        citrius::Shape({1, 1}));
}

TEST(ReductionOperationsTest, ReducesDimensionsAndKeepsDimensions) {
    EXPECT_EQ(values(citrius::sum(matrix(), 0)), std::vector<float>({5, 7, 9}));
    EXPECT_EQ(values(citrius::sum(matrix(), -1)), std::vector<float>({6, 15}));
    EXPECT_EQ(citrius::mean(matrix(), 1, true).shape(), citrius::Shape({2, 1}));
    EXPECT_EQ(values(citrius::mean(matrix(), 1, true)), std::vector<float>({2, 5}));
    EXPECT_EQ(values(citrius::max(matrix(), 0)), std::vector<float>({4, 5, 6}));
}

TEST(ReductionOperationsTest, ReducesMultipleDimensions) {
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8},
        {2, 2, 2});

    EXPECT_EQ(values(citrius::sum(input, std::vector<std::int64_t>{0, 2})),
              std::vector<float>({14, 22}));
    EXPECT_EQ(citrius::sum(input, std::vector<std::int64_t>{0, 2}, true).shape(),
              citrius::Shape({1, 2, 1}));
}

TEST(ReductionOperationsTest, ComputesPopulationVariance) {
    EXPECT_EQ(values(citrius::variance(matrix(), 1)), std::vector<float>({2.0f / 3.0f, 2.0f / 3.0f}));
    EXPECT_FLOAT_EQ(values(citrius::variance(matrix()))[0], 35.0f / 12.0f);
}

TEST(ReductionOperationsTest, ArgmaxReturnsInt64Indices) {
    const citrius::Tensor input(std::vector<float>{-4, -1, -1, -2, 9, 3, 4, 9}, {2, 4});
    const auto rows = citrius::argmax(input, -1);
    EXPECT_EQ(rows.dtype(), citrius::DType::Int64);
    EXPECT_EQ(rows.shape(), citrius::Shape({2}));
    EXPECT_EQ(indices(rows), std::vector<std::int64_t>({1, 0}));
    EXPECT_EQ(indices(citrius::argmax(input, 0)), std::vector<std::int64_t>({1, 1, 1, 1}));
    EXPECT_EQ(indices(citrius::argmax(input)), std::vector<std::int64_t>({4}));
    EXPECT_EQ(citrius::argmax(input, 1, true).shape(), citrius::Shape({2, 1}));
}

TEST(ReductionOperationsTest, ArgmaxRespectsIndexedViewLayout) {
    const citrius::Tensor input(
        std::vector<float>{100, 90, 80, 70, 1, 7, 3, 2}, {2, 4});
    const auto final_row = input.index({-1, citrius::indexing::Slice()});
    const auto reversed = final_row.index(
        {citrius::indexing::Slice(std::nullopt, std::nullopt, -1)});

    EXPECT_EQ(final_row.storage_offset(), 4);
    EXPECT_EQ(reversed.storage_offset(), 7);
    EXPECT_EQ(citrius::argmax(final_row).item<std::int64_t>(), 1);
    EXPECT_EQ(citrius::argmax(final_row, -1).item<std::int64_t>(), 1);
    EXPECT_EQ(citrius::argmax(reversed).item<std::int64_t>(), 2);
}

TEST(ReductionOperationsTest, ArgmaxRejectsInvalidAndEmptyInputs) {
    EXPECT_THROW(citrius::argmax(matrix(), 2), std::out_of_range);
    EXPECT_THROW(citrius::argmax(citrius::Tensor({0}, citrius::DType::Float32)), std::invalid_argument);
}

TEST(ReductionOperationsTest, RejectsInvalidDimensions) {
    EXPECT_THROW(citrius::sum(matrix(), 2), std::out_of_range);
    EXPECT_THROW(
        citrius::sum(matrix(), std::vector<std::int64_t>{0, -2}),
        std::invalid_argument);
    EXPECT_THROW(
        citrius::sum(matrix(), std::vector<std::int64_t>{}),
        std::invalid_argument);
}
