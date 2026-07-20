#include "shape_operations.h"

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

} // namespace

TEST(ShapeOperationsTest, ReshapeAndViewShareStorage) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const auto reshaped = citrius::reshape(input, {3, -1});
    const auto viewed = citrius::view(input, {6});

    EXPECT_EQ(reshaped.shape(), citrius::Shape({3, 2}));
    EXPECT_EQ(viewed.shape(), citrius::Shape({6}));
    EXPECT_EQ(reshaped.storage(), input.storage());
    EXPECT_EQ(viewed.storage(), input.storage());
    EXPECT_THROW(citrius::reshape(input, {4, 2}), std::invalid_argument);
}

TEST(ShapeOperationsTest, FlattensSqueezesAndUnsqueezes) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4, 5, 6}, {1, 2, 3, 1});

    EXPECT_EQ(citrius::flatten(input, 1, 2).shape(), citrius::Shape({1, 6, 1}));
    EXPECT_EQ(citrius::squeeze(input).shape(), citrius::Shape({2, 3}));
    EXPECT_EQ(citrius::squeeze(input, 0).shape(), citrius::Shape({2, 3, 1}));
    EXPECT_EQ(citrius::unsqueeze(input, -1).shape(), citrius::Shape({1, 2, 3, 1, 1}));
}

TEST(ShapeOperationsTest, TransposesAndPermutesValues) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const auto transposed = citrius::transpose(input, 0, 1);

    EXPECT_EQ(transposed.shape(), citrius::Shape({3, 2}));
    EXPECT_EQ(transposed.storage(), input.storage());
    EXPECT_FALSE(transposed.is_contiguous());
    EXPECT_EQ(values(citrius::contiguous(transposed)), std::vector<float>({1, 4, 2, 5, 3, 6}));

    const citrius::Tensor cube(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    const auto permuted = citrius::permute(cube, {1, 0, 2});
    EXPECT_EQ(permuted.storage(), cube.storage());
    EXPECT_EQ(values(citrius::contiguous(permuted)), std::vector<float>({1, 2, 5, 6, 3, 4, 7, 8}));
}

TEST(ShapeOperationsTest, SplitsChunksAndConcatenates) {
    const citrius::Tensor input(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const auto parts = citrius::split(input, 2, 1);

    ASSERT_EQ(parts.size(), 2u);
    EXPECT_EQ(parts[0].shape(), citrius::Shape({2, 2}));
    EXPECT_EQ(values(parts[0]), std::vector<float>({1, 2, 4, 5}));
    EXPECT_EQ(values(parts[1]), std::vector<float>({3, 6}));
    EXPECT_EQ(values(citrius::concat(parts, 1)), values(input));

    const auto chunks = citrius::chunk(input, 3, 1);
    ASSERT_EQ(chunks.size(), 3u);
    EXPECT_EQ(values(citrius::concat(chunks, 1)), values(input));
}

TEST(ShapeOperationsTest, ContiguousMaterializesAStridedView) {
    const citrius::Tensor input(
        std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                           12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
        {2, 3, 4});
    const auto selected = input.select(1, 2);
    const auto packed = citrius::contiguous(selected);

    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ(packed.strides(), citrius::Strides({4, 1}));
    EXPECT_EQ(packed.storage_offset(), 0);
    EXPECT_NE(packed.storage(), input.storage());
    EXPECT_EQ(values(packed), std::vector<float>({8, 9, 10, 11, 20, 21, 22, 23}));
}
