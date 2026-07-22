#include "nn/embedding.h"

#include "impl/cpu_storage.h"
#include "tensor_factory.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(EmbeddingTest, LooksUpBatchedTokenIds) {
    citrius::nn::Embedding embedding(4, 3);
    embedding.weight() = citrius::Tensor(
        std::vector<float>{0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32},
        {4, 3});
    const citrius::Tensor token_ids =
        citrius::from_vector(std::vector<std::int64_t>{2, 0, 3, 1}, {2, 2});

    const citrius::Tensor output = embedding(token_ids);

    EXPECT_EQ(output.shape(), citrius::Shape({2, 2, 3}));
    EXPECT_EQ(
        values(output),
        (std::vector<float>{20, 21, 22, 0, 1, 2, 30, 31, 32, 10, 11, 12}));
    EXPECT_EQ(embedding.weight().shape(), citrius::Shape({4, 3}));
    EXPECT_EQ(embedding.named_parameters().size(), 1u);
}

TEST(EmbeddingTest, RejectsInvalidDimensionsIdsAndDtypes) {
    EXPECT_THROW(citrius::nn::Embedding(0, 3), std::invalid_argument);
    EXPECT_THROW(citrius::nn::Embedding(3, 0), std::invalid_argument);

    citrius::nn::Embedding embedding(3, 2);
    EXPECT_THROW(
        embedding(citrius::from_vector(std::vector<std::int64_t>{0, 3})),
        std::out_of_range);
    EXPECT_THROW(
        embedding(citrius::Tensor(std::vector<float>{0, 1})),
        std::invalid_argument);
}

TEST(EmbeddingTest, StoresReducedPrecisionWeightsAndReturnsFloat32) {
    citrius::nn::Embedding embedding(
        2, 2, citrius::Device::cpu(), citrius::DType::Float16);
    embedding.weight() = citrius::from_vector(
        std::vector<float>{1, 2, 3, 4}, {2, 2}, citrius::DType::Float16);

    const auto output = embedding(citrius::from_vector(std::vector<std::int64_t>{1}));

    EXPECT_EQ(embedding.weight().dtype(), citrius::DType::Float16);
    EXPECT_EQ(output.dtype(), citrius::DType::Float32);
    EXPECT_EQ(values(output), std::vector<float>({3, 4}));
}
