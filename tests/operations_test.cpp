#include "exceptions.h"
#include "operations.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace {

std::vector<float> values(const citrius::Tensor& tensor) {
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(tensor.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + tensor.numel());
}

} // namespace

TEST(OperationsTest, AddsUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3});
    const citrius::Tensor right(std::vector<float>{10, 20, 30});

    EXPECT_EQ(values(citrius::add(left, right)), std::vector<float>({11, 22, 33}));
}

TEST(OperationsTest, SubtractsUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{10, 20, 30});
    const citrius::Tensor right(std::vector<float>{1, 2, 3});

    EXPECT_EQ(values(citrius::sub(left, right)), std::vector<float>({9, 18, 27}));
}

TEST(OperationsTest, MultipliesUsingTensorDevice) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3});
    const citrius::Tensor right(
        std::vector<float>{7, 8, 9, 10, 11, 12},
        {3, 2});

    EXPECT_EQ(
        values(citrius::matmul(left, right)),
        std::vector<float>({58, 64, 139, 154}));
}

TEST(OperationsTest, ThrowsDeviceMismatchException) {
    const citrius::Tensor cpu(std::vector<float>{1, 2});
    const citrius::Tensor other(
        cpu.shape(),
        cpu.dtype(),
        citrius::Device::cuda(),
        cpu.storage());

    EXPECT_THROW(citrius::add(cpu, other), citrius::DeviceMismatchException);
}

TEST(OperationsTest, OperatorsDelegateToTopLevelOperations) {
    const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2});
    const citrius::Tensor right(std::vector<float>{5, 6, 7, 8}, {2, 2});

    EXPECT_EQ(values(left + right), std::vector<float>({6, 8, 10, 12}));
    EXPECT_EQ(values(left - right), std::vector<float>({-4, -4, -4, -4}));
    EXPECT_EQ(values(left * right), std::vector<float>({19, 22, 43, 50}));
}
