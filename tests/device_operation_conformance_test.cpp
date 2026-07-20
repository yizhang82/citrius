#include "indexing_operations.h"
#include "nn/functional.h"
#include "operations.h"
#include "reduction_operations.h"
#include "shape_operations.h"
#include "tensor_factory.h"

#include "impl/cpu_storage.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

struct DeviceCase {
    const char* name;
    citrius::Device device;
};

std::vector<DeviceCase> device_cases() {
    std::vector<DeviceCase> devices{{"CPU", citrius::Device::cpu()}};
#ifdef CITRIUS_TEST_HAS_CUDA
    devices.push_back({"CUDA", citrius::Device::cuda()});
#endif
#ifdef CITRIUS_TEST_HAS_METAL
    devices.push_back({"Metal", citrius::Device::metal()});
#endif
    return devices;
}

std::vector<float> values(const citrius::Tensor& tensor) {
    const citrius::Tensor cpu = citrius::contiguous(tensor).to(citrius::Device::cpu());
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(cpu.storage());
    const float* data = storage->data_as<float>();
    return std::vector<float>(data, data + cpu.numel());
}

std::vector<std::int64_t> int64_values(const citrius::Tensor& tensor) {
    const citrius::Tensor cpu = citrius::contiguous(tensor).to(citrius::Device::cpu());
    const auto storage =
        std::static_pointer_cast<citrius::impl::CpuMemTensorStorageImpl>(cpu.storage());
    const auto* data = storage->data_as<std::int64_t>();
    return std::vector<std::int64_t>(data, data + cpu.numel());
}

void expect_near(const std::vector<float>& actual, const std::vector<float>& expected,
                 float tolerance = 1e-5f) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
        EXPECT_NEAR(actual[index], expected[index], tolerance) << "element " << index;
}

class DeviceOperationConformanceTest : public testing::TestWithParam<DeviceCase> {
  protected:
    citrius::Device device() const {
        return GetParam().device;
    }

    citrius::Tensor tensor(std::vector<float> data, citrius::Shape shape) const {
        return citrius::Tensor(std::move(data), std::move(shape), device());
    }
};

TEST_P(DeviceOperationConformanceTest, ElementwiseBroadcastAndScalarMatchTorchSemantics) {
    const auto matrix = tensor({-2, -1, 0, 1, 2, 4}, {2, 3});
    const auto row = tensor({2, 4, 8}, {1, 3});

    EXPECT_EQ(values(citrius::add(matrix, row)), (std::vector<float>{0, 3, 8, 3, 6, 12}));
    EXPECT_EQ(values(citrius::sub(row, matrix)), (std::vector<float>{4, 5, 8, 1, 2, 4}));
    EXPECT_EQ(values(citrius::mul(matrix, row)), (std::vector<float>{-4, -4, 0, 2, 8, 32}));
    expect_near(values(citrius::div(matrix, row)), {-1, -0.25f, 0, 0.5f, 0.5f, 0.5f});
    EXPECT_EQ(values(8.0f - matrix), (std::vector<float>{10, 9, 8, 7, 6, 4}));
    EXPECT_EQ(values(citrius::maximum(matrix, row)), (std::vector<float>{2, 4, 8, 2, 4, 8}));
}

TEST_P(DeviceOperationConformanceTest, UnaryMathMatchesTorchValues) {
    const auto input = tensor({-2, -1, 0, 1, 2}, {5});
    expect_near(values(citrius::exp(input)),
                {0.13533528f, 0.36787945f, 1.0f, 2.71828175f, 7.3890561f}, 2e-5f);
    EXPECT_EQ(values(citrius::sqrt(tensor({0, 1, 4, 9}, {4}))), (std::vector<float>{0, 1, 2, 3}));
    expect_near(values(citrius::pow(input, 3.0f)), {-8, -1, 0, 1, 8});
}

TEST_P(DeviceOperationConformanceTest, ViewsAndContiguousPreserveLogicalValues) {
    const auto input = tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 2, 3});
    const auto view = citrius::permute(input, {1, 0, 2}).select(0, 1);

    EXPECT_EQ(view.shape(), (citrius::Shape{2, 3}));
    EXPECT_EQ(values(view), (std::vector<float>{3, 4, 5, 9, 10, 11}));
    EXPECT_EQ(values(citrius::transpose(tensor({1, 2, 3, 4, 5, 6}, {2, 3}), 0, 1)),
              (std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST_P(DeviceOperationConformanceTest, ReductionsMatchTorchDimensionsAndKeepdim) {
    const auto input = tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 2, 3});

    EXPECT_EQ(values(citrius::sum(input, -1)), (std::vector<float>{6, 15, 24, 33}));
    EXPECT_EQ(values(citrius::sum(input, std::vector<std::int64_t>{0, 2})),
              (std::vector<float>{30, 48}));
    EXPECT_EQ(values(citrius::mean(input, 1, true)),
              (std::vector<float>{2.5f, 3.5f, 4.5f, 8.5f, 9.5f, 10.5f}));
    EXPECT_EQ(values(citrius::max(input, 0)), (std::vector<float>{7, 8, 9, 10, 11, 12}));
    expect_near(values(citrius::variance(input, -1)),
                {2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f});
}

TEST_P(DeviceOperationConformanceTest, LinearStyleMatmulConsumesTransposedWeightView) {
    const auto input = tensor({1, 2, 3, 4, 5, 6}, {2, 3});
    const auto weight = tensor({1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {4, 3});
    const auto output = citrius::matmul(input, citrius::transpose(weight, 0, 1));

    EXPECT_EQ(output.shape(), (citrius::Shape{2, 4}));
    EXPECT_EQ(values(output), (std::vector<float>{1, 2, 3, 6, 4, 5, 6, 15}));
}

TEST_P(DeviceOperationConformanceTest, BatchedMatmulBroadcastsAndHandlesStridedRightOperand) {
    const auto left = tensor({1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2});
    const auto right_storage = tensor({1, 0, 2, 0, 1, 3}, {3, 2});
    const auto right = citrius::transpose(right_storage, 0, 1);
    const auto output = citrius::matmul(left, right);

    EXPECT_EQ(output.shape(), (citrius::Shape{2, 2, 3}));
    EXPECT_EQ(values(output), (std::vector<float>{1, 2, 7, 3, 6, 15, 5, 10, 23, 7, 14, 31}));
}

TEST_P(DeviceOperationConformanceTest, MaskedFillAndSoftmaxMatchTorchValues) {
    const auto input = tensor({1, 2, 3, 10000, 10001, 10002}, {2, 3});
    const auto mask = citrius::from_vector(std::vector<bool>{false, true, false}, {1, 3}, device());
    const auto masked = citrius::masked_fill(input, mask, -10000.0f);
    const auto probabilities = citrius::nn::functional::softmax(masked, -1);

    expect_near(values(probabilities),
                {0.11920292f, 0.0f, 0.88079708f, 0.11920292f, 0.0f, 0.88079708f}, 2e-5f);
}

TEST_P(DeviceOperationConformanceTest, ScaledDotProductAttentionMatchesTorchReference) {
    const auto query = tensor({1, 0, 0, 1}, {1, 1, 2, 2});
    const auto key = tensor({1, 0, 0, 1}, {1, 1, 2, 2});
    const auto value = tensor({1, 2, 3, 4}, {1, 1, 2, 2});
    const auto mask =
        citrius::from_vector(std::vector<bool>{false, true, false, false}, {2, 2}, device());
    const auto output =
        citrius::nn::functional::scaled_dot_product_attention(query, key, value, mask);

    expect_near(values(output), {1.0f, 2.0f, 2.3395231f, 3.3395231f}, 2e-5f);
}

TEST_P(DeviceOperationConformanceTest, GatherAndArgmaxMatchTorchIndexingSemantics) {
    const auto table = tensor({10, 11, 12, 20, 21, 22, 30, 31, 32}, {3, 3});
    const auto indices = citrius::from_vector(std::vector<std::int64_t>{2, 0, 2}, {3}, device());
    EXPECT_EQ(values(citrius::gather_rows(table, indices)),
              (std::vector<float>{30, 31, 32, 10, 11, 12, 30, 31, 32}));

    const auto logits = tensor({1, 7, 7, -2, -3, -1, -1, -4}, {2, 4});
    EXPECT_EQ(int64_values(citrius::argmax(logits, -1)), (std::vector<std::int64_t>{1, 1}));
}

INSTANTIATE_TEST_SUITE_P(AllCompiledDevices, DeviceOperationConformanceTest,
                         testing::ValuesIn(device_cases()),
                         [](const testing::TestParamInfo<DeviceCase>& info) {
                             return std::string(info.param.name);
                         });

} // namespace
