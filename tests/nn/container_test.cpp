#include "nn/dropout.h"
#include "nn/linear.h"
#include "nn/module_list.h"

#include "impl/cpu_storage.h"

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

TEST(DropoutTest, IsIdentityDuringInference) {
    citrius::nn::Dropout dropout(0.25f);
    const citrius::Tensor input(std::vector<float>{1.0f, 2.0f, 3.0f});

    const citrius::Tensor output = dropout(input);

    EXPECT_FLOAT_EQ(dropout.probability(), 0.25f);
    EXPECT_EQ(output.storage(), input.storage());
    EXPECT_THROW(citrius::nn::Dropout(-0.1f), std::invalid_argument);
    EXPECT_THROW(citrius::nn::Dropout(1.1f), std::invalid_argument);
}

TEST(ModuleListTest, AppliesRegisteredModulesInOrder) {
    auto first = std::make_shared<citrius::nn::Linear>(2, 2);
    first->weight() = citrius::Tensor(std::vector<float>{1, 0, 0, 2}, {2, 2});
    first->bias() = citrius::Tensor(std::vector<float>{0, 0});
    auto second = std::make_shared<citrius::nn::Linear>(2, 1);
    second->weight() = citrius::Tensor(std::vector<float>{3, 4}, {1, 2});
    second->bias() = citrius::Tensor(std::vector<float>{5});
    citrius::nn::ModuleList modules({first, second});

    const auto output = modules(citrius::Tensor(std::vector<float>{1, 2}, {1, 2}));
    const auto named = modules.named_parameters();

    EXPECT_EQ(values(output), std::vector<float>({24}));
    ASSERT_EQ(named.size(), 4u);
    EXPECT_EQ(named[0].first, "0.weight");
    EXPECT_EQ(named[1].first, "0.bias");
    EXPECT_EQ(named[2].first, "1.weight");
    EXPECT_EQ(named[3].first, "1.bias");
    EXPECT_EQ(modules.size(), 2u);
    EXPECT_EQ(modules.at(0), first);
}

TEST(ModuleListTest, RejectsNullModules) {
    citrius::nn::ModuleList modules;
    EXPECT_TRUE(modules.empty());
    EXPECT_THROW(modules.append(nullptr), std::invalid_argument);
}
