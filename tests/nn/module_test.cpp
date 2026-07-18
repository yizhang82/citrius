#include "nn/module.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

class TestModule final : public citrius::nn::Module {
public:
    TestModule() {
        register_parameter("weight", citrius::Tensor(std::vector<float>{2.0f}));
    }

    citrius::Tensor forward(const citrius::Tensor& input) override {
        return input;
    }

    citrius::Tensor& add_parameter(std::string name, citrius::Tensor value) {
        return register_parameter(std::move(name), std::move(value));
    }

    std::shared_ptr<TestModule> add_child(
        std::string name,
        std::shared_ptr<TestModule> child) {
        return register_module(std::move(name), std::move(child));
    }
};

} // namespace

TEST(ModuleTest, CallsForward) {
    TestModule module;
    const citrius::Tensor input(std::vector<float>{1.0f, 2.0f});

    const auto output = module(input);

    EXPECT_EQ(output.storage(), input.storage());
}

TEST(ModuleTest, RegistersAndFindsParameters) {
    TestModule module;

    EXPECT_EQ(module.parameters(false).size(), 1u);
    EXPECT_EQ(module.named_parameters(false).front().first, "weight");
    EXPECT_EQ(module.parameter("weight").shape(), citrius::Shape({1}));
    EXPECT_THROW(module.parameter("missing"), std::out_of_range);
}

TEST(ModuleTest, RecursivelyFindsChildParameters) {
    TestModule parent;
    auto child = std::make_shared<TestModule>();

    EXPECT_EQ(parent.add_child("layer", child), child);

    const auto named = parent.named_parameters();
    ASSERT_EQ(named.size(), 2u);
    EXPECT_EQ(named[0].first, "weight");
    EXPECT_EQ(named[1].first, "layer.weight");
    EXPECT_EQ(parent.children().front(), child);
    EXPECT_EQ(parent.named_children().front().first, "layer");
}

TEST(ModuleTest, RejectsInvalidOrDuplicateNames) {
    TestModule module;

    EXPECT_THROW(
        module.add_parameter("bad.name", citrius::Tensor(std::vector<float>{1.0f})),
        std::invalid_argument);
    EXPECT_THROW(
        module.add_parameter("weight", citrius::Tensor(std::vector<float>{1.0f})),
        std::invalid_argument);
    EXPECT_THROW(module.add_child("child", nullptr), std::invalid_argument);
    EXPECT_THROW(module.add_child("weight", std::make_shared<TestModule>()), std::invalid_argument);
}
