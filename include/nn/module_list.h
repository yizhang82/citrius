#pragma once

#include "nn/module.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace citrius::nn {

/// Registered child-module container that applies its modules in order.
class ModuleList final : public Module {
public:
    ModuleList() = default;
    explicit ModuleList(std::vector<std::shared_ptr<Module>> modules);

    void append(std::shared_ptr<Module> module);
    std::size_t size() const;
    bool empty() const;
    std::shared_ptr<Module> at(std::size_t index) const;
    std::shared_ptr<Module> operator[](std::size_t index) const;

    Tensor forward(const Tensor& input) override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace citrius::nn
