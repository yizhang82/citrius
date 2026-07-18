#pragma once

#include "tensor.h"

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace citrius::nn {

class Module {
public:
    using NamedParameter = std::pair<std::string, Tensor>;
    using NamedModule = std::pair<std::string, std::shared_ptr<Module>>;

    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    Tensor operator()(const Tensor& input);

    std::vector<Tensor> parameters(bool recurse = true) const;
    std::vector<NamedParameter> named_parameters(bool recurse = true) const;
    Tensor& parameter(const std::string& name);
    const Tensor& parameter(const std::string& name) const;

    std::vector<std::shared_ptr<Module>> children() const;
    std::vector<NamedModule> named_children() const;

protected:
    Tensor& register_parameter(std::string name, Tensor parameter);

    template <typename ModuleType>
    std::shared_ptr<ModuleType> register_module(
        std::string name,
        std::shared_ptr<ModuleType> module) {
        static_assert(std::is_base_of_v<Module, ModuleType>);
        register_module_impl(std::move(name), module);
        return module;
    }

private:
    void register_module_impl(std::string name, std::shared_ptr<Module> module);
    bool contains_name(const std::string& name) const;

    std::vector<NamedParameter> parameters_;
    std::vector<NamedModule> children_;
};

} // namespace citrius::nn
