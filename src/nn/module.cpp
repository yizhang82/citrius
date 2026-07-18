#include "nn/module.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace citrius::nn {
namespace {

void validate_name(const std::string& name) {
    if (name.empty()) throw std::invalid_argument("module item name cannot be empty");
    if (name.find('.') != std::string::npos) {
        throw std::invalid_argument("module item name cannot contain '.'");
    }
}

} // namespace

Tensor Module::operator()(const Tensor& input) {
    return forward(input);
}

std::vector<Tensor> Module::parameters(bool recurse) const {
    const auto named = named_parameters(recurse);
    std::vector<Tensor> result;
    result.reserve(named.size());
    for (const auto& [name, value] : named) {
        (void)name;
        result.push_back(value);
    }
    return result;
}

std::vector<Module::NamedParameter> Module::named_parameters(bool recurse) const {
    std::vector<NamedParameter> result = parameters_;
    if (!recurse) return result;

    for (const auto& [child_name, child] : children_) {
        for (auto [parameter_name, value] : child->named_parameters(true)) {
            result.emplace_back(child_name + "." + parameter_name, std::move(value));
        }
    }
    return result;
}

Tensor& Module::parameter(const std::string& name) {
    const auto found = std::find_if(
        parameters_.begin(), parameters_.end(),
        [&name](const NamedParameter& item) { return item.first == name; });
    if (found == parameters_.end()) throw std::out_of_range("parameter not found: " + name);
    return found->second;
}

const Tensor& Module::parameter(const std::string& name) const {
    const auto found = std::find_if(
        parameters_.begin(), parameters_.end(),
        [&name](const NamedParameter& item) { return item.first == name; });
    if (found == parameters_.end()) throw std::out_of_range("parameter not found: " + name);
    return found->second;
}

std::vector<std::shared_ptr<Module>> Module::children() const {
    std::vector<std::shared_ptr<Module>> result;
    result.reserve(children_.size());
    for (const auto& [name, child] : children_) {
        (void)name;
        result.push_back(child);
    }
    return result;
}

std::vector<Module::NamedModule> Module::named_children() const {
    return children_;
}

Tensor& Module::register_parameter(std::string name, Tensor value) {
    validate_name(name);
    if (contains_name(name)) throw std::invalid_argument("module item already registered: " + name);
    parameters_.emplace_back(std::move(name), std::move(value));
    return parameters_.back().second;
}

void Module::register_module_impl(std::string name, std::shared_ptr<Module> module) {
    validate_name(name);
    if (!module) throw std::invalid_argument("cannot register a null module");
    if (contains_name(name)) throw std::invalid_argument("module item already registered: " + name);
    children_.emplace_back(std::move(name), std::move(module));
}

bool Module::contains_name(const std::string& name) const {
    const auto parameter = std::find_if(
        parameters_.begin(), parameters_.end(),
        [&name](const NamedParameter& item) { return item.first == name; });
    const auto child = std::find_if(
        children_.begin(), children_.end(),
        [&name](const NamedModule& item) { return item.first == name; });
    return parameter != parameters_.end() || child != children_.end();
}

} // namespace citrius::nn
