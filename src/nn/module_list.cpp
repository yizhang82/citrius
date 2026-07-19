#include "nn/module_list.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace citrius::nn {

ModuleList::ModuleList(std::vector<std::shared_ptr<Module>> modules) {
    for (auto& module : modules) append(std::move(module));
}

void ModuleList::append(std::shared_ptr<Module> module) {
    if (!module) throw std::invalid_argument("ModuleList cannot contain a null module");
    modules_.push_back(register_module(std::to_string(modules_.size()), std::move(module)));
}

std::size_t ModuleList::size() const {
    return modules_.size();
}

bool ModuleList::empty() const {
    return modules_.empty();
}

std::shared_ptr<Module> ModuleList::at(std::size_t index) const {
    return modules_.at(index);
}

std::shared_ptr<Module> ModuleList::operator[](std::size_t index) const {
    return modules_[index];
}

Tensor ModuleList::forward(const Tensor& input) {
    if (!input.defined()) throw std::invalid_argument("ModuleList input must be defined");
    Tensor output = input;
    for (const auto& module : modules_) output = (*module)(output);
    return output;
}

} // namespace citrius::nn
