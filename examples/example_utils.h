#pragma once

#include "citrius.h"
#include <iostream>
#include <stdexcept>
#include <string>

namespace citrius::examples {

inline Device parse_device(int argc, char** argv, const char* program_name) {
    if (argc != 2) {
        throw std::invalid_argument(
            std::string("Usage: ") + program_name + " --cpu|--cuda|--metal");
    }

    const std::string argument = argv[1];
    if (argument == "--cpu") return Device::cpu();
    if (argument == "--cuda") return Device::cuda();
    if (argument == "--metal") return Device::metal();

    throw std::invalid_argument(
        std::string("Usage: ") + program_name + " --cpu|--cuda|--metal");
}

inline void print_tensor(const Tensor& tensor) {
    std::cout << tensor << '\n';
}

template <typename Example>
int run(int argc, char** argv, const char* program_name, Example example) {
    try {
        example(parse_device(argc, argv, program_name));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace citrius::examples
