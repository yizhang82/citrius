#pragma once

#include "citrius.h"
#include <cstdlib>
#include <iostream>
#include <string>

namespace citrius::examples {

inline void print_device_help(std::ostream& stream, const char* program_name) {
    stream
        << "Usage: " << program_name << " [--cpu|--cuda|--metal]\n"
        << "\n"
        << "Device options:\n"
        << "  --cpu    Run on CPU (default)\n"
        << "  --cuda   Run on CUDA\n"
        << "  --metal  Run on Metal\n";
}

inline Device parse_device(int argc, char** argv, const char* program_name) {
    if (argc == 1) return Device::cpu();

    if (argc == 2) {
        const std::string argument = argv[1];
        if (argument == "--help" || argument == "-h") {
            print_device_help(std::cout, program_name);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--cpu") return Device::cpu();
        if (argument == "--cuda") return Device::cuda();
        if (argument == "--metal") return Device::metal();
        std::cerr << "Unrecognized argument: " << argument << "\n\n";
    } else {
        std::cerr << "Unexpected number of arguments.\n\n";
    }

    print_device_help(std::cerr, program_name);
    std::exit(EXIT_FAILURE);
}

} // namespace citrius::examples
