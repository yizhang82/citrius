#include "example_utils.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    return citrius::examples::run(
        argc,
        argv,
        "example_operations",
        [](citrius::Device device) {
            const citrius::Tensor left(std::vector<float>{1, 2, 3, 4}, {2, 2}, device);
            const citrius::Tensor right(std::vector<float>{5, 6, 7, 8}, {2, 2}, device);

            std::cout << "add: ";
            citrius::examples::print_tensor(citrius::add(left, right));

            std::cout << "sub: ";
            citrius::examples::print_tensor(citrius::sub(left, right));

            std::cout << "matmul: ";
            citrius::examples::print_tensor(citrius::matmul(left, right));
        });
}
