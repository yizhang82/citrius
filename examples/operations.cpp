#include "example_utils.h"

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const citrius::Device device =
        citrius::examples::parse_device(argc, argv, "example_operations");
    const citrius::Tensor left(
        std::vector<float>{1, 2, 3, 4},
        {2, 2},
        device);
    const citrius::Tensor right(
        std::vector<float>{5, 6, 7, 8},
        {2, 2},
        device);

    std::cout << "add: " << citrius::add(left, right) << '\n';
    std::cout << "sub: " << citrius::sub(left, right) << '\n';
    std::cout << "matmul: " << citrius::matmul(left, right) << '\n';
}
