#include "citrius.h"
#include "example_utils.h"

#include <iostream>
#include <memory>
#include <vector>

class CustomModule final : public citrius::nn::Module {
public:
    explicit CustomModule(citrius::Device device)
        : linear_(register_module(
              "linear",
              std::make_shared<citrius::nn::Linear>(3, 2, true, device))) {}

    citrius::Tensor forward(const citrius::Tensor& input) override {
        return (*linear_)(input);
    }

private:
    std::shared_ptr<citrius::nn::Linear> linear_;
};

int main(int argc, char** argv) {
    const citrius::Device device =
        citrius::examples::parse_device(argc, argv, "example_custom_module");
    CustomModule model(device);
    const citrius::Tensor input(
        std::vector<float>{1, 2, 3, 4, 5, 6},
        {2, 3},
        device);

    std::cout << "input: " << input << '\n';
    std::cout << "output: " << model(input) << '\n';
    std::cout << "parameters:\n";
    for (const auto& [name, parameter] : model.named_parameters()) {
        std::cout << "  " << name << ": " << parameter << '\n';
    }
}
