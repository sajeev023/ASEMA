#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    std::string user_prompt = "Explain what a neural network is in simple terms.";
    std::string formatted = "<｜User｜>" + user_prompt + "<｜Assistant｜>";

    std::cout << "Testing prompt: \"" << formatted << "\"\n";
    std::cout << "Output: ";
    auto tokens = runner.generate(formatted, 32, [](const asema::m8::TokenGenerationTelemetry& tel) {
        std::cout << tel.token_str << std::flush;
    });
    std::cout << "\n\nTotal tokens generated: " << tokens.size() - runner.tokenizer().encode(formatted).size() << "\n";
    return 0;
}
