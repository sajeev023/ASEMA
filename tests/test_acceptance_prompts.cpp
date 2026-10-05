#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

void test_acceptance(asema::m8::M8ModelRunner& runner, const std::string& prompt, int max_tokens) {
    runner.reset_state();
    std::cout << "======================================================================\n";
    std::cout << "[ACCEPTANCE PROMPT]: \"" << prompt << "\"\n";
    std::cout << "[GENERATED OUTPUT]: ";

    runner.generate(prompt, max_tokens, [](const asema::m8::TokenGenerationTelemetry& tel) {
        std::cout << tel.token_str << std::flush;
    });
    std::cout << "\n\n";
}

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    test_acceptance(runner, "Explain what a neural network is in simple terms.", 32);
    test_acceptance(runner, "What is machine learning?", 32);
    test_acceptance(runner, "Explain gravity to a beginner.", 32);
    test_acceptance(runner, "Write a short explanation of how computers store information.", 32);

    return 0;
}
