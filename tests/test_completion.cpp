#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

void test_prompt_gen(asema::m8::M8ModelRunner& runner, const std::string& prompt, int num_tokens) {
    runner.reset_state();
    std::cout << "======================================================================\n";
    std::cout << "PROMPT: \"" << prompt << "\"\n";
    std::cout << "Output: " << prompt;

    auto tokens = runner.generate(prompt, num_tokens, [](const asema::m8::TokenGenerationTelemetry& tel) {
        std::cout << tel.token_str << std::flush;
    });
    std::cout << "\n";
}

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    test_prompt_gen(runner, "A neural network is a ", 16);
    test_prompt_gen(runner, "Machine learning is a subset of ", 16);
    test_prompt_gen(runner, "In simple terms, a computer is ", 16);
    test_prompt_gen(runner, "DeepSeek is an artificial intelligence ", 16);

    return 0;
}
