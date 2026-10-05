#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

void test_prompt(asema::m8::M8ModelRunner& runner, const std::string& prompt) {
    runner.reset_state();
    auto prompt_ids = runner.tokenizer().encode(prompt);
    std::cout << "======================================================================\n";
    std::cout << "PROMPT: \"" << prompt << "\" (" << prompt_ids.size() << " tokens)\n";
    std::cout << "Tokens: ";
    for (int id : prompt_ids) {
        std::cout << id << " ('" << runner.tokenizer().decode({id}) << "') ";
    }
    std::cout << "\n";

    std::vector<float> logits;
    std::vector<asema::m8::LayerTelemetry> layer_tels;
    for (size_t p = 0; p + 1 < prompt_ids.size(); ++p) {
        runner.step(prompt_ids[p], static_cast<int>(p), logits, layer_tels, /*compute_logits=*/false);
    }
    int last_p = static_cast<int>(prompt_ids.size()) - 1;
    runner.step(prompt_ids[last_p], last_p, logits, layer_tels, /*compute_logits=*/true);

    std::vector<std::pair<float, int>> ranked;
    for (int i = 0; i < (int)logits.size(); ++i) {
        ranked.push_back({logits[i], i});
    }
    std::sort(ranked.rbegin(), ranked.rend());
    std::cout << "Top 10 predicted next tokens:\n";
    for (int k = 0; k < 10; ++k) {
        std::cout << "  #" << k + 1 << ": ID " << std::setw(6) << ranked[k].second
                  << " (logit: " << std::fixed << std::setprecision(4) << ranked[k].first << ")"
                  << " -> '" << runner.tokenizer().decode({ranked[k].second}) << "'\n";
    }
}

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    test_prompt(runner, "Explain what a neural network is in simple terms.");
    test_prompt(runner, "Explain what a neural network is in simple terms:\n");
    test_prompt(runner, "A neural network is ");
    test_prompt(runner, "What is machine learning?");
    test_prompt(runner, "Explain gravity to a beginner.");

    return 0;
}
