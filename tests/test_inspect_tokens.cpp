#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int main() {
    asema::m8::M8ModelRunner runner;
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }
    std::string prompt = "Explain what a neural network is in simple terms.";
    auto prompt_ids = runner.tokenizer().encode(prompt);

    runner.reset_state();
    std::vector<float> logits;
    std::vector<asema::m8::LayerTelemetry> layer_tels;

    for (size_t p = 0; p < prompt_ids.size(); ++p) {
        runner.step(prompt_ids[p], static_cast<int>(p), logits, layer_tels, /*compute_logits=*/true);

        std::vector<std::pair<float, int>> top;
        for (size_t v = 0; v < logits.size(); ++v) {
            top.push_back({logits[v], static_cast<int>(v)});
        }
        std::partial_sort(top.begin(), top.begin() + 3, top.end(), std::greater<std::pair<float, int>>());

        std::cout << "Pos " << p << " [Token " << prompt_ids[p] << " '" << runner.tokenizer().decode({prompt_ids[p]}) << "'] -> Top-1: ID "
                  << top[0].second << " ('" << runner.tokenizer().decode({top[0].second}) << "') logit " << top[0].first
                  << ", Top-2: ID " << top[1].second << " ('" << runner.tokenizer().decode({top[1].second}) << "') logit " << top[1].first << "\n";
    }
    return 0;
}
