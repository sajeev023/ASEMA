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
    std::cout << "Testing prompt: \"" << prompt << "\"\n";

    auto prompt_ids = runner.tokenizer().encode(prompt);
    std::cout << "Encoded tokens (" << prompt_ids.size() << "): ";
    for (int id : prompt_ids) std::cout << id << " ";
    std::cout << "\n\n";

    // Trace Layer 0 MoE execution
    const int D = 5120;
    std::vector<float> h(D);
    runner.lookup_token_embedding(prompt_ids.back(), h.data());

    std::vector<float> normed_h(D);
    std::vector<float> logits;
    runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), D);
    runner.compute_lm_head_logits(normed_h.data(), logits);

    std::vector<std::pair<float, int>> top;
    for (size_t v = 0; v < logits.size(); ++v) top.push_back({logits[v], (int)v});
    std::partial_sort(top.begin(), top.begin() + 10, top.end(), std::greater<std::pair<float, int>>());

    std::cout << "Top-10 tokens for last prompt token '" << runner.tokenizer().decode({prompt_ids.back()}) << "':\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << "  #" << (i + 1) << " | ID " << std::setw(6) << top[i].second
                  << " | Logit " << std::fixed << std::setprecision(4) << top[i].first
                  << " | Piece: \"" << runner.tokenizer().decode({top[i].second}) << "\"\n";
    }

    return 0;
}
