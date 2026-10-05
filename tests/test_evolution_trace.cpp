#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    std::string prompt = "Explain what a neural network is in simple terms.";
    auto prompt_ids = runner.tokenizer().encode(prompt);
    int last_tok = prompt_ids.back();

    // Get embedding of last prompt token
    std::vector<float> h(5120);
    runner.lookup_token_embedding(last_tok, h.data());

    std::vector<float> normed_h(5120);
    std::vector<float> logits;

    auto print_top = [&](int layer_idx) {
        runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), 5120);
        runner.compute_lm_head_logits(normed_h.data(), logits);

        std::vector<std::pair<float, int>> ranked;
        for (int i = 0; i < (int)logits.size(); ++i) {
            ranked.push_back({logits[i], i});
        }
        std::sort(ranked.rbegin(), ranked.rend());
        std::cout << "Layer " << std::setw(2) << layer_idx << ": "
                  << "Top 1: ID " << std::setw(6) << ranked[0].second << " ('" << runner.tokenizer().decode({ranked[0].second}) << "') logit=" << std::fixed << std::setprecision(2) << ranked[0].first
                  << " | Top 2: ID " << std::setw(6) << ranked[1].second << " ('" << runner.tokenizer().decode({ranked[1].second}) << "') logit=" << ranked[1].first
                  << " | Top 3: ID " << std::setw(6) << ranked[2].second << " ('" << runner.tokenizer().decode({ranked[2].second}) << "') logit=" << ranked[2].first << "\n";
    };

    std::cout << "Tracking prompt: \"" << prompt << "\" across 40 layers:\n";
    print_top(-1); // Pure embedding

    std::vector<float> next_h(5120);
    asema::m8::LayerTelemetry tel;
    for (int l = 0; l < 40; ++l) {
        runner.active_layer().set_layer_id(l);
        runner.active_layer().forward(h.data(), next_h.data(), 0, tel);
        h = next_h;
        print_top(l);
    }

    return 0;
}
