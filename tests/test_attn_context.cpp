#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
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
    std::cout << "Prompt: \"" << prompt << "\" (" << prompt_ids.size() << " tokens)\n";

    // Let's inspect each prompt token's embedding alignment with LM head:
    std::cout << "\nIndividual prompt tokens and their top predicted next token:\n";
    for (int tok : prompt_ids) {
        std::vector<float> h(5120);
        runner.lookup_token_embedding(tok, h.data());
        std::vector<float> normed_h(5120);
        std::vector<float> logits;
        runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), 5120);
        runner.compute_lm_head_logits(normed_h.data(), logits);

        int best_id = 0;
        float best_l = -1e9f;
        for (int v = 0; v < (int)logits.size(); ++v) {
            if (logits[v] > best_l) { best_l = logits[v]; best_id = v; }
        }
        std::cout << "  Token " << std::setw(6) << tok << " ('" << runner.tokenizer().decode({tok}) << "')"
                  << " -> Top-1: ID " << std::setw(6) << best_id << " ('" << runner.tokenizer().decode({best_id}) << "') logit=" << best_l << "\n";
    }

    // Now, what if the hidden state at position N is the mean or attention-weighted sum of prompt tokens?
    std::cout << "\nContextual average of all prompt token embeddings -> LM head:\n";
    std::vector<float> avg_h(5120, 0.0f);
    for (int tok : prompt_ids) {
        std::vector<float> h(5120);
        runner.lookup_token_embedding(tok, h.data());
        for (int d = 0; d < 5120; ++d) avg_h[d] += h[d];
    }
    for (int d = 0; d < 5120; ++d) avg_h[d] /= prompt_ids.size();

    std::vector<float> normed_avg(5120);
    std::vector<float> avg_logits;
    runner.rms_norm(avg_h.data(), runner.final_norm().data(), normed_avg.data(), 5120);
    runner.compute_lm_head_logits(normed_avg.data(), avg_logits);

    std::vector<std::pair<float, int>> top_avg;
    for (size_t v = 0; v < avg_logits.size(); ++v) top_avg.push_back({avg_logits[v], (int)v});
    std::sort(top_avg.rbegin(), top_avg.rend());
    std::cout << "Top 10 predicted tokens from contextual average:\n";
    for (int k = 0; k < 10; ++k) {
        std::cout << "  #" << k + 1 << ": ID " << std::setw(6) << top_avg[k].second
                  << " ('" << runner.tokenizer().decode({top_avg[k].second}) << "') logit=" << std::fixed << std::setprecision(4) << top_avg[k].first << "\n";
    }

    return 0;
}
