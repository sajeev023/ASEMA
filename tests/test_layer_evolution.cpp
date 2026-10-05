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
    const int D = 5120;
    std::vector<float> h(D);
    runner.lookup_token_embedding(65106, h.data()); // "Explain"

    asema::m8::LayerTelemetry tel;
    std::vector<float> layer_out(D);
    std::vector<float> normed_h(D);
    std::vector<float> logits;

    // Check embedding directly:
    runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), D);
    runner.compute_lm_head_logits(normed_h.data(), logits);
    auto get_top = [&](const std::vector<float>& lg) {
        std::vector<std::pair<float, int>> top;
        for (size_t v = 0; v < lg.size(); ++v) top.push_back({lg[v], (int)v});
        std::partial_sort(top.begin(), top.begin() + 3, top.end(), std::greater<std::pair<float, int>>());
        return top;
    };
    auto top0 = get_top(logits);
    std::cout << "Embed direct -> Top-1: " << top0[0].second << " ('" << runner.tokenizer().decode({top0[0].second}) << "') logit " << top0[0].first
              << ", Top-2: " << top0[1].second << " ('" << runner.tokenizer().decode({top0[1].second}) << "') logit " << top0[1].first << "\n";

    for (int l = 0; l < 40; ++l) {
        runner.active_layer().set_layer_id(l);
        runner.active_layer().forward(h.data(), layer_out.data(), 0, tel);
        h = layer_out;

        if (l == 0 || l == 1 || l == 2 || l == 3 || l == 4 || l == 9 || l == 19 || l == 39) {
            runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), D);
            runner.compute_lm_head_logits(normed_h.data(), logits);
            auto top = get_top(logits);
            std::cout << "After Layer " << std::setw(2) << l << " -> Top-1: " << top[0].second << " ('" << runner.tokenizer().decode({top[0].second}) << "') logit " << top[0].first
                      << ", Top-2: " << top[1].second << " ('" << runner.tokenizer().decode({top[1].second}) << "') logit " << top[1].first << "\n";
        }
    }
    return 0;
}
