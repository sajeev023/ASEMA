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
    const int D = 5120;
    std::vector<float> h(D);
    runner.lookup_token_embedding(65106, h.data()); // "Explain"

    asema::m8::LayerTelemetry tel;
    std::vector<float> layer_out(D);

    for (int l = 0; l < 40; ++l) {
        runner.active_layer().set_layer_id(l);
        runner.active_layer().forward(h.data(), layer_out.data(), 0, tel);
        h = layer_out;

        if (l == 0 || l == 1 || l == 9 || l == 19 || l == 39) {
            std::vector<float> normed_h(D);
            std::vector<float> logits;
            runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), D);
            runner.compute_lm_head_logits(normed_h.data(), logits);

            std::vector<std::pair<float, int>> top;
            for (size_t v = 0; v < logits.size(); ++v) top.push_back({logits[v], (int)v});
            std::partial_sort(top.begin(), top.begin() + 3, top.end(), std::greater<std::pair<float, int>>());

            std::cout << "After Layer " << std::setw(2) << l
                      << " -> Top-1: ID " << top[0].second << " ('" << runner.tokenizer().decode({top[0].second}) << "') logit " << top[0].first
                      << ", Top-2: ID " << top[1].second << " ('" << runner.tokenizer().decode({top[1].second}) << "') logit " << top[1].first << "\n";
        }
    }
    return 0;
}
