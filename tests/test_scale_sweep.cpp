#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

int main() {
    asema::m8::M8ModelRunner runner;
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }
    const int D = 5120;
    std::vector<float> h0(D);
    runner.lookup_token_embedding(65106, h0.data()); // "Explain"

    std::vector<float> scales = { 0.0125f, 0.005f, 0.0025f, 0.00125f, 0.000625f, 0.00025f };

    for (float s : scales) {
        std::vector<float> h = h0;
        std::vector<float> norm_in(D);
        std::vector<float> attn_out(D);
        std::vector<float> residual_attn(D);
        std::vector<float> norm_ffn(D);
        std::vector<float> moe_out(D);
        std::vector<float> ones(D, 1.0f);

        for (int l = 0; l < 40; ++l) {
            runner.active_layer().set_layer_id(l);
            runner.rms_norm(h.data(), ones.data(), norm_in.data(), D);
            runner.active_layer().attn().forward(norm_in.data(), attn_out.data(), 0);

            for (int i = 0; i < D; ++i) residual_attn[i] = h[i] + s * attn_out[i];

            runner.rms_norm(residual_attn.data(), ones.data(), norm_ffn.data(), D);
            auto sel = runner.active_layer().router().route(norm_ffn.data());

            std::fill(moe_out.begin(), moe_out.end(), 0.0f);
            asema::m8::ExpertPayload payload;
            runner.byte_loader()->load_expert_payload(l, 0, payload);

            if (runner.gpu_kernel() && runner.gpu_kernel()->is_initialized() && sel.expert_indices.size() == 6) {
                std::vector<const uint8_t*> sc(6, payload.scales.data());
                std::vector<const uint8_t*> wt(6, payload.weights.data());
                runner.gpu_kernel()->forward_top6_layer(sc, wt, sel.expert_weights, norm_ffn.data(), moe_out.data());
            }

            for (int i = 0; i < D; ++i) h[i] = residual_attn[i] + s * moe_out[i];
        }

        std::vector<float> normed_h(D);
        std::vector<float> logits;
        runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), D);
        runner.compute_lm_head_logits(normed_h.data(), logits);

        std::vector<std::pair<float, int>> top;
        for (size_t v = 0; v < logits.size(); ++v) top.push_back({logits[v], (int)v});
        std::partial_sort(top.begin(), top.begin() + 3, top.end(), std::greater<std::pair<float, int>>());

        std::cout << "Scale " << std::fixed << std::setprecision(6) << s
                  << " -> Top-1: ID " << top[0].second << " ('" << runner.tokenizer().decode({top[0].second}) << "') logit " << top[0].first
                  << ", Top-2: ID " << top[1].second << " ('" << runner.tokenizer().decode({top[1].second}) << "') logit " << top[1].first
                  << ", Top-3: ID " << top[2].second << " ('" << runner.tokenizer().decode({top[2].second}) << "') logit " << top[2].first << "\n";
    }
    return 0;
}
