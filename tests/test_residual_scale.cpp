#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    std::vector<float> h(5120);
    runner.lookup_token_embedding(65106, h.data()); // "Explain"

    double in_l2 = 0.0;
    for (float v : h) in_l2 += v * v;
    std::cout << "Embedding L2 norm: " << std::sqrt(in_l2) << "\n";

    // Run Layer 0
    std::vector<float> norm_in(5120);
    std::vector<float> attn_out(5120);
    std::vector<float> ones(5120, 1.0f);
    runner.rms_norm(h.data(), ones.data(), norm_in.data(), 5120);
    runner.active_layer().attn().forward(norm_in.data(), attn_out.data(), 0);

    double attn_l2 = 0.0;
    for (float v : attn_out) attn_l2 += v * v;
    std::cout << "Attn Out L2 norm: " << std::sqrt(attn_l2) << "\n";

    // MoE out
    std::vector<float> norm_ffn(5120);
    runner.rms_norm(h.data(), ones.data(), norm_ffn.data(), 5120);
    auto sel = runner.active_layer().router().route(norm_ffn.data());

    std::vector<float> moe_out(5120, 0.0f);
    asema::m8::ExpertPayload payload;
    runner.byte_loader()->load_expert_payload(0, 0, payload);
    if (runner.gpu_kernel() && runner.gpu_kernel()->is_initialized()) {
        std::vector<const uint8_t*> sc(6, payload.scales.data());
        std::vector<const uint8_t*> wt(6, payload.weights.data());
        runner.gpu_kernel()->forward_top6_layer(sc, wt, sel.expert_weights, norm_ffn.data(), moe_out.data());
    }
    double moe_l2 = 0.0;
    for (float v : moe_out) moe_l2 += v * v;
    std::cout << "MoE Out L2 norm: " << std::sqrt(moe_l2) << "\n";

    return 0;
}
