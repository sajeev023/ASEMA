#include "asema/m8/m8_reference_engine.hpp"
#include "asema/m8/m8_fp8_lut.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <cstring>

namespace asema {
namespace m8 {

void M8ReferenceEngine::rms_norm(const float* x, const float* w, float* out, int dim, float eps) {
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
        sum_sq += static_cast<double>(x[i]) * x[i];
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / dim) + eps);
    for (int i = 0; i < dim; ++i) {
        out[i] = x[i] * scale * w[i];
    }
}

void M8ReferenceEngine::matvec(const float* A, const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        const float* row = A + r * cols;
        for (int c = 0; c < cols; ++c) {
            acc += static_cast<double>(row[c]) * x[c];
        }
        out[r] = static_cast<float>(acc);
    }
}

void M8ReferenceEngine::swiglu(const float* gate, const float* up, float* act, int dim, float limit) {
    for (int i = 0; i < dim; ++i) {
        float g = (limit > 0.0f) ? std::min(gate[i], limit) : gate[i];
        float u = (limit > 0.0f) ? std::clamp(up[i], -limit, limit) : up[i];
        float silu = g / (1.0f + std::exp(-g));
        act[i] = silu * u;
    }
}

M8ReferenceEngine::RouterResult M8ReferenceEngine::route(const float* x, const float* gate_w, const float* gate_b,
                                                        int num_experts, int dim, int top_k, float route_scale) {
    std::vector<float> logits(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        double dot = 0.0;
        const float* w_row = gate_w + e * dim;
        for (int c = 0; c < dim; ++c) {
            dot += static_cast<double>(w_row[c]) * x[c];
        }
        logits[e] = static_cast<float>(dot + gate_b[e]);
    }

    std::vector<float> scores(num_experts);
    for (int e = 0; e < num_experts; ++e) {
        float l = logits[e];
        float sp = (l > 20.0f) ? l : std::log1p(std::exp(l));
        scores[e] = std::sqrt(sp);
    }

    std::vector<int> all_indices(num_experts);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::partial_sort(all_indices.begin(), all_indices.begin() + top_k, all_indices.end(),
                      [&](int a, int b) { return scores[a] > scores[b]; });

    RouterResult res;
    res.indices.assign(all_indices.begin(), all_indices.begin() + top_k);

    float max_s = -1e9f;
    for (int idx : res.indices) {
        if (scores[idx] > max_s) max_s = scores[idx];
    }

    float sum_exp = 0.0f;
    std::vector<float> exps(top_k);
    for (int k = 0; k < top_k; ++k) {
        exps[k] = std::exp(scores[res.indices[k]] - max_s);
        sum_exp += exps[k];
    }

    res.weights.resize(top_k);
    for (int k = 0; k < top_k; ++k) {
        res.weights[k] = (exps[k] / sum_exp) * route_scale;
    }

    return res;
}

void M8ReferenceEngine::dequant_fp8(const uint8_t* raw_w, const uint8_t* raw_s,
                                    float* out, int rows, int cols) {
    const int s_cols = cols / 32;
    for (int r = 0; r < rows; ++r) {
        int s_row = r / 32;
        const uint8_t* s_row_ptr = raw_s + s_row * s_cols;
        const uint8_t* w_row_ptr = raw_w + r * cols;
        float* out_row_ptr = out + r * cols;
        for (int c = 0; c < cols; ++c) {
            float scale = k_e8m0_lut[s_row_ptr[c / 32]];
            float val = k_e4m3_lut[w_row_ptr[c]];
            out_row_ptr[c] = val * scale;
        }
    }
}

void M8ReferenceEngine::expert_forward(const float* w1, const float* w2, const float* w3,
                                      const float* x, float* out, int hidden_dim, int inter_dim,
                                      float router_weight, bool accumulate, float swiglu_limit) {
    std::vector<float> gate(inter_dim);
    std::vector<float> up(inter_dim);
    std::vector<float> act(inter_dim);
    std::vector<float> raw_out(hidden_dim);

    matvec(w1, x, gate.data(), inter_dim, hidden_dim);
    matvec(w3, x, up.data(), inter_dim, hidden_dim);
    swiglu(gate.data(), up.data(), act.data(), inter_dim, swiglu_limit);
    matvec(w2, act.data(), raw_out.data(), hidden_dim, inter_dim);

    if (accumulate) {
        for (int i = 0; i < hidden_dim; ++i) out[i] += raw_out[i] * router_weight;
    } else {
        for (int i = 0; i < hidden_dim; ++i) out[i] = raw_out[i] * router_weight;
    }
}

void M8ReferenceEngine::layer_forward_scalar(const float* in, float* out,
                                            const LayerReferenceWeights& weights,
                                            int hidden_dim, int inter_dim) {
    // 1. Attention norm & residual
    std::vector<float> norm_in(hidden_dim);
    rms_norm(in, weights.attn_norm.data(), norm_in.data(), hidden_dim);

    std::vector<float> residual_attn(hidden_dim);
    for (int i = 0; i < hidden_dim; ++i) {
        residual_attn[i] = in[i]; // Identity attention path in pure scalar layer
    }

    // 2. FFN norm
    std::vector<float> norm_ffn(hidden_dim);
    rms_norm(residual_attn.data(), weights.ffn_norm.data(), norm_ffn.data(), hidden_dim);

    // 3. Shared expert
    std::vector<float> moe_out(hidden_dim, 0.0f);
    if (!weights.shared_w1.empty() && !weights.shared_w2.empty() && !weights.shared_w3.empty()) {
        expert_forward(weights.shared_w1.data(), weights.shared_w2.data(), weights.shared_w3.data(),
                       norm_ffn.data(), moe_out.data(), hidden_dim, inter_dim, 1.0f, false);
    }

    // 4. Output residual
    for (int i = 0; i < hidden_dim; ++i) {
        out[i] = residual_attn[i] + moe_out[i];
    }
}

int M8ReferenceEngine::lm_head_argmax(const float* final_hidden, const float* final_norm,
                                     const uint16_t* lm_head_bf16, int vocab_size, int hidden_dim,
                                     float* out_max_logit) {
    std::vector<float> normed(hidden_dim);
    rms_norm(final_hidden, final_norm, normed.data(), hidden_dim);

    int best_id = 0;
    float best_val = -1e30f;

    for (int v = 0; v < vocab_size; ++v) {
        const uint16_t* row = lm_head_bf16 + v * hidden_dim;
        double dot = 0.0;
        for (int i = 0; i < hidden_dim; ++i) {
            uint32_t u = static_cast<uint32_t>(row[i]) << 16;
            float val = *reinterpret_cast<const float*>(&u);
            dot += static_cast<double>(normed[i]) * val;
        }
        float logit = static_cast<float>(dot);
        if (logit > best_val) {
            best_val = logit;
            best_id = v;
        }
    }

    if (out_max_logit) {
        *out_max_logit = best_val;
    }
    return best_id;
}

} // namespace m8
} // namespace asema
