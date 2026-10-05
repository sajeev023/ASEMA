#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace asema {
namespace m8 {

// Independent, clean-room, unoptimized scalar CPU reference implementation
// Used strictly as the ground-truth baseline to verify numerical correctness
// of SIMD/GPU/fused production kernels (Section 19).
class M8ReferenceEngine {
public:
    M8ReferenceEngine() = default;

    // 1. RMSNorm Reference: y = (x / sqrt(mean(x^2) + eps)) * w
    static void rms_norm(const float* x, const float* w, float* out, int dim, float eps = 1e-20f);

    // 2. Scalar Matrix-Vector Multiply: out = A * x (A is [rows x cols] row-major)
    static void matvec(const float* A, const float* x, float* out, int rows, int cols);

    // 3. SwiGLU Reference: act = (silu(clamp(gate, limit)) * clamp(up, -limit, limit))
    static void swiglu(const float* gate, const float* up, float* act, int dim, float limit = 10.0f);

    // 4. Router Reference (sqrtsoftplus + noaux_tc top-k selection)
    struct RouterResult {
        std::vector<int> indices;
        std::vector<float> weights;
    };
    static RouterResult route(const float* x, const float* gate_w, const float* gate_b,
                              int num_experts, int dim, int top_k = 6, float route_scale = 1.5f);

    // 5. Dequantize FP8 Block (E4M3 weight * E8M0 scale) into pure float
    static void dequant_fp8(const uint8_t* raw_w, const uint8_t* raw_s,
                            float* out, int rows, int cols);

    // 6. Expert Forward Reference (FP32 precision)
    static void expert_forward(const float* w1, const float* w2, const float* w3,
                               const float* x, float* out, int hidden_dim, int inter_dim,
                               float router_weight = 1.0f, bool accumulate = false,
                               float swiglu_limit = 10.0f);

    // 7. Full Layer Forward Reference
    struct LayerReferenceWeights {
        std::vector<float> attn_norm;
        std::vector<float> ffn_norm;
        std::vector<float> router_gate_w; // [num_experts x hidden_dim]
        std::vector<float> router_gate_b; // [num_experts]
        std::vector<float> shared_w1;     // [inter_dim x hidden_dim]
        std::vector<float> shared_w2;     // [hidden_dim x inter_dim]
        std::vector<float> shared_w3;     // [inter_dim x hidden_dim]
    };

    static void layer_forward_scalar(const float* in, float* out,
                                     const LayerReferenceWeights& weights,
                                     int hidden_dim = 5120, int inter_dim = 2304);

    // 8. LM Head & Argmax Reference
    static int lm_head_argmax(const float* final_hidden, const float* final_norm,
                              const uint16_t* lm_head_bf16, int vocab_size, int hidden_dim,
                              float* out_max_logit = nullptr);
};

} // namespace m8
} // namespace asema
