#include "asema/m8/m8_router.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace asema {
namespace m8 {

namespace {

// Helper to convert BF16 to FP32
inline float bf16_to_fp32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

// Vectorized AVX2 dot product between two FP32 vectors of length N (N multiple of 8)
float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 16; i += 16) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
    }
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }

    __m256 sum = _mm256_add_ps(sum0, sum1);
    // Horizontal add of 8 floats in sum
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 v128 = _mm_add_ps(lo, hi);
    v128 = _mm_hadd_ps(v128, v128);
    v128 = _mm_hadd_ps(v128, v128);
    float total = _mm_cvtss_f32(v128);

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

} // namespace

M8Router::M8Router(RouterConfig config)
    : config_(config) {
    weights_fp32_.resize(config_.num_routed_experts * config_.hidden_dim, 0.0f);
    biases_fp32_.resize(config_.num_routed_experts, 0.0f);
}

bool M8Router::load_from_buffers(const uint16_t* bf16_weights, const float* fp32_biases) {
    if (!bf16_weights || !fp32_biases) {
        return false;
    }
    int total_weights = config_.num_routed_experts * config_.hidden_dim;
    for (int i = 0; i < total_weights; ++i) {
        weights_fp32_[i] = bf16_to_fp32(bf16_weights[i]);
    }
    std::memcpy(biases_fp32_.data(), fp32_biases, config_.num_routed_experts * sizeof(float));
    is_loaded_ = true;
    return true;
}

bool M8Router::load_from_files(const std::string& weight_path, const std::string& bias_path) {
    std::ifstream fw(weight_path, std::ios::binary);
    if (!fw.is_open()) {
        return false;
    }
    int total_weights = config_.num_routed_experts * config_.hidden_dim;
    std::vector<uint16_t> bf16_buf(total_weights);
    fw.read(reinterpret_cast<char*>(bf16_buf.data()), total_weights * sizeof(uint16_t));
    if (!fw) {
        return false;
    }

    std::ifstream fb(bias_path, std::ios::binary);
    if (!fb.is_open()) {
        return false;
    }
    std::vector<float> bias_buf(config_.num_routed_experts);
    fb.read(reinterpret_cast<char*>(bias_buf.data()), config_.num_routed_experts * sizeof(float));
    if (!fb) {
        return false;
    }

    return load_from_buffers(bf16_buf.data(), bias_buf.data());
}

bool M8Router::load_from_vol_mgr(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id) {
    if (!vol_mgr) return false;

    std::string w_name = "layers." + std::to_string(layer_id) + ".ffn.gate.weight";
    std::string b_name = "layers." + std::to_string(layer_id) + ".ffn.gate.bias";

    int total_weights = config_.num_routed_experts * config_.hidden_dim;
    weights_fp32_.resize(total_weights);
    biases_fp32_.resize(config_.num_routed_experts);

    // Read gate weight [384, 5120] BF16 -> FP32
    if (!vol_mgr->read_tensor_bf16_to_fp32(w_name, weights_fp32_.data(), total_weights)) {
        return false;
    }

    // Read gate bias [384] F32
    if (!vol_mgr->read_tensor(b_name, reinterpret_cast<uint8_t*>(biases_fp32_.data()),
                              config_.num_routed_experts * sizeof(float))) {
        return false;
    }

    is_loaded_ = true;
    return true;
}

RouterSelection M8Router::route(const float* x) const {
    if (!is_loaded_ || !x) {
        return {};
    }

    const int E = config_.num_routed_experts;
    const int D = config_.hidden_dim;
    const int K = config_.top_k;

    std::vector<float> raw_scores(E);
    std::vector<float> selection_scores(E);

    for (int e = 0; e < E; ++e) {
        const float* row = &weights_fp32_[e * D];
        float dot = dot_product_avx2(row, x, D);
        float logit = dot / config_.gate_temp;

        // sqrtsoftplus: score = sqrt(softplus(logit))
        // softplus(z) = log(1 + exp(z))
        float sp;
        if (logit > 20.0f) {
            sp = logit;
        } else if (logit < -80.0f) {
            sp = std::exp(logit);
        } else {
            sp = std::log1p(std::exp(logit));
        }
        float score = std::sqrt(sp);
        raw_scores[e] = score;

        // Selection score includes bias: steers routing choice
        selection_scores[e] = score + biases_fp32_[e];
    }

    // Top-K selection based on selection_scores
    std::vector<int> all_indices(E);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::partial_sort(all_indices.begin(), all_indices.begin() + K, all_indices.end(),
                      [&](int a, int b) {
                          return selection_scores[a] > selection_scores[b];
                      });

    RouterSelection result;
    result.expert_indices.assign(all_indices.begin(), all_indices.begin() + K);
    result.expert_weights.resize(K);

    // DeepSeek routing weights come from unbiased raw_scores
    float score_sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float s = raw_scores[result.expert_indices[k]];
        result.expert_weights[k] = s;
        score_sum += s;
    }

    if (config_.norm_topk_prob && K > 1) {
        float inv_sum = 1.0f / (score_sum + 1e-20f);
        for (int k = 0; k < K; ++k) {
            result.expert_weights[k] = (result.expert_weights[k] * inv_sum) * config_.routed_scaling_factor;
        }
    } else {
        for (int k = 0; k < K; ++k) {
            result.expert_weights[k] *= config_.routed_scaling_factor;
        }
    }

    return result;
}

} // namespace m8
} // namespace asema
