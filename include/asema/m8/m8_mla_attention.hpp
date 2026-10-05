#pragma once

#include "asema/m8/m8_fp8_gemv.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asema {
namespace m8 {

class M8MultiVolumeManager;

struct MLAParams {
    int dim{5120};
    int n_heads{64};
    int head_dim{512};
    int rope_head_dim{64};
    int q_lora_rank{1280};
    int o_lora_rank{1024};
    int o_groups{8};
    int window_size{128};
    float eps{1e-20f};
    float softmax_scale{0.04419417382f}; // 1.0 / sqrt(512)
    float rope_theta{10000.0f};
};

class M8MLAAttention {
public:
    explicit M8MLAAttention(MLAParams params = MLAParams{});

    // Initialize/allocate weight storage
    void init_buffers();

    // Attach to loaded weight buffers (or load from disk files)
    bool load_mock_or_reference_weights();
    bool load_from_checkpoint_files(const std::string& chunk1_file, const std::string& chunk2_file);
    bool load_from_vol_mgr(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id);
    bool load_from_buffers(const float* wq_a, const float* q_norm, const float* wq_b,
                           const float* wkv, const float* kv_norm,
                           const float* wo_a, const float* wo_b, const float* attn_sink);

    // Compute single-token MLA forward pass with sliding-window KV cache
    // x: [dim]
    // out: [dim]
    // start_pos: sequence position (0, 1, 2, ...)
    void forward(const float* x, float* out, int start_pos);

    // Reset KV cache ring buffer
    void reset_kv_cache();

    // Select active layer KV cache (0..39)
    void set_active_layer(int layer_id);
    int active_layer() const { return current_layer_id_; }

    void set_gpu_mla(std::shared_ptr<class M8GpuMlaKernel> gpu_mla);
    std::shared_ptr<class M8GpuMlaKernel> gpu_mla() const { return gpu_mla_; }

    // True when the five FP8 projections are used in place from memory-mapped shards
    // (no FP32 copies exist). False when the legacy dequantized FP32 path is active.
    bool direct_fp8() const { return direct_fp8_; }

    // Timing of the last forward(): KV-cache ring-buffer store and attention over the cached window.
    double last_kv_update_ms() const { return last_kv_update_ms_; }
    double last_kv_attend_ms() const { return last_kv_attend_ms_; }

    const MLAParams& params() const { return params_; }
    size_t kv_cache_bytes() const { return static_cast<size_t>(params_.window_size) * params_.head_dim * sizeof(float); }
    const std::vector<float>& kv_cache() const { return kv_caches_[current_layer_id_]; }

private:
    MLAParams params_;
    bool is_initialized_{false};
    std::shared_ptr<class M8GpuMlaKernel> gpu_mla_{nullptr};

    // Weights
    std::vector<float> wq_a_;      // [1280, 5120]
    std::vector<float> q_norm_;    // [1280]
    std::vector<float> wq_b_;      // [32768, 1280]
    std::vector<float> wkv_;       // [512, 5120]
    std::vector<float> kv_norm_;   // [512]
    std::vector<float> wo_a_;      // [8 * 1024 * 4096]
    std::vector<float> wo_b_;      // [5120, 8192]
    std::vector<float> attn_sink_; // [64]

    // Zero-copy FP8 projections (valid while direct_fp8_ is true)
    bool direct_fp8_{false};
    double last_kv_update_ms_{0.0};
    double last_kv_attend_ms_{0.0};
    Fp8MatView wq_a_f8_;
    Fp8MatView wq_b_f8_;
    Fp8MatView wkv_f8_;
    Fp8MatView wo_a_f8_;
    Fp8MatView wo_b_f8_;
    bool bind_direct_fp8(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id);
    void release_fp32_projections();

    // Sliding window KV cache: [40 layers][window_size * head_dim]
    static constexpr int NUM_KV_LAYERS = 40;
    int current_layer_id_{0};
    std::vector<std::vector<float>> kv_caches_;
    std::vector<int> cached_tokens_per_layer_;

    // Precomputed RoPE frequencies
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
    void precompute_rope();

    void apply_rope(float* vec, int head_dim, int pos, bool inverse = false) const;
    void rms_norm(const float* in, const float* weight, float* out, int size) const;
};

} // namespace m8
} // namespace asema
