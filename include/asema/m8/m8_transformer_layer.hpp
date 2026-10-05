#pragma once

#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_gpu_expert_kernel.hpp"
#include "asema/m8/m8_mla_attention.hpp"
#include "asema/m8/m8_router.hpp"

#include <memory>
#include <vector>

namespace asema {
namespace m8 {

struct LayerTelemetry {
    int layer_id{0};
    std::vector<int> selected_experts;
    std::vector<float> routing_weights;
    double dense_load_time_ms{0.0};
    size_t dense_bytes_loaded{0};
    double attn_time_ms{0.0};
    double router_time_ms{0.0};
    double shared_expert_time_ms{0.0};
    double expert_load_time_ms{0.0};
    size_t expert_bytes_loaded{0};
    int expert_cache_hits{0};
    int expert_cache_misses{0};
    double expert_compute_time_ms{0.0};
    double gpu_upload_time_ms{0.0};
    double gpu_kernel_time_ms{0.0};
    double gpu_readback_time_ms{0.0};
    double total_layer_time_ms{0.0};
    double kv_update_time_ms{0.0};    // KV-cache ring-buffer store inside MLA
    double kv_attend_time_ms{0.0};    // attention over the cached window inside MLA
    double gpu_busy_upload_ms{0.0};   // GPU-side time executing expert uploads (timestamp queries)
    double gpu_busy_compute_ms{0.0};  // GPU-side time executing expert kernels (timestamp queries)
    uint64_t storage_read_ops{0};     // ReadFile operations issued by this layer's expert loads
    size_t ram_working_set_bytes{0};
    size_t vram_working_set_bytes{0};
    bool gpu_accelerated{false};
};

class M8TransformerLayer {
public:
    explicit M8TransformerLayer(int layer_id, std::shared_ptr<M8ByteRangeLoader> byte_loader);

    void init();
    void set_layer_id(int id) {
        layer_id_ = id;
        self_attn_.set_active_layer(id);
    }
    void set_gpu_kernel(std::shared_ptr<M8GpuExpertKernel> gpu_kernel) { gpu_kernel_ = gpu_kernel; }
    void set_gpu_mla(std::shared_ptr<class M8GpuMlaKernel> gpu_mla) {
        self_attn_.set_gpu_mla(gpu_mla);
        loaded_layer_id_ = -1; // force reload so weights match the selected (CPU/GPU) path
    }

    bool load_router(const std::string& gate_weight_file, const std::string& gate_bias_file);

    bool load_layer(int layer_id);
    int loaded_layer_id() const { return loaded_layer_id_; }

    void forward(const float* in, float* out, int start_pos, LayerTelemetry& tel);
    void forward_hc(const float* in, float* out, const float* pre_mix_in, float* pre_mix_out, int start_pos, LayerTelemetry& tel);
    bool has_hyper_connections() const { return has_hc_; }

    int layer_id() const { return layer_id_; }
    M8MLAAttention& attn() { return self_attn_; }
    M8Router& router() { return router_; }
    void reset_kv_cache() { self_attn_.reset_kv_cache(); }

private:
    int layer_id_{0};
    int loaded_layer_id_{-1};
    std::shared_ptr<M8ByteRangeLoader> byte_loader_;
    std::shared_ptr<M8GpuExpertKernel> gpu_kernel_{nullptr};

    M8MLAAttention self_attn_;
    M8Router router_;
    M8ExpertKernel routed_kernel_;

    bool has_shared_expert_{false};
    // Zero-copy FP8 shared expert (memory-mapped checkpoint bytes); FP32 vectors below are the legacy path.
    bool shared_direct_fp8_{false};
    Fp8MatView shared_w1_f8_;
    Fp8MatView shared_w2_f8_;
    Fp8MatView shared_w3_f8_;
    std::vector<float> shared_w1_;
    std::vector<float> shared_w2_;
    std::vector<float> shared_w3_;

    std::vector<float> attn_norm_;
    std::vector<float> ffn_norm_;

    bool has_hc_{false};
    std::vector<float> hc_attn_fn_;    // [24, 20480]
    std::vector<float> hc_attn_scale_; // [3]
    std::vector<float> hc_attn_base_;  // [24]
    std::vector<float> hc_ffn_fn_;     // [24, 20480]
    std::vector<float> hc_ffn_scale_;  // [3]
    std::vector<float> hc_ffn_base_;   // [24]

    void compute_hc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base,
                          float* pre, float* post, float* comb) const;
    void rms_norm(const float* in, const float* weight, float* out, int size, float eps = 1e-20f) const;
    void compute_shared_expert(const float* in, float* out_accum) const;
};

} // namespace m8
} // namespace asema
