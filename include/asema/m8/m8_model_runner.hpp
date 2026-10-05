#pragma once

#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_gpu_expert_kernel.hpp"
#include "asema/m8/m8_gpu_mla_kernel.hpp"
#include "asema/m8/m8_model_adapter.hpp"
#include "asema/m8/m8_multi_volume.hpp"
#include "asema/m8/m8_transformer_layer.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace asema {
namespace m8 {

// Why a generation call returned. Generation must never end silently.
enum class GenerationEndReason {
    NONE,               // no generation has run yet
    EOS,                // the model produced an end/stop token (see last_stop_token_id())
    MAX_TOKENS,         // the requested token limit was reached
    USER_STOP,          // Ctrl+C / shutdown request
    FAILURE,            // an exception was thrown (printed as "ERROR"; the name avoids the Win32 ERROR macro)
    RESOURCE_GOVERNOR,  // reserved: the governor throttles but currently never stops generation
    TIMEOUT             // reserved: no timeout is currently implemented
};

inline const char* to_string(GenerationEndReason r) {
    switch (r) {
        case GenerationEndReason::NONE: return "NONE";
        case GenerationEndReason::EOS: return "EOS";
        case GenerationEndReason::MAX_TOKENS: return "MAX_TOKENS";
        case GenerationEndReason::USER_STOP: return "USER_STOP";
        case GenerationEndReason::FAILURE: return "ERROR";
        case GenerationEndReason::RESOURCE_GOVERNOR: return "RESOURCE_GOVERNOR";
        case GenerationEndReason::TIMEOUT: return "TIMEOUT";
    }
    return "UNKNOWN";
}

// Per-stage wall-clock timing of one decode step (everything outside the 40 layers).
struct StepTiming {
    double embed_ms{0.0};
    double final_norm_ms{0.0};
    double lm_head_ms{0.0};
    double layers_ms{0.0};
};

struct TokenGenerationTelemetry {
    int step{0};

    // Stage breakdown of the step that produced this token (milliseconds).
    StepTiming stage;
    double select_ms{0.0};          // repetition penalty + argmax
    double decode_ms{0.0};          // detokenization + UTF-8 streaming
    double dense_load_ms{0.0};      // per-layer dense weight binding (sum over layers)
    double attn_ms{0.0};            // MLA attention incl. KV (sum over layers)
    double kv_update_ms{0.0};       // KV-cache ring-buffer store (sum over layers)
    double kv_attend_ms{0.0};       // attention over the cached window (sum over layers)
    double router_ms{0.0};
    double shared_expert_ms{0.0};
    double expert_load_ms{0.0};     // expert cache lookup + storage reads (sum over layers)
    double expert_gpu_ms{0.0};      // GPU expert stage wall time: upload + compute + wait (sum)
    double gpu_upload_call_ms{0.0}; // CPU time spent inside GPU upload calls
    double gpu_wait_ms{0.0};        // CPU blocked waiting for the GPU result
    double gpu_busy_upload_ms{0.0}; // GPU-side time executing uploads (timestamp queries)
    double gpu_busy_compute_ms{0.0};// GPU-side time executing expert kernels (timestamp queries)
    double cpu_other_ms{0.0};       // hyper-connection mixing, norms, residuals (layer time not above)
    uint64_t storage_read_ops{0};   // ReadFile operations issued during this step
    uint64_t storage_bytes{0};      // bytes read from storage during this step
    int expert_hits{0};
    int expert_misses{0};
    int token_id{0};
    std::string token_str;
    double token_latency_ms{0.0};
    double total_elapsed_ms{0.0};
    size_t active_experts_materialized{0};
    size_t ram_working_set_bytes{0};
    size_t peak_ram_working_set_bytes{0};
    size_t available_ram_bytes{0};
    size_t ram_ceiling_bytes{20480ULL * 1024 * 1024};
    size_t vram_working_set_bytes{0};
    size_t peak_vram_working_set_bytes{0};
    size_t available_vram_bytes{0};
    size_t vram_ceiling_bytes{5120ULL * 1024 * 1024};
    uint64_t gpu_allocation_failures{0};
    int layer_count_executed{40};
    float top_logit{0.0f};
    bool kv_cache_valid{true};
    std::string diagnostic_info;
    size_t bytes_read_from_storage{0};
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    uint64_t useful_prefetches{0};
    uint64_t wasted_prefetches{0};
    bool gpu_accelerated{false};
    std::vector<int> layer0_top6_experts;
};

class M8ModelRunner {
public:
    M8ModelRunner();
    ~M8ModelRunner();

    // Verify all 40 layers are physically present on disk and executable
    bool verify_40_layers_executed(std::string& failure_reason);

    // Initialize the model runner with physical storage volumes and tokenizer
    bool init(const std::string& hf_root,
              const std::string& vol_primary = asema::m8::paths::primary_shards(),
              const std::string& vol_secondary = asema::m8::paths::secondary_shards());

    // Enable/disable GPU acceleration (DirectCompute Direct3D 11)
    bool set_gpu_acceleration(bool enable);
    bool is_gpu_accelerated() const { return gpu_enabled_ && gpu_kernel_ && gpu_kernel_->is_initialized(); }

    // Enable/disable GPU MLA projection acceleration
    bool set_gpu_mla_acceleration(bool enable);
    bool is_gpu_mla_accelerated() const { return gpu_mla_enabled_ && gpu_mla_kernel_ && gpu_mla_kernel_->is_initialized(); }

    // Enable/disable asynchronous double buffering / prefetching
    void set_async_double_buffering(bool enable) { async_double_buffering_ = enable; }
    bool is_async_double_buffering() const { return async_double_buffering_; }

    // Set expert RAM cache capacity in MB (e.g. 256, 512, 1024, 2048, 4096)
    void set_cache_capacity_mb(size_t mb);

    // Reset KV cache and state for new prompt generation
    void reset_state();

    // Single token step forward pass through all 40 transformer layers
    void step(int token_id, int pos, std::vector<float>& out_logits, std::vector<LayerTelemetry>& layer_tels, bool compute_logits = true);

    // Autoregressive generation
    std::vector<int> generate(const std::string& prompt, int max_new_tokens,
                              std::function<void(const TokenGenerationTelemetry&)> on_token = nullptr,
                              float repetition_penalty = 1.0f);

    int num_layers() const { return total_layers_; }
    void set_num_layers(int n) { total_layers_ = std::max(1, n); }
    const Tokenizer& tokenizer() const { return tokenizer_; }
    std::shared_ptr<M8ByteRangeLoader> byte_loader() { return byte_loader_; }
    std::shared_ptr<M8GpuExpertKernel> gpu_kernel() { return gpu_kernel_; }
    bool has_real_embeddings() const { return has_real_embeddings_; }
    bool has_real_lm_head() const { return has_real_lm_head_; }
    bool lookup_token_embedding(int token_id, float* out_emb) const;
    void compute_lm_head_logits(const float* normed_h, std::vector<float>& out_logits);
    M8TransformerLayer& active_layer() { return *active_layer_; }
    const std::vector<float>& final_norm() const { return final_norm_; }
    void rms_norm(const float* in, const float* weight, float* out, int size, float eps = 1e-20f) const;

private:
    std::shared_ptr<M8MultiVolumeManager> vol_mgr_;
    std::shared_ptr<M8ByteRangeLoader> byte_loader_;
    std::shared_ptr<M8GpuExpertKernel> gpu_kernel_{nullptr};
    std::shared_ptr<class M8GpuMlaKernel> gpu_mla_kernel_{nullptr};

    bool gpu_enabled_{false};
    bool gpu_mla_enabled_{false};
    bool async_double_buffering_{false};

    int total_layers_{40};
    // Single active layer working memory: zero redundant layer duplication
    std::unique_ptr<M8TransformerLayer> active_layer_;

    Tokenizer tokenizer_;
    std::vector<float> final_norm_;

    struct RealWeightsMapping;
    std::unique_ptr<RealWeightsMapping> real_weights_;
    bool has_real_embeddings_{false};
    bool has_real_lm_head_{false};

    void init_mock_or_reference_embeddings();

    // Memory-pressure governor: sizes the expert cache and read concurrency from live Windows
    // available memory. Called once per decode step (cheap: one GlobalMemoryStatusEx).
    // allow_grow: the cache may grow only once per decode step; shrink/throttle checks run per layer.
    void apply_resource_governor(bool allow_grow = true);
    // Under memory pressure, hand memory-mapped checkpoint pages back to Windows' standby cache.
    void trim_working_set();
    size_t expert_cache_cap_mb_{6144}; // growth ceiling (a ceiling, never a target)
    int governor_state_{0};            // 0 = SAFE, 1 = WARNING, 2 = CRITICAL

    GenerationEndReason last_end_reason_{GenerationEndReason::NONE};
    int last_stop_token_id_{-1};
    StepTiming last_step_timing_;

public:
    int governor_state() const { return governor_state_; }

    // Why the most recent generate() call ended (never silent), and the stop token if EOS.
    GenerationEndReason last_end_reason() const { return last_end_reason_; }
    int last_stop_token_id() const { return last_stop_token_id_; }
    const StepTiming& last_step_timing() const { return last_step_timing_; }
};

} // namespace m8
} // namespace asema
