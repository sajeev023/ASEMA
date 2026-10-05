#pragma once

#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_model_runner.hpp"
#include <string>
#include <memory>
#include <functional>

namespace asema {
namespace m8 {

struct EngineTelemetry {
    size_t ram_ceiling_bytes{20480ULL * 1024 * 1024};
    size_t ram_working_set_bytes{0};
    size_t peak_ram_working_set_bytes{0};
    size_t available_ram_bytes{0};
    size_t vram_ceiling_bytes{5120ULL * 1024 * 1024};
    size_t vram_working_set_bytes{0};
    size_t peak_vram_working_set_bytes{0};
    size_t available_vram_bytes{0};
    uint64_t gpu_allocation_failures{0};
    int layer_count_executed{40};
    double last_token_latency_ms{0.0};
    double tokens_per_second{0.0};
    uint64_t total_tokens_generated{0};
    uint64_t total_storage_bytes_read{0};
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    bool gpu_accelerated{false};
};

class AsemaEngine {
public:
    AsemaEngine();
    ~AsemaEngine();

    // Load model from physical storage volumes
    bool load_model(const std::string& hf_root = asema::m8::paths::hf_dir(),
                    const std::string& vol_primary = asema::m8::paths::primary_shards(),
                    const std::string& vol_secondary = asema::m8::paths::secondary_shards());

    // Single-shot text generation
    std::string generate_text(const std::string& prompt, int max_tokens = 32);

    // Real-time streamed text generation
    void stream_text(const std::string& prompt,
                     std::function<void(const std::string& piece)> on_token,
                     int max_tokens = 32);

    // Release model working set
    void release_model();

    // Query active telemetry
    EngineTelemetry get_telemetry() const;

    // Verify all 40 physical layers are executed without fallback
    bool verify_40_layers(std::string& failure_reason);

    // Why the most recent generation ended (EOS, MAX_TOKENS, USER_STOP, ERROR, ...).
    GenerationEndReason last_end_reason() const {
        return runner_ ? runner_->last_end_reason() : GenerationEndReason::NONE;
    }
    int last_stop_token_id() const { return runner_ ? runner_->last_stop_token_id() : -1; }

    // True only if the Direct3D 11 expert kernel initialised and is in use (not a CPU fallback).
    bool gpu_active() const { return runner_ && runner_->is_gpu_accelerated(); }

    bool is_loaded() const { return is_loaded_; }
    int num_layers() const { return runner_ ? runner_->num_layers() : 0; }

private:
    std::unique_ptr<M8ModelRunner> runner_;
    bool is_loaded_{false};
    EngineTelemetry telemetry_;
};

} // namespace m8
} // namespace asema
