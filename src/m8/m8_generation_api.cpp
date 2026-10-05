#include "asema/m8/m8_generation_api.hpp"
#include <sstream>
#include <iostream>

namespace asema {
namespace m8 {

AsemaEngine::AsemaEngine() : runner_(std::make_unique<M8ModelRunner>()) {}

AsemaEngine::~AsemaEngine() {
    release_model();
}

bool AsemaEngine::load_model(const std::string& hf_root,
                             const std::string& vol_primary,
                             const std::string& vol_secondary) {
    if (!runner_) {
        runner_ = std::make_unique<M8ModelRunner>();
    }

    runner_->set_gpu_acceleration(true);
    runner_->set_gpu_mla_acceleration(false);
    runner_->set_async_double_buffering(true);
    runner_->set_cache_capacity_mb(7168);

    is_loaded_ = runner_->init(hf_root, vol_primary, vol_secondary);
    return is_loaded_;
}

std::string AsemaEngine::generate_text(const std::string& prompt, int max_tokens) {
    if (!is_loaded_ || !runner_) return "";

    std::ostringstream ss;
    stream_text(prompt, [&](const std::string& piece) {
        ss << piece;
    }, max_tokens);

    return ss.str();
}

void AsemaEngine::stream_text(const std::string& prompt,
                              std::function<void(const std::string& piece)> on_token,
                              int max_tokens) {
    if (!is_loaded_ || !runner_) return;

    double total_time_ms = 0.0;
    uint64_t generated_count = 0;

    runner_->generate(prompt, max_tokens, [&](const TokenGenerationTelemetry& tel) {
        generated_count++;
        total_time_ms += tel.token_latency_ms;

        telemetry_.last_token_latency_ms = tel.token_latency_ms;
        telemetry_.ram_ceiling_bytes = tel.ram_ceiling_bytes;
        telemetry_.ram_working_set_bytes = tel.ram_working_set_bytes;
        telemetry_.peak_ram_working_set_bytes = tel.peak_ram_working_set_bytes;
        telemetry_.available_ram_bytes = tel.available_ram_bytes;
        telemetry_.vram_ceiling_bytes = tel.vram_ceiling_bytes;
        telemetry_.vram_working_set_bytes = tel.vram_working_set_bytes;
        telemetry_.peak_vram_working_set_bytes = tel.peak_vram_working_set_bytes;
        telemetry_.available_vram_bytes = tel.available_vram_bytes;
        telemetry_.gpu_allocation_failures = tel.gpu_allocation_failures;
        telemetry_.layer_count_executed = tel.layer_count_executed;
        telemetry_.total_storage_bytes_read = tel.bytes_read_from_storage;
        telemetry_.cache_hits = tel.cache_hits;
        telemetry_.cache_misses = tel.cache_misses;
        telemetry_.gpu_accelerated = tel.gpu_accelerated;

        if (on_token) {
            on_token(tel.token_str);
        }
    });

    telemetry_.total_tokens_generated += generated_count;
    if (total_time_ms > 0.0) {
        telemetry_.tokens_per_second = (generated_count * 1000.0) / total_time_ms;
    }
}

void AsemaEngine::release_model() {
    if (runner_) {
        runner_ = nullptr;
    }
    is_loaded_ = false;
}

EngineTelemetry AsemaEngine::get_telemetry() const {
    return telemetry_;
}

bool AsemaEngine::verify_40_layers(std::string& failure_reason) {
    if (!runner_) {
        failure_reason = "Model runner not initialized";
        return false;
    }
    return runner_->verify_40_layers_executed(failure_reason);
}

} // namespace m8
} // namespace asema
