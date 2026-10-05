#pragma once

// ASEMA v0.1 — Milestone 5: ASEMARuntime
// -----------------------------------------------------------------------------
// A minimal end-to-end runtime that orchestrates the cache, queue, prefetch
// engine, telemetry, and Agent #1's async loader.
//
// The runtime does NOT implement a real DeepSeek inference engine. It
// simulates routing/execution using the synthetic MoE model. Its job is to
// prove the architecture works end-to-end and to produce benchmark numbers.
//
// Two execution modes:
//   * Mode A (baseline): every expert is loaded directly via the loader
//     (no cache, no prefetch).
//   * Mode B (ASEMA): the cache + queue + prefetch pipeline is in use.
// -----------------------------------------------------------------------------

#include "loader_adapter.hpp"
#include "cache.hpp"
#include "request_queue.hpp"
#include "prefetch.hpp"
#include "telemetry.hpp"
#include "manifest.hpp"
#include "expert.hpp"
#include "async_loader.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asema {

// -----------------------------------------------------------------------------
// Runtime configuration
// -----------------------------------------------------------------------------
struct RuntimeConfig {
    std::string container_path;
    std::string manifest_path;

    // Synthetic model parameters (must match generator)
    uint32_t num_layers{4};
    uint32_t experts_per_layer{16};
    uint32_t top_k{2};

    // RAM cache budget in bytes. 0 = disabled (Mode A).
    uint64_t ram_cache_bytes{0};

    // Prefetch lookahead K (0 = disabled).
    size_t prefetch_lookahead{0};

    // Number of synthetic tokens to "infer".
    uint32_t num_tokens{64};

    // Random seed for the synthetic router.
    uint64_t router_seed{42};

    // Loader worker count.
    size_t loader_workers{4};

    // Telemetry output paths (empty = disabled).
    std::string telemetry_jsonl;
    std::string report_text_path;
    std::string report_json_path;
};

// -----------------------------------------------------------------------------
// Run result — populated by run()
// -----------------------------------------------------------------------------
struct RuntimeResult {
    uint64_t total_tokens{0};
    double total_wall_time_ms{0.0};
    double first_token_latency_ms{0.0};
    bool completed{false};
    std::string error;
};

// -----------------------------------------------------------------------------
// ASEMARuntime
// -----------------------------------------------------------------------------
class ASEMARuntime {
public:
    explicit ASEMARuntime(RuntimeConfig cfg);
    ~ASEMARuntime();

    ASEMARuntime(const ASEMARuntime&) = delete;
    ASEMARuntime& operator=(const ASEMARuntime&) = delete;

    // Run the synthetic workload end-to-end.
    RuntimeResult run();

    // Accessors for tests/benchmarks.
    ExpertRAMCache* cache() { return cache_.get(); }
    ExpertRequestQueue* queue() { return queue_.get(); }
    PrefetchEngine* prefetch() { return prefetch_.get(); }
    TelemetryEngine* telemetry() { return telemetry_.get(); }
    IExpertLoader* loader() { return loader_.get(); }
    const ModelManifest& manifest() const { return manifest_; }

private:
    // Routing: deterministic locality-aware choice.
    std::vector<ExpertCoord> route_token(uint64_t token_id, uint32_t layer_id);

    // Load + execute a single expert (with telemetry).
    void load_and_execute(uint64_t token_id, ExpertCoord c, LoadPriority prio);

    // Synthetic "execution": touch the buffer to enforce residency,
    // record latency. The product of weight bytes summed is the output.
    void execute_synthetic(uint64_t token_id, ExpertCoord c, std::shared_ptr<ExpertBuffer> buf);

    RuntimeConfig cfg_;
    ModelManifest manifest_;
    std::unique_ptr<IExpertLoader> loader_;
    std::unique_ptr<ExpertRAMCache> cache_;
    std::unique_ptr<ExpertRequestQueue> queue_;
    std::unique_ptr<PrefetchEngine> prefetch_;
    std::unique_ptr<TelemetryEngine> telemetry_;
    std::atomic<bool> shutdown_{false};
};

} // namespace asema
