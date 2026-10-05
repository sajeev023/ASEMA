#pragma once

// ASEMA v0.1 — Milestone 4: PrefetchEngine
// -----------------------------------------------------------------------------
// A deterministic lookahead prefetcher for expert weights.
//
// Design (v0.1, deterministic, no ML):
//   * Lookahead K layers: enqueue PREFETCH requests for the predicted experts
//     of layers (current + 1) through (current + K).
//   * The prediction comes from a pluggable IRoutingOracle. The default oracle
//     uses a locality-heuristic: it predicts the same experts as the previous
//     token, plus a small set of recent experts from a sliding history.
//   * The engine never starves REQUIRED_NOW: every prefetch is gated on
//     queue depth, cache capacity, and a per-tick budget.
//   * Cancelled prefetches are tracked. Useful prefetches are defined as
//     prefetches that result in a cache hit before they are evicted or
//     cancelled.
// -----------------------------------------------------------------------------

#include "request_queue.hpp"
#include "cache.hpp"
#include "expert.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace asema {

// -----------------------------------------------------------------------------
// Routing oracle (pluggable). The runtime can supply its own; the default is
// LocalityRoutingOracle.
// -----------------------------------------------------------------------------
class IRoutingOracle {
public:
    virtual ~IRoutingOracle() = default;
    // Given the current layer and a recent history of expert choices, return
    // a list of predicted experts for `layer_id`. Returned coord vector is
    // un-ordered; the prefetcher will cap it to a configurable maximum.
    virtual std::vector<ExpertCoord> predict(
        uint32_t layer_id,
        const std::vector<std::vector<ExpertCoord>>& recent_history) = 0;
};

// -----------------------------------------------------------------------------
// Default locality-based oracle: predicts the experts used at layer_id for
// the last N tokens. Lightweight and deterministic.
// -----------------------------------------------------------------------------
class LocalityRoutingOracle final : public IRoutingOracle {
public:
    explicit LocalityRoutingOracle(size_t top_n_per_layer = 4,
                                   size_t history_depth = 8)
        : top_n_per_layer_(top_n_per_layer), history_depth_(history_depth) {}

    std::vector<ExpertCoord> predict(
        uint32_t layer_id,
        const std::vector<std::vector<ExpertCoord>>& recent_history) override;

private:
    size_t top_n_per_layer_;
    size_t history_depth_;
};

// -----------------------------------------------------------------------------
// Prefetch statistics
// -----------------------------------------------------------------------------
struct PrefetchStats {
    uint64_t prefetch_requests{0};        // Total issued
    uint64_t useful_prefetches{0};        // Resulted in cache hit before eviction/cancel
    uint64_t wasted_prefetches{0};        // Evicted before being used
    uint64_t duplicate_prefetches{0};     // Coalesced or already in queue
    uint64_t cancelled_prefetches{0};     // Cancelled before completion
    uint64_t prefetch_bytes{0};           // Total bytes targeted
    double   prefetch_hit_rate{0.0};      // useful / (useful + wasted + cancelled) where not duplicate
    uint64_t avg_prefetch_latency_us{0};  // Approximate
    uint64_t lookahead{0};
    uint64_t gated_by_queue{0};
    uint64_t gated_by_cache{0};
    uint64_t gated_by_budget{0};
};

// -----------------------------------------------------------------------------
// Prefetch engine
// -----------------------------------------------------------------------------
class PrefetchEngine {
public:
    PrefetchEngine(
        ExpertRequestQueue& queue,
        ExpertRAMCache& cache,
        size_t lookahead = 2,
        size_t max_experts_per_layer = 8,
        size_t history_depth = 8,
        size_t max_prefetch_bytes_per_tick = 64ull * 1024 * 1024);

    ~PrefetchEngine();

    PrefetchEngine(const PrefetchEngine&) = delete;
    PrefetchEngine& operator=(const PrefetchEngine&) = delete;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    void set_lookahead(size_t K);
    void set_oracle(std::shared_ptr<IRoutingOracle> oracle);

    size_t lookahead() const noexcept { return lookahead_.load(std::memory_order_acquire); }

    // -------------------------------------------------------------------------
    // Tick
    // -------------------------------------------------------------------------

    // Inform the engine of the experts the runtime actually used for the
    // current token (one entry per layer).
    void on_token_complete(
        uint32_t current_layer,
        const std::vector<std::vector<ExpertCoord>>& chosen_experts_per_layer);

    // Trigger a prefetch tick. Generates PREFETCH requests for layers
    // (current_layer+1) ... (current_layer+lookahead) using the oracle.
    // Honours queue depth, cache capacity, and per-tick byte budget.
    void tick(uint32_t current_layer);

    // -------------------------------------------------------------------------
    // Outcome recording
    // -------------------------------------------------------------------------

    // Call when a prefetched expert is consumed (becomes a cache hit).
    void record_useful(ExpertCoord coord);

    // Call when a prefetched expert is evicted without being used.
    void record_wasted(ExpertCoord coord);

    // Call when a prefetch is cancelled.
    void record_cancelled(ExpertCoord coord);

    // -------------------------------------------------------------------------
    // State & metrics
    // -------------------------------------------------------------------------
    PrefetchStats stats() const;
    void reset_stats();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------
    void shutdown();

private:
    ExpertRequestQueue& queue_;
    ExpertRAMCache& cache_;

    std::atomic<size_t> lookahead_{2};
    size_t max_experts_per_layer_;
    size_t history_depth_;
    size_t max_prefetch_bytes_per_tick_;

    std::shared_ptr<IRoutingOracle> oracle_;
    std::shared_ptr<IRoutingOracle> default_oracle_;

    mutable std::mutex history_mu_;
    std::vector<std::vector<ExpertCoord>> recent_history_;

    // Tracking which coords were prefetched (for useful/wasted accounting).
    mutable std::mutex tracking_mu_;
    std::unordered_map<uint64_t, uint64_t> prefetched_coord_to_count_;

    std::atomic<bool> shutdown_{false};

    // Stats
    std::atomic<uint64_t> prefetch_requests_{0};
    std::atomic<uint64_t> useful_prefetches_{0};
    std::atomic<uint64_t> wasted_prefetches_{0};
    std::atomic<uint64_t> duplicate_prefetches_{0};
    std::atomic<uint64_t> cancelled_prefetches_{0};
    std::atomic<uint64_t> prefetch_bytes_{0};
    std::atomic<uint64_t> gated_by_queue_{0};
    std::atomic<uint64_t> gated_by_cache_{0};
    std::atomic<uint64_t> gated_by_budget_{0};
    std::atomic<uint64_t> total_latency_us_{0};
};

} // namespace asema
