// ASEMA v0.1 — Milestone 4: PrefetchEngine implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/prefetch.hpp"

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace asema {

// =============================================================================
// LocalityRoutingOracle
// =============================================================================
std::vector<ExpertCoord> LocalityRoutingOracle::predict(
    uint32_t layer_id,
    const std::vector<std::vector<ExpertCoord>>& recent_history)
{
    // For locality: count occurrences of each expert at the same layer in
    // recent history. Return the top-N.
    std::unordered_map<uint64_t, uint64_t> counts;
    for (const auto& token_experts : recent_history) {
        for (const auto& c : token_experts) {
            if (c.layer_id == layer_id) {
                uint64_t key = (static_cast<uint64_t>(c.layer_id) << 32) | c.expert_id;
                counts[key]++;
            }
        }
    }

    // Sort by count desc, take top-N.
    std::vector<std::pair<uint64_t, uint64_t>> sorted(counts.begin(), counts.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const std::pair<uint64_t, uint64_t>& a, const std::pair<uint64_t, uint64_t>& b) {
            return a.second > b.second;
        });

    std::vector<ExpertCoord> out;
    out.reserve(std::min(top_n_per_layer_, sorted.size()));
    for (size_t i = 0; i < sorted.size() && i < top_n_per_layer_; ++i) {
        ExpertCoord c;
        c.layer_id = static_cast<uint32_t>(sorted[i].first >> 32);
        c.expert_id = static_cast<uint32_t>(sorted[i].first & 0xFFFFFFFFu);
        out.push_back(c);
    }
    return out;
}

// =============================================================================
// PrefetchEngine
// =============================================================================
PrefetchEngine::PrefetchEngine(
    ExpertRequestQueue& queue,
    ExpertRAMCache& cache,
    size_t lookahead,
    size_t max_experts_per_layer,
    size_t history_depth,
    size_t max_prefetch_bytes_per_tick)
    : queue_(queue),
      cache_(cache),
      max_experts_per_layer_(max_experts_per_layer),
      history_depth_(history_depth),
      max_prefetch_bytes_per_tick_(max_prefetch_bytes_per_tick)
{
    lookahead_.store(lookahead, std::memory_order_release);
    default_oracle_ = std::make_shared<LocalityRoutingOracle>(max_experts_per_layer, history_depth);
    oracle_ = default_oracle_;
}

PrefetchEngine::~PrefetchEngine() {
    shutdown();
}

void PrefetchEngine::set_lookahead(size_t K) {
    lookahead_.store(K, std::memory_order_release);
}

void PrefetchEngine::set_oracle(std::shared_ptr<IRoutingOracle> oracle) {
    if (oracle) oracle_ = std::move(oracle);
}

void PrefetchEngine::on_token_complete(
    uint32_t /*current_layer*/,
    const std::vector<std::vector<ExpertCoord>>& chosen_experts_per_layer)
{
    std::lock_guard<std::mutex> lock(history_mu_);
    recent_history_.push_back({});
    for (const auto& layer_experts : chosen_experts_per_layer) {
        for (const auto& c : layer_experts) {
            recent_history_.back().push_back(c);
        }
    }
    if (recent_history_.size() > history_depth_) {
        recent_history_.erase(recent_history_.begin());
    }
}

void PrefetchEngine::tick(uint32_t current_layer) {
    if (shutdown_.load(std::memory_order_acquire)) return;

    size_t K = lookahead_.load(std::memory_order_acquire);
    if (K == 0) return;

    // Snapshot history under lock
    std::vector<std::vector<ExpertCoord>> history_snapshot;
    {
        std::lock_guard<std::mutex> lock(history_mu_);
        history_snapshot = recent_history_;
    }

    size_t bytes_this_tick = 0;
    for (size_t k = 1; k <= K; ++k) {
        uint32_t target_layer = current_layer + static_cast<uint32_t>(k);

        auto predictions = oracle_->predict(target_layer, history_snapshot);
        if (predictions.empty()) continue;

        // Cap to max_experts_per_layer_
        if (predictions.size() > max_experts_per_layer_) {
            predictions.resize(max_experts_per_layer_);
        }

        for (const auto& c : predictions) {
            // Skip if cache already has it
            if (cache_.contains(c)) continue;

            // Skip if already in-flight or already queued.
            if (queue_.is_in_flight(c) || queue_.is_queued(c)) {
                duplicate_prefetches_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Back-pressure: if queue depth is high, gate
            // (rough heuristic: queue capacity / 4)
            if (queue_.size() >= queue_.capacity() / 4) {
                gated_by_queue_.fetch_add(1, std::memory_order_relaxed);
                return;
            }

            // Budget: stop issuing if per-tick byte budget exceeded.
            // We don't have access to the expert size without the manifest;
            // assume a conservative upper bound per expert of 1 MB.
            if (bytes_this_tick + (1ull << 20) > max_prefetch_bytes_per_tick_) {
                gated_by_budget_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            bytes_this_tick += (1ull << 20);

            // Submit the PREFETCH request.
            auto id = queue_.push_prefetch(c);
            if (id != 0) {
                prefetch_requests_.fetch_add(1, std::memory_order_relaxed);
                prefetch_bytes_.fetch_add(1ull << 20, std::memory_order_relaxed);

                std::lock_guard<std::mutex> lock(tracking_mu_);
                uint64_t key = (static_cast<uint64_t>(c.layer_id) << 32) | c.expert_id;
                prefetched_coord_to_count_[key]++;
            } else {
                // Queue refused (full or shutdown). Count as gated.
                gated_by_queue_.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

void PrefetchEngine::record_useful(ExpertCoord coord) {
    {
        std::lock_guard<std::mutex> lock(tracking_mu_);
        uint64_t key = (static_cast<uint64_t>(coord.layer_id) << 32) | coord.expert_id;
        auto it = prefetched_coord_to_count_.find(key);
        if (it != prefetched_coord_to_count_.end() && it->second > 0) {
            it->second--;
            if (it->second == 0) prefetched_coord_to_count_.erase(it);
            useful_prefetches_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    // If we didn't track this prefetch, it's a regular hit (not useful for
    // accounting). Don't double-count.
}

void PrefetchEngine::record_wasted(ExpertCoord coord) {
    {
        std::lock_guard<std::mutex> lock(tracking_mu_);
        uint64_t key = (static_cast<uint64_t>(coord.layer_id) << 32) | coord.expert_id;
        auto it = prefetched_coord_to_count_.find(key);
        if (it != prefetched_coord_to_count_.end() && it->second > 0) {
            it->second--;
            if (it->second == 0) prefetched_coord_to_count_.erase(it);
            wasted_prefetches_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

void PrefetchEngine::record_cancelled(ExpertCoord coord) {
    {
        std::lock_guard<std::mutex> lock(tracking_mu_);
        uint64_t key = (static_cast<uint64_t>(coord.layer_id) << 32) | coord.expert_id;
        auto it = prefetched_coord_to_count_.find(key);
        if (it != prefetched_coord_to_count_.end() && it->second > 0) {
            it->second--;
            if (it->second == 0) prefetched_coord_to_count_.erase(it);
            cancelled_prefetches_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}

PrefetchStats PrefetchEngine::stats() const {
    PrefetchStats s;
    s.prefetch_requests = prefetch_requests_.load(std::memory_order_relaxed);
    s.useful_prefetches = useful_prefetches_.load(std::memory_order_relaxed);
    s.wasted_prefetches = wasted_prefetches_.load(std::memory_order_relaxed);
    s.duplicate_prefetches = duplicate_prefetches_.load(std::memory_order_relaxed);
    s.cancelled_prefetches = cancelled_prefetches_.load(std::memory_order_relaxed);
    s.prefetch_bytes = prefetch_bytes_.load(std::memory_order_relaxed);
    s.lookahead = lookahead_.load(std::memory_order_acquire);
    s.gated_by_queue = gated_by_queue_.load(std::memory_order_relaxed);
    s.gated_by_cache = gated_by_cache_.load(std::memory_order_relaxed);
    s.gated_by_budget = gated_by_budget_.load(std::memory_order_relaxed);

    uint64_t reqs = s.prefetch_requests;
    s.avg_prefetch_latency_us = reqs == 0 ? 0 :
        total_latency_us_.load(std::memory_order_relaxed) / reqs;

    uint64_t denom = s.useful_prefetches + s.wasted_prefetches + s.cancelled_prefetches;
    s.prefetch_hit_rate = denom == 0 ? 0.0 :
        static_cast<double>(s.useful_prefetches) / static_cast<double>(denom);
    return s;
}

void PrefetchEngine::reset_stats() {
    prefetch_requests_.store(0);
    useful_prefetches_.store(0);
    wasted_prefetches_.store(0);
    duplicate_prefetches_.store(0);
    cancelled_prefetches_.store(0);
    prefetch_bytes_.store(0);
    gated_by_queue_.store(0);
    gated_by_cache_.store(0);
    gated_by_budget_.store(0);
    total_latency_us_.store(0);
    {
        std::lock_guard<std::mutex> lock(tracking_mu_);
        prefetched_coord_to_count_.clear();
    }
}

void PrefetchEngine::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;
    // Cancel all queued prefetches.
    queue_.cancel_all_prefetches();
}

} // namespace asema
