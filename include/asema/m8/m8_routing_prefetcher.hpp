#pragma once

#include "asema/m8/m8_expert_cache.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_router.hpp"

#include <vector>
#include <unordered_set>
#include <memory>
#include <mutex>

namespace asema {
namespace m8 {

struct PrefetchTelemetry {
    uint64_t total_predictions{0};
    uint64_t useful_prefetches{0};
    uint64_t wasted_prefetches{0};
    uint64_t duplicate_suppressed{0};
    double prediction_accuracy{0.0};
    double latency_reduction_ms{0.0};
    size_t additional_nvme_bytes{0};
};

class M8RoutingPrefetcher {
public:
    explicit M8RoutingPrefetcher(std::shared_ptr<M8ExpertCacheEngine> cache,
                                 std::shared_ptr<M8ByteRangeLoader> loader);

    // Predicts and prefetches top candidates for next layer or subsequent token
    void on_layer_routed(int current_layer,
                         const std::vector<int>& active_top6,
                         const std::vector<float>& full_router_scores);

    // Evaluates whether an actual required expert was successfully pre-paged
    void on_expert_demanded(int layer, int expert_id);

    PrefetchTelemetry get_telemetry() const;
    void reset();

private:
    std::shared_ptr<M8ExpertCacheEngine> cache_;
    std::shared_ptr<M8ByteRangeLoader> loader_;
    mutable std::mutex mutex_;

    PrefetchTelemetry telemetry_;
    std::unordered_set<int64_t> in_flight_or_speculative_;

    // Transition co-occurrence matrix between layer L and L+1
    std::vector<std::vector<int>> transition_matrix_; // [40][384] top co-occurring expert
};

} // namespace m8
} // namespace asema
