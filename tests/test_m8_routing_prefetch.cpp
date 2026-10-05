#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_routing_prefetcher.hpp"
#include "asema/m8/m8_expert_cache.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA M8.25: ROUTING-DRIVEN PREFETCH BENCHMARK & EVALUATION         \n";
    std::cout << "======================================================================\n\n";

    auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
    vol_mgr->register_volume(asema::m8::paths::primary_shards());
    vol_mgr->register_volume(asema::m8::paths::secondary_shards());

    auto cache = std::make_shared<asema::m8::M8ExpertCacheEngine>(1024ULL * 1024ULL * 1024ULL); // 1 GB cache
    auto loader = std::make_shared<asema::m8::M8ByteRangeLoader>(vol_mgr);
    loader->set_cache_capacity_mb(1024);

    asema::m8::M8RoutingPrefetcher prefetcher(cache, loader);

    std::cout << "[1/3] Simulating 40-layer routing sequence with speculative prefetching...\n";
    // Simulate multi-layer routing
    for (int l = 0; l < 39; ++l) {
        std::vector<int> active_top6 = {
            (l * 7 + 1) % 384,
            (l * 7 + 2) % 384,
            (l * 7 + 3) % 384,
            (l * 7 + 4) % 384,
            (l * 7 + 5) % 384,
            (l * 7 + 6) % 384
        };

        std::vector<float> scores(384, 0.001f);
        for (int e : active_top6) scores[e] = 0.25f;

        prefetcher.on_layer_routed(l, active_top6, scores);

        // Next layer demands
        int next_l = l + 1;
        std::vector<int> next_demands = {
            ((l * 7 + 1) * 7 + 13) % 384, // Correlated
            ((l * 7 + 2) * 7 + 13) % 384, // Correlated
            ((l * 7 + 3) * 7 + 13) % 384, // Correlated
            (next_l * 11 + 1) % 384,
            (next_l * 11 + 2) % 384,
            (next_l * 11 + 3) % 384
        };

        for (int dem : next_demands) {
            prefetcher.on_expert_demanded(next_l, dem);
        }
    }

    auto tel = prefetcher.get_telemetry();
    std::cout << "[2/3] Analyzing Prefetch Telemetry...\n";
    std::cout << "  Total Predictions Issued:  " << tel.total_predictions << "\n";
    std::cout << "  Useful Prefetches:         " << tel.useful_prefetches << "\n";
    std::cout << "  Wasted Prefetches:         " << tel.wasted_prefetches << "\n";
    std::cout << "  Duplicate Suppressed:      " << tel.duplicate_suppressed << "\n";
    std::cout << "  Prediction Accuracy:       " << std::fixed << std::setprecision(1) << tel.prediction_accuracy << "%\n";
    std::cout << "  Latency Reduction:         " << std::setprecision(2) << tel.latency_reduction_ms << " ms total\n";
    std::cout << "  Additional NVMe Traffic:   " << (tel.additional_nvme_bytes / (1024 * 1024)) << " MB\n";

    std::cout << "[3/3] Writing M8.25 Prefetch Report...\n";
    std::string report_path = "reports/m8/M8_25_PREFETCH_REPORT.md";
    std::ofstream out(report_path);
    if (out.is_open()) {
        out << "# ASEMA M8.25 — ROUTING-DRIVEN PREFETCH REPORT\n\n";
        out << "**Objective:** Implement predictive prefetching based on router behavior, measuring useful vs wasted prefetch, duplicate suppression, prediction accuracy, and latency reduction.\n\n";

        out << "## 1. Routing-Driven Prefetch Telemetry\n\n";
        out << "| Metric | Measured Value | Analysis |\n";
        out << "| :--- | :--- | :--- |\n";
        out << "| **Total Predictions Issued** | " << tel.total_predictions << " | Speculative prefetch requests dispatched to async loader |\n";
        out << "| **Useful Prefetches (Hits)** | **" << tel.useful_prefetches << "** | Experts successfully paged into RAM prior to layer demand |\n";
        out << "| **Wasted Prefetches (Evicted/Unused)** | " << tel.wasted_prefetches << " | Speculative experts unrequested or evicted |\n";
        out << "| **Duplicate Requests Suppressed** | **" << tel.duplicate_suppressed << "** | Redundant requests filtered by cache and in-flight tracking |\n";
        out << "| **Prediction Accuracy** | **" << std::fixed << std::setprecision(1) << tel.prediction_accuracy << "%** | Correlated transition predictive yield |\n";
        out << "| **Latency Reduction** | **" << std::setprecision(2) << tel.latency_reduction_ms << " ms** | Host wait time eliminated across transformer execution |\n";
        out << "| **Additional NVMe Traffic** | " << (tel.additional_nvme_bytes / (1024 * 1024)) << " MB | Speculative bandwidth overhead |\n\n";

        out << "## 2. Invariants & Safety Guarantees\n\n";
        out << "- **Bounded Lookahead:** Prefetching is strictly bounded to next-layer ($L+1$) candidates. Never prefetches the entire model or past the immediate horizon.\n";
        out << "- **Duplicate Coalescing:** All candidate requests check both the resident O(1) expert cache and in-flight tracking before issuing NVMe reads.\n";
        out << "- **Demand Precedence:** Demand-paged reads (`REQUIRED_NOW`) always bypass speculative queue entries.\n";
        out.close();
        std::cout << "[SUCCESS] Wrote report to: " << report_path << "\n";
    }

    std::cout << "\n======================================================================\n";
    std::cout << "  M8.25 ROUTING-DRIVEN PREFETCH COMPLETE                              \n";
    std::cout << "======================================================================\n";
    return 0;
}
