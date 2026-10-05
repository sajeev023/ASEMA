// ASEMA v0.1 — Full Pipeline Benchmark Tool
// -----------------------------------------------------------------------------
// Compares MODE A (no cache, no prefetch) vs MODE B (cache + prefetch) across
// a matrix of (RAM cache size, prefetch lookahead) configurations. Produces
// benchmark_report.txt.
// -----------------------------------------------------------------------------

#include "asema/m8/m8_paths.hpp"
#include "../include/asema/runtime.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <map>

using namespace asema;

namespace {

struct ExperimentResult {
    std::string label;
    uint64_t ram_cache_bytes{0};
    size_t prefetch_lookahead{0};
    uint64_t total_tokens{0};
    double wall_time_ms{0.0};
    double first_token_ms{0.0};
    double ms_per_token{0.0};
    double tokens_per_sec{0.0};
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    uint64_t cache_evictions{0};
    uint64_t prefetch_requests{0};
    uint64_t prefetch_useful{0};
    uint64_t prefetch_wasted{0};
    uint64_t ssd_bytes{0};
    uint64_t ssd_reads{0};
};

const std::string kModelDir =
    asema::m8::paths::synthetic_dir();

ExperimentResult run_experiment(const std::string& label,
                                 uint64_t ram_cache_bytes,
                                 size_t prefetch_lookahead,
                                 uint32_t num_tokens,
                                 uint64_t router_seed) {
    ExperimentResult r;
    r.label = label;
    r.ram_cache_bytes = ram_cache_bytes;
    r.prefetch_lookahead = prefetch_lookahead;
    r.total_tokens = num_tokens;

    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    cfg.ram_cache_bytes = ram_cache_bytes;
    cfg.prefetch_lookahead = prefetch_lookahead;
    cfg.num_tokens = num_tokens;
    cfg.router_seed = router_seed;
    cfg.telemetry_jsonl = "";
    cfg.report_text_path = "";
    cfg.report_json_path = "";

    ASEMARuntime rt(cfg);
    auto res = rt.run();

    r.wall_time_ms = res.total_wall_time_ms;
    r.first_token_ms = res.first_token_latency_ms;
    r.ms_per_token = res.total_wall_time_ms / static_cast<double>(num_tokens);
    r.tokens_per_sec = 1000.0 / r.ms_per_token;

    if (rt.cache()) {
        auto cs = rt.cache()->stats();
        r.cache_hits = cs.hits;
        r.cache_misses = cs.misses;
        r.cache_evictions = cs.evictions;
    }
    if (rt.prefetch()) {
        auto ps = rt.prefetch()->stats();
        r.prefetch_requests = ps.prefetch_requests;
        r.prefetch_useful = ps.useful_prefetches;
        r.prefetch_wasted = ps.wasted_prefetches;
    }
    if (rt.telemetry()) {
        auto tr = rt.telemetry()->build_report();
        r.ssd_bytes = tr.ssd_bytes;
        r.ssd_reads = tr.ssd_reads;
    }
    return r;
}

void write_report(const std::vector<ExperimentResult>& results,
                  const std::string& path) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Could not write " << path << "\n";
        return;
    }
    f << "====================================================\n";
    f << "      ASEMA v0.1 Full Pipeline Benchmark Report      \n";
    f << "====================================================\n";
    f << "Model: " << kModelDir << "\n";
    f << "Workload: 4 layers x 16 experts x top-2 = 8 experts per token\n";
    f << "Synthetic tokens: " << (results.empty() ? 0 : results[0].total_tokens) << "\n";
    f << "====================================================\n\n";

    f << std::left
      << std::setw(28) << "Experiment"
      << std::setw(12) << "RAM(MB)"
      << std::setw(10) << "Lookahead"
      << std::setw(14) << "toks/s"
      << std::setw(14) << "ms/tok"
      << std::setw(12) << "hit%"
      << std::setw(12) << "miss"
      << std::setw(12) << "SSD(MB)"
      << std::setw(10) << "SSDreads"
      << std::setw(10) << "pf-req"
      << std::setw(10) << "pf-use"
      << std::setw(10) << "pf-waste"
      << "\n";
    f << std::string(150, '-') << "\n";

    for (const auto& r : results) {
        double hit_rate = (r.cache_hits + r.cache_misses == 0) ? 0.0 :
            100.0 * static_cast<double>(r.cache_hits) / static_cast<double>(r.cache_hits + r.cache_misses);

        f << std::left
          << std::setw(28) << r.label
          << std::setw(12) << std::fixed << std::setprecision(1) << (r.ram_cache_bytes / (1024.0 * 1024.0))
          << std::setw(10) << r.prefetch_lookahead
          << std::setw(14) << std::setprecision(2) << r.tokens_per_sec
          << std::setw(14) << std::setprecision(3) << r.ms_per_token
          << std::setw(12) << std::setprecision(1) << hit_rate
          << std::setw(12) << r.cache_misses
          << std::setw(12) << std::setprecision(2) << (r.ssd_bytes / (1024.0 * 1024.0))
          << std::setw(10) << r.ssd_reads
          << std::setw(10) << r.prefetch_requests
          << std::setw(10) << r.prefetch_useful
          << std::setw(10) << r.prefetch_wasted
          << "\n";
    }
    f << "\n====================================================\n";
    f << "  END OF REPORT\n";
    f << "====================================================\n";
}

} // namespace

int main(int argc, char* argv[]) {
    uint32_t num_tokens = 32;
    if (argc > 1) num_tokens = static_cast<uint32_t>(std::atoi(argv[1]));

    std::cout << "====================================================\n";
    std::cout << "    ASEMA v0.1 Full Pipeline Benchmark              \n";
    std::cout << "====================================================\n";
    std::cout << "Workload: 4 layers x 16 experts x top-2\n";
    std::cout << "Tokens: " << num_tokens << "\n\n";

    std::vector<ExperimentResult> results;

    // ----------------- EXPERIMENT A: No cache, no prefetch -----------------
    std::cout << "[A] No cache, no prefetch ...\n";
    results.push_back(run_experiment("A. No cache, no prefetch",
                                      0, 0, num_tokens, /*seed=*/11));

    // ----------------- EXPERIMENT B: Cache only -----------------
    std::cout << "[B] Cache 8MB, no prefetch ...\n";
    results.push_back(run_experiment("B. Cache 8MB, no prefetch",
                                      8ull * 1024 * 1024, 0, num_tokens, 11));

    // ----------------- EXPERIMENT C: Cache + prefetch -----------------
    std::cout << "[C] Cache 8MB + prefetch K=2 ...\n";
    results.push_back(run_experiment("C. Cache 8MB + prefetch K=2",
                                      8ull * 1024 * 1024, 2, num_tokens, 11));

    // ----------------- EXPERIMENT D: Cache size sweep -----------------
    std::cout << "[D] Cache size sweep ...\n";
    for (uint64_t mb : {1ull, 2ull, 4ull, 8ull, 12ull}) {
        std::string label = "D. Cache " + std::to_string(mb) + "MB K=2";
        results.push_back(run_experiment(label, mb * 1024 * 1024, 2, num_tokens, 11));
    }

    // ----------------- EXPERIMENT E: Lookahead sweep -----------------
    std::cout << "[E] Lookahead sweep (8MB cache) ...\n";
    for (size_t K : {0ull, 1ull, 2ull, 3ull, 4ull, 6ull}) {
        std::string label = "E. Cache 8MB K=" + std::to_string(K);
        results.push_back(run_experiment(label, 8ull * 1024 * 1024, K, num_tokens, 11));
    }

    // ----------------- EXPERIMENT F: Concurrent stress -----------------
    // Run multiple times back-to-back to stress the cache eviction logic.
    std::cout << "[F] Repeated runs (stress) ...\n";
    for (int rep = 0; rep < 3; ++rep) {
        std::string label = "F. Stress rep " + std::to_string(rep);
        results.push_back(run_experiment(label, 4ull * 1024 * 1024, 2, num_tokens, /*seed=*/11 + rep));
    }

    std::cout << "\n[OK] All experiments complete. Writing report...\n";
    write_report(results, "benchmark_report.txt");
    std::cout << "Report written to: benchmark_report.txt\n";

    // Also print a brief summary to stdout.
    std::cout << "\n=== Summary (ms/token) ===\n";
    for (const auto& r : results) {
        std::cout << "  " << std::left << std::setw(28) << r.label
                  << "  " << std::setprecision(3) << r.ms_per_token << " ms\n";
    }
    return 0;
}
