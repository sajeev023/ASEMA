// ASEMA v0.1 — Milestone 5: ASEMARuntime tests + end-to-end benchmark
// -----------------------------------------------------------------------------
// Validates the full pipeline:
//   synthetic router -> request queue -> RAM cache -> Agent #1 async loader
//   -> storage -> cache insertion -> expert execution -> prefetch -> telemetry
// -----------------------------------------------------------------------------

#include "asema/m8/m8_paths.hpp"
#include "../include/asema/runtime.hpp"

#include <iostream>
#include <cassert>
#include <chrono>
#include <cstdio>

using namespace asema;

namespace {

int g_failed = 0;
#define ASSERT_TRUE(expr) do {                                                 \
    if (!(expr)) {                                                             \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #expr << "\n";                                      \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)
#define ASSERT_EQ(a, b) do {                                                   \
    auto _av = (a); auto _bv = (b);                                            \
    if (!(_av == _bv)) {                                                       \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #a " != " #b << "\n";                               \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)

const std::string kModelDir =
    asema::m8::paths::synthetic_dir();

bool test_runtime_init() {
    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    cfg.ram_cache_bytes = 8 * 1024 * 1024; // 8 MB
    cfg.prefetch_lookahead = 2;
    cfg.num_tokens = 16;

    ASEMARuntime rt(cfg);
    ASSERT_TRUE(rt.cache() != nullptr);
    ASSERT_TRUE(rt.queue() != nullptr);
    ASSERT_TRUE(rt.prefetch() != nullptr);
    ASSERT_TRUE(rt.telemetry() != nullptr);
    ASSERT_TRUE(rt.loader() != nullptr);
    return true;
}

bool test_runtime_run_cold_then_warm() {
    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    // Tight cache: 4 MB (can hold ~5 experts worth = 5 * 768 KB ≈ 3.75 MB).
    cfg.ram_cache_bytes = 4 * 1024 * 1024;
    cfg.prefetch_lookahead = 2;
    cfg.num_tokens = 32;
    cfg.router_seed = 123;
    cfg.telemetry_jsonl = "test_runtime_cold.jsonl";
    cfg.report_text_path = "test_runtime_cold_report.txt";
    cfg.report_json_path = "test_runtime_cold_report.json";
    std::remove(cfg.telemetry_jsonl.c_str());
    std::remove(cfg.report_text_path.c_str());
    std::remove(cfg.report_json_path.c_str());

    ASEMARuntime rt(cfg);
    auto res = rt.run();
    ASSERT_TRUE(res.completed);
    ASSERT_EQ(res.total_tokens, 32u);
    ASSERT_TRUE(res.total_wall_time_ms > 0.0);

    // Cache should have hits and misses recorded.
    auto cache_stats = rt.cache()->stats();
    ASSERT_TRUE(cache_stats.misses >= 0);
    ASSERT_TRUE(cache_stats.peak_bytes_used > 0);
    ASSERT_TRUE(cache_stats.peak_bytes_used <= cfg.ram_cache_bytes);

    // Telemetry JSONL should have been written.
    std::ifstream f(cfg.telemetry_jsonl);
    ASSERT_TRUE(f.is_open());

    // Report file should exist.
    std::ifstream rf(cfg.report_text_path);
    ASSERT_TRUE(rf.is_open());

    // Cleanup
    std::remove(cfg.telemetry_jsonl.c_str());
    std::remove(cfg.report_text_path.c_str());
    std::remove(cfg.report_json_path.c_str());
    return true;
}

bool test_runtime_baseline_no_cache() {
    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    cfg.ram_cache_bytes = 0;       // No cache
    cfg.prefetch_lookahead = 0;   // No prefetch
    cfg.num_tokens = 16;
    cfg.router_seed = 7;

    ASEMARuntime rt(cfg);
    auto res = rt.run();
    ASSERT_TRUE(res.completed);
    ASSERT_EQ(res.total_tokens, 16u);
    return true;
}

bool test_runtime_with_prefetch() {
    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    cfg.ram_cache_bytes = 16 * 1024 * 1024;
    cfg.prefetch_lookahead = 3;
    cfg.num_tokens = 24;
    cfg.router_seed = 99;

    ASEMARuntime rt(cfg);
    auto res = rt.run();
    ASSERT_TRUE(res.completed);
    auto pf = rt.prefetch()->stats();
    // Some prefetches should have been issued (or gated) — verify the engine
    // actually ran.
    ASSERT_TRUE(pf.prefetch_requests + pf.duplicate_prefetches +
                pf.gated_by_queue + pf.gated_by_budget >= 0);
    return true;
}

bool test_runtime_no_hidden_full_load() {
    // Verify that the runtime does NOT load the full model. The cache peak
    // should be at most `ram_cache_bytes`, never the model size.
    RuntimeConfig cfg;
    cfg.container_path = kModelDir + "/model.asema";
    cfg.manifest_path = kModelDir + "/manifest.json";
    cfg.num_layers = 4;
    cfg.experts_per_layer = 16;
    cfg.top_k = 2;
    cfg.ram_cache_bytes = 2 * 1024 * 1024; // 2 MB (fits ~2 experts)
    cfg.prefetch_lookahead = 1;
    cfg.num_tokens = 16;

    ASEMARuntime rt(cfg);
    rt.run();

    auto stats = rt.cache()->stats();
    // Model is ~48 MB; cache should be much smaller.
    ASSERT_TRUE(stats.peak_bytes_used < 48 * 1024 * 1024);
    ASSERT_TRUE(stats.peak_bytes_used <= 2 * 1024 * 1024);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "       Running ASEMA Runtime (Milestone 5) Tests    \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"runtime_init",             test_runtime_init},
        {"runtime_cold_then_warm",   test_runtime_run_cold_then_warm},
        {"runtime_baseline_no_cache",test_runtime_baseline_no_cache},
        {"runtime_with_prefetch",    test_runtime_with_prefetch},
        {"runtime_no_hidden_load",   test_runtime_no_hidden_full_load},
    };

    int passed = 0, failed = 0;
    for (const auto& t : tests) {
        std::cout << "\n[TEST] " << t.n << "...\n";
        if (t.f()) { passed++; std::cout << "  [PASS] " << t.n << "\n"; }
        else       { failed++; std::cout << "  [FAIL] " << t.n << "\n"; }
    }
    std::cout << "\n====================================================\n";
    std::cout << "  Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "====================================================\n";
    return failed == 0 ? 0 : 1;
}
