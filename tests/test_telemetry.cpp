// ASEMA v0.1 — Milestone 5: TelemetryEngine tests
// -----------------------------------------------------------------------------

#include "../include/asema/telemetry.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include <cassert>

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

bool test_basic_emit() {
    TelemetryConfig cfg;
    cfg.jsonl_path = ""; // stdout
    cfg.background_flush = false;
    TelemetryEngine t(cfg);

    TelemetryEvent e;
    e.type = EventType::EXPERT_LOAD_END;
    e.layer_id = 5;
    e.expert_id = 17;
    e.bytes = 786432;
    e.latency_us = 1542;
    e.success = true;
    t.emit(e);
    t.flush();
    return true;
}

bool test_latency_percentiles() {
    // Verify the LatencyStats percentile computation.
    std::vector<double> values;
    for (int i = 1; i <= 100; ++i) values.push_back(i);
    auto stats = LatencyStats::from(values);
    ASSERT_EQ(stats.count, 100u);
    // For [1..100] sorted: median=50, p95=95.
    if (stats.p50_us < 49.0 || stats.p50_us > 51.0) {
        std::cerr << "p50 out of range: " << stats.p50_us << "\n";
        return false;
    }
    if (stats.p95_us < 94.0 || stats.p95_us > 96.0) {
        std::cerr << "p95 out of range: " << stats.p95_us << "\n";
        return false;
    }
    if (stats.avg_us < 50.0 || stats.avg_us > 51.0) {
        std::cerr << "avg out of range: " << stats.avg_us << "\n";
        return false;
    }
    return true;
}

bool test_counters_and_report() {
    TelemetryConfig cfg;
    cfg.background_flush = false;
    TelemetryEngine t(cfg);

    t.increment_cache_hit();
    t.increment_cache_hit();
    t.increment_cache_miss();
    t.increment_cache_eviction(1024);
    t.increment_cache_insertion(2048);
    t.increment_ssd_read(4096);
    t.increment_prefetch_request();
    t.increment_prefetch_useful();
    t.increment_prefetch_wasted();
    t.set_ram_bytes(2048, 4096, 16384);

    auto r = t.build_report();
    ASSERT_EQ(r.cache_hits, 2u);
    ASSERT_EQ(r.cache_misses, 1u);
    ASSERT_EQ(r.cache_evictions, 1u);
    ASSERT_EQ(r.ssd_bytes, 4096u);
    ASSERT_TRUE(r.ram_bytes_used > 0);
    return true;
}

bool test_jsonl_output() {
    std::string path = "test_telemetry_output.jsonl";
    std::remove(path.c_str());
    TelemetryConfig cfg;
    cfg.jsonl_path = path;
    cfg.background_flush = false;
    TelemetryEngine t(cfg);

    t.expert_load_end(1, ExpertCoord{0, 0}, 1024, 500, true);
    t.expert_cache_hit(1, ExpertCoord{0, 1}, 5);
    t.flush();
    t.shutdown();

    std::ifstream f(path);
    ASSERT_TRUE(f.is_open());
    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    // Each line must be valid JSON.
    int line_count = 0;
    std::istringstream lines(content);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) continue;
        // Must start with '{' and end with '}'
        ASSERT_TRUE(line.front() == '{');
        ASSERT_TRUE(line.back() == '}');
        // Must contain "event":
        ASSERT_TRUE(line.find("\"event\":") != std::string::npos);
        line_count++;
    }
    ASSERT_TRUE(line_count >= 2);
    std::remove(path.c_str());
    return true;
}

bool test_event_names() {
    ASSERT_TRUE(std::string(event_name(EventType::TOKEN_BEGIN)) == "TOKEN_BEGIN");
    ASSERT_TRUE(std::string(event_name(EventType::EXPERT_LOAD_END)) == "EXPERT_LOAD_END");
    ASSERT_TRUE(std::string(event_name(EventType::PREFETCH_USEFUL)) == "PREFETCH_USEFUL");
    return true;
}

bool test_runtime_begin_end() {
    TelemetryConfig cfg;
    cfg.background_flush = false;
    TelemetryEngine t(cfg);
    t.runtime_begin();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    t.runtime_end(100.0);
    auto r = t.build_report();
    ASSERT_TRUE(r.total_wall_time_ms >= 0.0);
    return true;
}

bool test_concurrent_emit() {
    TelemetryConfig cfg;
    cfg.background_flush = false;
    cfg.buffer_capacity = 1 << 16;
    TelemetryEngine t(cfg);

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int tid = 0; tid < 4; ++tid) {
        threads.emplace_back([&, tid]() {
            while (!start.load()) std::this_thread::yield();
            for (int i = 0; i < 1000; ++i) {
                t.increment_cache_hit();
            }
        });
    }
    start.store(true);
    for (auto& th : threads) th.join();

    auto r = t.build_report();
    ASSERT_EQ(r.cache_hits, 4000u);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "      Running ASEMA TelemetryEngine (M5) Tests       \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"basic_emit",                test_basic_emit},
        {"latency_percentiles",       test_latency_percentiles},
        {"counters_and_report",       test_counters_and_report},
        {"jsonl_output",              test_jsonl_output},
        {"event_names",               test_event_names},
        {"runtime_begin_end",         test_runtime_begin_end},
        {"concurrent_emit",           test_concurrent_emit},
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
