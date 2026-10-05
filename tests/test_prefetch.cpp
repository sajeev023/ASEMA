// ASEMA v0.1 — Milestone 4: PrefetchEngine tests
// -----------------------------------------------------------------------------

#include "../include/asema/prefetch.hpp"
#include "../include/asema/request_queue.hpp"
#include "../include/asema/cache.hpp"
#include "../include/asema/expert.hpp"

#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

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

std::shared_ptr<ExpertBuffer> make_buf(size_t n) {
    return std::make_shared<ExpertBuffer>(n, 64);
}

ExpertMetadata make_meta(uint32_t l, uint32_t e, uint64_t sz) {
    ExpertMetadata m;
    m.layer_id = l; m.expert_id = e;
    m.storage_offset = 0; m.storage_length = sz; m.alignment = 64;
    m.dtype = "fp16"; m.quant_type = "none";
    return m;
}

bool test_lookahead_0() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/0);

    // With lookahead 0, no prefetch requests should ever be issued.
    std::vector<std::vector<ExpertCoord>> hist = {{{0, 0}, {0, 1}}};
    pf.tick(0);
    ASSERT_EQ(pf.stats().prefetch_requests, 0u);
    return true;
}

bool test_lookahead_1() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/1);

    // Seed history with layer 1 experts (what we want to predict).
    std::vector<std::vector<ExpertCoord>> hist = {
        {{1, 3}, {1, 7}},
        {{1, 3}, {1, 7}},
    };
    pf.on_token_complete(0, hist);

    // Tick at layer 0 -> prefetch layer 1 experts.
    pf.tick(0);

    auto stats = pf.stats();
    ASSERT_TRUE(stats.prefetch_requests >= 1);
    // Queue should now have prefetch entries.
    auto p = queue.try_pop();
    ASSERT_TRUE(p.has_value());
    ASSERT_EQ(p->priority, LoadPriority::PREFETCH);
    return true;
}

bool test_lookahead_2() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/2);

    // History must contain layer-1 and layer-2 experts for lookahead=2.
    std::vector<std::vector<ExpertCoord>> hist = {
        {{1, 0}, {2, 5}},
    };
    pf.on_token_complete(0, hist);
    pf.tick(0);

    auto stats = pf.stats();
    ASSERT_TRUE(stats.prefetch_requests >= 1);
    return true;
}

bool test_prefetch_priority() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/1);

    std::vector<std::vector<ExpertCoord>> hist = {{{1, 0}}};
    pf.on_token_complete(0, hist);

    // Add a higher-priority REQUIRED_NOW after a prefetch tick.
    pf.tick(0);
    queue.push(ExpertCoord{5, 5}, LoadPriority::REQUIRED_NOW);

    // Pop order: REQUIRED_NOW first, then PREFETCH.
    auto p1 = queue.try_pop(); ASSERT_TRUE(p1.has_value());
    ASSERT_EQ(p1->priority, LoadPriority::REQUIRED_NOW);
    ASSERT_EQ(p1->coord.expert_id, 5u);

    auto p2 = queue.try_pop(); ASSERT_TRUE(p2.has_value());
    ASSERT_EQ(p2->priority, LoadPriority::PREFETCH);
    return true;
}

bool test_prefetch_duplicate_coalescing() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/1);

    std::vector<std::vector<ExpertCoord>> hist = {{{1, 4}}};
    pf.on_token_complete(0, hist);

    // Tick twice — the same coord should be deduplicated via in-flight tracking.
    pf.tick(0);
    pf.tick(0);

    auto stats = pf.stats();
    ASSERT_TRUE(stats.duplicate_prefetches >= 1);
    return true;
}

bool test_prefetch_cancellation() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/2);

    std::vector<std::vector<ExpertCoord>> hist = {{{1, 0}, {2, 0}}};
    pf.on_token_complete(0, hist);
    pf.tick(0);

    size_t cancelled = queue.cancel_all_prefetches();
    ASSERT_TRUE(cancelled > 0);
    // After cancel, queue should have no items.
    auto p = queue.try_pop();
    ASSERT_TRUE(!p.has_value());

    auto qstats = queue.stats();
    ASSERT_TRUE(qstats.cancelled >= 1);
    return true;
}

bool test_prefetch_cache_interaction() {
    // If the expert is already in the cache, the prefetcher should NOT issue
    // a request.
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/1);

    // Pre-populate cache with layer 1 expert.
    ExpertCoord target{1, 0};
    cache.put(target, make_meta(1, 0, 1024), make_buf(1024));

    // Build a history that would predict target.
    std::vector<std::vector<ExpertCoord>> hist = {{{1, 0}}};
    pf.on_token_complete(0, hist);

    pf.tick(0);
    auto stats = pf.stats();
    // The prefetch request for (1, 0) was deduplicated by cache.contains.
    // Should not have incremented prefetch_requests.
    ASSERT_EQ(stats.prefetch_requests, 0u);
    return true;
}

bool test_required_request_starvation_protection() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/2);

    std::vector<std::vector<ExpertCoord>> hist = {{{1, 0}, {2, 0}}};
    pf.on_token_complete(0, hist);
    pf.tick(0);

    // Push a REQUIRED_NOW. It must come out first.
    queue.push(ExpertCoord{99, 99}, LoadPriority::REQUIRED_NOW);
    auto p = queue.try_pop();
    ASSERT_TRUE(p.has_value());
    ASSERT_EQ(p->priority, LoadPriority::REQUIRED_NOW);
    return true;
}

bool test_useful_wasted_accounting() {
    ExpertRAMCache cache(64 * 1024);
    ExpertRequestQueue queue(128);
    PrefetchEngine pf(queue, cache, /*lookahead=*/1);

    // Seed history with two layer-1 experts.
    std::vector<std::vector<ExpertCoord>> hist = {{{1, 0}, {1, 1}}};
    pf.on_token_complete(0, hist);
    pf.tick(0);

    // The prefetcher should have pushed requests for both (1,0) and (1,1).
    pf.record_useful(ExpertCoord{1, 0});
    pf.record_wasted(ExpertCoord{1, 1});

    auto stats = pf.stats();
    ASSERT_TRUE(stats.useful_prefetches >= 1);
    ASSERT_TRUE(stats.wasted_prefetches >= 1);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "      Running ASEMA PrefetchEngine (M4) Tests        \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"lookahead_0",                   test_lookahead_0},
        {"lookahead_1",                   test_lookahead_1},
        {"lookahead_2",                   test_lookahead_2},
        {"prefetch_priority",             test_prefetch_priority},
        {"prefetch_duplicate_coalescing", test_prefetch_duplicate_coalescing},
        {"prefetch_cancellation",         test_prefetch_cancellation},
        {"prefetch_cache_interaction",    test_prefetch_cache_interaction},
        {"required_starvation",           test_required_request_starvation_protection},
        {"useful_wasted_accounting",      test_useful_wasted_accounting},
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
