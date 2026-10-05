// ASEMA v0.1 — Milestone 4: ExpertRequestQueue tests
// -----------------------------------------------------------------------------

#include "../include/asema/request_queue.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
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

bool test_basic_push_pop() {
    ExpertRequestQueue q(64);
    auto id = q.push(ExpertCoord{0, 0}, LoadPriority::REQUIRED_NOW);
    ASSERT_TRUE(id != 0);
    auto popped = q.try_pop();
    ASSERT_TRUE(popped.has_value());
    ASSERT_EQ(popped->coord.layer_id, 0u);
    ASSERT_EQ(popped->coord.expert_id, 0u);
    ASSERT_EQ(popped->priority, LoadPriority::REQUIRED_NOW);
    return true;
}

bool test_priority_ordering() {
    ExpertRequestQueue q(64);

    // Push in reverse-priority order
    q.push(ExpertCoord{1, 0}, LoadPriority::BACKGROUND);
    q.push(ExpertCoord{2, 0}, LoadPriority::PREFETCH);
    q.push(ExpertCoord{3, 0}, LoadPriority::REQUIRED_NOW);

    auto p1 = q.try_pop(); ASSERT_TRUE(p1.has_value());
    ASSERT_EQ(p1->coord.layer_id, 3u); // REQUIRED_NOW first

    auto p2 = q.try_pop(); ASSERT_TRUE(p2.has_value());
    ASSERT_EQ(p2->coord.layer_id, 2u); // PREFETCH second

    auto p3 = q.try_pop(); ASSERT_TRUE(p3.has_value());
    ASSERT_EQ(p3->coord.layer_id, 1u); // BACKGROUND third

    return true;
}

bool test_fifo_within_priority() {
    ExpertRequestQueue q(64);
    q.push(ExpertCoord{0, 0}, LoadPriority::REQUIRED_NOW);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    q.push(ExpertCoord{0, 1}, LoadPriority::REQUIRED_NOW);
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    q.push(ExpertCoord{0, 2}, LoadPriority::REQUIRED_NOW);

    auto p1 = q.try_pop(); ASSERT_EQ(p1->coord.expert_id, 0u);
    auto p2 = q.try_pop(); ASSERT_EQ(p2->coord.expert_id, 1u);
    auto p3 = q.try_pop(); ASSERT_EQ(p3->coord.expert_id, 2u);
    return true;
}

bool test_request_coalescing() {
    ExpertRequestQueue q(64);
    auto id1 = q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    auto id2 = q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    auto id3 = q.push(ExpertCoord{0, 0}, LoadPriority::REQUIRED_NOW);

    // All three pushes returned the same logical id (the first one).
    ASSERT_EQ(id1, id2);
    ASSERT_EQ(id1, id3);

    // Only one physical node in the queue, with waiter_count == 3.
    auto popped = q.try_pop();
    ASSERT_TRUE(popped.has_value());
    ASSERT_EQ(popped->waiter_count, 3u);

    auto stats = q.stats();
    ASSERT_EQ(stats.coalesced, 2u); // 2 of the 3 pushes were coalesced
    return true;
}

bool test_priority_boost() {
    ExpertRequestQueue q(64);
    q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    auto id = q.push(ExpertCoord{0, 0}, LoadPriority::REQUIRED_NOW);

    auto popped = q.try_pop();
    ASSERT_TRUE(popped.has_value());
    ASSERT_EQ(popped->priority, LoadPriority::REQUIRED_NOW);
    return true;
}

bool test_cancel() {
    ExpertRequestQueue q(64);
    auto id = q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    bool cancelled = q.cancel(id);
    ASSERT_TRUE(cancelled);

    // try_pop should NOT return the cancelled entry.
    auto popped = q.try_pop();
    ASSERT_TRUE(!popped.has_value());
    return true;
}

bool test_cancel_coord() {
    ExpertRequestQueue q(64);
    q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH); // coalesced
    size_t n = q.cancel_coord(ExpertCoord{0, 0});
    ASSERT_TRUE(n >= 1);
    auto popped = q.try_pop();
    ASSERT_TRUE(!popped.has_value());
    return true;
}

bool test_cancel_all_prefetches() {
    ExpertRequestQueue q(64);
    q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    q.push(ExpertCoord{0, 1}, LoadPriority::PREFETCH);
    q.push(ExpertCoord{0, 2}, LoadPriority::REQUIRED_NOW);
    size_t n = q.cancel_all_prefetches();
    ASSERT_TRUE(n >= 2);

    // Only the REQUIRED_NOW should remain.
    auto popped = q.try_pop();
    ASSERT_TRUE(popped.has_value());
    ASSERT_EQ(popped->coord.layer_id, 0u);
    ASSERT_EQ(popped->coord.expert_id, 2u);
    return true;
}

bool test_deprioritize() {
    ExpertRequestQueue q(64);
    auto id = q.push(ExpertCoord{0, 0}, LoadPriority::PREFETCH);
    bool ok = q.deprioritize(id);
    ASSERT_TRUE(ok);
    auto popped = q.try_pop();
    ASSERT_EQ(popped->priority, LoadPriority::BACKGROUND);
    return true;
}

bool test_inflight_tracking() {
    ExpertRequestQueue q(64);
    ExpertCoord c{0, 0};

    ASSERT_TRUE(!q.is_in_flight(c));
    bool first = q.mark_in_flight(c);
    ASSERT_TRUE(first);
    bool second = q.mark_in_flight(c);
    ASSERT_TRUE(!second); // already inflight
    q.clear_in_flight(c);
    ASSERT_TRUE(!q.is_in_flight(c));
    return true;
}

bool test_concurrent_push_pop() {
    ExpertRequestQueue q(1024);
    std::atomic<bool> start{false};
    std::atomic<int> produced{0};
    std::atomic<int> consumed{0};
    std::atomic<int> errors{0};

    constexpr int N = 1000;

    std::thread producer([&]() {
        while (!start.load()) std::this_thread::yield();
        for (int i = 0; i < N; ++i) {
            uint32_t layer = (uint32_t)(i / 32);
            uint32_t expert = (uint32_t)(i % 32);
            q.push(ExpertCoord{layer, expert}, LoadPriority::REQUIRED_NOW);
            produced.fetch_add(1);
        }
    });

    std::thread consumer([&]() {
        while (!start.load()) std::this_thread::yield();
        for (int i = 0; i < N; ++i) {
            std::optional<PoppedRequest> p;
            while (!(p = q.try_pop()).has_value()) {
                std::this_thread::yield();
            }
            consumed.fetch_add(1);
        }
    });

    start.store(true);
    producer.join();
    consumer.join();

    ASSERT_EQ(produced.load(), N);
    ASSERT_EQ(consumed.load(), N);
    return true;
}

bool test_capacity_limit() {
    ExpertRequestQueue q(4);
    // Push 5 distinct coords
    for (uint32_t i = 0; i < 5; ++i) {
        q.push(ExpertCoord{0, i}, LoadPriority::REQUIRED_NOW);
    }
    ASSERT_EQ(q.size(), 4u);
    return true;
}

bool test_wait_pop() {
    ExpertRequestQueue q(64);
    std::atomic<bool> got{false};
    std::thread consumer([&]() {
        auto p = q.wait_pop();
        got.store(p.has_value());
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.push(ExpertCoord{5, 5}, LoadPriority::REQUIRED_NOW);
    consumer.join();
    ASSERT_TRUE(got.load());
    return true;
}

bool test_required_now_starvation_protection() {
    ExpertRequestQueue q(64);

    // Saturate with prefetches.
    for (uint32_t i = 0; i < 30; ++i) {
        q.push(ExpertCoord{0, i}, LoadPriority::PREFETCH);
    }

    // Inject a REQUIRED_NOW.
    q.push(ExpertCoord{1, 1}, LoadPriority::REQUIRED_NOW);

    // The first pop must be the REQUIRED_NOW.
    auto p = q.try_pop();
    ASSERT_TRUE(p.has_value());
    ASSERT_EQ(p->coord.layer_id, 1u);
    ASSERT_EQ(p->coord.expert_id, 1u);
    ASSERT_EQ(p->priority, LoadPriority::REQUIRED_NOW);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "      Running ASEMA RequestQueue (M4) Tests         \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"basic_push_pop",              test_basic_push_pop},
        {"priority_ordering",           test_priority_ordering},
        {"fifo_within_priority",        test_fifo_within_priority},
        {"request_coalescing",          test_request_coalescing},
        {"priority_boost",              test_priority_boost},
        {"cancel",                      test_cancel},
        {"cancel_coord",                test_cancel_coord},
        {"cancel_all_prefetches",       test_cancel_all_prefetches},
        {"deprioritize",                test_deprioritize},
        {"inflight_tracking",           test_inflight_tracking},
        {"concurrent_push_pop",         test_concurrent_push_pop},
        {"capacity_limit",              test_capacity_limit},
        {"wait_pop",                    test_wait_pop},
        {"required_now_starvation",     test_required_now_starvation_protection},
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
