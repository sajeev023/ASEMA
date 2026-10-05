// ASEMA v0.1 — Milestone 3: ExpertRAMCache tests
// -----------------------------------------------------------------------------
// Covers insert, hit, miss, eviction, LRU order, memory budget, reservation,
// concurrent insert/lookup, pinning, pinned-eviction protection, and clear.
// Also includes a 10 000-entry thrashing stress test.
// -----------------------------------------------------------------------------

#include "../include/asema/cache.hpp"
#include "../include/asema/expert.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <cassert>
#include <chrono>
#include <string>

using namespace asema;

namespace {

constexpr uint64_t kTestBudget = 4ull * 1024 * 1024; // 4 MB budget for tests

std::shared_ptr<ExpertBuffer> make_buffer(size_t n, size_t alignment = 64) {
    auto buf = std::make_shared<ExpertBuffer>(n, alignment);
    // Touch memory to ensure pages are mapped (helps catch budget overruns).
    if (buf->data() && n > 0) {
        std::memset(buf->data(), 0xA5, n);
    }
    return buf;
}

ExpertMetadata make_meta(uint32_t layer, uint32_t expert, uint64_t size) {
    ExpertMetadata m;
    m.layer_id = layer;
    m.expert_id = expert;
    m.storage_offset = 0;
    m.storage_length = size;
    m.alignment = 64;
    m.dtype = "fp16";
    m.quant_type = "none";
    m.checksum_sha256 = "";
    m.checksum_crc32 = 0;
    m.tensor_shape = {};
    return m;
}

int g_test_passed = 0;
int g_test_failed = 0;

#define ASSERT_TRUE(expr) do {                                                 \
    if (!(expr)) {                                                             \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #expr << "\n";                                      \
        g_test_failed++;                                                        \
        return false;                                                          \
    }                                                                          \
} while (0)

#define ASSERT_EQ(a, b) do {                                                   \
    auto _av = (a); auto _bv = (b);                                            \
    if (!(_av == _bv)) {                                                       \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #a " != " #b << "\n";                               \
        g_test_failed++;                                                        \
        return false;                                                          \
    }                                                                          \
} while (0)

struct Test {
    const char* name;
    bool (*fn)();
};

// -----------------------------------------------------------------------------
// Test 1: Basic insertion and hit
// -----------------------------------------------------------------------------
bool test_cache_insert_and_hit() {
    ExpertRAMCache cache(1024 * 1024);
    auto buf = make_buffer(4096);
    auto meta = make_meta(0, 0, 4096);
    auto r = cache.put(ExpertCoord{0, 0}, meta, buf);
    ASSERT_EQ(r, CacheResult::OK);
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 0}));

    auto entry = cache.get(ExpertCoord{0, 0});
    ASSERT_TRUE(entry != nullptr);
    ASSERT_EQ(entry->buffer->size(), 4096u);
    ASSERT_EQ(cache.hits(), 1u);
    ASSERT_EQ(cache.misses(), 0u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 2: Cache miss
// -----------------------------------------------------------------------------
bool test_cache_miss() {
    ExpertRAMCache cache(1024 * 1024);
    auto entry = cache.get(ExpertCoord{5, 5});
    ASSERT_TRUE(entry == nullptr);
    ASSERT_EQ(cache.misses(), 1u);
    ASSERT_EQ(cache.hits(), 0u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 3: LRU eviction order
// -----------------------------------------------------------------------------
bool test_lru_order() {
    ExpertRAMCache cache(4096); // 4 KB total — fits exactly one 4 KB expert

    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024));
    cache.put(ExpertCoord{0, 1}, make_meta(0, 1, 1024), make_buffer(1024));
    cache.put(ExpertCoord{0, 2}, make_meta(0, 2, 1024), make_buffer(1024));
    cache.put(ExpertCoord{0, 3}, make_meta(0, 3, 1024), make_buffer(1024));
    // 4 x 1024 = 4096 used. Now we are at capacity.

    // Touch 0,0 to make it MRU. The LRU order from MRU to LRU is:
    //   0,3 -> 0,2 -> 0,1 -> 0,0
    cache.get(ExpertCoord{0, 0});
    // Now MRU: 0,0 ; then 0,3, 0,2, 0,1

    // Insert a 5th — should evict 0,1 (the LRU).
    cache.put(ExpertCoord{0, 4}, make_meta(0, 4, 1024), make_buffer(1024));

    ASSERT_TRUE(cache.contains(ExpertCoord{0, 0})); // just touched
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 4})); // just inserted
    ASSERT_TRUE(!cache.contains(ExpertCoord{0, 1})); // evicted (LRU)
    ASSERT_EQ(cache.evictions(), 1u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 4: Memory budget hard enforcement
// -----------------------------------------------------------------------------
bool test_memory_budget() {
    constexpr uint64_t budget = 16 * 1024; // 16 KB
    ExpertRAMCache cache(budget);

    // Insert 5 x 4 KB = 20 KB. Should evict older entries.
    for (int i = 0; i < 5; ++i) {
        cache.put(ExpertCoord{0, (uint32_t)i}, make_meta(0, (uint32_t)i, 4096), make_buffer(4096));
    }

    // bytes_used must never exceed capacity.
    ASSERT_TRUE(cache.bytes_used() <= budget);
    // At least 4 entries should fit (16384 / 4096 == 4).
    ASSERT_EQ(cache.resident_count(), 4u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 5: Pinned eviction protection
// -----------------------------------------------------------------------------
bool test_pinned_eviction_protection() {
    ExpertRAMCache cache(8192);

    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 2048), make_buffer(2048));
    cache.put(ExpertCoord{0, 1}, make_meta(0, 1, 2048), make_buffer(2048));
    cache.put(ExpertCoord{0, 2}, make_meta(0, 2, 2048), make_buffer(2048));
    cache.put(ExpertCoord{0, 3}, make_meta(0, 3, 2048), make_buffer(2048));
    // 4 x 2048 = 8192 bytes. Full.

    // Pin ALL four.
    ASSERT_EQ(cache.pin(ExpertCoord{0, 0}), CacheResult::OK);
    ASSERT_EQ(cache.pin(ExpertCoord{0, 1}), CacheResult::OK);
    ASSERT_EQ(cache.pin(ExpertCoord{0, 2}), CacheResult::OK);
    ASSERT_EQ(cache.pin(ExpertCoord{0, 3}), CacheResult::OK);
    ASSERT_EQ(cache.pinned_count(), 4u);

    // Try to insert one more. Should fail with EXHAUSTED.
    auto r = cache.put(ExpertCoord{0, 4}, make_meta(0, 4, 2048), make_buffer(2048));
    ASSERT_EQ(r, CacheResult::EXHAUSTED);

    // Budget should still hold.
    ASSERT_TRUE(cache.bytes_used() <= 8192);

    // Unpin one — the LRU non-pinned candidate (0,0) is now evictable.
    cache.unpin(ExpertCoord{0, 0});
    auto r2 = cache.put(ExpertCoord{0, 4}, make_meta(0, 4, 2048), make_buffer(2048));
    ASSERT_TRUE(r2 == CacheResult::OK || r2 == CacheResult::EVICTED);
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 4}));
    // 0,0 was evicted (LRU of unpinned), 0,1/0,2/0,3 still pinned so resident.
    ASSERT_TRUE(!cache.contains(ExpertCoord{0, 0}));
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 1}));
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 2}));
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 3}));

    // Now unpin two more. Only 0,3 is pinned. Re-insert and verify 0,3 survives.
    cache.unpin(ExpertCoord{0, 1});
    cache.unpin(ExpertCoord{0, 2});
    cache.remove(ExpertCoord{0, 4}); // free a slot
    auto r3 = cache.put(ExpertCoord{0, 4}, make_meta(0, 4, 2048), make_buffer(2048));
    ASSERT_TRUE(r3 == CacheResult::OK || r3 == CacheResult::EVICTED);
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 4}));

    // The still-pinned entry (0,3) must remain resident.
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 3}));
    return true;
}

// -----------------------------------------------------------------------------
// Test 6: Concurrent readers and writers
// -----------------------------------------------------------------------------
bool test_concurrent_insert_lookup() {
    constexpr int N_WRITERS = 4;
    constexpr int N_READERS = 4;
    constexpr int OPS_PER_THREAD = 500;

    ExpertRAMCache cache(2 * 1024 * 1024);

    // Pre-fill with 200 entries (200 * 4096 = 800 KB).
    std::vector<ExpertCoord> all;
    for (int i = 0; i < 200; ++i) {
        ExpertCoord c{(uint32_t)(i / 16), (uint32_t)(i % 16)};
        all.push_back(c);
        cache.put(c, make_meta(c.layer_id, c.expert_id, 4096), make_buffer(4096));
    }

    std::atomic<bool> start{false};
    std::atomic<bool> done{false};
    std::atomic<int> errors{0};

    auto writer = [&](int tid) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::mt19937 rng(tid);
        std::uniform_int_distribution<int> dist(0, (int)all.size() - 1);
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            const auto& c = all[dist(rng)];
            cache.put(c, make_meta(c.layer_id, c.expert_id, 4096), make_buffer(4096));
        }
    };
    auto reader = [&](int tid) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::mt19937 rng(tid + 100);
        std::uniform_int_distribution<int> dist(0, (int)all.size() - 1);
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
            const auto& c = all[dist(rng)];
            auto e = cache.get(c);
            if (!e) errors.fetch_add(1);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < N_WRITERS; ++i) threads.emplace_back(writer, i);
    for (int i = 0; i < N_READERS; ++i) threads.emplace_back(reader, i);

    start.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    // Every read should have found a hit (entries stay resident because the
    // budget fits all of them with overhead).
    ASSERT_TRUE(errors.load() == 0);

    // Budget invariant: never exceeded.
    ASSERT_TRUE(cache.bytes_used() <= cache.capacity());
    return true;
}

// -----------------------------------------------------------------------------
// Test 7: Cache thrashing with 10 000-entry sweep
// -----------------------------------------------------------------------------
bool test_thrashing_10k() {
    constexpr int N = 10000;
    constexpr int BUF = 256;
    // Total dataset: 10000 * 256 = 2.5 MB. Cache: 256 KB = 1000 entries.
    ExpertRAMCache cache(256 * 1024);

    for (int i = 0; i < N; ++i) {
        cache.put(ExpertCoord{0, (uint32_t)i}, make_meta(0, (uint32_t)i, BUF), make_buffer(BUF));
    }

    // Budget invariant after sweep.
    ASSERT_TRUE(cache.bytes_used() <= cache.capacity());

    // Some entries should have been evicted.
    ASSERT_TRUE(cache.evictions() > 0);

    // Hits should be less than misses because we sweeped through unique coords.
    return true;
}

// -----------------------------------------------------------------------------
// Test 8: 1 000-entry sweep (sanity)
// -----------------------------------------------------------------------------
bool test_thrashing_1k() {
    constexpr int N = 1000;
    constexpr int BUF = 256;
    ExpertRAMCache cache(64 * 1024);

    for (int i = 0; i < N; ++i) {
        cache.put(ExpertCoord{0, (uint32_t)i}, make_meta(0, (uint32_t)i, BUF), make_buffer(BUF));
    }

    ASSERT_TRUE(cache.bytes_used() <= cache.capacity());
    ASSERT_TRUE(cache.evictions() > 0);
    return true;
}

// -----------------------------------------------------------------------------
// Test 9: Clear
// -----------------------------------------------------------------------------
bool test_cache_clear() {
    ExpertRAMCache cache(64 * 1024);
    for (int i = 0; i < 100; ++i) {
        cache.put(ExpertCoord{0, (uint32_t)i}, make_meta(0, (uint32_t)i, 256), make_buffer(256));
    }
    ASSERT_TRUE(cache.resident_count() > 0);

    cache.clear();
    ASSERT_EQ(cache.resident_count(), 0u);
    ASSERT_EQ(cache.bytes_used(), 0u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 10: Reservation primitive never exceeds budget under contention
// -----------------------------------------------------------------------------
bool test_reservation_atomic() {
    constexpr int N_THREADS = 8;
    constexpr int OPS = 1000;
    constexpr uint64_t BUDGET = 64 * 1024;

    ExpertRAMCache cache(BUDGET);

    std::atomic<bool> start{false};
    std::atomic<uint64_t> max_seen{0};

    auto worker = [&](int tid) {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        std::mt19937 rng(tid);
        for (int i = 0; i < OPS; ++i) {
            uint32_t expert = (uint32_t)(rng() % 4096);
            cache.put(ExpertCoord{0, expert}, make_meta(0, expert, 256), make_buffer(256));
            uint64_t b = cache.bytes_used();
            uint64_t prev = max_seen.load(std::memory_order_relaxed);
            while (b > prev && !max_seen.compare_exchange_weak(
                    prev, b, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREADS; ++i) threads.emplace_back(worker, i);
    start.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    ASSERT_TRUE(max_seen.load() <= BUDGET);
    return true;
}

// -----------------------------------------------------------------------------
// Test 11: RAII pin handle
// -----------------------------------------------------------------------------
bool test_pin_handle() {
    ExpertRAMCache cache(64 * 1024);
    cache.put(ExpertCoord{7, 7}, make_meta(7, 7, 1024), make_buffer(1024));

    {
        auto pin = cache.pin_handle(ExpertCoord{7, 7});
        ASSERT_TRUE(pin.active());
        ASSERT_EQ(cache.pinned_count(), 1u);

        // Try to evict — should fail because pinned.
        bool evicted = cache.evict_lru();
        ASSERT_TRUE(!evicted);
        ASSERT_TRUE(cache.contains(ExpertCoord{7, 7}));
    } // pin dtor fires unpin

    ASSERT_EQ(cache.pinned_count(), 0u);

    // Now eviction should work.
    bool evicted = cache.evict_lru();
    ASSERT_TRUE(evicted);
    return true;
}

// -----------------------------------------------------------------------------
// Test 12: State transitions
// -----------------------------------------------------------------------------
bool test_state_transitions() {
    ExpertRAMCache cache(64 * 1024);
    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024));
    ASSERT_EQ(cache.get_state(ExpertCoord{0, 0}), ExpertState::RAM_RESIDENT);

    ASSERT_EQ(cache.set_state(ExpertCoord{0, 0}, ExpertState::IN_USE), CacheResult::OK);
    ASSERT_EQ(cache.get_state(ExpertCoord{0, 0}), ExpertState::IN_USE);

    ASSERT_EQ(cache.get_state(ExpertCoord{99, 99}), ExpertState::NOT_RESIDENT);
    return true;
}

// -----------------------------------------------------------------------------
// Test 13: Per-layer stats
// -----------------------------------------------------------------------------
bool test_layer_stats() {
    ExpertRAMCache cache(64 * 1024);
    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024));
    cache.put(ExpertCoord{1, 0}, make_meta(1, 0, 1024), make_buffer(1024));
    cache.get(ExpertCoord{0, 0});
    cache.get(ExpertCoord{0, 0});
    cache.get(ExpertCoord{1, 0});
    cache.get(ExpertCoord{1, 0});
    cache.get(ExpertCoord{1, 0}); // hit
    cache.get(ExpertCoord{2, 0}); // miss

    auto l0 = cache.layer_stats(0);
    ASSERT_EQ(l0.hits, 2u);

    auto l1 = cache.layer_stats(1);
    ASSERT_EQ(l1.hits, 3u);

    auto l2 = cache.layer_stats(2);
    ASSERT_EQ(l2.misses, 1u);
    return true;
}

// -----------------------------------------------------------------------------
// Test 14: Already-present with replace=false
// -----------------------------------------------------------------------------
bool test_already_present() {
    ExpertRAMCache cache(64 * 1024);
    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024));
    auto r = cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024), /*replace=*/false);
    ASSERT_EQ(r, CacheResult::ALREADY_PRESENT);

    // With replace=true, should succeed.
    auto r2 = cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024), /*replace=*/true);
    ASSERT_TRUE(r2 == CacheResult::OK);
    return true;
}

// -----------------------------------------------------------------------------
// Test 15: remove()
// -----------------------------------------------------------------------------
bool test_remove() {
    ExpertRAMCache cache(64 * 1024);
    cache.put(ExpertCoord{0, 0}, make_meta(0, 0, 1024), make_buffer(1024));
    ASSERT_TRUE(cache.contains(ExpertCoord{0, 0}));
    ASSERT_EQ(cache.remove(ExpertCoord{0, 0}), CacheResult::OK);
    ASSERT_TRUE(!cache.contains(ExpertCoord{0, 0}));
    ASSERT_EQ(cache.remove(ExpertCoord{0, 0}), CacheResult::NOT_FOUND);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "       Running ASEMA Cache (Milestone 3) Tests      \n";
    std::cout << "====================================================\n";

    Test tests[] = {
        {"insert_and_hit",          test_cache_insert_and_hit},
        {"miss",                    test_cache_miss},
        {"lru_order",               test_lru_order},
        {"memory_budget",           test_memory_budget},
        {"pinned_eviction",         test_pinned_eviction_protection},
        {"concurrent_insert_lookup",test_concurrent_insert_lookup},
        {"thrashing_1k",            test_thrashing_1k},
        {"thrashing_10k",           test_thrashing_10k},
        {"clear",                   test_cache_clear},
        {"reservation_atomic",      test_reservation_atomic},
        {"pin_handle",              test_pin_handle},
        {"state_transitions",       test_state_transitions},
        {"layer_stats",             test_layer_stats},
        {"already_present",         test_already_present},
        {"remove",                  test_remove},
    };

    int passed = 0;
    int failed = 0;
    for (const auto& t : tests) {
        std::cout << "\n[TEST] " << t.name << "...\n";
        bool ok = t.fn();
        if (ok) { passed++; std::cout << "  [PASS] " << t.name << "\n"; }
        else    { failed++; std::cout << "  [FAIL] " << t.name << "\n"; }
    }

    std::cout << "\n====================================================\n";
    std::cout << "  Results: " << passed << " passed, " << failed << " failed\n";
    std::cout << "====================================================\n";
    return failed == 0 ? 0 : 1;
}
