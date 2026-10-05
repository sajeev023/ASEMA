#pragma once

// ASEMA v0.1 — Milestone 3: ExpertRAMCache
// -----------------------------------------------------------------------------
// A bounded, thread-safe, LRU RAM cache for expert weight buffers.
//
// Design properties (per Agent #2 directive):
//   * O(1) lookup via unordered_map keyed by (layer_id, expert_id).
//   * O(1) LRU promotion via a doubly-linked list of entries.
//   * Hard memory budget enforced atomically — reservation precedes allocation.
//   * Pinning prevents eviction of in-use entries.
//   * State integration with asema::ExpertState.
//   * Per-layer hit/miss statistics.
//
// The cache does NOT depend on Agent #1's implementation details; it depends
// only on ExpertCoord, ExpertMetadata, ExpertBuffer, and ExpertState from
// include/asema/expert.hpp.
// -----------------------------------------------------------------------------

#include "expert.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace asema {

// -----------------------------------------------------------------------------
// Cache result codes
// -----------------------------------------------------------------------------
enum class CacheResult : uint8_t {
    OK              = 0, // Operation succeeded
    MISS            = 1, // get() did not find the entry
    EVICTED         = 2, // put() evicted one or more entries to make room
    EXHAUSTED       = 3, // put() could not free enough room (everything pinned)
    ALREADY_PRESENT = 4, // put() with replace=false found existing entry
    NOT_FOUND       = 5, // remove()/pin()/unpin() on unknown coord
    INVALID_ARG     = 6, // size == 0 or budget == 0
};

inline const char* to_string(CacheResult r) {
    switch (r) {
        case CacheResult::OK:              return "OK";
        case CacheResult::MISS:            return "MISS";
        case CacheResult::EVICTED:         return "EVICTED";
        case CacheResult::EXHAUSTED:       return "EXHAUSTED";
        case CacheResult::ALREADY_PRESENT: return "ALREADY_PRESENT";
        case CacheResult::NOT_FOUND:       return "NOT_FOUND";
        case CacheResult::INVALID_ARG:     return "INVALID_ARG";
        default:                           return "UNKNOWN";
    }
}

// -----------------------------------------------------------------------------
// Cache statistics snapshot
// -----------------------------------------------------------------------------
struct CacheStats {
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t insertions{0};
    uint64_t evictions{0};
    uint64_t resident_entries{0};
    uint64_t pinned_entries{0};
    uint64_t bytes_used{0};
    uint64_t bytes_reserved{0};
    uint64_t peak_bytes_used{0};
    uint64_t capacity_bytes{0};
    uint64_t failed_reservations{0};

    double hit_rate() const noexcept {
        const uint64_t total = hits + misses;
        return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    }
};

// -----------------------------------------------------------------------------
// Per-layer hit/miss counters
// -----------------------------------------------------------------------------
struct LayerCacheStats {
    uint64_t hits{0};
    uint64_t misses{0};
    double hit_rate() const noexcept {
        const uint64_t total = hits + misses;
        return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    }
};

// -----------------------------------------------------------------------------
// Cache entry (immutable from the outside, mutated under cache lock)
// -----------------------------------------------------------------------------
struct CacheEntry {
    ExpertCoord coord;
    ExpertMetadata metadata;
    std::shared_ptr<ExpertBuffer> buffer;
    ExpertState state{ExpertState::RAM_RESIDENT};
    std::atomic<uint64_t> creation_timestamp_ns{0};
    std::atomic<uint64_t> last_access_timestamp_ns{0};
    std::atomic<uint64_t> access_count{0};
    std::atomic<uint32_t> pin_count{0};   // execution-time pin (cache-level)
    std::atomic<uint64_t> generation{0};  // bumped on every put() of this coord
};

// -----------------------------------------------------------------------------
// RAII pin handle returned by pin(). The expert is auto-unpinned on dtor.
// -----------------------------------------------------------------------------
class CachePin {
public:
    CachePin() = default;
    CachePin(class ExpertRAMCache* cache, ExpertCoord coord, bool active);
    ~CachePin();
    CachePin(const CachePin&) = delete;
    CachePin& operator=(const CachePin&) = delete;
    CachePin(CachePin&& other) noexcept;
    CachePin& operator=(CachePin&& other) noexcept;

    bool active() const noexcept { return active_; }
    void release() noexcept;

    const ExpertCoord& coord() const noexcept { return coord_; }

private:
    class ExpertRAMCache* cache_{nullptr};
    ExpertCoord coord_{0, 0};
    bool active_{false};
};

// -----------------------------------------------------------------------------
// The cache
// -----------------------------------------------------------------------------
class ExpertRAMCache {
public:
    explicit ExpertRAMCache(uint64_t capacity_bytes);
    ~ExpertRAMCache();

    // Non-copyable, non-movable (holds internal state)
    ExpertRAMCache(const ExpertRAMCache&) = delete;
    ExpertRAMCache& operator=(const ExpertRAMCache&) = delete;

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    uint64_t capacity() const noexcept;
    void set_capacity(uint64_t new_capacity);

    // -------------------------------------------------------------------------
    // Core operations (thread-safe)
    // -------------------------------------------------------------------------

    // Look up an entry without changing LRU order or stats.
    std::shared_ptr<CacheEntry> peek(ExpertCoord coord) const;

    // Look up an entry. On hit, promotes to MRU and increments hits.
    // On miss, increments misses and returns nullptr.
    std::shared_ptr<CacheEntry> get(ExpertCoord coord);

    // Test membership without touching stats or LRU order.
    bool contains(ExpertCoord coord) const;

    // Insert or replace an entry. The cache takes shared ownership of `buffer`.
    // Returns:
    //   OK              if inserted and bytes fit (after any evictions)
    //   EVICTED         if inserted and other entries were evicted
    //   EXHAUSTED       if not enough room and everything else is pinned
    //   ALREADY_PRESENT if `replace=false` and entry already exists
    //   INVALID_ARG     if size==0
    //
    // The hard memory budget is enforced via reservation: before accepting the
    // buffer, the cache atomically increments bytes_used. If reservation fails,
    // the function returns EXHAUSTED and the buffer is dropped.
    CacheResult put(
        ExpertCoord coord,
        const ExpertMetadata& metadata,
        std::shared_ptr<ExpertBuffer> buffer,
        bool replace = true);

    // Remove an entry. Does not require it to be unpinned.
    CacheResult remove(ExpertCoord coord);

    // Clear all entries. Pinned entries are forcibly evicted (force=true) or
    // left untouched (force=false, in which case their byte count remains
    // reserved against the budget). Default: false.
    void clear(bool force = false);

    // -------------------------------------------------------------------------
    // Pinning
    // -------------------------------------------------------------------------
    // Pin an entry, preventing eviction. Returns OK if pinned, NOT_FOUND if
    // the entry is not resident. While pinned, eviction candidates skip this
    // entry.
    CacheResult pin(ExpertCoord coord);

    // Decrement pin count. If the entry has been removed while pinned, the
    // pin count is still decremented (and the entry forgotten).
    CacheResult unpin(ExpertCoord coord);

    // RAII wrapper.
    CachePin pin_handle(ExpertCoord coord);

    // -------------------------------------------------------------------------
    // Eviction
    // -------------------------------------------------------------------------
    // Evict a single LRU non-pinned entry. Returns true if one was evicted.
    bool evict_lru();

    // Evict until bytes_used <= target. Returns number of entries evicted.
    size_t evict_until(uint64_t target_bytes);

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    CacheResult set_state(ExpertCoord coord, ExpertState new_state);
    ExpertState get_state(ExpertCoord coord) const;

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    CacheStats stats() const;
    LayerCacheStats layer_stats(uint32_t layer_id) const;
    std::vector<LayerCacheStats> all_layer_stats() const;

    uint64_t hits() const noexcept;
    uint64_t misses() const noexcept;
    uint64_t insertions() const noexcept;
    uint64_t evictions() const noexcept;
    uint64_t bytes_used() const noexcept;
    uint64_t bytes_reserved() const noexcept;   // == bytes_used; alias for clarity
    uint64_t resident_count() const noexcept;
    uint64_t pinned_count() const noexcept;
    uint64_t peak_bytes_used() const noexcept;

    // Memory budget diagnostics
    uint64_t free_bytes() const noexcept;
    double utilization() const noexcept;        // bytes_used / capacity

private:
    // -------------------------------------------------------------------------
    // LRU list node (internal)
    // -------------------------------------------------------------------------
    struct ListNode {
        std::shared_ptr<CacheEntry> entry;
        ListNode* prev{nullptr};
        ListNode* next{nullptr};
        explicit ListNode(std::shared_ptr<CacheEntry> e) : entry(std::move(e)) {}
    };

    // -------------------------------------------------------------------------
    // Reservation primitives (atomic, lock-free fast path)
    // -------------------------------------------------------------------------
    bool try_reserve_bytes(uint64_t n) noexcept;
    void commit_reserved_bytes(uint64_t n) noexcept;
    void release_reserved_bytes(uint64_t n) noexcept;

    // LRU manipulation (called under write lock)
    void detach_node(ListNode* node) noexcept;
    void attach_mru(ListNode* node) noexcept;

    // Move existing node to MRU position (called on hit)
    void touch(ListNode* node) noexcept;

    // Find a non-pinned eviction candidate from LRU end. Returns nullptr if
    // every entry is pinned.
    ListNode* find_eviction_candidate() const;

    // Convert coord to key.
    static uint64_t key_of(ExpertCoord c) noexcept {
        return (static_cast<uint64_t>(c.layer_id) << 32) | static_cast<uint64_t>(c.expert_id);
    }

    static uint64_t now_ns() noexcept;

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    mutable std::shared_mutex mu_;             // RW lock: shared for reads, exclusive for writes
    uint64_t capacity_{0};
    std::atomic<uint64_t> bytes_used_{0};      // live reservation
    std::atomic<uint64_t> peak_bytes_used_{0};
    std::atomic<uint64_t> next_generation_{1};

    std::unordered_map<uint64_t, std::unique_ptr<ListNode>> index_;

    // Doubly-linked list: head = MRU, tail = LRU.
    ListNode* head_{nullptr};
    ListNode* tail_{nullptr};
    size_t resident_count_{0};

    // Statistics
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> insertions_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> failed_reservations_{0};

    // Per-layer statistics (indexed by layer_id). Lazily grown.
    mutable std::mutex layer_mu_;
    std::unordered_map<uint32_t, LayerCacheStats> layer_stats_;
};

} // namespace asema
