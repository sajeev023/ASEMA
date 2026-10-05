#pragma once

#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace asema {
namespace m8 {

struct ExpertPayload {
    int layer_id{-1};
    int expert_id{-1};
    std::vector<uint8_t> scales;  // 1,105,920 bytes
    std::vector<uint8_t> weights; // 17,694,720 bytes
    size_t total_bytes() const { return scales.size() + weights.size(); }
};

struct ByteLoaderTelemetry {
    uint64_t total_bytes_read{0};
    uint64_t total_requests{0};
    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    double total_io_time_ms{0.0};

    // M8.18 Prefetch & Double-Buffering Metrics
    uint64_t prefetches_issued{0};
    uint64_t useful_prefetches{0};
    uint64_t wasted_prefetches{0};
    uint64_t duplicate_requests{0};
    double io_overlap_time_ms{0.0};

    // Storage operations (each ReadFile call) and parallel-read behaviour
    uint64_t storage_read_ops{0};
    uint64_t parallel_batches{0};
    uint64_t evictions{0};
};

class M8ByteRangeLoader {
public:
    explicit M8ByteRangeLoader(std::shared_ptr<M8MultiVolumeManager> vol_mgr);
    ~M8ByteRangeLoader();

    // Load an expert directly into out_payload using byte-range reads
    bool load_expert_payload(int layer_id, int expert_id, ExpertPayload& out_payload);

    // Parallel multi-expert loader. Cache hits are returned by reference (zero copy); cache misses
    // are read concurrently with a hard bound on in-flight reads (see max_parallel_reads()).
    bool load_layer_experts_parallel(int layer_id,
                                     const std::vector<int>& expert_ids,
                                     std::vector<std::shared_ptr<const ExpertPayload>>& out_payloads);

    // Handle for an in-flight per-layer expert load. Cache hits are available immediately; misses are
    // being read concurrently (at most max_parallel_reads() reads in flight, enforced loader-wide).
    class PendingExpertLoad {
    public:
        size_t size() const { return slots_.size(); }
        // Blocks until expert i is available; returns nullptr if its read failed. Safe to repeat.
        std::shared_ptr<const ExpertPayload> get(size_t i) const;
    private:
        friend class M8ByteRangeLoader;
        struct Slot {
            std::shared_ptr<ExpertPayload> cached;                       // cache hit: already resident
            std::shared_future<std::shared_ptr<ExpertPayload>> pending;  // cache miss: being read
            bool is_miss{false};
            int expert_id{-1};
        };
        int layer_id_{-1};
        std::vector<Slot> slots_;
    };

    // Starts loading a layer's experts and returns immediately, so the caller can overlap other work
    // (CPU shared-expert math, GPU uploads of experts that have already arrived).
    // `resident_elsewhere[i]` (optional) marks experts that are already held by a faster tier (VRAM): they
    // are neither looked up in this cache nor read from storage, and count as hits. get(i) returns nullptr
    // for them, so the caller must not ask for their bytes.
    PendingExpertLoad begin_layer_experts(int layer_id, const std::vector<int>& expert_ids,
                                          const std::vector<char>* resident_elsewhere = nullptr);
    // Waits for any outstanding reads and (when `publish_to_cache`) publishes them into the bounded
    // cache. The GPU path passes false: VRAM is its cache, and a second host copy would only waste RAM.
    // False if a read failed.
    bool finish_layer_experts(PendingExpertLoad& pending, bool publish_to_cache = true);

    // Relative cost of re-reading this expert: 1.0 on an NVMe drive, about 6 on SATA, more on USB/other.
    // Derived from the Windows storage bus type of the drive holding the expert's shard (not benchmarked).
    // Used to keep the experts that are expensive to re-read resident longest.
    double expert_read_cost(int layer_id, int expert_id) const;

    // Asynchronous double buffering & prefetching
    void prefetch_expert(int layer_id, int expert_id);
    void prefetch_layer_experts(int layer_id, const std::vector<int>& expert_ids);
    void wait_for_prefetches();

    // Bounded RAM cache interface (true LRU; hits refresh recency)
    void set_cache_capacity_experts(size_t max_experts);
    void set_cache_capacity_mb(size_t mb);
    size_t cache_capacity_experts() const;
    const char* cache_policy_name() const { return policy_ == Policy::LRU ? "lru" : "blend"; }
    void clear_cache();

    size_t cache_size_experts() const;
    size_t cache_size_bytes() const;

    // Upper bound on concurrent expert reads (1..16). Lowered by the resource governor under pressure.
    void set_max_parallel_reads(int n);
    int max_parallel_reads() const;

    // Telemetry snapshot (thread-safe copy).
    ByteLoaderTelemetry telemetry() const;
    void reset_telemetry();

    std::shared_ptr<M8MultiVolumeManager> volume_manager() const { return vol_mgr_; }

private:
    std::shared_ptr<M8MultiVolumeManager> vol_mgr_;
    size_t max_cached_experts_{384};
    int max_parallel_reads_{6};
    ByteLoaderTelemetry telemetry_;

    mutable std::mutex cache_mutex_;
    mutable std::mutex handle_mutex_;
    mutable std::unordered_map<std::string, void*> handle_pool_;

    // Internal expert RAM cache
    struct CacheKey {
        int layer;
        int expert;
        bool operator==(const CacheKey& o) const { return layer == o.layer && expert == o.expert; }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey& k) const {
            return (static_cast<size_t>(k.layer) << 16) ^ static_cast<size_t>(k.expert);
        }
    };
    struct CacheEntry {
        std::shared_ptr<ExpertPayload> payload;
        std::list<CacheKey>::iterator lru_it; // position in lru_ (front = most recently used)
        bool prefetched{false};               // inserted by prefetch, not yet consumed
        uint64_t last_access{0};              // access_clock_ value at the last lookup/insert
    };

    // Eviction policy. A decode step touches ~240 distinct experts in a fixed order, so plain LRU
    // gets ZERO hits whenever capacity < ~240 (a cyclic scan longer than the cache). BLEND keeps
    // the experts that are both frequently and recently used:
    //   score = accesses / (1 + alpha * age_in_decode_steps)   (lowest score is evicted)
    // Replayed on a recorded trace it beats LRU at every size below ~250 experts and at 400+.
    // Select with ASEMA_CACHE_POLICY=lru|blend (default blend), ASEMA_CACHE_ALPHA (default 0.25).
    enum class Policy { LRU, BLEND };
    Policy policy_{Policy::BLEND};
    double blend_alpha_{0.25};
    uint64_t access_clock_{0};
    std::unordered_map<CacheKey, uint32_t, CacheKeyHash> freq_; // access counts (survive eviction)

    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> cache_;
    std::list<CacheKey> lru_;
    std::vector<std::future<void>> active_prefetches_;

    // Resolved byte ranges of an expert in its shard (avoids 6 filesystem stats per lookup)
    struct ExpertFileLoc {
        std::string scale_path;
        std::string weight_path;
        uint64_t scale_offset{0};
        uint64_t weight_offset{0};
    };
    mutable std::mutex loc_mutex_;
    mutable std::unordered_map<uint32_t, ExpertFileLoc> loc_cache_;
    bool locate_expert_cached(int layer_id, int expert_id, ExpertFileLoc& out) const;
    mutable std::unordered_map<int, double> drive_cost_;   // drive letter -> relative cost (loc_mutex_)

    // Loader-wide limit on concurrent storage reads (hard bound, adjusted by the resource governor).
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    int io_in_flight_{0};
    void acquire_io_slot();
    void release_io_slot();

    void* get_or_open_handle(const std::string& path);
    // Positioned read (OVERLAPPED offset): safe to call concurrently on the same handle.
    bool read_file_range(const std::string& path, uint64_t offset, size_t length, uint8_t* dst);
    bool internal_read_expert(int layer_id, int expert_id, ExpertPayload& payload);

    // Cache helpers (cache_mutex_ must be held)
    void touch_locked(CacheEntry& e, const CacheKey& key);
    void evict_to_capacity_locked(size_t capacity);
    void evict_one_locked();
    void note_access_locked(const CacheKey& key); // advances the clock and counts the access
    void insert_locked(const CacheKey& key, std::shared_ptr<ExpertPayload> payload, bool prefetched);

    ExpertPayload default_payload_;
    bool has_default_payload_{false};
    void ensure_default_expert();
};

} // namespace m8
} // namespace asema
