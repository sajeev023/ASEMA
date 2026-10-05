#pragma once

#include "asema/m8/m8_expert_kernel.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <optional>
#include <chrono>

namespace asema {
namespace m8 {

enum class CachePriority : uint8_t {
    BACKGROUND = 0,
    SPECULATIVE_PREFETCH = 1,
    NEXT_LAYER = 2,
    REQUIRED_NOW = 3
};

struct CacheNode {
    int layer_id{-1};
    int expert_id{-1};
    CachePriority priority{CachePriority::REQUIRED_NOW};
    std::shared_ptr<std::vector<uint8_t>> weights; // 17.69 MB
    std::shared_ptr<std::vector<uint8_t>> scales;  // 1.11 MB
    size_t total_bytes{0};

    std::atomic<int32_t> pin_count{0};
    std::atomic<int32_t> ref_count{0};

    CacheNode* prev{nullptr};
    CacheNode* next{nullptr};
};

struct CacheStats {
    uint64_t capacity_bytes{0};
    uint64_t used_bytes{0};
    size_t count{0};
    size_t pinned_count{0};
    uint64_t hits{0};
    uint64_t misses{0};
    uint64_t evictions{0};
    uint64_t rejected_admissions{0};
    double hit_rate{0.0};
};

class M8ExpertCacheEngine {
public:
    explicit M8ExpertCacheEngine(size_t capacity_bytes);
    ~M8ExpertCacheEngine();

    void set_capacity_bytes(size_t capacity_bytes);
    void set_capacity_mb(size_t capacity_mb);
    size_t capacity_bytes() const;
    size_t used_bytes() const;

    // O(1) lookup & touch (promotes to MRU if unpinned)
    std::shared_ptr<CacheNode> get(int layer_id, int expert_id);

    // O(1) put with admission policy & priority
    bool put(int layer_id,
             int expert_id,
             std::shared_ptr<std::vector<uint8_t>> weights,
             std::shared_ptr<std::vector<uint8_t>> scales,
             CachePriority priority = CachePriority::REQUIRED_NOW);

    // Pinning / Unpinning
    bool pin(int layer_id, int expert_id);
    bool unpin(int layer_id, int expert_id);

    // Eviction
    size_t evict_to_fit(size_t required_bytes);
    void clear();

    CacheStats get_stats() const;
    void reset_stats();

private:
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

    mutable std::mutex mutex_;
    size_t capacity_bytes_{1024ULL * 1024ULL * 1024ULL}; // default 1 GB
    size_t used_bytes_{0};

    std::unordered_map<CacheKey, CacheNode*, CacheKeyHash> map_;
    CacheNode* head_{nullptr}; // MRU
    CacheNode* tail_{nullptr}; // LRU

    mutable uint64_t hits_{0};
    mutable uint64_t misses_{0};
    uint64_t evictions_{0};
    uint64_t rejected_admissions_{0};

    void detach_node(CacheNode* node);
    void attach_mru(CacheNode* node);
    void remove_node(CacheNode* node);
};

} // namespace m8
} // namespace asema
