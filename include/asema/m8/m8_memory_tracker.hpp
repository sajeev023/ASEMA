#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <chrono>

namespace asema {
namespace m8 {

enum class MemoryCategory {
    MODEL_METADATA,
    TOKENIZER,
    EMBEDDINGS,
    ENGRAM_DATA,
    EXPERT_BUFFERS,
    MLA_BUFFERS,
    KV_CACHE,
    GPU_BUFFERS,
    STAGING_BUFFERS,
    ALLOCATOR_OVERHEAD,
    OS_FILE_CACHE,
    RUNTIME_BUFFERS,
    COUNT
};

inline const char* memory_category_name(MemoryCategory cat) {
    switch (cat) {
        case MemoryCategory::MODEL_METADATA:     return "Model Metadata";
        case MemoryCategory::TOKENIZER:          return "Tokenizer";
        case MemoryCategory::EMBEDDINGS:         return "Embeddings";
        case MemoryCategory::ENGRAM_DATA:        return "Engram Data";
        case MemoryCategory::EXPERT_BUFFERS:     return "Expert Buffers";
        case MemoryCategory::MLA_BUFFERS:        return "MLA Buffers";
        case MemoryCategory::KV_CACHE:           return "KV Cache";
        case MemoryCategory::GPU_BUFFERS:        return "GPU VRAM Buffers";
        case MemoryCategory::STAGING_BUFFERS:    return "Staging Buffers";
        case MemoryCategory::ALLOCATOR_OVERHEAD: return "Allocator Overhead";
        case MemoryCategory::OS_FILE_CACHE:      return "OS File Cache";
        case MemoryCategory::RUNTIME_BUFFERS:    return "Runtime Buffers";
        default:                                 return "Unknown";
    }
}

enum class AllocationLocation {
    CPU_RAM,
    GPU_VRAM,
    HOST_PINNED_STAGING
};

inline const char* allocation_location_name(AllocationLocation loc) {
    switch (loc) {
        case AllocationLocation::CPU_RAM:              return "CPU RAM";
        case AllocationLocation::GPU_VRAM:             return "GPU VRAM";
        case AllocationLocation::HOST_PINNED_STAGING:  return "Host Pinned / Staging";
        default:                                       return "Unknown";
    }
}

enum class AllocationLifetime {
    STATIC_PROCESS,     // Persists for application lifetime
    LAYER_EPHEMERAL,    // Reused / allocated per layer
    TOKEN_EPHEMERAL,    // Persists across single autoregressive step
    DYNAMIC_CACHE       // Managed by LRU cache eviction
};

inline const char* allocation_lifetime_name(AllocationLifetime life) {
    switch (life) {
        case AllocationLifetime::STATIC_PROCESS:   return "Static Process";
        case AllocationLifetime::LAYER_EPHEMERAL:  return "Layer Ephemeral";
        case AllocationLifetime::TOKEN_EPHEMERAL:  return "Token Ephemeral";
        case AllocationLifetime::DYNAMIC_CACHE:    return "Dynamic Cache";
        default:                                   return "Unknown";
    }
}

struct MemoryRecord {
    std::string tag;
    MemoryCategory category;
    AllocationLocation location;
    AllocationLifetime lifetime;
    std::string owner;
    std::string reason;
    size_t size_bytes{0};
    uint64_t allocation_id{0};
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct OSProcessMemoryInfo {
    size_t working_set_bytes{0};
    size_t peak_working_set_bytes{0};
    size_t private_bytes{0};
    size_t peak_private_bytes{0};
    size_t system_cache_bytes{0};
    size_t available_physical_bytes{0};
    size_t total_physical_bytes{0};
};

class M8MemoryTracker {
public:
    static M8MemoryTracker& instance();

    uint64_t record_allocation(const std::string& tag,
                               MemoryCategory category,
                               AllocationLocation location,
                               AllocationLifetime lifetime,
                               const std::string& owner,
                               const std::string& reason,
                               size_t size_bytes);

    void record_deallocation(uint64_t allocation_id);

    size_t get_category_bytes(MemoryCategory category) const;
    size_t get_location_bytes(AllocationLocation location) const;
    size_t get_total_tracked_bytes() const;

    OSProcessMemoryInfo query_os_memory() const;

    std::vector<MemoryRecord> get_active_allocations() const;
    std::unordered_map<MemoryCategory, size_t> get_category_breakdown() const;

    std::string generate_markdown_report() const;
    void reset();

private:
    M8MemoryTracker() = default;
    mutable std::mutex mutex_;
    uint64_t next_id_{1};
    std::unordered_map<uint64_t, MemoryRecord> active_allocations_;
    std::unordered_map<MemoryCategory, size_t> category_totals_;
    std::unordered_map<AllocationLocation, size_t> location_totals_;
};

} // namespace m8
} // namespace asema
