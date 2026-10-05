#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <unordered_map>

namespace asema {
namespace m8 {

struct StorageRequestRecord {
    std::string volume;       // "D:" or "E:"
    std::string shard_name;   // e.g. "model-00003-of-00048.safetensors"
    int layer_id{-1};
    int expert_id{-1};
    uint64_t offset{0};
    size_t length{0};
    double latency_ms{0.0};
    bool is_prefetch{false};
    bool is_sequential{false};
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct VolumeStorageStats {
    std::string volume_name;
    uint64_t total_requests{0};
    uint64_t total_bytes{0};
    uint64_t sequential_requests{0};
    uint64_t random_requests{0};
    double total_io_time_ms{0.0};
    double throughput_mb_s{0.0};
    std::vector<double> latencies_ms;
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
    double avg_latency_ms{0.0};
};

class M8StorageForensics {
public:
    static M8StorageForensics& instance();

    void record_request(const std::string& volume,
                        const std::string& shard_name,
                        int layer_id,
                        int expert_id,
                        uint64_t offset,
                        size_t length,
                        double latency_ms,
                        bool is_prefetch);

    void record_cache_hit(int layer_id, int expert_id);
    void record_cache_miss(int layer_id, int expert_id);
    void record_prefetch_result(bool useful);
    void record_duplicate_request();
    void update_queue_depth(size_t depth);

    VolumeStorageStats get_volume_stats(const std::string& volume) const;
    std::unordered_map<std::string, VolumeStorageStats> get_all_volume_stats() const;

    size_t total_requests() const;
    size_t total_bytes() const;
    double cache_hit_rate() const;
    size_t max_queue_depth() const;

    std::string generate_markdown_report() const;
    void reset();

private:
    M8StorageForensics() = default;
    mutable std::recursive_mutex mutex_;

    std::vector<StorageRequestRecord> records_;
    uint64_t last_offset_d_{0};
    uint64_t last_offset_e_{0};

    uint64_t cache_hits_{0};
    uint64_t cache_misses_{0};
    uint64_t useful_prefetches_{0};
    uint64_t wasted_prefetches_{0};
    uint64_t duplicate_requests_{0};
    size_t max_queue_depth_{0};
};

} // namespace m8
} // namespace asema
