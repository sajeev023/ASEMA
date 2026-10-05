#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

namespace asema {
namespace m8 {

enum class RequestPriority : uint8_t {
    BACKGROUND = 0,
    LIKELY_NEXT = 1,
    NEXT_LAYER = 2,
    REQUIRED_NOW = 3
};

inline const char* priority_name(RequestPriority p) {
    switch (p) {
        case RequestPriority::BACKGROUND:   return "BACKGROUND";
        case RequestPriority::LIKELY_NEXT:  return "LIKELY_NEXT";
        case RequestPriority::NEXT_LAYER:   return "NEXT_LAYER";
        case RequestPriority::REQUIRED_NOW: return "REQUIRED_NOW";
        default:                            return "UNKNOWN";
    }
}

struct ScheduledIORequest {
    uint64_t request_id{0};
    std::string volume;     // "D:" or "E:"
    std::string path;
    uint64_t offset{0};
    size_t length{0};
    uint8_t* destination{nullptr};
    RequestPriority priority{RequestPriority::REQUIRED_NOW};
    uint64_t age_counter{0}; // For starvation prevention
    bool cancelled{false};
    std::function<void(bool success)> completion_cb;

    // Strict priority comparison: higher priority outranks lower; if equal, older age outranks newer
    bool operator<(const ScheduledIORequest& o) const {
        if (priority != o.priority) {
            return static_cast<uint8_t>(priority) < static_cast<uint8_t>(o.priority);
        }
        return age_counter < o.age_counter;
    }
};

struct SchedulerTelemetry {
    uint64_t total_scheduled{0};
    uint64_t required_now_dispatched{0};
    uint64_t speculative_dispatched{0};
    uint64_t requests_coalesced{0};
    uint64_t requests_cancelled{0};
    uint64_t starvation_boosts{0};
    double avg_wait_time_ms{0.0};
};

class M8StorageScheduler {
public:
    explicit M8StorageScheduler(size_t max_concurrency_per_volume = 4);
    ~M8StorageScheduler();

    uint64_t submit_request(const std::string& volume,
                            const std::string& path,
                            uint64_t offset,
                            size_t length,
                            uint8_t* destination,
                            RequestPriority priority,
                            std::function<void(bool success)> completion_cb = nullptr);

    bool cancel_request(uint64_t request_id);
    void cancel_all_speculative();

    // Pull next best request for given volume honoring priority and aging
    std::shared_ptr<ScheduledIORequest> dispatch_next(const std::string& volume);

    void notify_request_complete(const std::string& volume, uint64_t request_id);

    size_t pending_count(const std::string& volume) const;
    SchedulerTelemetry get_telemetry() const;
    void reset();

private:
    size_t max_concurrency_per_volume_{4};
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    uint64_t next_request_id_{1};
    uint64_t global_tick_{0};

    // Priority queues per volume
    std::unordered_map<std::string, std::priority_queue<ScheduledIORequest>> queues_;
    std::unordered_map<std::string, size_t> in_flight_per_volume_;

    // Coalescing map: (path + ":" + offset) -> request_id
    std::unordered_map<std::string, uint64_t> active_read_keys_;

    SchedulerTelemetry telemetry_;
};

} // namespace m8
} // namespace asema
