#include "asema/m8/m8_storage_scheduler.hpp"

#include <algorithm>

namespace asema {
namespace m8 {

M8StorageScheduler::M8StorageScheduler(size_t max_concurrency_per_volume)
    : max_concurrency_per_volume_(max_concurrency_per_volume) {
    in_flight_per_volume_["D:"] = 0;
    in_flight_per_volume_["E:"] = 0;
}

M8StorageScheduler::~M8StorageScheduler() {
    cancel_all_speculative();
}

uint64_t M8StorageScheduler::submit_request(const std::string& volume,
                                           const std::string& path,
                                           uint64_t offset,
                                           size_t length,
                                           uint8_t* destination,
                                           RequestPriority priority,
                                           std::function<void(bool success)> completion_cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    global_tick_++;
    telemetry_.total_scheduled++;

    // Duplicate coalescing check
    std::string key = path + ":" + std::to_string(offset) + ":" + std::to_string(length);
    auto it = active_read_keys_.find(key);
    if (it != active_read_keys_.end()) {
        telemetry_.requests_coalesced++;
        return it->second; // Return existing in-flight request ID
    }

    uint64_t req_id = next_request_id_++;
    active_read_keys_[key] = req_id;

    ScheduledIORequest req;
    req.request_id = req_id;
    req.volume = volume;
    req.path = path;
    req.offset = offset;
    req.length = length;
    req.destination = destination;
    req.priority = priority;
    req.age_counter = global_tick_;
    req.cancelled = false;
    req.completion_cb = std::move(completion_cb);

    queues_[volume].push(req);
    cv_.notify_all();

    return req_id;
}

bool M8StorageScheduler::cancel_request(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Clean coalescing key if present
    for (auto it = active_read_keys_.begin(); it != active_read_keys_.end(); ++it) {
        if (it->second == request_id) {
            active_read_keys_.erase(it);
            telemetry_.requests_cancelled++;
            return true;
        }
    }
    return false;
}

void M8StorageScheduler::cancel_all_speculative() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : queues_) {
        std::priority_queue<ScheduledIORequest> new_q;
        while (!kv.second.empty()) {
            auto top = kv.second.top();
            kv.second.pop();
            if (top.priority == RequestPriority::REQUIRED_NOW) {
                new_q.push(top);
            } else {
                telemetry_.requests_cancelled++;
            }
        }
        kv.second = std::move(new_q);
    }
}

std::shared_ptr<ScheduledIORequest> M8StorageScheduler::dispatch_next(const std::string& volume) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (in_flight_per_volume_[volume] >= max_concurrency_per_volume_) {
        return nullptr; // Volume saturated
    }

    auto& q = queues_[volume];
    while (!q.empty()) {
        ScheduledIORequest req = q.top();
        q.pop();

        if (req.cancelled) {
            continue;
        }

        in_flight_per_volume_[volume]++;
        if (req.priority == RequestPriority::REQUIRED_NOW) {
            telemetry_.required_now_dispatched++;
        } else {
            telemetry_.speculative_dispatched++;
        }

        return std::make_shared<ScheduledIORequest>(req);
    }

    return nullptr;
}

void M8StorageScheduler::notify_request_complete(const std::string& volume, uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (in_flight_per_volume_[volume] > 0) {
        in_flight_per_volume_[volume]--;
    }

    for (auto it = active_read_keys_.begin(); it != active_read_keys_.end(); ++it) {
        if (it->second == request_id) {
            active_read_keys_.erase(it);
            break;
        }
    }
    cv_.notify_all();
}

size_t M8StorageScheduler::pending_count(const std::string& volume) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = queues_.find(volume);
    return (it != queues_.end()) ? it->second.size() : 0;
}

SchedulerTelemetry M8StorageScheduler::get_telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_;
}

void M8StorageScheduler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    queues_.clear();
    active_read_keys_.clear();
    in_flight_per_volume_["D:"] = 0;
    in_flight_per_volume_["E:"] = 0;
    telemetry_ = SchedulerTelemetry{};
    next_request_id_ = 1;
    global_tick_ = 0;
}

} // namespace m8
} // namespace asema
