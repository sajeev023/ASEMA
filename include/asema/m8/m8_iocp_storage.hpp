#pragma once

#include <windows.h>
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace asema {
namespace m8 {

struct IOCPOperation {
    OVERLAPPED overlapped{};
    uint64_t request_id{0};
    uint8_t* buffer{nullptr};
    size_t length{0};
    HANDLE file_handle{INVALID_HANDLE_VALUE};
    std::function<void(bool success, size_t bytes)> callback;
    std::chrono::high_resolution_clock::time_point start_time;
};

struct IOCPBenchmarkMetrics {
    double throughput_mb_s{0.0};
    double avg_latency_ms{0.0};
    double cpu_user_ms{0.0};
    double cpu_kernel_ms{0.0};
    uint64_t total_reads{0};
    uint64_t total_bytes{0};
};

class M8IOCPStorageBackend {
public:
    explicit M8IOCPStorageBackend(size_t worker_threads = 4);
    ~M8IOCPStorageBackend();

    bool initialize();
    void shutdown();

    HANDLE open_overlapped_file(const std::string& path);

    bool submit_read_async(HANDLE file_handle,
                           uint64_t offset,
                           size_t length,
                           uint8_t* destination,
                           std::function<void(bool success, size_t bytes)> callback);

    IOCPBenchmarkMetrics get_metrics() const;
    void reset_metrics();

private:
    size_t worker_threads_{4};
    HANDLE iocp_handle_{INVALID_HANDLE_VALUE};
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};

    mutable std::mutex mutex_;
    std::unordered_map<std::string, HANDLE> file_handles_;

    std::atomic<uint64_t> total_reads_{0};
    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<uint64_t> total_io_time_us_{0};

    void worker_loop();
};

} // namespace m8
} // namespace asema
