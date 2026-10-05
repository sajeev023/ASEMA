#pragma once

#include "expert.hpp"
#include "manifest.hpp"
#include "storage.hpp"

#include <cstdint>
#include <string>
#include <memory>
#include <future>
#include <functional>
#include <optional>
#include <vector>
#include <chrono>

namespace asema {

// Priority tier for expert loads
enum class LoadPriority : uint8_t {
    REQUIRED_NOW = 0, // Critical execution path
    PREFETCH     = 1, // Lookahead speculation
    BACKGROUND   = 2  // Low-priority / warmup
};

inline const char* to_string(LoadPriority p) {
    switch (p) {
        case LoadPriority::REQUIRED_NOW: return "REQUIRED_NOW";
        case LoadPriority::PREFETCH:     return "PREFETCH";
        case LoadPriority::BACKGROUND:   return "BACKGROUND";
        default:                         return "UNKNOWN";
    }
}

// Result of an asynchronous expert load
struct ExpertLoadResult {
    bool success{false};
    ExpertCoord coord{0, 0};
    std::shared_ptr<ExpertBuffer> buffer;
    size_t bytes_read{0};

    // Sub-millisecond timing breakdown
    double queue_latency_ms{0.0};
    double io_latency_ms{0.0};
    double crc_latency_ms{0.0};
    double total_latency_ms{0.0};

    std::string error_message;
    bool is_corrupt{false};
    bool is_cancelled{false};
};

using LoadHandle = uint64_t;
using LoadCallback = std::function<void(const ExpertLoadResult&)>;

// Forward declaration of internal engine state
class AsyncLoaderImpl;

class AsyncExpertLoader {
public:
    AsyncExpertLoader(
        const std::string& container_path,
        const ModelManifest& manifest,
        size_t num_workers = 4
    );

    ~AsyncExpertLoader();

    // Disable copy, enable move
    AsyncExpertLoader(const AsyncExpertLoader&) = delete;
    AsyncExpertLoader& operator=(const AsyncExpertLoader&) = delete;
    AsyncExpertLoader(AsyncExpertLoader&&) noexcept;
    AsyncExpertLoader& operator=(AsyncExpertLoader&&) noexcept;

    // Submit asynchronous load with optional completion callback
    LoadHandle submit(
        ExpertCoord coord,
        LoadPriority priority = LoadPriority::REQUIRED_NOW,
        LoadCallback callback = nullptr
    );

    // Submit asynchronous load and obtain a std::future
    std::future<ExpertLoadResult> submit_future(
        ExpertCoord coord,
        LoadPriority priority = LoadPriority::REQUIRED_NOW
    );

    // Cancel an in-flight or queued request
    bool cancel(LoadHandle handle);

    // Non-blocking poll for result
    std::optional<ExpertLoadResult> try_get_result(LoadHandle handle);

    // Synchronous block until specific request finishes
    ExpertLoadResult wait(LoadHandle handle);

    // Stop and clean up all worker threads and I/O resources
    void shutdown();

    // Telemetry and statistics counters
    uint64_t total_requests_submitted() const noexcept;
    uint64_t total_coalesced_requests() const noexcept;
    uint64_t total_io_dispatches() const noexcept;
    uint64_t total_bytes_loaded() const noexcept;
    uint64_t total_checksum_failures() const noexcept;
    uint64_t active_in_flight() const noexcept;

private:
    std::unique_ptr<AsyncLoaderImpl> impl_;
};

} // namespace asema
