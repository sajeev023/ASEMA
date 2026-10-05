#include "../../include/asema/async_loader.hpp"
#include "../../include/asema/storage.hpp"

#include <iostream>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cassert>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace asema {

namespace {

inline uint64_t coord_to_key(const ExpertCoord& c) noexcept {
    return (static_cast<uint64_t>(c.layer_id) << 32) | static_cast<uint64_t>(c.expert_id);
}

constexpr ULONG_PTR SHUTDOWN_COMPLETION_KEY = 0xDEADBEEF;

} // anonymous namespace

struct Listener {
    LoadHandle handle{0};
    LoadCallback callback;
    std::shared_ptr<std::promise<ExpertLoadResult>> promise;
};

// Represents an in-flight asynchronous operation
struct AsyncOpContext {
#if defined(_WIN32)
    OVERLAPPED ov{};
#endif
    LoadHandle primary_handle{0};
    ExpertCoord coord{0, 0};
    ExpertMetadata meta;
    LoadPriority priority{LoadPriority::REQUIRED_NOW};

    std::chrono::steady_clock::time_point submit_time;
    std::chrono::steady_clock::time_point io_start_time;
    std::chrono::steady_clock::time_point io_end_time;
    std::chrono::steady_clock::time_point crc_start_time;
    std::chrono::steady_clock::time_point crc_end_time;

    std::shared_ptr<ExpertBuffer> buffer;
    std::vector<Listener> listeners;
    std::atomic<bool> cancelled{false};
    bool in_flight{false};

    AsyncOpContext() {
#if defined(_WIN32)
        std::memset(&ov, 0, sizeof(ov));
#endif
    }
};

class AsyncLoaderImpl {
public:
    AsyncLoaderImpl(
        const std::string& container_path,
        const ModelManifest& manifest,
        size_t num_workers
    ) : path_(container_path), manifest_(manifest), num_workers_(num_workers) {
        open_container();
        start_threads();
    }

    ~AsyncLoaderImpl() {
        shutdown();
    }

    LoadHandle submit(
        ExpertCoord coord,
        LoadPriority priority,
        LoadCallback callback,
        std::shared_ptr<std::promise<ExpertLoadResult>> promise
    ) {
        if (shutdown_.load()) {
            ExpertLoadResult res;
            res.coord = coord;
            res.error_message = "Loader is shutting down";
            if (callback) callback(res);
            if (promise) promise->set_value(res);
            return 0;
        }

        const auto* meta = manifest_.get_expert_metadata(coord.layer_id, coord.expert_id);
        if (!meta) {
            ExpertLoadResult res;
            res.coord = coord;
            res.error_message = "Expert coordinate not found in manifest";
            if (callback) callback(res);
            if (promise) promise->set_value(res);
            return 0;
        }

        uint64_t handle = next_handle_.fetch_add(1, std::memory_order_relaxed);
        uint64_t key = coord_to_key(coord);

        std::lock_guard<std::mutex> lock(map_mutex_);

        // Request Coalescing: check if this expert is already being processed
        auto it = in_flight_map_.find(key);
        if (it != in_flight_map_.end()) {
            auto op = it->second;
            op->listeners.push_back(Listener{handle, std::move(callback), std::move(promise)});
            // Priority boost if newer request is higher priority
            if (static_cast<uint8_t>(priority) < static_cast<uint8_t>(op->priority)) {
                op->priority = priority;
            }
            handle_to_coord_[handle] = coord;
            coalesced_count_.fetch_add(1, std::memory_order_relaxed);
            requests_submitted_.fetch_add(1, std::memory_order_relaxed);
            return handle;
        }

        // New operation
        auto op = std::make_shared<AsyncOpContext>();
        op->primary_handle = handle;
        op->coord = coord;
        op->meta = *meta;
        op->priority = priority;
        op->submit_time = std::chrono::steady_clock::now();
        op->listeners.push_back(Listener{handle, std::move(callback), std::move(promise)});

        in_flight_map_[key] = op;
        handle_to_coord_[handle] = coord;
        requests_submitted_.fetch_add(1, std::memory_order_relaxed);

        // Enqueue into dispatch queue
        {
            std::lock_guard<std::mutex> qlock(queue_mutex_);
            dispatch_queue_.push(op);
        }
        dispatch_cv_.notify_one();

        return handle;
    }

    bool cancel(LoadHandle handle) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it_coord = handle_to_coord_.find(handle);
        if (it_coord == handle_to_coord_.end()) {
            return false;
        }

        uint64_t key = coord_to_key(it_coord->second);
        auto it_op = in_flight_map_.find(key);
        if (it_op != in_flight_map_.end()) {
            auto op = it_op->second;
            // Mark listener cancelled
            for (auto& l : op->listeners) {
                if (l.handle == handle) {
                    ExpertLoadResult res;
                    res.coord = op->coord;
                    res.is_cancelled = true;
                    res.error_message = "Request cancelled by caller";
                    if (l.callback) l.callback(res);
                    if (l.promise) l.promise->set_value(res);
                    l.callback = nullptr;
                    l.promise = nullptr;
                    break;
                }
            }
            // If all listeners are cancelled, mark operation cancelled
            bool all_cancelled = true;
            for (const auto& l : op->listeners) {
                if (l.callback || l.promise) {
                    all_cancelled = false;
                    break;
                }
            }
            if (all_cancelled) {
                op->cancelled.store(true);
            }
            return true;
        }
        return false;
    }

    std::optional<ExpertLoadResult> try_get_result(LoadHandle handle) {
        std::lock_guard<std::mutex> lock(results_mutex_);
        auto it = completed_results_.find(handle);
        if (it != completed_results_.end()) {
            ExpertLoadResult res = it->second;
            completed_results_.erase(it);
            return res;
        }
        return std::nullopt;
    }

    ExpertLoadResult wait(LoadHandle handle) {
        std::unique_lock<std::mutex> lock(results_mutex_);
        results_cv_.wait(lock, [this, handle]() {
            return shutdown_.load() || completed_results_.find(handle) != completed_results_.end();
        });

        auto it = completed_results_.find(handle);
        if (it != completed_results_.end()) {
            ExpertLoadResult res = it->second;
            completed_results_.erase(it);
            return res;
        }

        ExpertLoadResult res;
        res.error_message = "Request terminated without completion (shutdown)";
        return res;
    }

    void shutdown() {
        bool expected = false;
        if (!shutdown_.compare_exchange_strong(expected, true)) {
            return;
        }

        // Wake dispatch thread
        dispatch_cv_.notify_all();
        if (dispatcher_thread_.joinable()) {
            dispatcher_thread_.join();
        }

        // Post shutdown sentinels to IOCP workers
#if defined(_WIN32)
        if (iocp_handle_ != NULL) {
            for (size_t i = 0; i < num_workers_; ++i) {
                PostQueuedCompletionStatus(iocp_handle_, 0, SHUTDOWN_COMPLETION_KEY, NULL);
            }
        }
#endif

        for (auto& w : worker_threads_) {
            if (w.joinable()) {
                w.join();
            }
        }
        worker_threads_.clear();

        // Close handles
#if defined(_WIN32)
        if (iocp_handle_ != NULL) {
            CloseHandle(iocp_handle_);
            iocp_handle_ = NULL;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#endif
        results_cv_.notify_all();
    }

    uint64_t total_requests_submitted() const noexcept { return requests_submitted_.load(); }
    uint64_t total_coalesced_requests() const noexcept { return coalesced_count_.load(); }
    uint64_t total_io_dispatches() const noexcept { return io_dispatches_.load(); }
    uint64_t total_bytes_loaded() const noexcept { return bytes_loaded_.load(); }
    uint64_t total_checksum_failures() const noexcept { return checksum_failures_.load(); }
    uint64_t active_in_flight() const noexcept { return active_ops_.load(); }

private:
    struct PriorityComparator {
        bool operator()(const std::shared_ptr<AsyncOpContext>& a, const std::shared_ptr<AsyncOpContext>& b) const {
            if (a->priority != b->priority) {
                return static_cast<uint8_t>(a->priority) > static_cast<uint8_t>(b->priority);
            }
            return a->submit_time > b->submit_time;
        }
    };

    void open_container() {
#if defined(_WIN32)
        file_handle_ = CreateFileA(
            path_.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
            NULL
        );

        if (file_handle_ == INVALID_HANDLE_VALUE) {
            std::cerr << "[ASEMA ERROR] Failed to open container: " << path_ << std::endl;
            return;
        }

        iocp_handle_ = CreateIoCompletionPort(file_handle_, NULL, 0, static_cast<DWORD>(num_workers_));
        if (!iocp_handle_) {
            std::cerr << "[ASEMA ERROR] Failed to create I/O completion port." << std::endl;
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
#endif
    }

    void start_threads() {
        dispatcher_thread_ = std::thread(&AsyncLoaderImpl::dispatcher_loop, this);
        for (size_t i = 0; i < num_workers_; ++i) {
            worker_threads_.emplace_back(&AsyncLoaderImpl::iocp_worker_loop, this);
        }
    }

    void dispatcher_loop() {
        while (!shutdown_.load()) {
            std::shared_ptr<AsyncOpContext> op;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                dispatch_cv_.wait(lock, [this]() {
                    return shutdown_.load() || !dispatch_queue_.empty();
                });

                if (shutdown_.load() && dispatch_queue_.empty()) {
                    break;
                }

                op = dispatch_queue_.top();
                dispatch_queue_.pop();
            }

            if (!op) continue;

            if (op->cancelled.load()) {
                finalize_operation(op, false, "Cancelled before I/O dispatch", false);
                continue;
            }

#if defined(_WIN32)
            if (file_handle_ == INVALID_HANDLE_VALUE) {
                finalize_operation(op, false, "Container file handle is invalid", false);
                continue;
            }

            op->buffer = std::make_shared<ExpertBuffer>(op->meta.storage_length, op->meta.alignment);
            LARGE_INTEGER li;
            li.QuadPart = static_cast<LONGLONG>(op->meta.storage_offset);
            op->ov.Offset = li.LowPart;
            op->ov.OffsetHigh = li.HighPart;
            op->io_start_time = std::chrono::steady_clock::now();
            op->in_flight = true;

            // Pin context in active map before issuing read
            {
                std::lock_guard<std::mutex> lock(active_mutex_);
                active_context_map_[&op->ov] = op;
            }
            active_ops_.fetch_add(1, std::memory_order_relaxed);
            io_dispatches_.fetch_add(1, std::memory_order_relaxed);

            DWORD bytes_read = 0;
            BOOL ok = ReadFile(
                file_handle_,
                op->buffer->data(),
                static_cast<DWORD>(op->meta.storage_length),
                &bytes_read,
                &op->ov
            );

            if (!ok && GetLastError() != ERROR_IO_PENDING) {
                // Immediate failure
                {
                    std::lock_guard<std::mutex> lock(active_mutex_);
                    active_context_map_.erase(&op->ov);
                }
                active_ops_.fetch_sub(1, std::memory_order_relaxed);
                finalize_operation(op, false, "ReadFile Win32 error code: " + std::to_string(GetLastError()), false);
            }
#else
            // Fallback for non-Windows platforms
            finalize_operation(op, false, "IOCP is only supported on Windows", false);
#endif
        }
    }

    void iocp_worker_loop() {
#if defined(_WIN32)
        DWORD bytes_transferred = 0;
        ULONG_PTR completion_key = 0;
        LPOVERLAPPED p_ov = nullptr;

        while (true) {
            BOOL ok = GetQueuedCompletionStatus(
                iocp_handle_,
                &bytes_transferred,
                &completion_key,
                &p_ov,
                INFINITE
            );

            if (completion_key == SHUTDOWN_COMPLETION_KEY) {
                break;
            }

            if (!p_ov) {
                continue;
            }

            std::shared_ptr<AsyncOpContext> op;
            {
                std::lock_guard<std::mutex> lock(active_mutex_);
                auto it = active_context_map_.find(p_ov);
                if (it != active_context_map_.end()) {
                    op = it->second;
                    active_context_map_.erase(it);
                }
            }
            active_ops_.fetch_sub(1, std::memory_order_relaxed);

            if (!op) continue;

            op->io_end_time = std::chrono::steady_clock::now();

            if (!ok || bytes_transferred != op->meta.storage_length) {
                finalize_operation(op, false, "I/O completion failure or incomplete transfer", false);
                continue;
            }

            if (op->cancelled.load()) {
                finalize_operation(op, false, "Operation cancelled during I/O", false);
                continue;
            }

            // CRC32 Validation Phase
            op->crc_start_time = std::chrono::steady_clock::now();
            bool valid = true;
            if (op->meta.checksum_crc32 != 0) {
                uint32_t computed = ChecksumUtil::compute_crc32(op->buffer->data(), op->meta.storage_length);
                if (computed != op->meta.checksum_crc32) {
                    checksum_failures_.fetch_add(1, std::memory_order_relaxed);
                    valid = false;
                }
            }
            op->crc_end_time = std::chrono::steady_clock::now();

            if (!valid) {
                finalize_operation(op, false, "CRC32 checksum mismatch (data corruption detected)", true);
            } else {
                bytes_loaded_.fetch_add(op->meta.storage_length, std::memory_order_relaxed);
                finalize_operation(op, true, "", false);
            }
        }
#endif
    }

    void finalize_operation(
        std::shared_ptr<AsyncOpContext> op,
        bool success,
        const std::string& err,
        bool is_corrupt
    ) {
        auto now = std::chrono::steady_clock::now();

        ExpertLoadResult res;
        res.success = success;
        res.coord = op->coord;
        res.bytes_read = success ? op->meta.storage_length : 0;
        res.error_message = err;
        res.is_corrupt = is_corrupt;
        res.is_cancelled = op->cancelled.load();

        if (success) {
            res.buffer = op->buffer;
        }

        // Sub-millisecond timing calculations
        if (op->io_start_time.time_since_epoch().count() > 0) {
            res.queue_latency_ms = std::chrono::duration<double, std::milli>(op->io_start_time - op->submit_time).count();
        }
        if (op->io_end_time.time_since_epoch().count() > 0) {
            res.io_latency_ms = std::chrono::duration<double, std::milli>(op->io_end_time - op->io_start_time).count();
        }
        if (op->crc_end_time.time_since_epoch().count() > 0 && op->crc_start_time.time_since_epoch().count() > 0) {
            res.crc_latency_ms = std::chrono::duration<double, std::milli>(op->crc_end_time - op->crc_start_time).count();
        }
        res.total_latency_ms = std::chrono::duration<double, std::milli>(now - op->submit_time).count();

        // Store results for synchronous callers
        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            for (const auto& l : op->listeners) {
                completed_results_[l.handle] = res;
            }
        }
        results_cv_.notify_all();

        // Dispatch callbacks and promises
        for (const auto& l : op->listeners) {
            if (l.callback) {
                l.callback(res);
            }
            if (l.promise) {
                l.promise->set_value(res);
            }
        }

        // Remove from in-flight and handle registries
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            in_flight_map_.erase(coord_to_key(op->coord));
            for (const auto& l : op->listeners) {
                handle_to_coord_.erase(l.handle);
            }
        }
    }

    std::string path_;
    ModelManifest manifest_;
    size_t num_workers_{4};

#if defined(_WIN32)
    HANDLE file_handle_{INVALID_HANDLE_VALUE};
    HANDLE iocp_handle_{NULL};
#endif

    std::atomic<bool> shutdown_{false};
    std::atomic<uint64_t> next_handle_{1};

    // Metrics counters
    std::atomic<uint64_t> requests_submitted_{0};
    std::atomic<uint64_t> coalesced_count_{0};
    std::atomic<uint64_t> io_dispatches_{0};
    std::atomic<uint64_t> bytes_loaded_{0};
    std::atomic<uint64_t> checksum_failures_{0};
    std::atomic<uint64_t> active_ops_{0};

    // Priority dispatch queue
    std::priority_queue<
        std::shared_ptr<AsyncOpContext>,
        std::vector<std::shared_ptr<AsyncOpContext>>,
        PriorityComparator
    > dispatch_queue_;
    std::mutex queue_mutex_;
    std::condition_variable dispatch_cv_;

    // Registry of in-flight operations (for request coalescing)
    std::unordered_map<uint64_t, std::shared_ptr<AsyncOpContext>> in_flight_map_;
    std::unordered_map<LoadHandle, ExpertCoord> handle_to_coord_;
    std::mutex map_mutex_;

    // Context tracking for active IOCP transfers
    std::unordered_map<void*, std::shared_ptr<AsyncOpContext>> active_context_map_;
    std::mutex active_mutex_;

    // Completed results map for wait/try_get_result
    std::unordered_map<LoadHandle, ExpertLoadResult> completed_results_;
    std::mutex results_mutex_;
    std::condition_variable results_cv_;

    std::thread dispatcher_thread_;
    std::vector<std::thread> worker_threads_;
};

// Public AsyncExpertLoader forwarding methods
AsyncExpertLoader::AsyncExpertLoader(
    const std::string& container_path,
    const ModelManifest& manifest,
    size_t num_workers
) : impl_(std::make_unique<AsyncLoaderImpl>(container_path, manifest, num_workers)) {}

AsyncExpertLoader::~AsyncExpertLoader() = default;
AsyncExpertLoader::AsyncExpertLoader(AsyncExpertLoader&&) noexcept = default;
AsyncExpertLoader& AsyncExpertLoader::operator=(AsyncExpertLoader&&) noexcept = default;

LoadHandle AsyncExpertLoader::submit(
    ExpertCoord coord,
    LoadPriority priority,
    LoadCallback callback
) {
    return impl_->submit(coord, priority, std::move(callback), nullptr);
}

std::future<ExpertLoadResult> AsyncExpertLoader::submit_future(
    ExpertCoord coord,
    LoadPriority priority
) {
    auto promise = std::make_shared<std::promise<ExpertLoadResult>>();
    std::future<ExpertLoadResult> fut = promise->get_future();
    impl_->submit(coord, priority, nullptr, promise);
    return fut;
}

bool AsyncExpertLoader::cancel(LoadHandle handle) {
    return impl_->cancel(handle);
}

std::optional<ExpertLoadResult> AsyncExpertLoader::try_get_result(LoadHandle handle) {
    return impl_->try_get_result(handle);
}

ExpertLoadResult AsyncExpertLoader::wait(LoadHandle handle) {
    return impl_->wait(handle);
}

void AsyncExpertLoader::shutdown() {
    impl_->shutdown();
}

uint64_t AsyncExpertLoader::total_requests_submitted() const noexcept {
    return impl_->total_requests_submitted();
}

uint64_t AsyncExpertLoader::total_coalesced_requests() const noexcept {
    return impl_->total_coalesced_requests();
}

uint64_t AsyncExpertLoader::total_io_dispatches() const noexcept {
    return impl_->total_io_dispatches();
}

uint64_t AsyncExpertLoader::total_bytes_loaded() const noexcept {
    return impl_->total_bytes_loaded();
}

uint64_t AsyncExpertLoader::total_checksum_failures() const noexcept {
    return impl_->total_checksum_failures();
}

uint64_t AsyncExpertLoader::active_in_flight() const noexcept {
    return impl_->active_in_flight();
}

} // namespace asema
