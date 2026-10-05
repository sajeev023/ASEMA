#include "asema/m8/m8_iocp_storage.hpp"

#include <iostream>

namespace asema {
namespace m8 {

M8IOCPStorageBackend::M8IOCPStorageBackend(size_t worker_threads)
    : worker_threads_(worker_threads) {}

M8IOCPStorageBackend::~M8IOCPStorageBackend() {
    shutdown();
}

bool M8IOCPStorageBackend::initialize() {
    if (running_.load()) return true;

    iocp_handle_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, static_cast<DWORD>(worker_threads_));
    if (!iocp_handle_ || iocp_handle_ == INVALID_HANDLE_VALUE) {
        return false;
    }

    running_.store(true);
    for (size_t i = 0; i < worker_threads_; ++i) {
        workers_.emplace_back(&M8IOCPStorageBackend::worker_loop, this);
    }
    return true;
}

void M8IOCPStorageBackend::shutdown() {
    if (!running_.load()) return;
    running_.store(false);

    if (iocp_handle_ && iocp_handle_ != INVALID_HANDLE_VALUE) {
        for (size_t i = 0; i < workers_.size(); ++i) {
            PostQueuedCompletionStatus(iocp_handle_, 0, 0, nullptr);
        }
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();

        CloseHandle(iocp_handle_);
        iocp_handle_ = INVALID_HANDLE_VALUE;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : file_handles_) {
        if (kv.second != INVALID_HANDLE_VALUE) {
            CloseHandle(kv.second);
        }
    }
    file_handles_.clear();
}

HANDLE M8IOCPStorageBackend::open_overlapped_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = file_handles_.find(path);
    if (it != file_handles_.end()) {
        return it->second;
    }

    std::wstring wpath(path.begin(), path.end());
    HANDLE h = CreateFileW(wpath.c_str(),
                           GENERIC_READ,
                           FILE_SHARE_READ,
                           nullptr,
                           OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS,
                           nullptr);

    if (h != INVALID_HANDLE_VALUE && iocp_handle_ != INVALID_HANDLE_VALUE) {
        CreateIoCompletionPort(h, iocp_handle_, reinterpret_cast<ULONG_PTR>(h), 0);
        file_handles_[path] = h;
    }
    return h;
}

bool M8IOCPStorageBackend::submit_read_async(HANDLE file_handle,
                                             uint64_t offset,
                                             size_t length,
                                             uint8_t* destination,
                                             std::function<void(bool success, size_t bytes)> callback) {
    if (file_handle == INVALID_HANDLE_VALUE || !destination || length == 0) {
        return false;
    }

    auto* op = new IOCPOperation();
    ZeroMemory(&op->overlapped, sizeof(OVERLAPPED));
    op->overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    op->overlapped.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
    op->buffer = destination;
    op->length = length;
    op->file_handle = file_handle;
    op->callback = std::move(callback);
    op->start_time = std::chrono::high_resolution_clock::now();

    DWORD bytesRead = 0;
    BOOL ok = ReadFile(file_handle, destination, static_cast<DWORD>(length), &bytesRead, &op->overlapped);
    if (!ok) {
        DWORD err = GetLastError();
        if (err != ERROR_IO_PENDING) {
            delete op;
            return false;
        }
    }

    total_reads_++;
    total_bytes_ += length;
    return true;
}

void M8IOCPStorageBackend::worker_loop() {
    DWORD bytesTransferred = 0;
    ULONG_PTR completionKey = 0;
    LPOVERLAPPED pOverlapped = nullptr;

    while (running_.load()) {
        BOOL ok = GetQueuedCompletionStatus(iocp_handle_,
                                            &bytesTransferred,
                                            &completionKey,
                                            &pOverlapped,
                                            100);

        if (!running_.load()) break;

        if (pOverlapped) {
            auto* op = reinterpret_cast<IOCPOperation*>(pOverlapped);
            auto end_time = std::chrono::high_resolution_clock::now();
            uint64_t us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - op->start_time).count();
            total_io_time_us_ += us;

            if (op->callback) {
                op->callback(ok, bytesTransferred);
            }
            delete op;
        }
    }
}

IOCPBenchmarkMetrics M8IOCPStorageBackend::get_metrics() const {
    IOCPBenchmarkMetrics m;
    m.total_reads = total_reads_.load();
    m.total_bytes = total_bytes_.load();

    uint64_t reads = m.total_reads;
    if (reads > 0) {
        m.avg_latency_ms = (total_io_time_us_.load() / 1000.0) / reads;
        double total_sec = (total_io_time_us_.load() / 1000000.0);
        if (total_sec > 0.0) {
            m.throughput_mb_s = (m.total_bytes / (1024.0 * 1024.0)) / total_sec;
        }
    }
    return m;
}

void M8IOCPStorageBackend::reset_metrics() {
    total_reads_ = 0;
    total_bytes_ = 0;
    total_io_time_us_ = 0;
}

} // namespace m8
} // namespace asema
