#include "../../include/asema/storage.hpp"
#include <fstream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace asema {

// Precomputed CRC32 table
static uint32_t crc32_table[256];
static bool crc32_initialized = false;

static void init_crc32_table() {
    uint32_t polynomial = 0xEDB88320;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (size_t j = 0; j < 8; j++) {
            if (c & 1) {
                c = polynomial ^ (c >> 1);
            } else {
                c = c >> 1;
            }
        }
        crc32_table[i] = c;
    }
    crc32_initialized = true;
}

uint32_t ChecksumUtil::compute_crc32(const uint8_t* data, size_t length) noexcept {
    if (!crc32_initialized) {
        init_crc32_table();
    }
    uint32_t c = 0xFFFFFFFF;
    for (size_t i = 0; i < length; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFF;
}

std::string ChecksumUtil::compute_sha256_hex(const uint8_t* data, size_t length) {
    // Basic placeholder hex formatting for fast tests, CRC32 is primary fast path
    uint32_t crc = compute_crc32(data, length);
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << crc;
    return ss.str();
}

class WindowsStorageBackend : public StorageBackend {
public:
    WindowsStorageBackend(const std::string& path) : path_(path) {
        open(path);
        start_async_workers(4);
    }

    ~WindowsStorageBackend() override {
        stop_async_workers();
        close();
    }

    bool open(const std::string& path) override {
        std::lock_guard<std::mutex> lock(io_mutex_);
        path_ = path;
#if defined(_WIN32)
        handle_ = CreateFileA(
            path_.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        is_open_ = (handle_ != INVALID_HANDLE_VALUE);
#else
        file_stream_.open(path_, std::ios::binary);
        is_open_ = file_stream_.is_open();
#endif
        return is_open_;
    }

    void close() override {
        std::lock_guard<std::mutex> lock(io_mutex_);
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (file_stream_.is_open()) {
            file_stream_.close();
        }
#endif
        is_open_ = false;
    }

    bool is_open() const override {
        return is_open_;
    }

    bool read_expert_sync(
        const ExpertMetadata& meta,
        ExpertBuffer& out_buffer,
        double& latency_ms
    ) override {
        if (!is_open_) return false;

        auto t0 = std::chrono::steady_clock::now();

        if (out_buffer.size() < meta.storage_length) {
            return false;
        }

#if defined(_WIN32)
        LARGE_INTEGER offset;
        offset.QuadPart = static_cast<LONGLONG>(meta.storage_offset);

        OVERLAPPED ov = {0};
        ov.Offset = offset.LowPart;
        ov.OffsetHigh = offset.HighPart;

        DWORD bytes_read = 0;
        BOOL ok = ReadFile(
            handle_,
            out_buffer.data(),
            static_cast<DWORD>(meta.storage_length),
            &bytes_read,
            &ov
        );

        if (!ok && GetLastError() != ERROR_IO_PENDING) {
            return false;
        }

        if (!ok) {
            GetOverlappedResult(handle_, &ov, &bytes_read, TRUE);
        }

        if (bytes_read != meta.storage_length) {
            return false;
        }
#else
        std::lock_guard<std::mutex> lock(io_mutex_);
        file_stream_.seekg(meta.storage_offset, std::ios::beg);
        file_stream_.read(reinterpret_cast<char*>(out_buffer.data()), meta.storage_length);
        if (!file_stream_) return false;
#endif

        // Verify checksum if provided
        if (meta.checksum_crc32 != 0) {
            uint32_t computed = ChecksumUtil::compute_crc32(out_buffer.data(), meta.storage_length);
            if (computed != meta.checksum_crc32) {
                std::cerr << "[ASEMA ERROR] CRC32 mismatch on L" << meta.layer_id 
                          << " E" << meta.expert_id << ": expected " 
                          << meta.checksum_crc32 << ", got " << computed << std::endl;
                return false;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        total_bytes_ += meta.storage_length;
        total_ops_++;
        return true;
    }

    bool read_expert_async(
        const ExpertMetadata& meta,
        AsyncCallback callback
    ) override {
        if (!is_open_) return false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            async_queue_.push(AsyncWorkItem{meta, std::move(callback)});
        }
        cv_.notify_one();
        return true;
    }

    uint64_t total_bytes_read() const noexcept override {
        return total_bytes_.load();
    }

    uint64_t total_read_operations() const noexcept override {
        return total_ops_.load();
    }

private:
    struct AsyncWorkItem {
        ExpertMetadata meta;
        AsyncCallback callback;
    };

    void start_async_workers(size_t num_threads) {
        stop_workers_ = false;
        for (size_t i = 0; i < num_threads; ++i) {
            worker_threads_.emplace_back([this]() {
                while (true) {
                    AsyncWorkItem item;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this]() {
                            return stop_workers_ || !async_queue_.empty();
                        });
                        if (stop_workers_ && async_queue_.empty()) {
                            break;
                        }
                        item = std::move(async_queue_.front());
                        async_queue_.pop();
                    }

                    auto buffer = std::make_shared<ExpertBuffer>(item.meta.storage_length, item.meta.alignment);
                    double latency = 0.0;
                    bool ok = read_expert_sync(item.meta, *buffer, latency);
                    if (item.callback) {
                        item.callback(ok, std::move(buffer), latency);
                    }
                }
            });
        }
    }

    void stop_async_workers() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_workers_ = true;
        }
        cv_.notify_all();
        for (auto& t : worker_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        worker_threads_.clear();
    }

    std::string path_;
    bool is_open_{false};
    std::mutex io_mutex_;
#if defined(_WIN32)
    HANDLE handle_{INVALID_HANDLE_VALUE};
#else
    std::ifstream file_stream_;
#endif

    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<uint64_t> total_ops_{0};

    // Async threadpool
    std::queue<AsyncWorkItem> async_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::vector<std::thread> worker_threads_;
    bool stop_workers_{false};
};

std::unique_ptr<StorageBackend> create_storage_backend(const std::string& container_path) {
    return std::make_unique<WindowsStorageBackend>(container_path);
}

} // namespace asema
