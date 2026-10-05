#pragma once

#include "expert.hpp"
#include "manifest.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <cstdint>

namespace asema {

// CRC32 calculation utility
class ChecksumUtil {
public:
    static uint32_t compute_crc32(const uint8_t* data, size_t length) noexcept;
    static std::string compute_sha256_hex(const uint8_t* data, size_t length);
};

// Abstract Storage Backend interface for synchronous and asynchronous I/O
class StorageBackend {
public:
    virtual ~StorageBackend() = default;

    virtual bool open(const std::string& path) = 0;
    virtual void close() = 0;
    virtual bool is_open() const = 0;

    // Synchronous read of an expert buffer
    virtual bool read_expert_sync(
        const ExpertMetadata& meta,
        ExpertBuffer& out_buffer,
        double& latency_ms
    ) = 0;

    // Asynchronous read callback signature
    using AsyncCallback = std::function<void(bool success, std::shared_ptr<ExpertBuffer> buffer, double latency_ms)>;

    // Queue an asynchronous read
    virtual bool read_expert_async(
        const ExpertMetadata& meta,
        AsyncCallback callback
    ) = 0;

    virtual uint64_t total_bytes_read() const noexcept = 0;
    virtual uint64_t total_read_operations() const noexcept = 0;
};

// Factory function to create Windows native I/O storage backend
std::unique_ptr<StorageBackend> create_storage_backend(const std::string& container_path);

} // namespace asema
