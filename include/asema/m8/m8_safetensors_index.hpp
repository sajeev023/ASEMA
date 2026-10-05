#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace asema {
namespace m8 {

struct SafetensorEntry {
    std::string name;
    std::string shard_path;
    uint64_t file_offset{0};
    uint64_t byte_length{0};
    std::string dtype;
    std::vector<int64_t> shape;
};

class M8SafetensorsIndex {
public:
    M8SafetensorsIndex();
    ~M8SafetensorsIndex();

    // Index a .safetensors file by parsing its JSON header
    bool index_shard(const std::string& shard_path);

    // Check if a tensor exists in the index
    bool has_tensor(const std::string& tensor_name) const;

    // Get tensor metadata (O(1) lookup)
    const SafetensorEntry* get_tensor_meta(const std::string& tensor_name) const;

    // Direct byte-range read from the safetensors file (with dense RAM cache)
    bool read_tensor_bytes(const std::string& tensor_name, uint8_t* dst, size_t dst_size) const;

    // Direct read with BF16 to FP32 dequantization
    bool read_tensor_bf16_to_fp32(const std::string& tensor_name, float* dst, size_t num_elements) const;

    // Zero-copy read-only view of a tensor inside a memory-mapped shard.
    // Pages are owned by the OS file cache, so Windows can reclaim them under memory pressure
    // (unlike a private cache). Returns nullptr if the tensor or mapping is unavailable.
    // The pointer stays valid until clear() / destruction.
    const uint8_t* map_tensor(const std::string& tensor_name, uint64_t* out_bytes = nullptr) const;

    // Remove all mapped shard pages from this process's working set (memory-pressure relief).
    // The data is not discarded: Windows keeps the pages in its standby cache, counts them as
    // available memory, and soft-faults them back on the next access if they were not reused.
    void trim_mapped_pages() const;

    // Number of indexed tensors
    size_t size() const;

    // Clear all index data and close handles
    void clear();

    // Dense tensor RAM cache metrics
    size_t dense_cache_bytes() const;
    static bool is_dense_tensor_name(const std::string& name);

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, SafetensorEntry> tensors_;
    std::unordered_map<std::string, uint64_t> indexed_shards_; // shard_path -> header_len
    mutable std::unordered_map<std::string, void*> handle_pool_; // shard_path -> HANDLE
    mutable std::unordered_map<std::string, std::vector<uint8_t>> dense_cache_;

    struct ShardMapping {
        void* file{nullptr};     // HANDLE
        void* mapping{nullptr};  // HANDLE
        const uint8_t* base{nullptr};
        uint64_t size{0};        // mapped file size in bytes (bounds-checks tensor ranges)
    };
    mutable std::unordered_map<std::string, ShardMapping> shard_maps_;

    void* get_or_open_handle(const std::string& shard_path) const;
};

} // namespace m8
} // namespace asema
