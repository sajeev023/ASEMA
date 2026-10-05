#include "asema/m8/m8_safetensors_index.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <windows.h>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

bool M8SafetensorsIndex::index_shard(const std::string& shard_path) {
    if (shard_path.empty() || !fs::exists(shard_path)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (indexed_shards_.find(shard_path) != indexed_shards_.end()) {
        return true; // Already indexed
    }

    std::wstring wpath(shard_path.begin(), shard_path.end());
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }

    uint64_t header_len = 0;
    DWORD bytesRead = 0;
    BOOL ok = ReadFile(hFile, &header_len, sizeof(header_len), &bytesRead, nullptr);
    if (!ok || bytesRead != sizeof(header_len) || header_len == 0 || header_len > 100000000ULL) {
        CloseHandle(hFile);
        return false;
    }

    std::vector<char> header_buf(header_len);
    ok = ReadFile(hFile, header_buf.data(), static_cast<DWORD>(header_len), &bytesRead, nullptr);
    CloseHandle(hFile);

    if (!ok || bytesRead != header_len) {
        return false;
    }

    try {
        nlohmann::json root = nlohmann::json::parse(header_buf.begin(), header_buf.end());
        const uint64_t base_data_offset = 8 + header_len;

        for (auto it = root.begin(); it != root.end(); ++it) {
            if (it.key() == "__metadata__") continue;

            const auto& val = it.value();
            if (!val.contains("data_offsets") || !val["data_offsets"].is_array() || val["data_offsets"].size() != 2) {
                continue;
            }

            uint64_t start_off = val["data_offsets"][0].get<uint64_t>();
            uint64_t end_off = val["data_offsets"][1].get<uint64_t>();

            SafetensorEntry meta;
            meta.name = it.key();
            meta.shard_path = shard_path;
            meta.file_offset = base_data_offset + start_off;
            meta.byte_length = end_off - start_off;

            if (val.contains("dtype") && val["dtype"].is_string()) {
                meta.dtype = val["dtype"].get<std::string>();
            }
            if (val.contains("shape") && val["shape"].is_array()) {
                for (const auto& s : val["shape"]) {
                    meta.shape.push_back(s.get<int64_t>());
                }
            }

            tensors_[meta.name] = std::move(meta);
        }

        indexed_shards_[shard_path] = header_len;
        return true;
    } catch (...) {
        return false;
    }
}

bool M8SafetensorsIndex::has_tensor(const std::string& tensor_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (tensors_.find(tensor_name) != tensors_.end());
}

const SafetensorEntry* M8SafetensorsIndex::get_tensor_meta(const std::string& tensor_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tensors_.find(tensor_name);
    if (it != tensors_.end()) {
        return &(it->second);
    }
    return nullptr;
}

M8SafetensorsIndex::M8SafetensorsIndex() = default;

M8SafetensorsIndex::~M8SafetensorsIndex() {
    clear();
}

void* M8SafetensorsIndex::get_or_open_handle(const std::string& shard_path) const {
    auto it = handle_pool_.find(shard_path);
    if (it != handle_pool_.end() && it->second != INVALID_HANDLE_VALUE && it->second != nullptr) {
        return it->second;
    }

    std::wstring wpath(shard_path.begin(), shard_path.end());
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (hFile != INVALID_HANDLE_VALUE) {
        handle_pool_[shard_path] = hFile;
    }
    return hFile;
}

bool M8SafetensorsIndex::is_dense_tensor_name(const std::string& name) {
    // Sparse routed experts are named "layers.<L>.ffn.experts.<E>."
    if (name.find(".ffn.experts.") != std::string::npos) {
        return false;
    }
    return true;
}

size_t M8SafetensorsIndex::dense_cache_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& pair : dense_cache_) {
        total += pair.second.size();
    }
    return total;
}

bool M8SafetensorsIndex::read_tensor_bytes(const std::string& tensor_name, uint8_t* dst, size_t dst_size) const {
    if (!dst || dst_size == 0) return false;

    // 1. Check dense RAM cache first (instant 0 ms return)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it_cache = dense_cache_.find(tensor_name);
        if (it_cache != dense_cache_.end()) {
            if (dst_size >= it_cache->second.size()) {
                std::memcpy(dst, it_cache->second.data(), it_cache->second.size());
                return true;
            }
        }
    }

    // 2. Locate tensor metadata
    SafetensorEntry meta;
    HANDLE hFile = INVALID_HANDLE_VALUE;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tensors_.find(tensor_name);
        if (it == tensors_.end()) return false;
        meta = it->second;

        hFile = static_cast<HANDLE>(get_or_open_handle(meta.shard_path));
    }

    if (dst_size < meta.byte_length) return false;
    if (hFile == INVALID_HANDLE_VALUE || hFile == nullptr) {
        return false;
    }

    // 3. Direct positioned read using persistent cached HANDLE
    LARGE_INTEGER li;
    li.QuadPart = static_cast<LONGLONG>(meta.file_offset);
    if (!SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN)) {
        return false;
    }

    DWORD bytesRead = 0;
    DWORD toRead = static_cast<DWORD>(meta.byte_length);
    if (!ReadFile(hFile, dst, toRead, &bytesRead, nullptr)) {
        return false;
    }
    if (bytesRead != toRead) {
        return false;
    }

    // 4. If dense tensor, store in dense RAM cache for subsequent zero-I/O tokens
    if (is_dense_tensor_name(tensor_name)) {
        std::lock_guard<std::mutex> lock(mutex_);
        dense_cache_[tensor_name].assign(dst, dst + meta.byte_length);
    }

    return true;
}

bool M8SafetensorsIndex::read_tensor_bf16_to_fp32(const std::string& tensor_name, float* dst, size_t num_elements) const {
    if (!dst || num_elements == 0) return false;

    SafetensorEntry meta;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tensors_.find(tensor_name);
        if (it == tensors_.end()) return false;
        meta = it->second;
    }

    size_t expected_bytes = num_elements * sizeof(uint16_t);
    if (meta.byte_length < expected_bytes) return false;

    std::vector<uint16_t> bf16_buf(num_elements);
    if (!read_tensor_bytes(tensor_name, reinterpret_cast<uint8_t*>(bf16_buf.data()), expected_bytes)) {
        return false;
    }

    for (size_t i = 0; i < num_elements; ++i) {
        uint32_t u = static_cast<uint32_t>(bf16_buf[i]) << 16;
        dst[i] = *reinterpret_cast<float*>(&u);
    }
    return true;
}

const uint8_t* M8SafetensorsIndex::map_tensor(const std::string& tensor_name, uint64_t* out_bytes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tensors_.find(tensor_name);
    if (it == tensors_.end()) return nullptr;
    const SafetensorEntry& meta = it->second;

    auto mit = shard_maps_.find(meta.shard_path);
    if (mit == shard_maps_.end()) {
        ShardMapping sm;
        std::wstring wpath(meta.shard_path.begin(), meta.shard_path.end());
        HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                   OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile != INVALID_HANDLE_VALUE) {
            HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (hMap != nullptr) {
                void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
                LARGE_INTEGER file_size{};
                if (view != nullptr && GetFileSizeEx(hFile, &file_size)) {
                    sm.file = hFile;
                    sm.mapping = hMap;
                    sm.base = static_cast<const uint8_t*>(view);
                    sm.size = static_cast<uint64_t>(file_size.QuadPart);
                } else {
                    if (view != nullptr) UnmapViewOfFile(view);
                    CloseHandle(hMap);
                    CloseHandle(hFile);
                }
            } else {
                CloseHandle(hFile);
            }
        }
        // Failed attempts are remembered (base == nullptr) so we do not retry on every token.
        mit = shard_maps_.emplace(meta.shard_path, sm).first;
    }
    if (!mit->second.base) return nullptr;
    // Never hand out a pointer that extends past the mapped file (corrupt / hostile header).
    if (meta.file_offset > mit->second.size || meta.byte_length > mit->second.size - meta.file_offset) {
        return nullptr;
    }
    if (out_bytes) *out_bytes = meta.byte_length;
    return mit->second.base + meta.file_offset;
}

void M8SafetensorsIndex::trim_mapped_pages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& pair : shard_maps_) {
        const ShardMapping& sm = pair.second;
        if (!sm.base || sm.size == 0) continue;
        // VirtualUnlock on a range that is not locked removes its pages from the working set
        // (it reports ERROR_NOT_LOCKED, which is expected and ignored).
        VirtualUnlock(const_cast<uint8_t*>(sm.base), static_cast<SIZE_T>(sm.size));
    }
}

size_t M8SafetensorsIndex::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tensors_.size();
}

void M8SafetensorsIndex::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : handle_pool_) {
        if (pair.second != INVALID_HANDLE_VALUE && pair.second != nullptr) {
            CloseHandle(static_cast<HANDLE>(pair.second));
        }
    }
    handle_pool_.clear();
    for (auto& pair : shard_maps_) {
        if (pair.second.base) UnmapViewOfFile(pair.second.base);
        if (pair.second.mapping) CloseHandle(static_cast<HANDLE>(pair.second.mapping));
        if (pair.second.file) CloseHandle(static_cast<HANDLE>(pair.second.file));
    }
    shard_maps_.clear();
    dense_cache_.clear();
    tensors_.clear();
    indexed_shards_.clear();
}

} // namespace m8
} // namespace asema
