#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "asema/m8/m8_safetensors_index.hpp"

namespace asema {
namespace m8 {

struct TensorLocation {
    std::string tensor_name;
    std::string shard_name;
    std::string physical_path;
    uint64_t offset_in_file{0};
    uint64_t size_bytes{0};
    bool exists_on_disk{false};
};

struct ExpertLocation {
    int layer_id{-1};
    int expert_id{-1};
    std::string shard_name;
    std::string physical_path;
    TensorLocation w1_scale;
    TensorLocation w2_scale;
    TensorLocation w3_scale;
    TensorLocation w1_weight;
    TensorLocation w2_weight;
    TensorLocation w3_weight;
    bool all_tensors_present{false};
};

class M8MultiVolumeManager {
public:
    M8MultiVolumeManager() = default;

    // Register a physical volume root path (e.g. the directories named by asema::m8::paths::primary_shards() / secondary_shards())
    void register_volume(const std::string& volume_root);

    // Load an explicit storage manifest ({"shards":[{"name","path","size"}]}). While a manifest is active it is the
    // single source of truth: only the listed files are indexed, volume directories are not scanned. Returns false
    // (and leaves the manager untouched) if the file is unreadable or any listed shard is missing / the wrong size.
    bool load_storage_manifest(const std::string& manifest_path);
    bool manifest_active() const { return manifest_active_; }

    // Load official Hugging Face model.safetensors.index.json
    bool load_index(const std::string& index_json_path);

    // Resolve where a shard file physically resides
    std::string resolve_shard_path(const std::string& shard_name) const;

    // Locate any tensor in the complete model (O(1) resolution)
    TensorLocation locate_tensor(const std::string& tensor_name) const;

    // Read tensor data directly from physical checkpoint
    bool read_tensor(const std::string& tensor_name, uint8_t* dst, size_t dst_size) const;
    bool read_tensor_bf16_to_fp32(const std::string& tensor_name, float* dst, size_t num_elements) const;

    // Locate the exact physical storage and offsets for any expert (layer 0..39, expert 0..383)
    ExpertLocation locate_expert(int layer_id, int expert_id) const;

    // Check if an expert is currently physically available on disk
    bool is_expert_available(int layer_id, int expert_id) const;

    // Get list of all registered volumes
    const std::vector<std::string>& volumes() const { return volumes_; }

    // Summary of capacity and distribution across volumes
    std::string get_storage_distribution_report() const;

    // Safetensors index accessor
    M8SafetensorsIndex& safetensors_index() { return safetensors_index_; }
    const M8SafetensorsIndex& safetensors_index() const { return safetensors_index_; }

private:
    std::vector<std::string> volumes_;
    bool manifest_active_{false};
    std::unordered_map<std::string, std::string> manifest_;   // shard_name -> absolute path (storage manifest)
    // Map tensor_name -> shard_name (e.g. "layers.0.ffn.experts.0.w1.weight" -> "model-00003-of-00048.safetensors")
    std::unordered_map<std::string, std::string> tensor_to_shard_;
    // Cache of shard_name -> resolved physical path
    mutable std::unordered_map<std::string, std::string> shard_to_path_cache_;
    mutable M8SafetensorsIndex safetensors_index_;
};

} // namespace m8
} // namespace asema
