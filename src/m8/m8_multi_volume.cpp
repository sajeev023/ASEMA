#include "asema/m8/m8_multi_volume.hpp"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

void M8MultiVolumeManager::register_volume(const std::string& volume_root) {
    if (volume_root.empty()) return;
    std::string norm = volume_root;
    while (!norm.empty() && (norm.back() == '/' || norm.back() == '\\')) {
        norm.pop_back();
    }
    volumes_.push_back(norm);

    if (fs::exists(norm)) {
        for (const auto& entry : fs::directory_iterator(norm)) {
            if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                safetensors_index_.index_shard(entry.path().string());
            }
        }
    }
}

bool M8MultiVolumeManager::load_index(const std::string& index_json_path) {
    std::ifstream f(index_json_path);
    if (!f.is_open()) {
        return false;
    }

    try {
        nlohmann::json j;
        f >> j;
        if (!j.contains("weight_map")) {
            return false;
        }

        tensor_to_shard_.clear();
        shard_to_path_cache_.clear();

        for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
            tensor_to_shard_[it.key()] = it.value().get<std::string>();
        }

        // Automatically index all physical .safetensors shards across all registered volumes
        for (const auto& vol : volumes_) {
            if (fs::exists(vol)) {
                for (const auto& entry : fs::directory_iterator(vol)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
                        safetensors_index_.index_shard(entry.path().string());
                    }
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

std::string M8MultiVolumeManager::resolve_shard_path(const std::string& shard_name) const {
    if (shard_name.empty()) return "";

    auto it = shard_to_path_cache_.find(shard_name);
    if (it != shard_to_path_cache_.end()) {
        return it->second;
    }

    // Search across registered volumes in order
    for (const auto& vol : volumes_) {
        fs::path p = fs::path(vol) / shard_name;
        if (fs::exists(p)) {
            std::string resolved = p.string();
            shard_to_path_cache_[shard_name] = resolved;
            safetensors_index_.index_shard(resolved);
            return resolved;
        }
    }

    // Default to first volume if not found on any
    if (!volumes_.empty()) {
        return (fs::path(volumes_[0]) / shard_name).string();
    }
    return shard_name;
}

TensorLocation M8MultiVolumeManager::locate_tensor(const std::string& tensor_name) const {
    TensorLocation tloc;
    tloc.tensor_name = tensor_name;

    const SafetensorEntry* meta = safetensors_index_.get_tensor_meta(tensor_name);
    if (meta) {
        tloc.physical_path = meta->shard_path;
        tloc.shard_name = fs::path(meta->shard_path).filename().string();
        tloc.exists_on_disk = fs::exists(tloc.physical_path);
        tloc.offset_in_file = meta->file_offset;
        tloc.size_bytes = meta->byte_length;
        return tloc;
    }

    auto it = tensor_to_shard_.find(tensor_name);
    if (it != tensor_to_shard_.end()) {
        tloc.shard_name = it->second;
        tloc.physical_path = resolve_shard_path(tloc.shard_name);
        tloc.exists_on_disk = fs::exists(tloc.physical_path);

        if (tloc.exists_on_disk) {
            meta = safetensors_index_.get_tensor_meta(tensor_name);
            if (meta) {
                tloc.offset_in_file = meta->file_offset;
                tloc.size_bytes = meta->byte_length;
            }
        }
    }
    return tloc;
}

bool M8MultiVolumeManager::read_tensor(const std::string& tensor_name, uint8_t* dst, size_t dst_size) const {
    return safetensors_index_.read_tensor_bytes(tensor_name, dst, dst_size);
}

bool M8MultiVolumeManager::read_tensor_bf16_to_fp32(const std::string& tensor_name, float* dst, size_t num_elements) const {
    return safetensors_index_.read_tensor_bf16_to_fp32(tensor_name, dst, num_elements);
}

ExpertLocation M8MultiVolumeManager::locate_expert(int layer_id, int expert_id) const {
    ExpertLocation loc;
    loc.layer_id = layer_id;
    loc.expert_id = expert_id;

    std::string prefix = "layers." + std::to_string(layer_id) + ".ffn.experts." + std::to_string(expert_id) + ".";

    loc.w1_scale = locate_tensor(prefix + "w1.scale");
    loc.w2_scale = locate_tensor(prefix + "w2.scale");
    loc.w3_scale = locate_tensor(prefix + "w3.scale");
    loc.w1_weight = locate_tensor(prefix + "w1.weight");
    loc.w2_weight = locate_tensor(prefix + "w2.weight");
    loc.w3_weight = locate_tensor(prefix + "w3.weight");

    loc.shard_name = loc.w1_weight.shard_name;
    loc.physical_path = loc.w1_weight.physical_path;

    loc.all_tensors_present = loc.w1_scale.exists_on_disk && loc.w2_scale.exists_on_disk &&
                              loc.w3_scale.exists_on_disk && loc.w1_weight.exists_on_disk &&
                              loc.w2_weight.exists_on_disk && loc.w3_weight.exists_on_disk;

    return loc;
}

bool M8MultiVolumeManager::is_expert_available(int layer_id, int expert_id) const {
    ExpertLocation loc = locate_expert(layer_id, expert_id);
    return loc.all_tensors_present;
}

std::string M8MultiVolumeManager::get_storage_distribution_report() const {
    std::ostringstream ss;
    ss << "=== ASEMA M8 Multi-Volume Storage Layout ===\n";
    ss << "Registered Physical Volumes: " << volumes_.size() << "\n";
    for (size_t i = 0; i < volumes_.size(); ++i) {
        ss << "  [" << i << "] " << volumes_[i];
        if (fs::exists(volumes_[i])) {
            auto space = fs::space(volumes_[i]);
            ss << " (Free: " << (space.available / (1024ULL * 1024ULL * 1024ULL)) << " GB / Total: "
               << (space.capacity / (1024ULL * 1024ULL * 1024ULL)) << " GB)\n";
        } else {
            ss << " (Directory does not exist yet)\n";
        }
    }
    ss << "Indexed Tensors: " << tensor_to_shard_.size() << "\n";
    return ss.str();
}

} // namespace m8
} // namespace asema
