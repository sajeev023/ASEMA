#include "asema/m8/m8_model_manifest.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>

namespace asema {
namespace m8 {

ModelManifest M8ModelManifestManager::create_production_manifest(
    const std::string& hf_root,
    const std::string& vol_d,
    const std::string& vol_e) {

    ModelManifest m;
    m.tokenizer_path = hf_root + "/tokenizer.json";
    m.config_path = hf_root + "/config.json";
    m.index_path = hf_root + "/model.safetensors.index.json";

    m.shards.reserve(48);
    for (int i = 1; i <= 48; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "model-%05d-of-00048.safetensors", i);
        ShardPlacement sp;
        sp.shard_id = i;
        sp.filename = buf;
        // Distribute: shards 1-32 to Volume D (~340 GB), shards 33-48 to Volume E (~170 GB)
        if (i <= 32) {
            sp.volume_path = vol_d + "/" + buf;
        } else {
            sp.volume_path = vol_e + "/" + buf;
        }
        sp.size_bytes = 10625000000ULL; // ~10.6 GB avg
        sp.sha256 = "verified_sha256_placeholder";
        m.shards.push_back(sp);
    }
    return m;
}

bool M8ModelManifestManager::save_manifest(const ModelManifest& manifest, const std::string& output_json_path) {
    nlohmann::json j;
    j["model_id"] = manifest.model_id;
    j["revision"] = manifest.revision;
    j["num_layers"] = manifest.num_layers;
    j["hidden_dim"] = manifest.hidden_dim;
    j["num_routed_experts"] = manifest.num_routed_experts;
    j["top_k"] = manifest.top_k;
    j["total_shards"] = manifest.total_shards;
    j["total_checkpoint_bytes"] = manifest.total_checkpoint_bytes;
    j["tokenizer_path"] = manifest.tokenizer_path;
    j["config_path"] = manifest.config_path;
    j["index_path"] = manifest.index_path;

    nlohmann::json shards_arr = nlohmann::json::array();
    for (const auto& sp : manifest.shards) {
        nlohmann::json item;
        item["shard_id"] = sp.shard_id;
        item["filename"] = sp.filename;
        item["volume_path"] = sp.volume_path;
        item["size_bytes"] = sp.size_bytes;
        item["sha256"] = sp.sha256;
        shards_arr.push_back(item);
    }
    j["shards"] = shards_arr;

    std::ofstream out(output_json_path);
    if (!out.is_open()) return false;
    out << j.dump(2) << std::endl;
    return true;
}

bool M8ModelManifestManager::load_manifest(const std::string& input_json_path, ModelManifest& manifest) {
    std::ifstream in(input_json_path);
    if (!in.is_open()) return false;
    nlohmann::json j;
    in >> j;

    manifest.model_id = j.value("model_id", "deepseek-ai/DeepSeek-V4.1-Flash");
    manifest.revision = j.value("revision", "");
    manifest.num_layers = j.value("num_layers", 40);
    manifest.hidden_dim = j.value("hidden_dim", 5120);
    manifest.num_routed_experts = j.value("num_routed_experts", 384);
    manifest.top_k = j.value("top_k", 6);
    manifest.total_shards = j.value("total_shards", 48);
    manifest.total_checkpoint_bytes = j.value("total_checkpoint_bytes", 0ULL);
    manifest.tokenizer_path = j.value("tokenizer_path", "");
    manifest.config_path = j.value("config_path", "");
    manifest.index_path = j.value("index_path", "");

    manifest.shards.clear();
    if (j.contains("shards") && j["shards"].is_array()) {
        for (const auto& item : j["shards"]) {
            ShardPlacement sp;
            sp.shard_id = item.value("shard_id", 0);
            sp.filename = item.value("filename", "");
            sp.volume_path = item.value("volume_path", "");
            sp.size_bytes = item.value("size_bytes", 0ULL);
            sp.sha256 = item.value("sha256", "");
            manifest.shards.push_back(sp);
        }
    }
    return true;
}

} // namespace m8
} // namespace asema
