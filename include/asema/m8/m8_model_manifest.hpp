#pragma once

#include "asema/m8/m8_paths.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace asema {
namespace m8 {

struct ShardPlacement {
    int shard_id{0};
    std::string filename;
    std::string volume_path;
    size_t size_bytes{0};
    std::string sha256;
};

struct ModelManifest {
    std::string model_id{"deepseek-ai/DeepSeek-V4.1-Flash"};
    std::string revision{"e1281f85d0ce4a3dfb63d41926fc4a47fa71f36b"};
    int num_layers{40};
    int hidden_dim{5120};
    int num_routed_experts{384};
    int top_k{6};
    int total_shards{48};
    size_t total_checkpoint_bytes{510000000000ULL}; // ~510 GB
    std::vector<ShardPlacement> shards;
    std::string tokenizer_path;
    std::string config_path;
    std::string index_path;
};

class M8ModelManifestManager {
public:
    static ModelManifest create_production_manifest(
        const std::string& hf_root,
        const std::string& vol_d = asema::m8::paths::primary_shards(),
        const std::string& vol_e = asema::m8::paths::secondary_shards());

    static bool save_manifest(const ModelManifest& manifest, const std::string& output_json_path);
    static bool load_manifest(const std::string& input_json_path, ModelManifest& manifest);
};

} // namespace m8
} // namespace asema
