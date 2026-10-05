#pragma once

#include "expert.hpp"
#include "../nlohmann/json.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <stdexcept>

namespace asema {

struct ManifestLayer {
    uint32_t layer_id{0};
    uint32_t num_experts{0};
    uint64_t shared_weights_offset{0};
    uint64_t shared_weights_size{0};
    std::string shared_weights_file;
    std::vector<ExpertMetadata> experts;
};

struct ModelManifest {
    std::string format{"ASEMA-SSF"};
    uint32_t version{1};
    std::string model_name;
    std::string base_architecture{"sparse-moe"};
    uint32_t num_layers{0};
    uint32_t experts_per_layer{0};
    uint32_t active_experts_per_token{2};
    uint32_t hidden_dim{0};
    uint32_t ffn_dim{0};
    uint32_t num_heads{0};
    uint64_t total_parameters{0};
    uint64_t total_storage_bytes{0};
    std::vector<ManifestLayer> layers;

    // Fast coordinate lookup index: (layer_id, expert_id) -> ExpertMetadata
    std::unordered_map<uint64_t, ExpertMetadata> lookup_index;

    static uint64_t make_key(uint32_t layer, uint32_t expert) noexcept {
        return (static_cast<uint64_t>(layer) << 32) | static_cast<uint64_t>(expert);
    }

    void build_index() {
        lookup_index.clear();
        for (const auto& layer : layers) {
            for (const auto& exp : layer.experts) {
                lookup_index[make_key(exp.layer_id, exp.expert_id)] = exp;
            }
        }
    }

    bool has_expert(uint32_t layer_id, uint32_t expert_id) const noexcept {
        return lookup_index.find(make_key(layer_id, expert_id)) != lookup_index.end();
    }

    const ExpertMetadata* get_expert_metadata(uint32_t layer_id, uint32_t expert_id) const noexcept {
        auto it = lookup_index.find(make_key(layer_id, expert_id));
        if (it != lookup_index.end()) {
            return &(it->second);
        }
        return nullptr;
    }

    std::string to_json() const {
        nlohmann::json j;
        j["format"] = format;
        j["version"] = version;
        j["model_name"] = model_name;
        j["base_architecture"] = base_architecture;
        j["num_layers"] = num_layers;
        j["experts_per_layer"] = experts_per_layer;
        j["active_experts_per_token"] = active_experts_per_token;
        j["hidden_dim"] = hidden_dim;
        j["ffn_dim"] = ffn_dim;
        j["num_heads"] = num_heads;
        j["total_parameters"] = total_parameters;
        j["total_storage_bytes"] = total_storage_bytes;

        nlohmann::json layers_json = nlohmann::json::array();
        for (const auto& l : layers) {
            nlohmann::json lj;
            lj["layer_id"] = l.layer_id;
            lj["num_experts"] = l.num_experts;
            lj["shared_weights_offset"] = l.shared_weights_offset;
            lj["shared_weights_size"] = l.shared_weights_size;
            lj["shared_weights_file"] = l.shared_weights_file;

            nlohmann::json exp_arr = nlohmann::json::array();
            for (const auto& e : l.experts) {
                nlohmann::json ej;
                ej["layer"] = e.layer_id;
                ej["expert"] = e.expert_id;
                ej["storage_offset"] = e.storage_offset;
                ej["storage_length"] = e.storage_length;
                ej["alignment"] = e.alignment;
                ej["dtype"] = e.dtype;
                ej["quant_type"] = e.quant_type;
                ej["checksum_sha256"] = e.checksum_sha256;
                ej["checksum_crc32"] = e.checksum_crc32;
                ej["tensor_shape"] = e.tensor_shape;
                exp_arr.push_back(ej);
            }
            lj["experts"] = exp_arr;
            layers_json.push_back(lj);
        }
        j["layers"] = layers_json;
        return j.dump(2);
    }

    static ModelManifest from_json(const std::string& json_str) {
        auto j = nlohmann::json::parse(json_str);
        ModelManifest m;
        m.format = j.value("format", "ASEMA-SSF");
        m.version = j.value("version", 1);
        m.model_name = j.value("model_name", "unnamed");
        m.base_architecture = j.value("base_architecture", "sparse-moe");
        m.num_layers = j.value("num_layers", 0u);
        m.experts_per_layer = j.value("experts_per_layer", 0u);
        m.active_experts_per_token = j.value("active_experts_per_token", 2u);
        m.hidden_dim = j.value("hidden_dim", 0u);
        m.ffn_dim = j.value("ffn_dim", 0u);
        m.num_heads = j.value("num_heads", 0u);
        m.total_parameters = j.value("total_parameters", 0ULL);
        m.total_storage_bytes = j.value("total_storage_bytes", 0ULL);

        if (j.contains("layers")) {
            for (const auto& lj : j["layers"]) {
                ManifestLayer ml;
                ml.layer_id = lj.value("layer_id", 0u);
                ml.num_experts = lj.value("num_experts", 0u);
                ml.shared_weights_offset = lj.value("shared_weights_offset", 0ULL);
                ml.shared_weights_size = lj.value("shared_weights_size", 0ULL);
                ml.shared_weights_file = lj.value("shared_weights_file", "");

                if (lj.contains("experts")) {
                    for (const auto& ej : lj["experts"]) {
                        ExpertMetadata em;
                        em.layer_id = ej.value("layer", 0u);
                        em.expert_id = ej.value("expert", 0u);
                        em.storage_offset = ej.value("storage_offset", 0ULL);
                        em.storage_length = ej.value("storage_length", 0ULL);
                        em.alignment = ej.value("alignment", 64u);
                        em.dtype = ej.value("dtype", "fp16");
                        em.quant_type = ej.value("quant_type", "none");
                        em.checksum_sha256 = ej.value("checksum_sha256", "");
                        em.checksum_crc32 = ej.value("checksum_crc32", 0u);
                        if (ej.contains("tensor_shape")) {
                            em.tensor_shape = ej["tensor_shape"].get<std::vector<int64_t>>();
                        }
                        ml.experts.push_back(em);
                    }
                }
                m.layers.push_back(ml);
            }
        }
        m.build_index();
        return m;
    }
};

} // namespace asema
