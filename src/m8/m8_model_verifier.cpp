#include "asema/m8/m8_model_verifier.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

VerificationResult M8ModelVerifier::verify_checkpoint(
    const std::string& hf_root,
    const std::string& vol_d,
    const std::string& vol_e) {

    VerificationResult res;

    // 1. Check config.json
    std::string config_file = hf_root + "/config.json";
    if (fs::exists(config_file)) {
        try {
            std::ifstream f(config_file);
            nlohmann::json j;
            f >> j;
            nlohmann::json cfg = j;
            if (j.contains("text_config")) {
                cfg = j["text_config"];
            }
            if (cfg.value("hidden_size", 0) == 5120 && cfg.value("num_hidden_layers", 0) == 40) {
                res.config_valid = true;
            } else {
                res.errors.push_back("config.json contains unexpected dimensions.");
            }
        } catch (...) {
            res.errors.push_back("config.json failed to parse as valid JSON.");
        }
    } else {
        res.errors.push_back("config.json is missing from " + hf_root);
    }

    // 2. Check tokenizer.json
    std::string tok_file = hf_root + "/tokenizer.json";
    if (fs::exists(tok_file) && fs::file_size(tok_file) > 1000) {
        res.tokenizer_valid = true;
    } else {
        res.errors.push_back("tokenizer.json is missing or truncated in " + hf_root);
    }

    // 3. Check model.safetensors.index.json
    std::string idx_file = hf_root + "/model.safetensors.index.json";
    if (fs::exists(idx_file)) {
        try {
            std::ifstream f(idx_file);
            nlohmann::json j;
            f >> j;
            if (j.contains("weight_map")) {
                res.index_valid = true;
                res.total_tensors_verified = j["weight_map"].size();
            }
        } catch (...) {
            res.errors.push_back("model.safetensors.index.json is malformed.");
        }
    } else {
        res.errors.push_back("model.safetensors.index.json is missing.");
    }

    // 4. Shard inspection across volumes
    for (int i = 1; i <= 48; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "model-%05d-of-00048.safetensors", i);
        std::string p_d = vol_d + "/" + buf;
        std::string p_e = vol_e + "/" + buf;

        if (fs::exists(p_d) || fs::exists(p_e)) {
            res.shards_found++;
        } else {
            res.shards_missing++;
        }
    }

    res.passed = res.config_valid && res.tokenizer_valid && res.index_valid;
    return res;
}

} // namespace m8
} // namespace asema
