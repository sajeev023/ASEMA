#pragma once

// Configurable model / storage locations.
//
// Nothing in the source tree hardcodes a machine-specific path. Locations are resolved, in order,
// from (1) an environment variable, (2) a `key=value` config file, (3) a neutral relative default.
//
//   key               environment variable        meaning
//   ---------------   -------------------------   --------------------------------------------
//   shards_primary    ASEMA_SHARDS_PRIMARY        directory with model-*.safetensors shards
//   shards_secondary  ASEMA_SHARDS_SECONDARY      optional second directory (e.g. another drive)
//   model_root        ASEMA_MODEL_ROOT            directory with index.json / tokenizer.json
//   hf_dir            ASEMA_HF_DIR                Hugging Face metadata dir (tokenizer, config)
//   synthetic_dir     ASEMA_SYNTHETIC_DIR         synthetic MoE test model directory
//
// The config file is `asema.config` in the working directory, or the file named by ASEMA_CONFIG.
// Lines starting with '#' and blank lines are ignored. See asema.config.example.

#include <cstdlib>
#include <fstream>
#include <map>
#include <string>

namespace asema {
namespace m8 {
namespace paths {

namespace detail {

inline std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n\"");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n\"");
    return s.substr(b, e - b + 1);
}

inline const std::map<std::string, std::string>& config_file_values() {
    static const std::map<std::string, std::string> values = [] {
        std::map<std::string, std::string> out;
        const char* explicit_path = std::getenv("ASEMA_CONFIG");
        std::ifstream in(explicit_path && *explicit_path ? explicit_path : "asema.config");
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            out[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
        }
        return out;
    }();
    return values;
}

inline std::string resolve(const char* env_name, const char* key, const char* fallback) {
    if (const char* env = std::getenv(env_name)) {
        if (*env) return env;
    }
    const auto& cfg = config_file_values();
    auto it = cfg.find(key);
    if (it != cfg.end() && !it->second.empty()) return it->second;
    return fallback;
}

} // namespace detail

inline std::string primary_shards() {
    return detail::resolve("ASEMA_SHARDS_PRIMARY", "shards_primary", "models/DeepSeek-V4.1-Flash/shards");
}

// Empty string means "no second volume".
inline std::string secondary_shards() {
    return detail::resolve("ASEMA_SHARDS_SECONDARY", "shards_secondary", "");
}

// Optional explicit shard-placement manifest (JSON). Empty string means "scan the shard directories".
inline std::string storage_manifest() {
    return detail::resolve("ASEMA_STORAGE_MANIFEST", "storage_manifest", "");
}

inline std::string model_root() {
    return detail::resolve("ASEMA_MODEL_ROOT", "model_root", "models/DeepSeek-V4.1-Flash");
}

inline std::string hf_dir() {
    return detail::resolve("ASEMA_HF_DIR", "hf_dir", "models/DeepSeek-V4.1-Flash/hf");
}

inline std::string synthetic_dir() {
    return detail::resolve("ASEMA_SYNTHETIC_DIR", "synthetic_dir", "examples/synthetic_moe/model_synthetic");
}

} // namespace paths
} // namespace m8
} // namespace asema
