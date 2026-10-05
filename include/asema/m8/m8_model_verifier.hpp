#pragma once

#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_model_manifest.hpp"

#include <string>
#include <vector>

namespace asema {
namespace m8 {

struct VerificationResult {
    bool passed{false};
    int shards_found{0};
    int shards_missing{0};
    int shards_corrupted{0};
    uint64_t total_tensors_verified{0};
    bool config_valid{false};
    bool tokenizer_valid{false};
    bool index_valid{false};
    std::vector<std::string> errors;
};

class M8ModelVerifier {
public:
    static VerificationResult verify_checkpoint(
        const std::string& hf_root,
        const std::string& vol_d = asema::m8::paths::primary_shards(),
        const std::string& vol_e = asema::m8::paths::secondary_shards());
};

} // namespace m8
} // namespace asema
