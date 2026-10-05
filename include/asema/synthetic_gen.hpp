#pragma once

// ASEMA v0.1 — Agent #3: C++ synthetic MoE generator
// -----------------------------------------------------------------------------
// Mirrors the Python generator's behavior (deterministic weights, 64-byte
// alignment, CRC32 checksums, manifest.json) but accepts parameters so we can
// produce multiple tier sizes for scaling experiments.
//
// The output format is identical to what the existing M1 storage and M2 loader
// already understand — no schema changes needed.
// -----------------------------------------------------------------------------

#include "experiments.hpp"

#include <cstdint>
#include <string>

namespace asema {
namespace gen {

// -----------------------------------------------------------------------------
// Generator parameters
// -----------------------------------------------------------------------------
struct GenParams {
    uint32_t num_layers{4};
    uint32_t experts_per_layer{16};
    uint32_t hidden_dim{128};
    uint32_t ffn_dim{512};
    uint32_t top_k{2};
    std::string dtype{"fp16"};          // "fp16" or "fp32"
    uint64_t seed{42};
    std::string output_dir;             // must be set
};

// -----------------------------------------------------------------------------
// Generation outcome
// -----------------------------------------------------------------------------
struct GenResult {
    bool success{false};
    std::string error;
    uint64_t experts_written{0};
    uint64_t bytes_written{0};
    double elapsed_ms{0.0};
    std::string manifest_path;
    std::string container_path;
};

// -----------------------------------------------------------------------------
// Pre-defined tiers (used by asema-bench and the experiment matrix)
// -----------------------------------------------------------------------------
GenParams tier1_small();      // 4L × 16E × 128 hidden × 512 ffn (≈ 48 MB fp16)
GenParams tier2_medium();     // 8L × 32E × 128 hidden × 512 ffn (≈ 192 MB fp16)
GenParams tier3_large();      // 12L × 64E × 128 hidden × 512 ffn (≈ 576 MB fp16)
GenParams tier_for_params(uint32_t layers, uint32_t experts, uint32_t hidden = 128, uint32_t ffn = 512);

// -----------------------------------------------------------------------------
// Main generator
// -----------------------------------------------------------------------------
GenResult generate(const GenParams& p);

// -----------------------------------------------------------------------------
// Free-space check (MB). Returns true if at least `need_bytes` is available.
// -----------------------------------------------------------------------------
bool has_free_disk_space(const std::string& path, uint64_t need_bytes);

// Estimate model bytes for given params (does not touch disk).
uint64_t estimate_model_bytes(const GenParams& p);

} // namespace gen
} // namespace asema
