// ASEMA v0.1 — Agent #3: C++ synthetic MoE generator implementation
// -----------------------------------------------------------------------------
// Produces an ASEMA-SSF container identical in schema to the Python generator.
// Deterministic, CRC32-checked, 64-byte aligned.
// -----------------------------------------------------------------------------

#include "../../include/asema/synthetic_gen.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <sys/stat.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace asema {
namespace gen {

namespace {

// Precomputed CRC32 (IEEE) — same algorithm as storage_ssf.cpp uses.
static uint32_t crc32_table[256];
static bool crc32_init = false;
static void init_crc32() {
    uint32_t poly = 0xEDB88320;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (poly ^ (c >> 1)) : (c >> 1);
        crc32_table[i] = c;
    }
    crc32_init = true;
}
uint32_t compute_crc32(const uint8_t* data, size_t len) {
    if (!crc32_init) init_crc32();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        c = crc32_table[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

// Cheap SHA256-hex stub matching storage_ssf.cpp's existing behavior.
std::string sha256_hex_stub(uint32_t crc) {
    std::ostringstream os;
    os << std::hex << std::setfill('0') << std::setw(8) << crc;
    return os.str();
}

} // anon

GenParams tier1_small() {
    GenParams p;
    p.num_layers = 4;
    p.experts_per_layer = 16;
    p.hidden_dim = 128;
    p.ffn_dim = 512;
    p.top_k = 2;
    p.dtype = "fp16";
    p.seed = 42;
    p.output_dir = "examples/synthetic_moe/model_synthetic";
    return p;
}

GenParams tier2_medium() {
    GenParams p;
    p.num_layers = 8;
    p.experts_per_layer = 32;
    p.hidden_dim = 128;
    p.ffn_dim = 512;
    p.top_k = 2;
    p.dtype = "fp16";
    p.seed = 43;
    p.output_dir = "examples/synthetic_moe/model_tier2";
    return p;
}

GenParams tier3_large() {
    GenParams p;
    p.num_layers = 12;
    p.experts_per_layer = 64;
    p.hidden_dim = 128;
    p.ffn_dim = 512;
    p.top_k = 2;
    p.dtype = "fp16";
    p.seed = 44;
    p.output_dir = "examples/synthetic_moe/model_tier3";
    return p;
}

GenParams tier_for_params(uint32_t layers, uint32_t experts, uint32_t hidden, uint32_t ffn) {
    GenParams p;
    p.num_layers = layers;
    p.experts_per_layer = experts;
    p.hidden_dim = hidden;
    p.ffn_dim = ffn;
    p.top_k = 2;
    p.dtype = "fp16";
    p.seed = 42;
    p.output_dir = "examples/synthetic_moe/model_custom";
    return p;
}

uint64_t estimate_model_bytes(const GenParams& p) {
    size_t bpe = (p.dtype == "fp32") ? 4 : 2;
    uint64_t expert_elems = (uint64_t)p.hidden_dim * p.ffn_dim * 2
                          + (uint64_t)p.ffn_dim * p.hidden_dim;
    return (uint64_t)p.num_layers * p.experts_per_layer * expert_elems * bpe;
}

bool has_free_disk_space(const std::string& path, uint64_t need_bytes) {
#if defined(_WIN32)
    // Find drive root from path.
    char root[4] = {'C', ':', '\\', '\0'};
    if (!path.empty() && std::isalpha(path[0]) && path.size() >= 2 && path[1] == ':') {
        root[0] = path[0];
    }
    ULARGE_INTEGER freeAvail, totalBytes, totalFree;
    if (!GetDiskFreeSpaceExA(root, &freeAvail, &totalBytes, &totalFree)) {
        return false;
    }
    return freeAvail.QuadPart >= need_bytes;
#else
    struct statvfs v {};
    if (statvfs(path.c_str(), &v) != 0) return false;
    return (uint64_t)v.f_bavail * v.f_frsize >= need_bytes;
#endif
}

GenResult generate(const GenParams& p) {
    GenResult r;
    if (p.output_dir.empty()) { r.error = "output_dir is required"; return r; }

    size_t bpe = (p.dtype == "fp32") ? 4 : 2;
    uint64_t expert_elems = (uint64_t)p.hidden_dim * p.ffn_dim * 2
                          + (uint64_t)p.ffn_dim * p.hidden_dim;
    uint64_t expert_bytes = expert_elems * bpe;
    uint64_t total_bytes = (uint64_t)p.num_layers * p.experts_per_layer * expert_bytes;

    // Disk space check (require 1.5× the model size for safety).
    if (!has_free_disk_space(p.output_dir, total_bytes + total_bytes / 2)) {
        r.error = "Insufficient free disk space; need " +
                  std::to_string(total_bytes + total_bytes / 2) + " bytes";
        return r;
    }

    try {
        std::filesystem::create_directories(p.output_dir);
    } catch (const std::exception& ex) {
        r.error = std::string("Cannot create output dir: ") + ex.what();
        return r;
    }

    auto t0 = std::chrono::steady_clock::now();

    std::string bin_path  = p.output_dir + "/model.asema";
    std::string mani_path = p.output_dir + "/manifest.json";

    std::ofstream f(bin_path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) { r.error = "Cannot open " + bin_path; return r; }

    // Build manifest JSON incrementally.
    std::ostringstream mani;
    mani << "{\n"
         << "  \"format\": \"ASEMA-SSF\",\n"
         << "  \"version\": 1,\n"
         << "  \"model_name\": \"Synthetic-MoE-" << p.num_layers << "L-"
         << p.experts_per_layer << "E-" << p.dtype << "\",\n"
         << "  \"base_architecture\": \"synthetic-sparse-moe\",\n"
         << "  \"num_layers\": " << p.num_layers << ",\n"
         << "  \"experts_per_layer\": " << p.experts_per_layer << ",\n"
         << "  \"active_experts_per_token\": " << p.top_k << ",\n"
         << "  \"hidden_dim\": " << p.hidden_dim << ",\n"
         << "  \"ffn_dim\": " << p.ffn_dim << ",\n"
         << "  \"num_heads\": 4,\n"
         << "  \"total_parameters\": "
         << (p.num_layers * p.experts_per_layer * expert_elems) << ",\n"
         << "  \"total_storage_bytes\": " << total_bytes << ",\n"
         << "  \"layers\": [\n";

    uint64_t current_offset = 0;
    uint64_t experts_written = 0;

    // Pre-allocate a buffer for one expert.
    std::vector<uint8_t> buf(expert_bytes);

    for (uint32_t layer_id = 0; layer_id < p.num_layers; ++layer_id) {
        if (layer_id > 0) mani << ",\n";
        mani << "    {\n"
             << "      \"layer_id\": " << layer_id << ",\n"
             << "      \"num_experts\": " << p.experts_per_layer << ",\n"
             << "      \"shared_weights_offset\": 0,\n"
             << "      \"shared_weights_size\": 0,\n"
             << "      \"shared_weights_file\": \"\",\n"
             << "      \"experts\": [\n";

        for (uint32_t eid = 0; eid < p.experts_per_layer; ++eid) {
            // Deterministic weights (mirrors Python: layer_id*1000 + eid).
            uint64_t seed = (uint64_t)layer_id * 1000ull + eid + p.seed;
            std::mt19937_64 rng(seed);
            std::normal_distribution<float> nd(0.0f, 1.0f);

            // Fill buffer. For fp16 we still emit 2 bytes per element.
            // Use fixed-point-ish: scale=0.02, then truncate.
            float scale = 0.02f;
            for (uint64_t i = 0; i < expert_elems; ++i) {
                float v = nd(rng) * scale;
                if (bpe == 4) {
                    std::memcpy(&buf[i * 4], &v, 4);
                } else {
                    // fp16 truncation via memcpy of half-precision bits.
                    // Approximation: store the upper 16 bits of the float.
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    uint16_t h = static_cast<uint16_t>((bits >> 16) & 0xFFFF);
                    std::memcpy(&buf[i * 2], &h, 2);
                }
            }

            uint32_t crc = compute_crc32(buf.data(), expert_bytes);
            std::string sha = sha256_hex_stub(crc);

            // Align offset to 64 bytes.
            uint64_t pad = (64 - (current_offset % 64)) % 64;
            if (pad > 0) {
                static const uint8_t zeros[64] = {0};
                f.write(reinterpret_cast<const char*>(zeros), pad);
                current_offset += pad;
            }

            uint64_t start = current_offset;
            f.write(reinterpret_cast<const char*>(buf.data()), expert_bytes);
            current_offset += expert_bytes;
            experts_written++;

            if (eid > 0) mani << ",\n";
            mani << "        {\n"
                 << "          \"layer\": " << layer_id << ",\n"
                 << "          \"expert\": " << eid << ",\n"
                 << "          \"storage_offset\": " << start << ",\n"
                 << "          \"storage_length\": " << expert_bytes << ",\n"
                 << "          \"alignment\": 64,\n"
                 << "          \"dtype\": \"" << p.dtype << "\",\n"
                 << "          \"quant_type\": \"none\",\n"
                 << "          \"checksum_sha256\": \"" << sha << "\",\n"
                 << "          \"checksum_crc32\": " << crc << ",\n"
                 << "          \"tensor_shape\": [3, " << p.hidden_dim << ", " << p.ffn_dim << "]\n"
                 << "        }";
        }
        mani << "\n      ]\n    }";
    }

    mani << "\n  ]\n}\n";
    f.close();

    // Write manifest.
    std::ofstream mf(mani_path, std::ios::out | std::ios::trunc);
    if (!mf.is_open()) { r.error = "Cannot write manifest"; return r; }
    mf << mani.str();
    mf.close();

    auto t1 = std::chrono::steady_clock::now();
    r.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.experts_written = experts_written;
    r.bytes_written = current_offset;
    r.manifest_path = mani_path;
    r.container_path = bin_path;
    r.success = true;
    return r;
}

} // namespace gen
} // namespace asema
