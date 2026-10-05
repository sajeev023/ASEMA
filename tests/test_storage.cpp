#include "asema/m8/m8_paths.hpp"
#include "../include/asema/manifest.hpp"
#include "../include/asema/storage.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

int main() {
    std::cout << "[TEST] Running ASEMA Storage & Manifest Tests...\n";

    std::string manifest_path = (asema::m8::paths::synthetic_dir() + "/manifest.json");
    std::string model_bin = (asema::m8::paths::synthetic_dir() + "/model.asema");

    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        std::cerr << "[FAIL] Could not open " << manifest_path << "\n";
        return 1;
    }

    std::stringstream buf;
    buf << f.rdbuf();
    auto manifest = asema::ModelManifest::from_json(buf.str());

    std::cout << "[PASS] Parsed manifest with " << manifest.num_layers << " layers, " 
              << manifest.lookup_index.size() << " indexed experts.\n";

    assert(manifest.num_layers == 4);
    assert(manifest.experts_per_layer == 16);
    assert(manifest.has_expert(0, 0));
    assert(manifest.has_expert(3, 15));
    assert(!manifest.has_expert(4, 0));

    auto backend = asema::create_storage_backend(model_bin);
    if (!backend->is_open()) {
        std::cerr << "[FAIL] Could not open model container: " << model_bin << "\n";
        return 1;
    }

    // Test synchronous read of Layer 0, Expert 0
    const auto* meta_0_0 = manifest.get_expert_metadata(0, 0);
    assert(meta_0_0 != nullptr);

    asema::ExpertBuffer exp_buf(meta_0_0->storage_length, meta_0_0->alignment);
    double latency = 0.0;
    bool read_ok = backend->read_expert_sync(*meta_0_0, exp_buf, latency);
    if (!read_ok) {
        std::cerr << "[FAIL] Failed to read Layer 0 Expert 0\n";
        return 1;
    }

    std::cout << "[PASS] Synchronous read: L0 E0 read in " << latency << " ms (" 
              << meta_0_0->storage_length << " bytes, checksum verified).\n";

    // Test synchronous read of Layer 2, Expert 7
    const auto* meta_2_7 = manifest.get_expert_metadata(2, 7);
    assert(meta_2_7 != nullptr);
    asema::ExpertBuffer exp_buf2(meta_2_7->storage_length, meta_2_7->alignment);
    read_ok = backend->read_expert_sync(*meta_2_7, exp_buf2, latency);
    assert(read_ok);
    std::cout << "[PASS] Synchronous read: L2 E7 read in " << latency << " ms.\n";

    // Test checksum corruption detection
    auto corrupted_meta = *meta_0_0;
    corrupted_meta.checksum_crc32 ^= 0xDEADBEEF; // Inject mismatch
    bool corrupt_ok = backend->read_expert_sync(corrupted_meta, exp_buf, latency);
    assert(!corrupt_ok);
    std::cout << "[PASS] Corrupted CRC32 safely rejected.\n";

    std::cout << "\n[ALL TESTS PASSED SUCCESSFULLY]\n";
    return 0;
}
