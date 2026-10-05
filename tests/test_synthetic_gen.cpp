// ASEMA v0.1 — Agent #3: synthetic generator tests
// -----------------------------------------------------------------------------

#include "../include/asema/synthetic_gen.hpp"
#include "../include/asema/manifest.hpp"
#include "../include/asema/storage.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

using namespace asema;

namespace {

int g_failed = 0;
#define ASSERT_TRUE(expr) do {                                                 \
    if (!(expr)) {                                                             \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #expr << "\n";                                      \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)
#define ASSERT_EQ(a, b) do {                                                   \
    auto _av = (a); auto _bv = (b);                                            \
    if (!(_av == _bv)) {                                                       \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #a " != " #b << "\n";                               \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)

bool test_tier_estimates() {
    auto t1 = gen::tier1_small();
    auto t2 = gen::tier2_medium();
    auto t3 = gen::tier3_large();
    uint64_t b1 = gen::estimate_model_bytes(t1);
    uint64_t b2 = gen::estimate_model_bytes(t2);
    uint64_t b3 = gen::estimate_model_bytes(t3);
    // Tier 1: 4 * 16 * (128*512*2 + 512*128) * 2 = 7 864 32 * 64 = 50 331 648 B
    // Actually: (128*512*2 + 512*128) = 131072 + 65536 = 196608 elements
    //           * 2 bytes = 393216 bytes per expert
    //           * 64 experts = 25 165 824 bytes ≈ 24 MB
    std::cout << "  tier1=" << b1 << " B  tier2=" << b2 << " B  tier3=" << b3 << " B\n";
    ASSERT_TRUE(b1 > 0);
    ASSERT_TRUE(b2 > b1);
    ASSERT_TRUE(b3 > b2);
    return true;
}

bool test_generate_and_verify() {
    auto p = gen::tier1_small();
    p.output_dir = "test_synthetic_gen_out";

    // Clean any previous run.
    std::filesystem::remove_all(p.output_dir);

    auto r = gen::generate(p);
    ASSERT_TRUE(r.success);
    ASSERT_TRUE(r.experts_written == p.num_layers * p.experts_per_layer);
    ASSERT_TRUE(std::filesystem::exists(r.manifest_path));
    ASSERT_TRUE(std::filesystem::exists(r.container_path));

    // Parse manifest and verify with M1 storage.
    std::ifstream mf(r.manifest_path);
    std::stringstream buf; buf << mf.rdbuf();
    auto manifest = ModelManifest::from_json(buf.str());
    ASSERT_EQ((int)manifest.num_layers, (int)p.num_layers);
    ASSERT_EQ((int)manifest.experts_per_layer, (int)p.experts_per_layer);

    auto backend = create_storage_backend(r.container_path);
    ASSERT_TRUE(backend->is_open());

    // Read every expert and verify CRC.
    uint32_t checked = 0;
    for (const auto& layer : manifest.layers) {
        for (const auto& exp : layer.experts) {
            ExpertBuffer eb(exp.storage_length, exp.alignment);
            double lat = 0.0;
            ASSERT_TRUE(backend->read_expert_sync(exp, eb, lat));
            uint32_t crc = ChecksumUtil::compute_crc32(eb.data(), eb.size());
            ASSERT_EQ(crc, exp.checksum_crc32);
            checked++;
        }
    }
    mf.close();
    backend.reset();
    std::error_code ec;
    std::filesystem::remove_all(p.output_dir, ec);
    return true;
}

bool test_disk_space_check() {
    ASSERT_TRUE(gen::has_free_disk_space("C:\\", 1024));   // tiny
    ASSERT_TRUE(!gen::has_free_disk_space("Z:\\nonexistent_drive_xyz", 1));
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "      Running ASEMA synthetic generator tests       \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"tier_estimates",         test_tier_estimates},
        {"generate_and_verify",    test_generate_and_verify},
        {"disk_space_check",       test_disk_space_check},
    };
    int passed = 0, failed = 0;
    for (const auto& t : tests) {
        std::cout << "\n[TEST] " << t.n << "...\n";
        if (t.f()) { passed++; std::cout << "  [PASS] " << t.n << "\n"; }
        else       { failed++; std::cout << "  [FAIL] " << t.n << "\n"; }
    }
    std::cout << "\n  Results: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
