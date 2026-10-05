// ASEMA v0.2 — M8 adapter tests.
// Verifies the model adapter parses the official DeepSeek-V4.1-Flash
// config.json + model.safetensors.index.json correctly, and that the
// tokenizer round-trips simple inputs.

#include "asema/m8/m8_paths.hpp"
#include "../include/asema/m8/m8_model_adapter.hpp"

#include <cassert>
#include <iostream>
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

const std::string kHfRoot =
    asema::m8::paths::hf_dir();

bool test_load_config() {
    auto adapter = m8::make_deepseek_v41_adapter();
    bool ok = adapter->load(kHfRoot);
    ASSERT_TRUE(ok);
    const auto& cfg = adapter->config();
    ASSERT_EQ(cfg.vocab_size, 129280);
    ASSERT_EQ(cfg.hidden_size, 5120);
    ASSERT_EQ(cfg.num_hidden_layers, 40);
    ASSERT_EQ(cfg.n_routed_experts, 384);
    ASSERT_EQ(cfg.n_shared_experts, 1);
    ASSERT_EQ(cfg.num_experts_per_tok, 6);
    ASSERT_EQ(cfg.moe_intermediate_size, 2304);
    ASSERT_TRUE(cfg.num_attention_heads == 64);
    ASSERT_TRUE(cfg.head_dim == 512);
    ASSERT_TRUE(cfg.qk_rope_head_dim == 64);
    return true;
}

bool test_tensor_inventory_size() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    // Per layer: 8 attn + 6 shared_expert + 6 norms + 6 HC + 6*384 routed
    //            = 28 + 2304 = 2332 tensors per text layer
    int total_routed = adapter->total_routed_expert_tensors();
    ASSERT_EQ(total_routed, 384 * 40 * 3); // w1, w3, w2 (no scales in count)
    int total_shared = adapter->total_shared_expert_tensors();
    ASSERT_EQ(total_shared, 1 * 40 * 3);
    return true;
}

bool test_group_layers() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    auto groups = adapter->group_layers();
    ASSERT_EQ((int)groups.size(), 40);
    // Layer 0 should have 384 routed expert tensors.
    int routed_in_layer0 = (int)groups[0].routed_expert_tensors.size();
    ASSERT_TRUE(routed_in_layer0 >= 384 * 3); // at least w1/w3/w2 of each expert
    return true;
}

bool test_expert_coord_mapping() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    auto c = adapter->routed_expert_coord(15, 42);
    ASSERT_EQ(c.kind, 0);
    ASSERT_EQ(c.layer_id, 15);
    ASSERT_EQ(c.expert_id, 42);

    auto s = adapter->shared_expert_coord(15);
    ASSERT_EQ(s.kind, 1);

    auto e = adapter->engram_coord(0);
    ASSERT_EQ(e.kind, 2);
    return true;
}

bool test_per_expert_byte_estimate() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    uint64_t bytes = adapter->bytes_per_routed_expert_fp8();
    // 3 tensors x 5120 x 2304 = 35 389 440 bytes FP8 ~= 33.75 MB
    ASSERT_EQ(bytes, (uint64_t)3 * 5120 * 2304);
    return true;
}

bool test_shard_lookup() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    std::string shard = adapter->shard_for_routed_expert(0, 0);
    ASSERT_TRUE(!shard.empty());
    ASSERT_TRUE(shard.find("model-") != std::string::npos);
    ASSERT_TRUE(shard.find(".safetensors") != std::string::npos);
    return true;
}

bool test_summary_json() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    std::string s = adapter->summary_json();
    // nlohmann::json::dump(2) inserts ": " (with space) after each key.
    ASSERT_TRUE(s.find("\"model_type\": \"deepseek_v41\"") != std::string::npos);
    ASSERT_TRUE(s.find("\"n_routed_experts\": 384") != std::string::npos);
    return true;
}

bool test_report_markdown() {
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    std::string r = adapter->report_markdown();
    ASSERT_TRUE(r.find("DeepSeek-V4.1-Flash") != std::string::npos);
    ASSERT_TRUE(r.find("384") != std::string::npos);
    ASSERT_TRUE(r.find("BLOCKED") != std::string::npos);
    return true;
}

bool test_tokenizer_roundtrip() {
    m8::Tokenizer tok;
    bool ok = tok.load(kHfRoot + "/tokenizer.json");
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tok.vocab_size() > 1000);
    // Single ASCII character round-trip.
    auto ids = tok.encode("Hi");
    ASSERT_TRUE(ids.size() >= 1);
    std::string dec = tok.decode(ids);
    ASSERT_EQ(dec, "Hi");
    return true;
}

bool test_no_disk_check() {
    // Verify the BLOCKED status is correctly reported.
    auto adapter = m8::make_deepseek_v41_adapter();
    ASSERT_TRUE(adapter->load(kHfRoot));
    std::string r = adapter->report_markdown();
    // The report should mention BLOCKED.
    ASSERT_TRUE(r.find("BLOCKED") != std::string::npos ||
                r.find("NOT downloaded") != std::string::npos);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "   ASEMA M8 DeepSeek-V4.1-Flash Adapter Tests     \n";
    std::cout << "====================================================\n";
    std::cout << "HF root: " << kHfRoot << "\n\n";

    if (!std::filesystem::exists(kHfRoot + "/config.json")) {
        std::cerr << "[SKIP] config.json not found at " << kHfRoot << "\n"
                  << "Run the download step from m8_forensics.md first.\n";
        return 1;
    }

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"load_config",                test_load_config},
        {"tensor_inventory_size",      test_tensor_inventory_size},
        {"group_layers",               test_group_layers},
        {"expert_coord_mapping",       test_expert_coord_mapping},
        {"per_expert_byte_estimate",   test_per_expert_byte_estimate},
        {"shard_lookup",               test_shard_lookup},
        {"summary_json",               test_summary_json},
        {"report_markdown",            test_report_markdown},
        {"tokenizer_roundtrip",        test_tokenizer_roundtrip},
        {"blocked_status_present",     test_no_disk_check},
    };

    int passed = 0, failed = 0;
    for (const auto& t : tests) {
        std::cout << "[TEST] " << t.n << "...\n";
        if (t.f()) { passed++; std::cout << "  [PASS] " << t.n << "\n"; }
        else       { failed++; std::cout << "  [FAIL] " << t.n << "\n"; }
    }
    std::cout << "\n  Results: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
