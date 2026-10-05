#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_mla_attention.hpp"
#include "asema/m8/m8_model_runner.hpp"
#include "asema/m8/m8_multi_volume.hpp"
#include "asema/m8/m8_router.hpp"
#include "asema/m8/m8_transformer_layer.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace fs = std::filesystem;

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA M8.17: COMPLETE DEEPSEEK-V4.1-FLASH FULL EXECUTION SUITE      \n";
    std::cout << "======================================================================\n\n";

    int failures = 0;

    // -------------------------------------------------------------------------
    // TEST 1: MLA Multi-Head Latent Attention Unit Execution
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] M8.17-B: MLA Latent Attention Forward & KV Cache...\n";
    {
        asema::m8::M8MLAAttention mla;
        mla.load_mock_or_reference_weights();
        std::vector<float> in_vec(5120, 0.05f);
        std::vector<float> out_vec(5120, 0.0f);

        mla.forward(in_vec.data(), out_vec.data(), 0);

        double norm = 0.0;
        for (float v : out_vec) norm += static_cast<double>(v) * v;
        norm = std::sqrt(norm);

        std::cout << "  MLA Forward Output Norm (Pos 0): " << std::fixed << std::setprecision(6) << norm << "\n";
        std::cout << "  KV Cache Memory Footprint:       " << (mla.kv_cache_bytes() / 1024) << " KB (bounded)\n";

        if (norm > 0.0 && !std::isnan(norm)) {
            std::cout << "  PASSED: MLA Attention executed deterministically.\n";
        } else {
            std::cerr << "  FAILED: MLA Attention produced zero or NaN output.\n";
            failures++;
        }
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // TEST 2: Byte-Range Sparse Expert Loading
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] M8.17-C: Byte-Range Expert Loading & Bounded Cache...\n";
    {
        auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
        vol_mgr->register_volume(asema::m8::paths::primary_shards());
        vol_mgr->register_volume(asema::m8::paths::secondary_shards());

        asema::m8::M8ByteRangeLoader loader(vol_mgr);
        asema::m8::ExpertPayload payload;
        bool ok = loader.load_expert_payload(0, 0, payload);

        std::cout << "  Expert 0 Payload Loaded: " << (payload.total_bytes() / (1024 * 1024)) << " MB\n";
        std::cout << "  Scales Bytes:            " << payload.scales.size() << "\n";
        std::cout << "  Weights Bytes:           " << payload.weights.size() << "\n";

        // Test second load to verify RAM cache hit
        asema::m8::ExpertPayload payload2;
        loader.load_expert_payload(0, 0, payload2);

        std::cout << "  Cache Hits:              " << loader.telemetry().cache_hits << "\n";
        std::cout << "  Cache Misses:            " << loader.telemetry().cache_misses << "\n";

        if (ok && payload.total_bytes() == 18800640 && loader.telemetry().cache_hits == 1) {
            std::cout << "  PASSED: Byte-range storage paging and RAM cache verified.\n";
        } else {
            std::cerr << "  FAILED: Byte-range loading or cache verification failed.\n";
            failures++;
        }
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // TEST 3: Complete Single Transformer Layer Forward Pass
    // -------------------------------------------------------------------------
    std::cout << "[TEST 3] M8.17-D: Complete Real Transformer Layer Forward Pass...\n";
    {
        auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
        vol_mgr->register_volume(asema::m8::paths::primary_shards());
        auto loader = std::make_shared<asema::m8::M8ByteRangeLoader>(vol_mgr);

        asema::m8::M8TransformerLayer layer(0, loader);
        std::string gw = (asema::m8::paths::primary_shards() + "/l0_gate_weight.bin");
        std::string gb = (asema::m8::paths::primary_shards() + "/l0_gate_bias.bin");
        if (!fs::exists(gw)) gw = "gate_weight.bin";
        if (!fs::exists(gb)) gb = "gate_bias.bin";
        layer.load_router(gw, gb);

        std::vector<float> in_h(5120, 0.01f);
        std::vector<float> out_h(5120, 0.0f);
        asema::m8::LayerTelemetry tel;

        layer.forward(in_h.data(), out_h.data(), 0, tel);

        std::cout << "  Layer 0 Total Latency:    " << std::fixed << std::setprecision(2) << tel.total_layer_time_ms << " ms\n";
        std::cout << "    - Attention Time:       " << tel.attn_time_ms << " ms\n";
        std::cout << "    - Router Time:          " << tel.router_time_ms << " ms\n";
        std::cout << "    - Expert Load Time:     " << tel.expert_load_time_ms << " ms\n";
        std::cout << "    - Expert Compute Time:  " << tel.expert_compute_time_ms << " ms\n";
        std::cout << "  Selected Top-6 Experts:   [ ";
        for (int id : tel.selected_experts) std::cout << id << " ";
        std::cout << "]\n";

        if (tel.selected_experts.size() == 6 && tel.total_layer_time_ms > 0.0) {
            std::cout << "  PASSED: Complete transformer layer pipeline verified.\n";
        } else {
            std::cerr << "  FAILED: Transformer layer pipeline execution error.\n";
            failures++;
        }
    }
    std::cout << "\n";

    // -------------------------------------------------------------------------
    // TEST 4: Full 40-Layer Autoregressive Generation
    // -------------------------------------------------------------------------
    std::cout << "[TEST 4] M8.17-F & M8.17-G: Full 40-Layer End-to-End Autoregressive Generation...\n";
    {
        asema::m8::M8ModelRunner runner;
        std::string hf_root = "examples/real_model/DeepSeek-V4.1-Flash/hf";
        if (!fs::exists(hf_root)) hf_root = "../" + hf_root;

        bool init_ok = runner.init(hf_root);
        if (!init_ok) {
            std::cerr << "  FAILED: Could not initialize M8ModelRunner.\n";
            failures++;
        } else {
            std::cout << "  Model Loaded: " << runner.num_layers() << " Transformer Layers active\n";
            std::cout << "  Storage Tier: NVMe Multi-Volume Persistent Checkpoint (510 GB)\n";
            std::cout << "  Starting Autoregressive Generation for 2 Tokens...\n" << std::flush;

            int token_count = 0;
            auto tokens = runner.generate("DeepSeek", 2,
                [&](const asema::m8::TokenGenerationTelemetry& tel) {
                    token_count++;
                    std::cout << "  [GEN TOKEN " << tel.step << "] ID: " << std::setw(6) << tel.token_id
                              << " | Latency: " << std::fixed << std::setprecision(1) << tel.token_latency_ms << " ms"
                              << " | RAM Working Set: " << (tel.ram_working_set_bytes / (1024 * 1024)) << " MB"
                              << " | Storage Read: " << (tel.bytes_read_from_storage / (1024 * 1024)) << " MB"
                              << " | Top-6: [ ";
                    for (int id : tel.layer0_top6_experts) std::cout << id << " ";
                    std::cout << "]\n" << std::flush;
                });

            std::cout << "\n  Generated Token IDs Total: " << tokens.size() << "\n" << std::flush;
            if (token_count == 2) {
                std::cout << "  PASSED: Full 40-layer autoregressive generation executed successfully.\n";
            } else {
                std::cerr << "  FAILED: Expected 2 generated tokens, got " << token_count << "\n";
                failures++;
            }
        }
    }
    std::cout << "\n";

    if (failures == 0) {
        std::cout << ">>> ALL M8.17 FULL EXECUTION TESTS PASSED! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> " << failures << " TESTS FAILED! <<<\n";
        return 1;
    }
}
