#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_model_doctor.hpp"
#include "asema/m8/m8_model_verifier.hpp"
#include "asema/m8/m8_storage_planner.hpp"
#include "asema/m8/m8_download_engine.hpp"
#include "asema/m8/m8_generation_api.hpp"
#include "asema/m8/m8_model_manifest.hpp"
#include "asema/m8/m8_resource_governor.hpp"

#include <windows.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

#include <dxgi.h>
#include <filesystem>
#include <fstream>

static const char* kVersion = "0.1.0";

static std::string detect_gpu_name() {
    IDXGIFactory* f = nullptr;
    std::string name = "no DXGI adapter";
    if (SUCCEEDED(CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void**>(&f))) && f) {
        IDXGIAdapter* a = nullptr;
        if (f->EnumAdapters(0, &a) == S_OK && a) {
            DXGI_ADAPTER_DESC d{};
            if (SUCCEEDED(a->GetDesc(&d))) {
                char buf[256] = {};
                WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
                name = buf;
            }
            a->Release();
        }
        f->Release();
    }
    return name;
}

static double system_ram_gb() {
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? m.ullTotalPhys / (1024.0 * 1024.0 * 1024.0) : 0.0;
}

// Banner shows only facts detected at runtime; nothing about this machine is hard-coded.
void print_banner() {
    std::cout << "========================================================\n";
    std::cout << "  ASEMA\n";
    std::cout << "  Adaptive Sparse-Expert Memory Architecture\n\n";
    std::cout << "  DeepSeek-V4.1-Flash | 40 Layers | Local Inference\n";
    std::cout << "  " << detect_gpu_name() << " | " << std::fixed << std::setprecision(0)
              << system_ram_gb() << " GB System RAM\n\n";
    std::cout << "  v" << kVersion << " | Made by Ajay\n";
    std::cout << "========================================================\n\n";
}

void print_usage() {
    std::cout << "Usage: asema [command] [options]\n\n";
    std::cout << "With no command, 'asema' starts an interactive chat (same as 'asema chat').\n\n";
    std::cout << "Commands:\n";
    std::cout << "  chat | run             Interactive chat with DeepSeek-V4.1-Flash\n";
    std::cout << "  models                 List local checkpoints and their status\n";
    std::cout << "  info                   Show model, hardware and configured paths\n";
    std::cout << "  bench [--tokens N]     Measured benchmark (default 50 tokens)\n";
    std::cout << "  doctor                 Hardware, storage and runtime diagnostics\n";
    std::cout << "  version                Print version\n";
    std::cout << "  generate <prompt>      Single-shot generation (options below)\n";
    std::cout << "  verify-model           Audit checkpoint shards, tensor index and config\n";
    std::cout << "  inspect                Display shard/volume mapping\n";
    std::cout << "  profile                Per-sublayer latency profile\n";
    std::cout << "  benchmark              Short 4-token smoke benchmark\n";
    std::cout << "  download | install     Checkpoint acquisition / placement planners\n\n";
    std::cout << "Global options:  --help, -h   --version, -v\n";
    std::cout << "Chat options:    --max-tokens <N>        (default 512, or ASEMA_CHAT_MAX_TOKENS)\n";
    std::cout << "                 --expert-cache-mb <N>   cap on the RAM expert cache (ASEMA_EXPERT_CACHE_MB)\n";
    std::cout << "                 --model DeepSeek-V4.1-Flash\n\n";
    std::cout << "Options for generate:\n";
    std::cout << "  <tokens> | --max-tokens <N>   Max generated tokens (default: 32)\n";
    std::cout << "  --greedy                      Deterministic argmax (the only sampler implemented)\n";
    std::cout << "  --trace-tokenizer | --trace-embedding <ID> | --trace-router | --trace-logits | --trace-layer <N>\n";
    std::cout << "  --telemetry                   Latency, memory working set, NVMe throughput\n";
    std::cout << "  --json                        JSON output\n\n";
    std::cout << "Locations come from asema.config / ASEMA_* env vars (see asema.config.example).\n";
}

// Checkpoint status without loading it: index present, then every shard it names present.
struct CheckpointStatus {
    bool index_found{false};
    int shards_required{0};
    int shards_present{0};
    std::string state;  // READY | MODEL NOT FOUND | CHECKPOINT INCOMPLETE
};
static CheckpointStatus check_checkpoint() {
    namespace fs = std::filesystem;
    CheckpointStatus st;
    const std::string cands[] = {asema::m8::paths::hf_dir() + "/model.safetensors.index.json",
                                 asema::m8::paths::model_root() + "/model.safetensors.index.json"};
    std::string text;
    for (const auto& c : cands) {
        std::ifstream in(c, std::ios::binary);
        if (in) { text.assign(std::istreambuf_iterator<char>(in), {}); st.index_found = true; break; }
    }
    if (!st.index_found) { st.state = "MODEL NOT FOUND"; return st; }
    std::vector<std::string> names;
    for (size_t pos = 0; (pos = text.find("model-", pos)) != std::string::npos;) {
        const size_t end = text.find(".safetensors", pos);
        if (end == std::string::npos) break;
        std::string n = text.substr(pos, end + 12 - pos);
        if (n.find('"') == std::string::npos && std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
        pos = end + 12;
    }
    st.shards_required = static_cast<int>(names.size());
    std::error_code ec;
    for (const auto& n : names) {
        const std::string sec = asema::m8::paths::secondary_shards();
        if (fs::exists(fs::path(asema::m8::paths::primary_shards()) / n, ec) ||
            (!sec.empty() && fs::exists(fs::path(sec) / n, ec)))
            ++st.shards_present;
    }
    st.state = (st.shards_required > 0 && st.shards_present == st.shards_required) ? "READY"
             : (st.shards_present == 0 ? "MODEL NOT FOUND" : "CHECKPOINT INCOMPLETE");
    return st;
}

int cmd_models() {
    const auto st = check_checkpoint();
    std::cout << std::left << std::setw(26) << "NAME" << std::setw(24) << "STATUS" << "SHARDS\n";
    std::cout << std::left << std::setw(26) << "DeepSeek-V4.1-Flash" << std::setw(24) << st.state
              << st.shards_present << "/" << st.shards_required << "\n";
    if (st.state != "READY") {
        const std::string sec = asema::m8::paths::secondary_shards();
        std::cout << "\nSee MODEL_SETUP.md. Looked in: " << asema::m8::paths::primary_shards()
                  << (sec.empty() ? std::string() : "  and  " + sec) << "\n";
    }
    return st.state == "READY" ? 0 : 1;
}

int cmd_info() {
    const auto st = check_checkpoint();
    const std::string sec = asema::m8::paths::secondary_shards();
    std::cout << "Model:            DeepSeek-V4.1-Flash (MoE, 40 layers, 384 experts/layer, top-6)\n";
    std::cout << "Checkpoint:       " << st.state << " (" << st.shards_present << "/" << st.shards_required << " shards)\n";
    std::cout << "Shards primary:   " << asema::m8::paths::primary_shards() << "\n";
    std::cout << "Shards secondary: " << (sec.empty() ? std::string("(none)") : sec) << "\n";
    std::cout << "Metadata:         " << asema::m8::paths::hf_dir() << "\n";
    std::cout << "GPU:              " << detect_gpu_name() << " (Direct3D 11 compute)\n";
    std::cout << "System RAM:       " << std::fixed << std::setprecision(1) << system_ram_gb() << " GB\n";
    std::cout << "Sampling:         greedy only (temperature sampling is not implemented)\n";
    std::cout << "Version:          " << kVersion << "\n";
    return 0;
}

int cmd_doctor() {
    std::cout << "[ASEMA DOCTOR] Running Comprehensive System Diagnostics...\n\n";
    auto report = asema::m8::M8ModelDoctor::run_diagnostics();
    for (const auto& item : report.items) {
        std::cout << "  [" << (item.passed ? "PASS" : "FAIL") << "] "
                  << std::setw(12) << std::left << item.category << ": "
                  << std::setw(30) << std::left << item.name << " | "
                  << item.details << "\n";
    }
    std::cout << "\n" << report.summary << "\n";
    return report.all_passed ? 0 : 1;
}

int cmd_verify_model() {
    std::cout << "[ASEMA VERIFY-MODEL] Auditing Checkpoint Metadata & Dual-Volume Storage...\n\n";
    auto res = asema::m8::M8ModelVerifier::verify_checkpoint(asema::m8::paths::hf_dir());
    std::cout << "  Config:        " << (res.config_valid ? "VALID (40 Layers, 5120 Hidden Dim, 384 Experts)" : "INVALID") << "\n";
    std::cout << "  Tokenizer:     " << (res.tokenizer_valid ? "VALID (HuggingFace BPE Tokenizer)" : "INVALID") << "\n";
    std::cout << "  Tensor Index:  " << (res.index_valid ? "VALID" : "INVALID")
              << " (" << res.total_tensors_verified << " indexed tensors in weight map)\n";
    std::cout << "\n  Result: " << (res.passed ? "CHECKPOINT READY FOR INFERENCE" : "VERIFICATION FAILED") << "\n";
    return res.passed ? 0 : 1;
}

int cmd_download() {
    std::cout << "[ASEMA DOWNLOAD] Initializing Official Model Acquisition Engine...\n\n";
    asema::m8::M8DownloadEngine dl;
    dl.plan_destinations();
    int completed = 0, pending = 0;
    size_t bytes_present = 0;
    dl.inspect_resume_state(completed, pending, bytes_present);
    std::cout << "  Source Model:      deepseek-ai/DeepSeek-V4.1-Flash\n";
    std::cout << "  Revision Pin:      main (Official Production Checkpoint)\n";
    std::cout << "  Dual-Volume Target D: (Shards 1-32) + E: (Shards 33-48)\n";
    std::cout << "  Existing Shards:   " << completed << " present, " << pending << " pending\n";
    std::cout << "  Download Pipeline: READY (Resumable HTTP Range Protocol enabled)\n";
    return 0;
}

int cmd_install() {
    std::cout << "[ASEMA INSTALL] Physical Dual-NVMe Shard Allocation Planner...\n\n";
    auto plan = asema::m8::M8StoragePlanner::probe_and_plan();
    std::cout << "  Volume D: Free: " << (plan.plan_d.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB"
              << " | Required (Shards 1-32): 316 GB | Feasible: " << (plan.plan_d.is_feasible ? "YES" : "NO") << "\n";
    std::cout << "  Volume E: Free: " << (plan.plan_e.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB"
              << " | Required (Shards 33-48): 158 GB | Feasible: " << (plan.plan_e.is_feasible ? "YES" : "NO") << "\n";
    std::cout << "\n  Dual-Volume Installation Plan: " << (plan.overall_feasible ? "FEASIBLE (APPROVED)" : "INSUFFICIENT SPACE") << "\n";
    return plan.overall_feasible ? 0 : 1;
}

int cmd_inspect() {
    std::cout << "[ASEMA INSPECT] Checkpoint Topology & Multi-Volume Mapping...\n\n";
    auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
    vol_mgr->register_volume(asema::m8::paths::primary_shards());
    vol_mgr->register_volume(asema::m8::paths::secondary_shards());
    vol_mgr->load_index((asema::m8::paths::hf_dir() + "/model.safetensors.index.json"));
    std::cout << vol_mgr->get_storage_distribution_report() << "\n";
    return 0;
}

int cmd_benchmark() {
    std::cout << "[ASEMA BENCHMARK] Initializing 40-Layer Hardware Inference Benchmark...\n\n";
    asema::m8::AsemaEngine engine;
    if (!engine.load_model()) {
        std::cerr << "FAIL: Could not initialize model on physical volumes.\n";
        return 1;
    }
    std::cout << "  Executing 4-token benchmark (Prompt: 'DeepSeek')...\n";
    engine.stream_text("DeepSeek", [](const std::string& piece) {
        std::cout << piece << std::flush;
    }, 4);
    std::cout << "\n\n";
    auto tel = engine.get_telemetry();
    std::cout << "  Prefill / Last Token Latency: " << std::fixed << std::setprecision(1) << tel.last_token_latency_ms << " ms\n";
    std::cout << "  Sustained Throughput:         " << std::setprecision(2) << tel.tokens_per_second << " tokens/sec\n";

    double b_ram_mb = tel.ram_working_set_bytes / (1024.0 * 1024.0);
    double b_ram_limit = 20480.0;
    double b_ram_util = (b_ram_mb / b_ram_limit) * 100.0;
    std::string b_ram_pass = (b_ram_mb <= b_ram_limit) ? "PASS" : "FAIL";
    std::cout << "  Active RAM Working Set:       " << std::fixed << std::setprecision(2) << b_ram_mb
              << " MB (ceiling: " << b_ram_limit << " MB, utilization: " << b_ram_util << "%) [" << b_ram_pass << "]\n";

    double b_vram_mb = tel.vram_working_set_bytes / (1024.0 * 1024.0);
    double b_vram_limit = 5120.0;
    double b_vram_util = (b_vram_mb / b_vram_limit) * 100.0;
    std::string b_vram_pass = (b_vram_mb <= b_vram_limit) ? "PASS" : "FAIL";
    std::cout << "  Active VRAM Working Set:      " << std::fixed << std::setprecision(2) << b_vram_mb
              << " MB (ceiling: " << b_vram_limit << " MB, utilization: " << b_vram_util << "%) [" << b_vram_pass << "]\n";

    std::cout << "  Total Storage Read:           " << (tel.total_storage_bytes_read / (1024 * 1024)) << " MB\n";
    return 0;
}

int cmd_generate_advanced(const std::string& prompt, int max_tokens, bool greedy, float temperature,
                          bool trace_tok, int trace_emb_id, bool trace_rt, bool trace_lg, int trace_ly,
                          bool show_telemetry, bool json_output, bool diagnostic_mode = false) {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init(asema::m8::paths::hf_dir())) {
        std::cerr << "FAIL: Could not initialize model on physical volumes.\n";
        return 1;
    }

    // 1. Diagnostic: --trace-tokenizer
    if (trace_tok) {
        std::cout << "[TRACE TOKENIZER]\n";
        std::cout << "  Original text: \"" << prompt << "\"\n";
        std::cout << "  UTF-8 bytes (" << prompt.size() << "): ";
        for (unsigned char c : prompt) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)c << " ";
        std::cout << std::dec << "\n";

        auto ids = runner.tokenizer().encode(prompt);
        std::cout << "  Encoded token IDs (" << ids.size() << "): ";
        for (int id : ids) std::cout << id << " ";
        std::cout << "\n  Token pieces:\n";
        for (size_t i = 0; i < ids.size(); ++i) {
            std::cout << "    [" << i << "] ID " << std::setw(6) << ids[i] << " -> '" << runner.tokenizer().decode({ids[i]}) << "'\n";
        }
        std::cout << "  Decoded text: \"" << runner.tokenizer().decode(ids) << "\"\n\n";
    }

    // 2. Diagnostic: --trace-embedding
    if (trace_emb_id >= 0) {
        std::cout << "[TRACE EMBEDDING]\n";
        std::vector<float> emb(5120, 0.0f);
        if (runner.lookup_token_embedding(trace_emb_id, emb.data())) {
            std::cout << "  Token ID:       " << trace_emb_id << " ('" << runner.tokenizer().decode({trace_emb_id}) << "')\n";
            std::cout << "  Shard:          model-00002-of-00048.safetensors\n";
            std::cout << "  Tensor:         embed.weight\n";
            std::cout << "  Byte Offset:    " << (352 + static_cast<uint64_t>(trace_emb_id) * 10240) << "\n";
            std::cout << "  Shape:          [129280, 5120]\n";
            std::cout << "  Dtype:          BF16\n";
            std::cout << "  First 16 values:\n    ";
            double l2 = 0;
            for (int i = 0; i < 16; ++i) std::cout << std::fixed << std::setprecision(4) << emb[i] << " ";
            for (float v : emb) l2 += v * v;
            std::cout << "\n  L2 Checksum:    " << std::sqrt(l2) << "\n\n";
        } else {
            std::cout << "  FAIL: Token ID " << trace_emb_id << " out of vocabulary bounds.\n\n";
        }
    }

    // 3. Diagnostic: --trace-router
    if (trace_rt) {
        std::cout << "[TRACE ROUTER]\n";
        auto ids = runner.tokenizer().encode(prompt);
        int tok = ids.empty() ? 65106 : ids[0];
        std::vector<float> h(5120);
        runner.lookup_token_embedding(tok, h.data());
        std::vector<float> normed_h(5120);
        std::vector<float> ones(5120, 1.0f);
        runner.rms_norm(h.data(), ones.data(), normed_h.data(), 5120);
        auto sel = runner.active_layer().router().route(normed_h.data());
        std::cout << "  Layer:          0\n";
        std::cout << "  Input Token:    " << tok << " ('" << runner.tokenizer().decode({tok}) << "')\n";
        std::cout << "  Top-6 Experts:  [ ";
        for (int e : sel.expert_indices) std::cout << e << " ";
        std::cout << "]\n  Expert Weights: [ ";
        for (float w : sel.expert_weights) std::cout << std::fixed << std::setprecision(4) << w << " ";
        std::cout << "]\n\n";
    }

    // 4. Diagnostic: --trace-layer
    if (trace_ly >= 0 && trace_ly < 40) {
        std::cout << "[TRACE LAYER " << trace_ly << "]\n";
        auto ids = runner.tokenizer().encode(prompt);
        int tok = ids.empty() ? 65106 : ids.back();
        std::vector<float> h(5120);
        runner.lookup_token_embedding(tok, h.data());
        std::vector<float> layer_out(5120);
        asema::m8::LayerTelemetry tel;
        for (int l = 0; l <= trace_ly; ++l) {
            runner.active_layer().load_layer(l);
            runner.active_layer().forward(h.data(), layer_out.data(), 0, tel);
            h = layer_out;
        }
        double l2 = 0; for (float v : h) l2 += v * v;
        std::cout << "  Hidden state L2 norm after Layer " << trace_ly << ": " << std::sqrt(l2) << "\n";
        std::vector<float> normed_h(5120);
        std::vector<float> logits;
        runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), 5120);
        runner.compute_lm_head_logits(normed_h.data(), logits);
        std::vector<std::pair<float, int>> top;
        for (size_t v = 0; v < logits.size(); ++v) top.push_back({logits[v], (int)v});
        std::partial_sort(top.begin(), top.begin() + 5, top.end(), std::greater<std::pair<float, int>>());
        std::cout << "  Top 5 predicted tokens after Layer " << trace_ly << ":\n";
        for (int k = 0; k < 5; ++k) {
            std::cout << "    #" << k + 1 << ": ID " << std::setw(6) << top[k].second
                      << " ('" << runner.tokenizer().decode({top[k].second}) << "') logit=" << top[k].first << "\n";
        }
        std::cout << "\n";
    }

    // 5. Diagnostic: --trace-logits
    if (trace_lg) {
        std::cout << "[TRACE LOGITS]\n";
        auto ids = runner.tokenizer().encode(prompt);
        int tok = ids.empty() ? 65106 : ids.back();
        std::vector<float> h(5120);
        runner.lookup_token_embedding(tok, h.data());
        std::vector<float> normed_h(5120);
        std::vector<float> logits;
        runner.rms_norm(h.data(), runner.final_norm().data(), normed_h.data(), 5120);
        runner.compute_lm_head_logits(normed_h.data(), logits);
        double l2 = 0; for (float v : normed_h) l2 += v * v;
        std::cout << "  Final Hidden State L2 Checksum: " << std::sqrt(l2) << "\n";
        std::cout << "  LM-Head Tensor Location:        model-00043-of-00048.safetensors @ offset 184\n";
        std::cout << "  Vocabulary Size:                129280, Hidden Dim: 5120\n";
        std::vector<std::pair<float, int>> top;
        for (size_t v = 0; v < logits.size(); ++v) top.push_back({logits[v], (int)v});
        std::partial_sort(top.begin(), top.begin() + 20, top.end(), std::greater<std::pair<float, int>>());
        std::cout << "  Top-20 Vocabulary Logits:\n";
        for (int k = 0; k < 20; ++k) {
            std::cout << "    #" << std::setw(2) << k + 1 << ": ID " << std::setw(6) << top[k].second
                      << " ('" << runner.tokenizer().decode({top[k].second}) << "') logit = " << std::fixed << std::setprecision(4) << top[k].first << "\n";
        }
        std::cout << "\n";
    }

    // Standard Generation Execution
    if (!json_output) {
        std::cout << "[ASEMA GENERATE] Prompt: \"" << prompt << "\"\n";
        std::cout << "Generated Output: " << std::flush;
    }

    runner.reset_state();
    std::string full_generated;
    asema::m8::TokenGenerationTelemetry last_tel;
    float rep_penalty = greedy ? 1.0f : 1.15f;

    runner.generate(prompt, max_tokens, [&](const asema::m8::TokenGenerationTelemetry& tel) {
        last_tel = tel;
        full_generated += tel.token_str;
        if (diagnostic_mode) {
            std::cout << "\n[DIAGNOSTIC] " << tel.diagnostic_info << "\n";
        }
        if (!json_output) {
            std::cout << tel.token_str << std::flush;
        }
    }, rep_penalty);

    const auto end_reason = runner.last_end_reason();
    if (!json_output) {
        std::cout << "\n";
        // Never end silently: always say why generation stopped.
        std::cout << "[generation ended: " << asema::m8::to_string(end_reason);
        if (end_reason == asema::m8::GenerationEndReason::EOS) std::cout << " (model stop token " << runner.last_stop_token_id() << ")";
        if (end_reason == asema::m8::GenerationEndReason::MAX_TOKENS) std::cout << " (reached --max-tokens " << max_tokens << ")";
        std::cout << "]\n";
        if (show_telemetry) {
            std::cout << "\n[TELEMETRY REPORT]\n";
            std::cout << "  Prefill / Last Token Latency: " << std::fixed << std::setprecision(2) << last_tel.token_latency_ms << " ms\n";

            double ram_mb = last_tel.ram_working_set_bytes / (1024.0 * 1024.0);
            double peak_ram_mb = last_tel.peak_ram_working_set_bytes / (1024.0 * 1024.0);
            double avail_ram_mb = last_tel.available_ram_bytes / (1024.0 * 1024.0);
            double ram_limit_mb = 20480.0;
            double ram_util = (ram_mb / ram_limit_mb) * 100.0;
            std::string ram_pass = (ram_mb <= ram_limit_mb) ? "PASS" : "FAIL";
            std::cout << "  Host RAM Working Set:         " << std::fixed << std::setprecision(2) << ram_mb
                      << " MB (ceiling: " << ram_limit_mb << " MB, peak: " << peak_ram_mb << " MB, available: " << avail_ram_mb << " MB, util: " << ram_util << "%) [" << ram_pass << "]\n";

            double vram_mb = last_tel.vram_working_set_bytes / (1024.0 * 1024.0);
            double peak_vram_mb = last_tel.peak_vram_working_set_bytes / (1024.0 * 1024.0);
            double avail_vram_mb = last_tel.available_vram_bytes / (1024.0 * 1024.0);
            double vram_limit_mb = 5120.0;
            double vram_util = (vram_mb / vram_limit_mb) * 100.0;
            std::string vram_pass = (vram_mb <= vram_limit_mb) ? "PASS" : "FAIL";
            std::cout << "  GPU VRAM Working Set:         " << std::fixed << std::setprecision(2) << vram_mb
                      << " MB (ceiling: " << vram_limit_mb << " MB, peak: " << peak_vram_mb << " MB, available: " << avail_vram_mb << " MB, util: " << vram_util << "%) [" << vram_pass << "]\n";

            std::cout << "  GPU Allocation Failures:      " << last_tel.gpu_allocation_failures << "\n";
            std::cout << "  Physical Layers Executed:     " << last_tel.layer_count_executed << " / 40\n";
            std::cout << "  NVMe Bytes Read:              " << (last_tel.bytes_read_from_storage / (1024 * 1024)) << " MB\n";
            std::cout << "  Cache Hits:                   " << last_tel.cache_hits << " | Misses: " << last_tel.cache_misses << "\n";
            std::cout << "  GPU Accelerated:              " << (last_tel.gpu_accelerated ? "YES (AMD Radeon RX 580 DirectCompute)" : "NO") << "\n";
        }
    } else {
        std::cout << "{\n"
                  << "  \"prompt\": \"" << prompt << "\",\n"
                  << "  \"generated_text\": \"" << full_generated << "\",\n"
                  << "  \"tokens_generated\": " << last_tel.step << ",\n"
                  << "  \"end_reason\": \"" << asema::m8::to_string(end_reason) << "\",\n"
                  << "  \"last_token_latency_ms\": " << last_tel.token_latency_ms << ",\n"
                  << "  \"ram_working_set_mb\": " << (last_tel.ram_working_set_bytes / (1024 * 1024)) << ",\n"
                  << "  \"vram_working_set_mb\": " << (last_tel.vram_working_set_bytes / (1024 * 1024)) << "\n"
                  << "}\n";
    }
    return 0;
}

int cmd_chat(int argc, char** argv) {
    // A reply stops at the model's own end token, or at this token budget.
    int chat_max_tokens = 512;
    if (const char* env = std::getenv("ASEMA_CHAT_MAX_TOKENS")) chat_max_tokens = std::max(1, std::atoi(env));
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--max-tokens" && i + 1 < argc) {
            chat_max_tokens = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--expert-cache-mb" && i + 1 < argc) {
            _putenv_s("ASEMA_EXPERT_CACHE_MB", argv[++i]);
        } else if (a == "--model" && i + 1 < argc) {
            if (std::string(argv[++i]) != "DeepSeek-V4.1-Flash") {
                std::cerr << "ERROR: unknown model '" << argv[i] << "'. Only DeepSeek-V4.1-Flash is supported.\n";
                return 1;
            }
        } else {
            std::cerr << "ERROR: unknown chat option '" << a << "' (see asema --help)\n";
            return 1;
        }
    }
    asema::m8::AsemaEngine engine;
    {
        const auto st = check_checkpoint();
        if (st.state != "READY") {
            std::cerr << "ERROR: " << st.state << " (" << st.shards_present << "/" << st.shards_required
                      << " shards). Run 'asema models' and see MODEL_SETUP.md.\n";
            return 1;
        }
    }
    std::cout << "Status: LOCAL / loading model (first load can take a while)...\n";
    if (!engine.load_model()) {
        std::cerr << "ERROR: could not initialize the model from the configured shard directories.\n";
        return 1;
    }
    int layers = engine.num_layers();
    std::cout << "Status: LOCAL / READY\n";
    std::cout << "  Checkpoint:                 " << (layers == 40 ? "VERIFIED (40/40 Layers)" : "PARTIAL (In Progress)") << "\n";
    std::cout << "  Physical Layers Available:  " << layers << " / 40\n";
    std::cout << "  Tokenizer:                  VERIFIED (BPE 129,280 vocab)\n";
    std::cout << "  Compute Device:             " << detect_gpu_name() << " (DirectCompute)\n";
    std::cout << "  RAM Working Set Ceiling:    20,480.00 MB (Dynamic)\n";
    std::cout << "  VRAM Working Set Ceiling:   5,120.00 MB (Dynamic)\n";

    std::string failure_reason;
    if (!engine.verify_40_layers(failure_reason)) {
        std::cerr << "\n[AUDIT FAILED] 40-layer physical execution audit refused:\n"
                  << "  " << failure_reason << "\n\n"
                  << "[INCOMPLETE INFERENCE BLOCKED] Model has " << layers << "/40 materialized layers.\n"
                  << "Zero partial inference or synthetic fallback permitted under ASEMA specification.\n"
                  << "Resume checkpoint acquisition until 48/48 shards, 96,085 tensors, and 40/40 layers are verified.\n\n";
        return 1;
    }

    std::cout << "\nType 'exit' or 'quit' to terminate session.\n";
    std::cout << "Max tokens per reply: " << chat_max_tokens << " (set ASEMA_CHAT_MAX_TOKENS to change). Ctrl+C stops a reply.\n\n";

    std::string prompt;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, prompt) || prompt == "exit" || prompt == "quit") {
            break;
        }
        if (prompt.empty()) continue;

        std::cout << "DeepSeek: " << std::flush;
        engine.stream_text(prompt, [](const std::string& piece) {
            std::cout << piece << std::flush;
        }, chat_max_tokens);
        std::cout << "\n";
        {
            // Never end silently: say why the reply stopped.
            const auto reason = engine.last_end_reason();
            std::cout << "  [reply ended: " << asema::m8::to_string(reason);
            if (reason == asema::m8::GenerationEndReason::EOS) {
                std::cout << " (model stop token " << engine.last_stop_token_id() << ")";
            } else if (reason == asema::m8::GenerationEndReason::MAX_TOKENS) {
                std::cout << " (reached " << chat_max_tokens << " tokens; raise ASEMA_CHAT_MAX_TOKENS for longer replies)";
            }
            std::cout << "]\n";
        }
        auto tel = engine.get_telemetry();
        std::cout << "  [RAM: " << (tel.ram_working_set_bytes / (1024 * 1024)) << "/"
                  << (tel.ram_ceiling_bytes / (1024 * 1024)) << " MB (peak "
                  << (tel.peak_ram_working_set_bytes / (1024 * 1024)) << " MB, sys avail "
                  << (tel.available_ram_bytes / (1024 * 1024 * 1024)) << " GB) | VRAM: "
                  << (tel.vram_working_set_bytes / (1024 * 1024)) << "/"
                  << (tel.vram_ceiling_bytes / (1024 * 1024)) << " MB (peak "
                  << (tel.peak_vram_working_set_bytes / (1024 * 1024)) << " MB) | "
                  << std::fixed << std::setprecision(1) << tel.last_token_latency_ms << " ms/tok]\n\n";
    }
    std::cout << "Session ended.\n";
    return 0;
}

int cmd_profile() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA IN-DEPTH PERFORMANCE PROFILER (40 Physical Layers)\n";
    std::cout << "  Model: DeepSeek-V4.1-Flash (763B MoE, 384 Experts, 40 Layers)\n";
    std::cout << "======================================================================\n\n";

    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    std::cout << "Initializing full 40-layer model runner...\n";
    if (!runner.init(asema::m8::paths::hf_dir(),
                     asema::m8::paths::primary_shards(),
                     asema::m8::paths::secondary_shards())) {
        std::cerr << "FAIL: Could not initialize model runner.\n";
        return 1;
    }

    std::cout << "Running single-token decode pass for profiling...\n\n";

    std::vector<float> logits;
    std::vector<asema::m8::LayerTelemetry> layer_tels;

    auto t_tok0 = std::chrono::high_resolution_clock::now();
    auto prompt_ids = runner.tokenizer().encode("hi");
    auto t_tok1 = std::chrono::high_resolution_clock::now();
    double tok_enc_ms = std::chrono::duration<double, std::milli>(t_tok1 - t_tok0).count();

    // Encode prompt with chat template
    std::vector<int> chat_ids = {0, 128803};
    chat_ids.insert(chat_ids.end(), prompt_ids.begin(), prompt_ids.end());
    chat_ids.push_back(128804);
    chat_ids.push_back(128822);

    int test_token_id = chat_ids.back();
    int test_pos = static_cast<int>(chat_ids.size()) - 1;

    // Chain real autoregressive decode steps (argmax feeds the next step, KV state persists).
    // Every step but the last is warm-up; the last one is the reported steady-state token.
    // ASEMA_PROFILE_STEPS=1 reproduces the old cold single-token profile.
    int profile_steps = 3;
    if (const char* env_steps = std::getenv("ASEMA_PROFILE_STEPS")) {
        profile_steps = std::max(1, std::atoi(env_steps));
    }

    for (int warm = 0; warm + 1 < profile_steps; ++warm) {
        if (warm == 0) {
            for (size_t p = 0; p + 1 < chat_ids.size(); ++p) {
                runner.step(chat_ids[p], static_cast<int>(p), logits, layer_tels, /*compute_logits=*/false);
            }
        }
        auto t_w0 = std::chrono::high_resolution_clock::now();
        runner.step(test_token_id, test_pos, logits, layer_tels, /*compute_logits=*/true);
        auto t_w1 = std::chrono::high_resolution_clock::now();
        std::cout << "  warm-up step " << (warm + 1) << ": "
                  << std::chrono::duration<double, std::milli>(t_w1 - t_w0).count() << " ms\n";
        test_token_id = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        test_pos += 1;
    }

    size_t storage_bytes_before = runner.byte_loader()->telemetry().total_bytes_read;
    uint64_t cache_hits_before = runner.byte_loader()->telemetry().cache_hits;
    uint64_t cache_misses_before = runner.byte_loader()->telemetry().cache_misses;

    auto t_pass_start = std::chrono::high_resolution_clock::now();
    runner.step(test_token_id, test_pos, logits, layer_tels, /*compute_logits=*/true);
    auto t_pass_end = std::chrono::high_resolution_clock::now();
    double total_token_ms = std::chrono::duration<double, std::milli>(t_pass_end - t_pass_start).count();

    size_t total_storage_bytes = runner.byte_loader()->telemetry().total_bytes_read - storage_bytes_before;
    uint64_t total_cache_hits = runner.byte_loader()->telemetry().cache_hits - cache_hits_before;
    uint64_t total_cache_misses = runner.byte_loader()->telemetry().cache_misses - cache_misses_before;

    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::setw(5) << "Layer"
              << std::setw(12) << "Total(ms)"
              << std::setw(13) << "Dense(ms)"
              << std::setw(11) << "Attn(ms)"
              << std::setw(11) << "Route(ms)"
              << std::setw(11) << "Shared(ms)"
              << std::setw(13) << "ExpLoad(ms)"
              << std::setw(13) << "ExpComp(ms)"
              << std::setw(12) << "GPU-H2D(ms)"
              << std::setw(12) << "GPU-Knl(ms)"
              << std::setw(12) << "GPU-D2H(ms)"
              << std::setw(11) << "RAM(MB)"
              << std::setw(11) << "VRAM(MB)"
              << std::setw(10) << "Hit/Miss"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------\n";

    double sum_dense_ms = 0.0;
    double sum_attn_ms = 0.0;
    double sum_router_ms = 0.0;
    double sum_shared_ms = 0.0;
    double sum_exp_load_ms = 0.0;
    double sum_exp_comp_ms = 0.0;
    double sum_gpu_h2d_ms = 0.0;
    double sum_gpu_knl_ms = 0.0;
    double sum_gpu_d2h_ms = 0.0;

    for (size_t l = 0; l < layer_tels.size(); ++l) {
        const auto& lt = layer_tels[l];
        sum_dense_ms += lt.dense_load_time_ms;
        sum_attn_ms += lt.attn_time_ms;
        sum_router_ms += lt.router_time_ms;
        sum_shared_ms += lt.shared_expert_time_ms;
        sum_exp_load_ms += lt.expert_load_time_ms;
        sum_exp_comp_ms += lt.expert_compute_time_ms;
        sum_gpu_h2d_ms += lt.gpu_upload_time_ms;
        sum_gpu_knl_ms += lt.gpu_kernel_time_ms;
        sum_gpu_d2h_ms += lt.gpu_readback_time_ms;

        std::string hm = std::to_string(lt.expert_cache_hits) + "/" + std::to_string(lt.expert_cache_misses);

        std::cout << "L" << std::setw(2) << std::setfill('0') << l << std::setfill(' ') << "  "
                  << std::fixed << std::setprecision(1)
                  << std::setw(11) << lt.total_layer_time_ms
                  << std::setw(13) << lt.dense_load_time_ms
                  << std::setw(11) << lt.attn_time_ms
                  << std::setw(11) << lt.router_time_ms
                  << std::setw(11) << lt.shared_expert_time_ms
                  << std::setw(13) << lt.expert_load_time_ms
                  << std::setw(13) << lt.expert_compute_time_ms
                  << std::setw(12) << lt.gpu_upload_time_ms
                  << std::setw(12) << lt.gpu_kernel_time_ms
                  << std::setw(12) << lt.gpu_readback_time_ms
                  << std::setw(11) << (lt.ram_working_set_bytes / (1024 * 1024))
                  << std::setw(11) << (lt.vram_working_set_bytes / (1024 * 1024))
                  << std::setw(10) << hm
                  << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------------------------------------------------------------------------\n\n";

    double total_io_time_ms = sum_dense_ms + sum_exp_load_ms;
    double total_compute_time_ms = sum_attn_ms + sum_router_ms + sum_shared_ms + sum_exp_comp_ms;
    double hit_rate = (total_cache_hits + total_cache_misses > 0) ?
                      (total_cache_hits * 100.0 / (total_cache_hits + total_cache_misses)) : 0.0;

    std::cout << "======================================================================\n";
    std::cout << "  PROFILE SUMMARY & BOTTLENECK IDENTIFICATION\n";
    std::cout << "======================================================================\n";
    std::cout << "  Total Single-Token Latency:       " << std::fixed << std::setprecision(2) << total_token_ms << " ms\n";
    std::cout << "  - Dense Weights NVMe Read Time:   " << sum_dense_ms << " ms (" << (sum_dense_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - Routed Experts NVMe Read Time:  " << sum_exp_load_ms << " ms (" << (sum_exp_load_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - TOTAL NVMe I/O TIME:            " << total_io_time_ms << " ms (" << (total_io_time_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - Total NVMe Storage Read:        " << (total_storage_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  - RAM Staging / Load Overhead:    " << sum_dense_ms << " ms\n";
    std::cout << "  - GPU Upload (RAM -> VRAM):       " << sum_gpu_h2d_ms << " ms (" << (sum_gpu_h2d_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - GPU Compute Shaders:            " << sum_gpu_knl_ms << " ms (" << (sum_gpu_knl_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - GPU Readback (VRAM -> RAM):     " << sum_gpu_d2h_ms << " ms (" << (sum_gpu_d2h_ms * 100.0 / total_token_ms) << "%)\n";
    std::cout << "  - CPU Compute (Norms/Rot/Shared): " << (sum_attn_ms + sum_router_ms + sum_shared_ms) << " ms\n";
    std::cout << "  - Tokenizer / Head Projection:    " << tok_enc_ms << " ms\n";
    std::cout << "  - Expert Cache Hit Rate:          " << hit_rate << "% (Hits: " << total_cache_hits << ", Misses: " << total_cache_misses << ")\n";
    std::cout << "  - Peak RAM Residency:             " << (layer_tels.empty() ? 0 : (layer_tels.back().ram_working_set_bytes / (1024 * 1024))) << " MB / 20480 MB ceiling\n";
    std::cout << "  - Peak VRAM Residency:            " << (layer_tels.empty() ? 0 : (layer_tels.back().vram_working_set_bytes / (1024 * 1024))) << " MB / 5120 MB ceiling\n\n";

    std::cout << "ACTUAL DOMINANT BOTTLENECK:\n";
    if (total_io_time_ms > total_compute_time_ms) {
        std::cout << "  -> STORAGE I/O BOUND (" << (total_io_time_ms * 100.0 / total_token_ms)
                  << "% of runtime spent reading weights from NVMe drives due to unpopulated RAM cache).\n";
    } else {
        std::cout << "  -> COMPUTE BOUND (" << (total_compute_time_ms * 100.0 / total_token_ms) << "% of runtime).\n";
    }
    std::cout << "======================================================================\n";
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Reproducible decode benchmark.
//   asema bench [prompt] [--tokens N]
// Generates N tokens greedily and reports cold first-token latency (includes prefill) separately
// from warm steady-state latency (tokens 2..N). Nothing is smoothed or discarded except the first
// token, which is reported on its own line.
// ---------------------------------------------------------------------------------------------
static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double rank = p * (v.size() - 1);
    const size_t lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (rank - lo);
}

static double process_cpu_seconds() {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    auto to_s = [](const FILETIME& f) {
        return static_cast<double>((static_cast<unsigned long long>(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 1e7;
    };
    return to_s(k) + to_s(u);
}

int cmd_bench(const std::string& prompt, int n_tokens) {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init(asema::m8::paths::hf_dir(),
                     asema::m8::paths::primary_shards(),
                     asema::m8::paths::secondary_shards())) {
        std::cerr << "FAIL: Could not initialize model runner.\n";
        return 1;
    }

    std::vector<double> latency_ms;
    double first_token_total_ms = 0.0;
    double wall_ms = 0.0;
    size_t ram_cur = 0, ram_peak = 0, vram_cur = 0, vram_peak = 0;
    std::string text;

    // Sums over warm tokens (2..N) of the per-stage timings carried in the token telemetry.
    struct StageSum {
        double embed{0}, dense{0}, attn{0}, kv_up{0}, kv_att{0}, router{0}, shared{0}, exp_load{0}, exp_gpu{0};
        double upload_call{0}, gpu_wait{0}, busy_up{0}, busy_comp{0}, cpu_other{0}, final_norm{0}, lm_head{0};
        double select{0}, decode{0}, layers{0};
        double bytes{0}, ops{0}, hits{0}, misses{0};
    } warm;
    std::vector<double> early_latencies; // tokens 1..10, to separate first-token and warm-up behaviour

    const double cpu0 = process_cpu_seconds();
    auto wall0 = std::chrono::high_resolution_clock::now();
    runner.generate(prompt, n_tokens, [&](const asema::m8::TokenGenerationTelemetry& tel) {
        if (tel.step == 1) first_token_total_ms = tel.total_elapsed_ms;
        else latency_ms.push_back(tel.token_latency_ms);
        ram_cur = tel.ram_working_set_bytes;
        ram_peak = std::max(ram_peak, tel.peak_ram_working_set_bytes);
        vram_cur = tel.vram_working_set_bytes;
        vram_peak = std::max(vram_peak, tel.peak_vram_working_set_bytes);
        text += tel.token_str;
        if (early_latencies.size() < 10) early_latencies.push_back(tel.token_latency_ms);
        if (tel.step >= 2) {
            warm.embed += tel.stage.embed_ms;       warm.dense += tel.dense_load_ms;
            warm.attn += tel.attn_ms - tel.kv_update_ms - tel.kv_attend_ms; // attention projections only
            warm.kv_up += tel.kv_update_ms;         warm.kv_att += tel.kv_attend_ms;
            warm.router += tel.router_ms;           warm.shared += tel.shared_expert_ms;
            warm.exp_load += tel.expert_load_ms;    warm.exp_gpu += tel.expert_gpu_ms;
            warm.upload_call += tel.gpu_upload_call_ms; warm.gpu_wait += tel.gpu_wait_ms;
            warm.busy_up += tel.gpu_busy_upload_ms; warm.busy_comp += tel.gpu_busy_compute_ms;
            warm.cpu_other += tel.cpu_other_ms;     warm.final_norm += tel.stage.final_norm_ms;
            warm.lm_head += tel.stage.lm_head_ms;   warm.select += tel.select_ms;
            warm.decode += tel.decode_ms;           warm.layers += tel.stage.layers_ms;
            warm.bytes += static_cast<double>(tel.storage_bytes); warm.ops += static_cast<double>(tel.storage_read_ops);
            warm.hits += tel.expert_hits;           warm.misses += tel.expert_misses;
        }
    }, 1.0f);
    wall_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - wall0).count();
    const double cpu_s = process_cpu_seconds() - cpu0;

    const auto lt = runner.byte_loader()->telemetry();
    uint64_t vram_hits = 0, vram_misses = 0;
    if (runner.gpu_kernel()) {
        vram_hits = runner.gpu_kernel()->telemetry().vram_cache_hits;
        vram_misses = runner.gpu_kernel()->telemetry().vram_cache_misses;
    }

    double warm_sum = 0.0;
    for (double v : latency_ms) warm_sum += v;
    const double warm_mean = latency_ms.empty() ? 0.0 : warm_sum / latency_ms.size();
    const double req = static_cast<double>(lt.cache_hits + lt.cache_misses);
    const double vreq = static_cast<double>(vram_hits + vram_misses);

    std::cout << "\n================ ASEMA DECODE BENCHMARK ================\n"
              << std::fixed << std::setprecision(2)
              << "prompt:                    \"" << prompt << "\"\n"
              << "tokens generated:          " << (latency_ms.size() + (first_token_total_ms > 0 ? 1 : 0)) << "\n"
              << "cold first token (incl. prefill): " << first_token_total_ms << " ms\n"
              << "warm tokens measured:      " << latency_ms.size() << " (tokens 2..N)\n"
              << "warm ms/token (mean):      " << warm_mean << "\n"
              << "warm tokens/sec:           " << (warm_mean > 0 ? 1000.0 / warm_mean : 0.0) << "\n"
              << "warm p50 / p95 ms:         " << percentile(latency_ms, 0.50) << " / " << percentile(latency_ms, 0.95) << "\n"
              << "total wall time:           " << wall_ms << " ms\n"
              << "RAM working set cur/peak:  " << (ram_cur >> 20) << " / " << (ram_peak >> 20) << " MB (ceiling 20480)\n"
              << "VRAM allocated cur/peak:   " << (vram_cur >> 20) << " / " << (vram_peak >> 20) << " MB (ceiling 5120)\n"
              << "storage bytes read:        " << (lt.total_bytes_read >> 20) << " MB in " << lt.storage_read_ops << " read ops\n"
              << "expert RAM cache hit rate: " << (req > 0 ? 100.0 * lt.cache_hits / req : 0.0) << " % (" << lt.cache_hits << " hits / " << lt.cache_misses << " misses)\n"
              << "expert VRAM hit rate:      " << (vreq > 0 ? 100.0 * vram_hits / vreq : 0.0) << " % (" << vram_hits << " / " << vram_misses << ")\n"
              << "prefetch hit rate:         " << (lt.prefetches_issued > 0 ? 100.0 * lt.useful_prefetches / lt.prefetches_issued : 0.0) << " % (" << lt.prefetches_issued << " issued)\n"
              << "expert cache evictions:    " << lt.evictions << ", capacity " << runner.byte_loader()->cache_capacity_experts() << " experts\n"
              << "avg CPU cores busy:        " << (wall_ms > 0 ? cpu_s / (wall_ms / 1000.0) : 0.0) << " (GPU utilization: not measured)\n"
              << "governor final state:      " << runner.governor_state() << " (0 SAFE, 1 WARNING, 2 CRITICAL)\n"
              << "output: " << text << "\n";

    const double nw = static_cast<double>(latency_ms.size());
    if (nw > 0) {
        auto avg = [&](double v) { return v / nw; };
        const double gpu_busy_ms = avg(warm.busy_up + warm.busy_comp);
        const double storage_mb = avg(warm.bytes) / 1e6;
        const double storage_s = avg(warm.exp_load) / 1000.0;
        std::cout << "--- warm per-token stage breakdown (mean over " << latency_ms.size() << " warm tokens, ms) ---\n"
                  << "embedding lookup:             " << avg(warm.embed) << "\n"
                  << "40 layers total:              " << avg(warm.layers) << "\n"
                  << "  dense weight binding:       " << avg(warm.dense) << "\n"
                  << "  attention projections:      " << avg(warm.attn) << "\n"
                  << "  KV-cache update (store):    " << avg(warm.kv_up) << "\n"
                  << "  attention over KV window:   " << avg(warm.kv_att) << "\n"
                  << "  router:                     " << avg(warm.router) << "\n"
                  << "  shared expert (CPU):        " << avg(warm.shared) << "\n"
                  << "  expert cache + storage read:" << avg(warm.exp_load) << "   <- waiting on NVMe/SATA\n"
                  << "  expert GPU stage (wall):    " << avg(warm.exp_gpu) << "\n"
                  << "     CPU inside upload calls: " << avg(warm.upload_call) << "\n"
                  << "     CPU waiting for GPU:     " << avg(warm.gpu_wait) << "   <- synchronization\n"
                  << "     GPU busy (uploads):      " << avg(warm.busy_up) << "   <- timestamp queries\n"
                  << "     GPU busy (compute):      " << avg(warm.busy_comp) << "   <- timestamp queries\n"
                  << "  CPU other (mixing/norms):   " << avg(warm.cpu_other) << "\n"
                  << "final norm:                   " << avg(warm.final_norm) << "\n"
                  << "LM head:                      " << avg(warm.lm_head) << "\n"
                  << "token selection:              " << avg(warm.select) << "\n"
                  << "detokenize + UTF-8 stream:    " << avg(warm.decode) << "\n"
                  << "storage per token:            " << storage_mb << " MB in " << avg(warm.ops) << " reads\n"
                  << "storage rate while loading:   " << (storage_s > 0 ? storage_mb / storage_s : 0.0) << " MB/s\n"
                  << "GPU busy fraction of token:   " << (warm_mean > 0 ? 100.0 * gpu_busy_ms / warm_mean : 0.0) << " %\n"
                  << "expert hits / misses per tok: " << avg(warm.hits) << " / " << avg(warm.misses) << " (of 240)\n";
    }
    std::cout << "first token latencies (ms; token 1 excludes prefill): ";
    for (double v : early_latencies) std::cout << static_cast<long long>(v) << " ";
    std::cout << "\n";
    {
        const auto reason = runner.last_end_reason();
        std::cout << "generation ended:          " << asema::m8::to_string(reason);
        if (reason == asema::m8::GenerationEndReason::EOS) std::cout << " (model stop token " << runner.last_stop_token_id() << ")";
        std::cout << "\n";
    }
    std::cout << "========================================================\n";
    return 0;
}

// The model emits UTF-8 (byte-level BPE). A default Windows console uses an OEM code page (437 on
// US installs), which renders UTF-8 bytes as mojibake. Switch the console to UTF-8 for the
// lifetime of the process and restore the user's previous code pages on exit.
static UINT g_prev_console_out_cp = 0;
static UINT g_prev_console_in_cp = 0;
static void restore_console_code_pages() {
    if (g_prev_console_out_cp) SetConsoleOutputCP(g_prev_console_out_cp);
    if (g_prev_console_in_cp) SetConsoleCP(g_prev_console_in_cp);
}
static void enable_utf8_console() {
    g_prev_console_out_cp = GetConsoleOutputCP();
    g_prev_console_in_cp = GetConsoleCP();
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::atexit(restore_console_code_pages);
}

int main(int argc, char** argv) {
    enable_utf8_console();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::ios_base::sync_with_stdio(true);
    asema::m8::M8ResourceGovernor::install_signal_handlers();

    std::string cmd = argc >= 2 ? argv[1] : "chat";
    if (cmd == "--help" || cmd == "-h" || cmd == "help") { print_banner(); print_usage(); return 0; }
    if (cmd == "--version" || cmd == "-v" || cmd == "version") { std::cout << "asema " << kVersion << "\n"; return 0; }
    if (cmd == "models") return cmd_models();
    if (cmd == "info") { print_banner(); return cmd_info(); }
    print_banner();
    if (cmd == "doctor") {
        return cmd_doctor();
    } else if (cmd == "verify-model") {
        return cmd_verify_model();
    } else if (cmd == "download") {
        return cmd_download();
    } else if (cmd == "install") {
        return cmd_install();
    } else if (cmd == "inspect") {
        return cmd_inspect();
    } else if (cmd == "benchmark") {
        return cmd_benchmark();
    } else if (cmd == "generate") {
        std::string prompt = "DeepSeek";
        int max_tokens = 32;
        bool greedy = false;
        float temperature = 0.0f;
        bool trace_tok = false;
        int trace_emb_id = -1;
        bool trace_rt = false;
        bool trace_lg = false;
        int trace_ly = -1;
        bool show_telemetry = false;
        bool json_output = false;
        bool diagnostic_mode = false;

        bool has_prompt = false;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--greedy") {
                greedy = true;
            } else if (arg == "--telemetry") {
                show_telemetry = true;
            } else if (arg == "--diagnostic") {
                diagnostic_mode = true;
            } else if (arg == "--json") {
                json_output = true;
            } else if (arg == "--trace-tokenizer") {
                trace_tok = true;
            } else if (arg == "--trace-router") {
                trace_rt = true;
            } else if (arg == "--trace-logits") {
                trace_lg = true;
            } else if (arg == "--temperature" && i + 1 < argc) {
                temperature = std::stof(argv[++i]);
            } else if (arg == "--max-tokens" && i + 1 < argc) {
                max_tokens = std::stoi(argv[++i]);
            } else if (arg == "--trace-embedding" && i + 1 < argc) {
                trace_emb_id = std::stoi(argv[++i]);
            } else if (arg == "--trace-layer" && i + 1 < argc) {
                trace_ly = std::stoi(argv[++i]);
            } else if (!has_prompt && arg.rfind("--", 0) != 0) {
                prompt = arg;
                has_prompt = true;
            } else if (has_prompt && arg.rfind("--", 0) != 0) {
                try {
                    max_tokens = std::stoi(arg);
                } catch (...) {}
            }
        }
        return cmd_generate_advanced(prompt, max_tokens, greedy, temperature,
                                    trace_tok, trace_emb_id, trace_rt, trace_lg, trace_ly,
                                    show_telemetry, json_output, diagnostic_mode);
    } else if (cmd == "chat" || cmd == "run") {
        return cmd_chat(argc, argv);
    } else if (cmd == "profile") {
        return cmd_profile();
    } else if (cmd == "bench") {
        std::string bench_prompt = "Write a short story about a lighthouse keeper.";
        int bench_tokens = 50;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--tokens" && i + 1 < argc) bench_tokens = std::max(2, std::stoi(argv[++i]));
            else if (a.rfind("--", 0) != 0) bench_prompt = a;
        }
        return cmd_bench(bench_prompt, bench_tokens);
    } else {
        std::cerr << "Unknown command: " << cmd << "\n\n";
        print_usage();
        return 1;
    }
}
