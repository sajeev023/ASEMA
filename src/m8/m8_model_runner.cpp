#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_model_runner.hpp"
#include "asema/m8/m8_gpu_mla_kernel.hpp"
#include "asema/m8/m8_resource_governor.hpp"
#include "asema/m8/m8_utf8.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <exception>
#include <future>
#include <iostream>
#include <unordered_map>
#include <immintrin.h>
#include <windows.h>
#include <psapi.h>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

static float dot_product_bf16_fp32(const uint16_t* b16_row, const float* x, int size) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= size - 32; i += 32) {
        __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i));
        __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 8));
        __m128i raw2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 16));
        __m128i raw3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 24));

        __m256i f32_0_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw0), 16);
        __m256i f32_1_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw1), 16);
        __m256i f32_2_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw2), 16);
        __m256i f32_3_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw3), 16);

        __m256 f32_0 = _mm256_castsi256_ps(f32_0_int);
        __m256 f32_1 = _mm256_castsi256_ps(f32_1_int);
        __m256 f32_2 = _mm256_castsi256_ps(f32_2_int);
        __m256 f32_3 = _mm256_castsi256_ps(f32_3_int);

        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        sum0 = _mm256_fmadd_ps(f32_0, x0, sum0);
        sum1 = _mm256_fmadd_ps(f32_1, x1, sum1);
        sum2 = _mm256_fmadd_ps(f32_2, x2, sum2);
        sum3 = _mm256_fmadd_ps(f32_3, x3, sum3);
    }

    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);

    for (; i < size; ++i) {
        uint32_t u = static_cast<uint32_t>(b16_row[i]) << 16;
        float val = *reinterpret_cast<float*>(&u);
        total += val * x[i];
    }
    return total;
}

struct M8ModelRunner::RealWeightsMapping {
    HANDLE h_embed_file{INVALID_HANDLE_VALUE};
    HANDLE m_embed_mapping{NULL};
    const uint8_t* embed_ptr{nullptr};
    uint64_t embed_size{0};

    HANDLE h_head_file{INVALID_HANDLE_VALUE};
    HANDLE m_head_mapping{NULL};
    const uint8_t* head_ptr{nullptr};
    uint64_t head_size{0};

    ~RealWeightsMapping() {
        if (embed_ptr) UnmapViewOfFile(embed_ptr);
        if (m_embed_mapping) CloseHandle(m_embed_mapping);
        if (h_embed_file != INVALID_HANDLE_VALUE) CloseHandle(h_embed_file);

        if (head_ptr) UnmapViewOfFile(head_ptr);
        if (m_head_mapping) CloseHandle(m_head_mapping);
        if (h_head_file != INVALID_HANDLE_VALUE) CloseHandle(h_head_file);
    }
};

M8ModelRunner::M8ModelRunner() {
    vol_mgr_ = std::make_shared<M8MultiVolumeManager>();
    byte_loader_ = std::make_shared<M8ByteRangeLoader>(vol_mgr_);
}

M8ModelRunner::~M8ModelRunner() = default;

void M8ModelRunner::rms_norm(const float* in, const float* weight, float* out, int size, float eps) const {
    double sq = 0.0;
    for (int i = 0; i < size; ++i) sq += static_cast<double>(in[i]) * in[i];
    float scale = 1.0f / std::sqrt(static_cast<float>(sq / size) + eps);
    for (int i = 0; i < size; ++i) out[i] = in[i] * scale * weight[i];
}

void M8ModelRunner::init_mock_or_reference_embeddings() {
    const int dim = 5120;
    final_norm_.resize(dim);
    if (has_real_lm_head_ && real_weights_ && real_weights_->head_ptr) {
        const uint16_t* norm_raw = reinterpret_cast<const uint16_t*>(real_weights_->head_ptr + 1323827384);
        for (int i = 0; i < dim; ++i) {
            uint32_t u = static_cast<uint32_t>(norm_raw[i]) << 16;
            final_norm_[i] = *reinterpret_cast<float*>(&u);
        }
    } else {
        final_norm_.assign(dim, 1.0f);
    }
}

void M8ModelRunner::set_cache_capacity_mb(size_t mb) {
    if (byte_loader_) {
        byte_loader_->set_cache_capacity_mb(mb);
    }
}

void M8ModelRunner::reset_state() {
    if (active_layer_) {
        active_layer_->reset_kv_cache();
    }
}

bool M8ModelRunner::set_gpu_acceleration(bool enable) {
    gpu_enabled_ = enable;
    if (enable) {
        if (!gpu_kernel_) {
            gpu_kernel_ = std::make_shared<M8GpuExpertKernel>();
            if (!gpu_kernel_->initialize()) {
                gpu_kernel_ = nullptr;
                gpu_enabled_ = false;
                return false;
            }
        }
        const char* cost_env = std::getenv("ASEMA_VRAM_COST_AWARE");
        const bool cost_aware = !(cost_env && std::string(cost_env) == "0");
        if (byte_loader_ && cost_aware) {
            // Keep experts that are slow to re-read (SATA) resident longer than ones on NVMe.
            // ASEMA_VRAM_COST_AWARE=0 turns this off (used for A/B measurements).
            auto loader = byte_loader_;
            gpu_kernel_->set_expert_cost_function([loader](int layer, int expert) {
                return loader->expert_read_cost(layer, expert);
            });
        }
        if (active_layer_) {
            active_layer_->set_gpu_kernel(gpu_kernel_);
        }
    } else {
        if (active_layer_) {
            active_layer_->set_gpu_kernel(nullptr);
        }
    }
    return true;
}

bool M8ModelRunner::set_gpu_mla_acceleration(bool enable) {
    gpu_mla_enabled_ = enable;
    if (enable) {
        if (!gpu_mla_kernel_) {
            gpu_mla_kernel_ = std::make_shared<M8GpuMlaKernel>();
            ID3D11Device* dev = gpu_kernel_ ? gpu_kernel_->device() : nullptr;
            ID3D11DeviceContext* ctx = gpu_kernel_ ? gpu_kernel_->context() : nullptr;
            if (!gpu_mla_kernel_->initialize(dev, ctx)) {
                gpu_mla_kernel_ = nullptr;
                gpu_mla_enabled_ = false;
                return false;
            }
        }
        if (active_layer_) {
            active_layer_->set_gpu_mla(gpu_mla_kernel_);
        }
    } else {
        if (active_layer_) {
            active_layer_->set_gpu_mla(nullptr);
        }
    }
    return true;
}

bool M8ModelRunner::init(const std::string& hf_root,
                         const std::string& vol_primary,
                         const std::string& vol_secondary) {
    vol_mgr_->register_volume(vol_primary);
    vol_mgr_->register_volume(vol_secondary);

    std::string index_file = hf_root + "/model.safetensors.index.json";
    if (!vol_mgr_->load_index(index_file)) {
        if (!vol_mgr_->load_index("../" + index_file)) {
            if (!vol_mgr_->load_index((asema::m8::paths::hf_dir() + "/model.safetensors.index.json"))) {
                if (!vol_mgr_->load_index("../examples/real_model/DeepSeek-V4.1-Flash/hf/model.safetensors.index.json")) {
                    vol_mgr_->load_index((asema::m8::paths::model_root() + "/model.safetensors.index.json"));
                }
            }
        }
    }

    std::string tok_path = hf_root + "/tokenizer.json";
    if (!fs::exists(tok_path)) tok_path = "../" + hf_root + "/tokenizer.json";
    if (!fs::exists(tok_path)) tok_path = (asema::m8::paths::hf_dir() + "/tokenizer.json");
    if (!fs::exists(tok_path)) tok_path = "../examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json";
    if (!fs::exists(tok_path)) tok_path = (asema::m8::paths::model_root() + "/tokenizer.json");
    tokenizer_.load(tok_path);

    // Initialize Real Checkpoint Weights (Shards 2 and 43)
    real_weights_ = std::make_unique<RealWeightsMapping>();

    std::string embed_candidate = vol_primary + "/model-00002-of-00048.safetensors";
    if (!fs::exists(embed_candidate)) embed_candidate = vol_secondary + "/model-00002-of-00048.safetensors";
    if (!fs::exists(embed_candidate)) embed_candidate = (asema::m8::paths::primary_shards() + "/model-00002-of-00048.safetensors");

    if (fs::exists(embed_candidate)) {
        real_weights_->h_embed_file = CreateFileA(embed_candidate.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (real_weights_->h_embed_file != INVALID_HANDLE_VALUE) {
            real_weights_->m_embed_mapping = CreateFileMappingA(real_weights_->h_embed_file, NULL, PAGE_READONLY, 0, 0, NULL);
            if (real_weights_->m_embed_mapping != NULL) {
                real_weights_->embed_ptr = static_cast<const uint8_t*>(MapViewOfFile(real_weights_->m_embed_mapping, FILE_MAP_READ, 0, 0, 0));
                if (real_weights_->embed_ptr != nullptr) {
                    LARGE_INTEGER sz{};
                    if (GetFileSizeEx(real_weights_->h_embed_file, &sz)) real_weights_->embed_size = static_cast<uint64_t>(sz.QuadPart);
                    has_real_embeddings_ = true;
                }
            }
        }
    }

    std::string head_candidate = vol_primary + "/model-00043-of-00048.safetensors";
    if (!fs::exists(head_candidate)) head_candidate = vol_secondary + "/model-00043-of-00048.safetensors";
    if (!fs::exists(head_candidate)) head_candidate = (asema::m8::paths::primary_shards() + "/model-00043-of-00048.safetensors");

    if (fs::exists(head_candidate)) {
        real_weights_->h_head_file = CreateFileA(head_candidate.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (real_weights_->h_head_file != INVALID_HANDLE_VALUE) {
            real_weights_->m_head_mapping = CreateFileMappingA(real_weights_->h_head_file, NULL, PAGE_READONLY, 0, 0, NULL);
            if (real_weights_->m_head_mapping != NULL) {
                real_weights_->head_ptr = static_cast<const uint8_t*>(MapViewOfFile(real_weights_->m_head_mapping, FILE_MAP_READ, 0, 0, 0));
                if (real_weights_->head_ptr != nullptr) {
                    LARGE_INTEGER sz{};
                    if (GetFileSizeEx(real_weights_->h_head_file, &sz)) real_weights_->head_size = static_cast<uint64_t>(sz.QuadPart);
                    has_real_lm_head_ = true;
                }
            }
        }
    }

    init_mock_or_reference_embeddings();

    int available = 0;
    while (available < 40 && vol_mgr_->locate_tensor("layers." + std::to_string(available) + ".attn_norm.weight").exists_on_disk) {
        available++;
    }
    total_layers_ = available;

    // Start the expert cache small; the governor grows it only while Windows reports healthy
    // available memory (see apply_resource_governor). ASEMA_EXPERT_CACHE_MB caps the growth,
    // ASEMA_EXPERT_CACHE_START_MB sets the initial size.
    size_t start_cache_mb = 2048;
    if (const char* env = std::getenv("ASEMA_EXPERT_CACHE_START_MB")) start_cache_mb = std::max<size_t>(256, std::strtoull(env, nullptr, 10));
    if (const char* env = std::getenv("ASEMA_EXPERT_CACHE_MB")) expert_cache_cap_mb_ = std::max<size_t>(256, std::strtoull(env, nullptr, 10));
    start_cache_mb = std::min(start_cache_mb, expert_cache_cap_mb_);
    set_cache_capacity_mb(start_cache_mb);

    active_layer_ = std::make_unique<M8TransformerLayer>(0, byte_loader_);
    active_layer_->load_layer(0);
    if (gpu_enabled_ && gpu_kernel_) active_layer_->set_gpu_kernel(gpu_kernel_);
    if (gpu_mla_enabled_ && gpu_mla_kernel_) active_layer_->set_gpu_mla(gpu_mla_kernel_);

    return true;
}

bool M8ModelRunner::verify_40_layers_executed(std::string& failure_reason) {
    if (!vol_mgr_) {
        failure_reason = "MultiVolumeManager not initialized";
        return false;
    }
    if (total_layers_ < 40) {
        failure_reason = "Incomplete checkpoint: only " + std::to_string(total_layers_) + "/40 physical layers available on NVMe.";
        return false;
    }
    for (int l = 0; l < 40; ++l) {
        std::string attn_tname = "layers." + std::to_string(l) + ".attn_norm.weight";
        auto loc_attn = vol_mgr_->locate_tensor(attn_tname);
        if (!loc_attn.exists_on_disk) {
            failure_reason = "Layer " + std::to_string(l) + " missing tensor " + attn_tname + " in shard " + loc_attn.shard_name;
            return false;
        }
        std::string ffn_tname = "layers." + std::to_string(l) + ".ffn_norm.weight";
        auto loc_ffn = vol_mgr_->locate_tensor(ffn_tname);
        if (!loc_ffn.exists_on_disk) {
            failure_reason = "Layer " + std::to_string(l) + " missing tensor " + ffn_tname + " in shard " + loc_ffn.shard_name;
            return false;
        }
    }
    return true;
}

bool M8ModelRunner::lookup_token_embedding(int token_id, float* out_emb) const {
    if (!has_real_embeddings_ || !real_weights_ || !real_weights_->embed_ptr) return false;
    if (token_id < 0 || token_id >= 129280) return false;
    const uint64_t offset = 352 + static_cast<uint64_t>(token_id) * 10240;
    const uint16_t* row = reinterpret_cast<const uint16_t*>(real_weights_->embed_ptr + offset);
    for (int i = 0; i < 5120; ++i) {
        uint32_t u = static_cast<uint32_t>(row[i]) << 16;
        out_emb[i] = *reinterpret_cast<float*>(&u);
    }
    return true;
}

void M8ModelRunner::compute_lm_head_logits(const float* normed_h, std::vector<float>& out_logits) {
    if (!has_real_lm_head_ || !real_weights_ || !real_weights_->head_ptr) {
        throw std::runtime_error("M8ModelRunner: Real LM head weights (model-00043-of-00048.safetensors) not mapped. Refusing synthetic inference.");
    }

    const int total_vocab = 129280;
    out_logits.resize(total_vocab);
    const uint16_t* head_rows = reinterpret_cast<const uint16_t*>(real_weights_->head_ptr + 184);

    const int num_threads = 8;
    int chunk_size = total_vocab / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? total_vocab : (t + 1) * chunk_size;
        futures.push_back(std::async(std::launch::async, [start, end, head_rows, normed_h, &out_logits]() {
            for (int v = start; v < end; ++v) {
                const uint16_t* row = head_rows + static_cast<size_t>(v) * 5120;
                out_logits[v] = dot_product_bf16_fp32(row, normed_h, 5120);
            }
        }));
    }
    for (auto& f : futures) {
        f.get();
    }
}

static size_t env_mb(const char* name, size_t fallback) {
    if (const char* v = std::getenv(name)) {
        char* end = nullptr;
        unsigned long long x = std::strtoull(v, &end, 10);
        if (end != v) return static_cast<size_t>(x);
    }
    return fallback;
}

void M8ModelRunner::trim_working_set() {
    vol_mgr_->safetensors_index().trim_mapped_pages();
    if (real_weights_) {
        // Embedding and LM-head views (VirtualUnlock on an unlocked range drops it from the working set).
        if (real_weights_->embed_ptr && real_weights_->embed_size) {
            VirtualUnlock(const_cast<uint8_t*>(real_weights_->embed_ptr), static_cast<SIZE_T>(real_weights_->embed_size));
        }
        if (real_weights_->head_ptr && real_weights_->head_size) {
            VirtualUnlock(const_cast<uint8_t*>(real_weights_->head_ptr), static_cast<SIZE_T>(real_weights_->head_size));
        }
    }
}

void M8ModelRunner::apply_resource_governor(bool allow_grow) {
    // Thresholds are on Windows *Available* memory (free + reclaimable standby cache).
    static const size_t warn_mb = env_mb("ASEMA_WARNING_AVAIL_MB", 7500);
    static const size_t crit_mb = env_mb("ASEMA_CRITICAL_AVAIL_MB", 4500);
    static const size_t grow_mb = env_mb("ASEMA_GROW_AVAIL_MB", 10000);
    constexpr size_t kMinExperts = 64;     // ~1.2 GB floor so decode never stalls completely
    constexpr size_t kGrowStep = 32;       // ~600 MB per decode step
    constexpr size_t kHysteresisMb = 500;  // extra headroom required to leave a worse state
    constexpr auto kActionInterval = std::chrono::seconds(2); // at most one shrink/trim per interval

    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (!GlobalMemoryStatusEx(&mem)) return;
    const size_t avail_mb = static_cast<size_t>(mem.ullAvailPhys / (1024 * 1024));

    const size_t cur = byte_loader_->cache_capacity_experts();
    const size_t cap = std::max<size_t>(kMinExperts, (expert_cache_cap_mb_ * 1024ULL * 1024ULL) / ExpertDimensions::TOTAL_EXPERT_BYTES);

    int state = 0;
    if (avail_mb < crit_mb || (governor_state_ == 2 && avail_mb < crit_mb + kHysteresisMb)) {
        state = 2;
    } else if (avail_mb < warn_mb || (governor_state_ >= 1 && avail_mb < warn_mb + kHysteresisMb)) {
        state = 1;
    }

    static std::chrono::steady_clock::time_point last_action{};
    const auto now = std::chrono::steady_clock::now();
    const bool may_act = (now - last_action) >= kActionInterval;
    size_t ws_before_mb = 0, ws_after_mb = 0;
    bool trimmed = false;

    if (state == 2) {
        byte_loader_->set_max_parallel_reads(2);
        if (may_act) {
            last_action = now;
            byte_loader_->set_cache_capacity_experts(std::max(kMinExperts, cur / 2));
            PROCESS_MEMORY_COUNTERS_EX pmc{};
            if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) ws_before_mb = pmc.WorkingSetSize >> 20;
            trim_working_set(); // give mapped checkpoint pages back to Windows' standby cache
            if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) ws_after_mb = pmc.WorkingSetSize >> 20;
            trimmed = true;
            Sleep(50); // give the Windows memory manager and desktop room to recover
        }
    } else if (state == 1) {
        byte_loader_->set_max_parallel_reads(4);
        if (may_act && cur > kMinExperts) {
            last_action = now;
            byte_loader_->set_cache_capacity_experts(std::max(kMinExperts, cur * 3 / 4));
        }
    } else {
        byte_loader_->set_max_parallel_reads(6);
        if (allow_grow && avail_mb > grow_mb && cur < cap) {
            byte_loader_->set_cache_capacity_experts(std::min(cap, cur + kGrowStep));
        }
    }

    if (state != governor_state_ || trimmed) {
        static const char* names[] = {"SAFE", "WARNING", "CRITICAL"};
        // SAFE<->WARNING flips are routine and would interleave with the streamed reply; print them
        // only with ASEMA_VERBOSE=1. Anything involving CRITICAL, or a working-set trim, always prints.
        static const bool verbose = std::getenv("ASEMA_VERBOSE") != nullptr;
        const bool notable = trimmed || state == 2 || governor_state_ == 2;
        if (verbose || notable) {
            std::cerr << "\n[ASEMA GOVERNOR] " << names[governor_state_] << " -> " << names[state]
                      << " (available " << avail_mb << " MB, expert cache "
                      << byte_loader_->cache_capacity_experts() << " experts";
            if (trimmed) std::cerr << ", working set " << ws_before_mb << " -> " << ws_after_mb << " MB after trim";
            std::cerr << ")\n";
        }
        governor_state_ = state;
    }
}

void M8ModelRunner::step(int token_id, int pos, std::vector<float>& out_logits, std::vector<LayerTelemetry>& layer_tels, bool compute_logits) {
    const int D = 5120;
    const int HC = 4;
    apply_resource_governor();
    StepTiming st;
    const auto t_step0 = std::chrono::high_resolution_clock::now();
    std::vector<float> emb(D);

    // 1. Embedding lookup
    if (!lookup_token_embedding(token_id, emb.data())) {
        throw std::runtime_error("M8ModelRunner: Real token embedding lookup failed for token " + std::to_string(token_id) + ". Refusing synthetic embedding.");
    }

    layer_tels.resize(total_layers_);

    // 2. Expand to HC copies for Hyper-Connections: [4, 5120]
    std::vector<float> h(HC * D);
    for (int j = 0; j < HC; ++j) {
        std::memcpy(h.data() + j * D, emb.data(), D * sizeof(float));
    }
    std::vector<float> pre_mix = {1.0f, 0.0f, 0.0f, 0.0f};

    std::vector<float> layer_out(HC * D);
    std::vector<float> next_pre_mix(HC);
    const auto t_layers0 = std::chrono::high_resolution_clock::now();
    st.embed_ms = std::chrono::duration<double, std::milli>(t_layers0 - t_step0).count();

    // 3. Pass sequentially through all 40 layers using single active layer working memory
    for (int l = 0; l < total_layers_; ++l) {
        if (M8ResourceGovernor::is_shutdown_requested()) {
            break;
        }
        apply_resource_governor(/*allow_grow=*/false); // react to memory pressure within a token, not only between tokens
        auto t_l_start = std::chrono::high_resolution_clock::now();
        M8TransformerLayer* cur_layer = active_layer_.get();
        auto t_dense_start = std::chrono::high_resolution_clock::now();
        const bool dense_was_resident = (cur_layer->loaded_layer_id() == l);
        cur_layer->load_layer(l);
        auto t_dense_end = std::chrono::high_resolution_clock::now();

        layer_tels[l].dense_load_time_ms = dense_was_resident
            ? 0.0
            : std::chrono::duration<double, std::milli>(t_dense_end - t_dense_start).count();
        layer_tels[l].dense_bytes_loaded = 0;

        if (cur_layer->has_hyper_connections()) {
            cur_layer->forward_hc(h.data(), layer_out.data(), pre_mix.data(), next_pre_mix.data(), pos, layer_tels[l]);
            h = layer_out;
            pre_mix = next_pre_mix;
        } else {
            std::vector<float> single_in(D);
            for (int d = 0; d < D; ++d) single_in[d] = h[d];
            std::vector<float> single_out(D);
            cur_layer->forward(single_in.data(), single_out.data(), pos, layer_tels[l]);
            for (int j = 0; j < HC; ++j) std::memcpy(h.data() + j * D, single_out.data(), D * sizeof(float));
        }

        auto t_l_end = std::chrono::high_resolution_clock::now();
        layer_tels[l].total_layer_time_ms = std::chrono::duration<double, std::milli>(t_l_end - t_l_start).count();

        PROCESS_MEMORY_COUNTERS_EX pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            layer_tels[l].ram_working_set_bytes = pmc.WorkingSetSize;
        }
        layer_tels[l].vram_working_set_bytes = (gpu_enabled_ && gpu_kernel_) ? gpu_kernel_->telemetry().vram_allocated_bytes : 0;
    }

    const auto t_layers1 = std::chrono::high_resolution_clock::now();
    st.layers_ms = std::chrono::duration<double, std::milli>(t_layers1 - t_layers0).count();

    if (compute_logits) {
        if (total_layers_ < 40) {
            throw std::runtime_error("M8ModelRunner: Incomplete checkpoint (" + std::to_string(total_layers_) +
                                     "/40 physical layers materialized). Full coherent inference strictly requires all 40 layers. Refusing to project partial residual stream into LM head.");
        }
        // 4. Collapse HC copies using final pre_mix: [4, 5120] -> [5120]
        std::vector<float> final_h(D, 0.0f);
        for (int j = 0; j < HC; ++j) {
            float m = pre_mix[j];
            const float* src = h.data() + j * D;
            for (int d = 0; d < D; ++d) {
                final_h[d] += m * src[d];
            }
        }

        // 5. Final normalization
        std::vector<float> normed_h(D);
        rms_norm(final_h.data(), final_norm_.data(), normed_h.data(), D);
        const auto t_head0 = std::chrono::high_resolution_clock::now();
        st.final_norm_ms = std::chrono::duration<double, std::milli>(t_head0 - t_layers1).count();

        // 6. Output logits projection
        compute_lm_head_logits(normed_h.data(), out_logits);
        st.lm_head_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_head0).count();
    }
    last_step_timing_ = st;
}

std::vector<int> M8ModelRunner::generate(const std::string& prompt, int max_new_tokens,
                                         std::function<void(const TokenGenerationTelemetry&)> on_token,
                                         float repetition_penalty) {
    // Generation must never end silently: record why it ended. If an exception escapes, the guard
    // marks the reason ERROR (the exception itself still propagates to the caller).
    last_end_reason_ = GenerationEndReason::NONE;
    last_stop_token_id_ = -1;
    struct EndGuard {
        GenerationEndReason& reason;
        int exceptions_at_entry;
        ~EndGuard() {
            if (std::uncaught_exceptions() > exceptions_at_entry) reason = GenerationEndReason::FAILURE;
        }
    } end_guard{last_end_reason_, std::uncaught_exceptions()};

    reset_state();
    auto prompt_ids = tokenizer_.encode(prompt);
    if (prompt_ids.empty()) {
        throw std::runtime_error("M8ModelRunner: Prompt produced 0 encoded tokens.");
    }

    // Apply official DeepSeek chat template if prompt does not already begin with BOS
    if (prompt_ids.front() != 0) {
        std::vector<int> chat_ids;
        chat_ids.push_back(0);      // BOS: <｜begin▁of▁sentence｜>
        chat_ids.push_back(128803); // <｜User｜>
        chat_ids.insert(chat_ids.end(), prompt_ids.begin(), prompt_ids.end());
        chat_ids.push_back(128804); // <｜Assistant｜>
        chat_ids.push_back(128822); // </think>
        prompt_ids = std::move(chat_ids);
    }

    for (int pid : prompt_ids) {
        if (pid < 0 || pid >= 129280) {
            throw std::runtime_error("M8ModelRunner: Prompt contains out-of-bounds token ID " + std::to_string(pid));
        }
    }

    std::vector<int> generated_ids = prompt_ids;
    size_t emitted_bytes = tokenizer_.decode(generated_ids).size(); // prompt text is not re-emitted
    std::vector<LayerTelemetry> layer_tels;
    std::vector<float> logits;
    ByteLoaderTelemetry lt_before = byte_loader_->telemetry(); // refreshed at the start of each token step

    auto t_start_all = std::chrono::high_resolution_clock::now();

    // Prefill sequence: process tokens 0..N-2 without computing LM head
    for (size_t p = 0; p + 1 < prompt_ids.size(); ++p) {
        if (M8ResourceGovernor::is_shutdown_requested()) break;
        step(prompt_ids[p], static_cast<int>(p), logits, layer_tels, /*compute_logits=*/false);
    }
    if (M8ResourceGovernor::is_shutdown_requested()) {
        last_end_reason_ = GenerationEndReason::USER_STOP;
        return generated_ids;
    }

    // Last prompt token: compute logits for the first generated token
    int last_prompt_pos = static_cast<int>(prompt_ids.size()) - 1;
    lt_before = byte_loader_->telemetry();
    auto t_step_start = std::chrono::high_resolution_clock::now();
    step(prompt_ids[last_prompt_pos], last_prompt_pos, logits, layer_tels, /*compute_logits=*/true);

    int current_pos = static_cast<int>(prompt_ids.size());

    for (int step_idx = 0; step_idx < max_new_tokens; ++step_idx) {
        if (M8ResourceGovernor::is_shutdown_requested()) {
            std::cerr << "\n[ASEMA] Clean shutdown requested. Breaking generation loop safely.\n";
            last_end_reason_ = GenerationEndReason::USER_STOP;
            break;
        }
        const auto t_sel0 = std::chrono::high_resolution_clock::now();
        // Repetition penalty on recently generated tokens (only if requested > 1.0f)
        if (repetition_penalty > 1.0f) {
            std::unordered_map<int, int> recent_counts;
            size_t start_r = std::max(prompt_ids.size(), generated_ids.size() >= 64 ? generated_ids.size() - 64 : prompt_ids.size());
            for (size_t r = start_r; r < generated_ids.size(); ++r) {
                recent_counts[generated_ids[r]]++;
            }
            for (auto& pair : recent_counts) {
                int tok = pair.first;
                if (tok >= 0 && tok < (int)logits.size()) {
                    float factor = 1.0f + (repetition_penalty - 1.0f) * pair.second;
                    if (logits[tok] > 0) logits[tok] /= factor;
                    else logits[tok] *= factor;
                }
            }
        }

        // Argmax selection
        int next_token = 0;
        float max_logit = -1e9f;
        for (size_t v = 0; v < logits.size(); ++v) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                next_token = static_cast<int>(v);
            }
        }

        if (next_token < 0 || next_token >= 129280) {
            throw std::runtime_error("M8ModelRunner: Generated invalid token ID " + std::to_string(next_token) + " out of vocabulary bounds [0..129279].");
        }
        if (!std::isfinite(max_logit)) {
            throw std::runtime_error("M8ModelRunner: Non-finite logit detected during token selection.");
        }
        const double select_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_sel0).count();

        if (next_token == tokenizer_.eos_id() || next_token == 1 || next_token == 128805 ||
            next_token == 128803 || next_token == 128804 || next_token == 129279) {
            last_end_reason_ = GenerationEndReason::EOS;
            last_stop_token_id_ = next_token;
            break;
        }

        generated_ids.push_back(next_token);
        const auto t_dec0 = std::chrono::high_resolution_clock::now();

        // Byte-level BPE can split one character across tokens. Only emit complete UTF-8
        // sequences; an incomplete tail is held back until a later token completes it.
        std::string cur_decoded_text = tokenizer_.decode(generated_ids);
        const size_t stable_len = utf8_complete_prefix_len(cur_decoded_text);
        std::string delta_token_str;
        if (stable_len > emitted_bytes) {
            delta_token_str = cur_decoded_text.substr(emitted_bytes, stable_len - emitted_bytes);
            emitted_bytes = stable_len;
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_dec0).count();

        auto t_step_end = std::chrono::high_resolution_clock::now();
        double step_latency_ms = std::chrono::duration<double, std::milli>(t_step_end - t_step_start).count();
        double total_elapsed_ms = std::chrono::duration<double, std::milli>(t_step_end - t_start_all).count();

        // Query Win32 process memory
        PROCESS_MEMORY_COUNTERS_EX pmc{};
        size_t cur_ram = 0, pk_ram = 0, av_ram = 0;
        if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
            cur_ram = pmc.WorkingSetSize;
            pk_ram = pmc.PeakWorkingSetSize;
        }
        MEMORYSTATUSEX mem_status{};
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            av_ram = mem_status.ullAvailPhys;
        }

        size_t cur_vram = (gpu_enabled_ && gpu_kernel_) ? gpu_kernel_->telemetry().vram_allocated_bytes : 0;
        static size_t peak_vram = 0;
        if (cur_vram > peak_vram) peak_vram = cur_vram;
        size_t av_vram = (8ULL * 1024 * 1024 * 1024 > cur_vram) ? (8ULL * 1024 * 1024 * 1024 - cur_vram) : 0;

        if (on_token) {
            TokenGenerationTelemetry tel;
            tel.step = step_idx + 1;
            tel.token_id = next_token;
            tel.token_str = delta_token_str;
            tel.token_latency_ms = step_latency_ms;
            tel.total_elapsed_ms = total_elapsed_ms;
            tel.active_experts_materialized = 6;
            tel.ram_working_set_bytes = cur_ram;
            tel.peak_ram_working_set_bytes = pk_ram;
            tel.available_ram_bytes = av_ram;
            tel.ram_ceiling_bytes = 20480ULL * 1024 * 1024;
            tel.vram_working_set_bytes = cur_vram;
            tel.peak_vram_working_set_bytes = peak_vram;
            tel.available_vram_bytes = av_vram;
            tel.vram_ceiling_bytes = 5120ULL * 1024 * 1024;
            tel.gpu_allocation_failures = 0;
            tel.layer_count_executed = total_layers_;
            tel.top_logit = max_logit;
            tel.kv_cache_valid = (active_layer_ != nullptr);
            tel.bytes_read_from_storage = byte_loader_->telemetry().total_bytes_read;
            tel.cache_hits = byte_loader_->telemetry().cache_hits;
            tel.cache_misses = byte_loader_->telemetry().cache_misses;
            tel.useful_prefetches = byte_loader_->telemetry().useful_prefetches;
            tel.wasted_prefetches = byte_loader_->telemetry().wasted_prefetches;
            tel.gpu_accelerated = (gpu_enabled_ && gpu_kernel_ && gpu_kernel_->is_initialized());
            if (!layer_tels.empty()) {
                tel.layer0_top6_experts = layer_tels[0].selected_experts;
            }

            // Stage breakdown of the step that produced this token.
            tel.stage = last_step_timing_;
            tel.select_ms = select_ms;
            tel.decode_ms = decode_ms;
            double layers_total = 0.0;
            for (const auto& ly : layer_tels) {
                tel.dense_load_ms += ly.dense_load_time_ms;
                tel.attn_ms += ly.attn_time_ms;
                tel.kv_update_ms += ly.kv_update_time_ms;
                tel.kv_attend_ms += ly.kv_attend_time_ms;
                tel.router_ms += ly.router_time_ms;
                tel.shared_expert_ms += ly.shared_expert_time_ms;
                tel.expert_load_ms += ly.expert_load_time_ms;
                tel.expert_gpu_ms += ly.expert_compute_time_ms;
                tel.gpu_upload_call_ms += ly.gpu_upload_time_ms;
                tel.gpu_wait_ms += ly.gpu_readback_time_ms;
                tel.gpu_busy_upload_ms += ly.gpu_busy_upload_ms;
                tel.gpu_busy_compute_ms += ly.gpu_busy_compute_ms;
                layers_total += ly.total_layer_time_ms;
            }
            tel.cpu_other_ms = layers_total - (tel.dense_load_ms + tel.attn_ms + tel.router_ms +
                                               tel.shared_expert_ms + tel.expert_load_ms + tel.expert_gpu_ms);
            const ByteLoaderTelemetry lt_after = byte_loader_->telemetry();
            tel.storage_read_ops = lt_after.storage_read_ops - lt_before.storage_read_ops;
            tel.storage_bytes = lt_after.total_bytes_read - lt_before.total_bytes_read;
            tel.expert_hits = static_cast<int>(lt_after.cache_hits - lt_before.cache_hits);
            tel.expert_misses = static_cast<int>(lt_after.cache_misses - lt_before.cache_misses);

            std::ostringstream diag;
            diag << "Token #" << tel.step << " (ID " << tel.token_id << "): '" << tel.token_str << "'"
                 << " | logit=" << std::fixed << std::setprecision(2) << tel.top_logit
                 << " | layers=" << tel.layer_count_executed << "/40"
                 << " | RAM: " << (cur_ram / (1024 * 1024)) << " MB (avail: " << (av_ram / (1024 * 1024 * 1024)) << " GB)"
                 << " | VRAM: " << (cur_vram / (1024 * 1024)) << " MB (avail: " << (av_vram / (1024 * 1024 * 1024)) << " GB)";
            tel.diagnostic_info = diag.str();

            on_token(tel);
        }

        if (step_idx + 1 == max_new_tokens) {
            last_end_reason_ = GenerationEndReason::MAX_TOKENS;
            break;
        }

        lt_before = byte_loader_->telemetry();
        t_step_start = std::chrono::high_resolution_clock::now();
        step(next_token, current_pos++, logits, layer_tels, /*compute_logits=*/true);
    }

    if (last_end_reason_ == GenerationEndReason::NONE) {
        last_end_reason_ = GenerationEndReason::MAX_TOKENS; // loop exhausted its token budget
    }
    return generated_ids;
}

} // namespace m8
} // namespace asema
