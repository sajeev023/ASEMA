#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_mla_attention.hpp"
#include "asema/m8/m8_fp8_lut.hpp"
#include "asema/m8/m8_gpu_mla_kernel.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <future>

namespace asema {
namespace m8 {

namespace {

float dot_avx2(const float* a, const float* b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 16; i += 16) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
    }
    for (; i <= n - 8; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }
    __m256 sum = _mm256_add_ps(sum0, sum1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 v128 = _mm_add_ps(lo, hi);
    v128 = _mm_hadd_ps(v128, v128);
    v128 = _mm_hadd_ps(v128, v128);
    float total = _mm_cvtss_f32(v128);
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

void gemv_dense(int rows, int cols, const float* W, const float* x, float* out) {
    for (int r = 0; r < rows; ++r) {
        out[r] = dot_avx2(W + r * cols, x, cols);
    }
}

static void dequant_fp8_block_range(const uint8_t* w_raw, const uint8_t* s_raw, float* out,
                                    int r_start, int r_end, int cols) {
    const int s_cols = (cols + 31) / 32;
    for (int r = r_start; r < r_end; ++r) {
        int s_row = r / 32;
        const uint8_t* s_row_ptr = s_raw + s_row * s_cols;
        const uint8_t* w_row_ptr = w_raw + r * cols;
        float* out_row_ptr = out + r * cols;

        int b = 0;
        for (; b <= cols - 32; b += 32) {
            float scale = k_e8m0_lut[s_row_ptr[b / 32]];
            __m256 vscale = _mm256_set1_ps(scale);

            for (int k = 0; k < 32; k += 8) {
                const uint8_t* w = w_row_ptr + b + k;
                __m256 v = _mm256_set_ps(
                    k_e4m3_lut[w[7]], k_e4m3_lut[w[6]],
                    k_e4m3_lut[w[5]], k_e4m3_lut[w[4]],
                    k_e4m3_lut[w[3]], k_e4m3_lut[w[2]],
                    k_e4m3_lut[w[1]], k_e4m3_lut[w[0]]
                );
                _mm256_storeu_ps(out_row_ptr + b + k, _mm256_mul_ps(v, vscale));
            }
        }
        for (; b < cols; ++b) {
            float scale = k_e8m0_lut[s_row_ptr[b / 32]];
            float val = k_e4m3_lut[w_row_ptr[b]];
            out_row_ptr[b] = val * scale;
        }
    }
}

void dequant_fp8_block(const uint8_t* w_raw, const uint8_t* s_raw, float* out, int rows, int cols) {
    if (rows >= 1024) {
        const int num_threads = 8;
        int chunk = rows / num_threads;
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            int r_start = t * chunk;
            int r_end = (t == num_threads - 1) ? rows : (t + 1) * chunk;
            futures.push_back(std::async(std::launch::async, [=]() {
                dequant_fp8_block_range(w_raw, s_raw, out, r_start, r_end, cols);
            }));
        }
        for (auto& f : futures) {
            f.get();
        }
    } else {
        dequant_fp8_block_range(w_raw, s_raw, out, 0, rows, cols);
    }
}

} // namespace

M8MLAAttention::M8MLAAttention(MLAParams params)
    : params_(params) {
    init_buffers();
    precompute_rope();
}

void M8MLAAttention::init_buffers() {
    wq_a_.assign(static_cast<size_t>(params_.q_lora_rank) * params_.dim, 0.0f);
    q_norm_.assign(params_.q_lora_rank, 1.0f);
    wq_b_.assign(static_cast<size_t>(params_.n_heads) * params_.head_dim * params_.q_lora_rank, 0.0f);
    wkv_.assign(static_cast<size_t>(params_.head_dim) * params_.dim, 0.0f);
    kv_norm_.assign(params_.head_dim, 1.0f);
    wo_a_.assign(static_cast<size_t>(params_.o_groups) * params_.o_lora_rank * (params_.n_heads / params_.o_groups * params_.head_dim), 0.0f);
    wo_b_.assign(static_cast<size_t>(params_.dim) * (params_.o_groups * params_.o_lora_rank), 0.0f);
    attn_sink_.assign(params_.n_heads, 0.0f);

    if (kv_caches_.empty()) {
        kv_caches_.resize(NUM_KV_LAYERS);
        cached_tokens_per_layer_.assign(NUM_KV_LAYERS, 0);
        for (int l = 0; l < NUM_KV_LAYERS; ++l) {
            kv_caches_[l].assign(static_cast<size_t>(params_.window_size) * params_.head_dim, 0.0f);
        }
    }
    is_initialized_ = true;
}

void M8MLAAttention::precompute_rope() {
    const int max_seq_len = 8192;
    int half_dim = params_.rope_head_dim / 2;
    cos_table_.resize(max_seq_len * half_dim);
    sin_table_.resize(max_seq_len * half_dim);

    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            float theta = std::pow(params_.rope_theta, -2.0f * i / params_.rope_head_dim);
            float angle = pos * theta;
            cos_table_[pos * half_dim + i] = std::cos(angle);
            sin_table_[pos * half_dim + i] = std::sin(angle);
        }
    }
}

void M8MLAAttention::apply_rope(float* vec, int head_dim, int pos, bool inverse) const {
    int offset = head_dim - params_.rope_head_dim;
    int half_dim = params_.rope_head_dim / 2;
    int p = std::clamp(pos, 0, 8191);

    for (int i = 0; i < half_dim; ++i) {
        int idx = offset + 2 * i;
        float x0 = vec[idx];
        float x1 = vec[idx + 1];
        float c = cos_table_[p * half_dim + i];
        float s = sin_table_[p * half_dim + i];

        if (inverse) {
            vec[idx]     =  x0 * c + x1 * s;
            vec[idx + 1] = -x0 * s + x1 * c;
        } else {
            vec[idx]     = x0 * c - x1 * s;
            vec[idx + 1] = x0 * s + x1 * c;
        }
    }
}

void M8MLAAttention::rms_norm(const float* in, const float* weight, float* out, int size) const {
    double sq_sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sq_sum += static_cast<double>(in[i]) * in[i];
    }
    float scale = 1.0f / std::sqrt(static_cast<float>(sq_sum / size) + params_.eps);
    for (int i = 0; i < size; ++i) {
        out[i] = in[i] * scale * weight[i];
    }
}

void M8MLAAttention::set_gpu_mla(std::shared_ptr<M8GpuMlaKernel> gpu_mla) {
    gpu_mla_ = std::move(gpu_mla);
    if (gpu_mla_ && direct_fp8_) {
        // GPU MLA needs FP32 uploads: leave direct mode. The owning layer must reload (it does,
        // via M8TransformerLayer::set_gpu_mla) before the next forward().
        direct_fp8_ = false;
        init_buffers();
    }
    if (gpu_mla_ && gpu_mla_->is_initialized() && is_initialized_) {
        gpu_mla_->upload_layer_weights(wq_a_.data(), q_norm_.data(), wq_b_.data(),
                                       wkv_.data(), kv_norm_.data(),
                                       wo_a_.data(), wo_b_.data());
        gpu_mla_->upload_rope_tables(cos_table_.data(), sin_table_.data());
        gpu_mla_->upload_attn_sink(attn_sink_.data());
    }
}

bool M8MLAAttention::load_mock_or_reference_weights() {
    // Forward to physical checkpoint files if present
    std::string c1 = (asema::m8::paths::primary_shards() + "/l0_chunk1_dense_norms_scales.bin");
    std::string c2 = (asema::m8::paths::primary_shards() + "/l0_chunk2_weights_shared.bin");
    if (load_from_checkpoint_files(c1, c2)) {
        return true;
    }
    init_buffers();
    return true;
}

bool M8MLAAttention::load_from_checkpoint_files(const std::string& chunk1_file, const std::string& chunk2_file) {
    std::ifstream f1(chunk1_file, std::ios::binary);
    std::ifstream f2(chunk2_file, std::ios::binary);
    if (!f1.is_open() || !f2.is_open()) {
        return false;
    }

    f1.seekg(0, std::ios::end);
    size_t sz1 = f1.tellg();
    f1.seekg(0, std::ios::beg);
    std::vector<uint8_t> c1(sz1);
    f1.read(reinterpret_cast<char*>(c1.data()), sz1);

    f2.seekg(0, std::ios::end);
    size_t sz2 = f2.tellg();
    f2.seekg(0, std::ios::beg);
    std::vector<uint8_t> c2(sz2);
    f2.read(reinterpret_cast<char*>(c2.data()), sz2);

    init_buffers();

    // 1. attn_sink [64] FP32 at offset 0
    std::memcpy(attn_sink_.data(), c1.data() + 0, 64 * sizeof(float));

    // 2. kv_norm.weight [512] BF16 at offset 3935704
    const uint16_t* raw_kv_norm = reinterpret_cast<const uint16_t*>(c1.data() + 3935704);
    for (int i = 0; i < 512; ++i) {
        uint32_t u = static_cast<uint32_t>(raw_kv_norm[i]) << 16;
        kv_norm_[i] = *reinterpret_cast<float*>(&u);
    }

    // 3. q_norm.weight [1280] BF16 at offset 3936728
    const uint16_t* raw_q_norm = reinterpret_cast<const uint16_t*>(c1.data() + 3936728);
    for (int i = 0; i < 1280; ++i) {
        uint32_t u = static_cast<uint32_t>(raw_q_norm[i]) << 16;
        q_norm_[i] = *reinterpret_cast<float*>(&u);
    }

    // Scales in chunk 1
    const uint8_t* s_wkv  = c1.data() + 7891928; // [16, 160]
    const uint8_t* s_wo_a = c1.data() + 7894488; // [256, 128]
    const uint8_t* s_wo_b = c1.data() + 7927256; // [160, 256]
    const uint8_t* s_wq_a = c1.data() + 7968216; // [40, 160]
    const uint8_t* s_wq_b = c1.data() + 7974616; // [1024, 40]

    // Weights in chunk 2 (offsets relative to 432688856)
    const uint64_t base2 = 432688856ULL;
    const uint8_t* w_wkv  = c2.data() + (432723416ULL - base2); // [512, 5120]
    const uint8_t* w_wo_a = c2.data() + (435344856ULL - base2); // [8192, 4096]
    const uint8_t* w_wo_b = c2.data() + (468899288ULL - base2); // [5120, 8192]
    const uint8_t* w_wq_a = c2.data() + (510842328ULL - base2); // [1280, 5120]
    const uint8_t* w_wq_b = c2.data() + (517395928ULL - base2); // [32768, 1280]

    // Dequantize all FP8 attention weights
    dequant_fp8_block(w_wq_a, s_wq_a, wq_a_.data(), 1280, 5120);
    dequant_fp8_block(w_wq_b, s_wq_b, wq_b_.data(), 32768, 1280);
    dequant_fp8_block(w_wkv,  s_wkv,  wkv_.data(),  512,  5120);
    dequant_fp8_block(w_wo_a, s_wo_a, wo_a_.data(), 8192, 4096);
    dequant_fp8_block(w_wo_b, s_wo_b, wo_b_.data(), 5120, 8192);

    is_initialized_ = true;

    if (gpu_mla_ && gpu_mla_->is_initialized()) {
        gpu_mla_->upload_layer_weights(wq_a_.data(), q_norm_.data(), wq_b_.data(),
                                       wkv_.data(), kv_norm_.data(),
                                       wo_a_.data(), wo_b_.data());
        gpu_mla_->upload_rope_tables(cos_table_.data(), sin_table_.data());
        gpu_mla_->upload_attn_sink(attn_sink_.data());
    }
    return true;
}

void M8MLAAttention::release_fp32_projections() {
    std::vector<float>().swap(wq_a_);
    std::vector<float>().swap(wq_b_);
    std::vector<float>().swap(wkv_);
    std::vector<float>().swap(wo_a_);
    std::vector<float>().swap(wo_b_);
}

bool M8MLAAttention::bind_direct_fp8(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id) {
    const auto& idx = vol_mgr->safetensors_index();
    const std::string prefix = "layers." + std::to_string(layer_id) + ".attn.";

    auto bind = [&](const char* name, int rows, int cols, Fp8MatView& out) -> bool {
        uint64_t w_bytes = 0, s_bytes = 0;
        const uint8_t* w = idx.map_tensor(prefix + name + ".weight", &w_bytes);
        const uint8_t* s = idx.map_tensor(prefix + name + ".scale", &s_bytes);
        if (!w || !s) return false;
        if (w_bytes != static_cast<uint64_t>(rows) * cols) return false;
        if (s_bytes != static_cast<uint64_t>(rows / 32) * (cols / 32)) return false;
        out = Fp8MatView{w, s, rows, cols};
        return out.valid();
    };

    Fp8MatView wq_a, wq_b, wkv, wo_a, wo_b;
    if (!bind("wq_a", params_.q_lora_rank, params_.dim, wq_a)) return false;
    if (!bind("wq_b", params_.n_heads * params_.head_dim, params_.q_lora_rank, wq_b)) return false;
    if (!bind("wkv", params_.head_dim, params_.dim, wkv)) return false;
    const int group_in_dim = (params_.n_heads / params_.o_groups) * params_.head_dim;
    if (!bind("wo_a", params_.o_groups * params_.o_lora_rank, group_in_dim, wo_a)) return false;
    if (!bind("wo_b", params_.dim, params_.o_groups * params_.o_lora_rank, wo_b)) return false;

    // Small tensors are still copied (a few KB); the large ones are used in place.
    if (attn_sink_.size() != 64 ||
        !vol_mgr->read_tensor(prefix + "attn_sink", reinterpret_cast<uint8_t*>(attn_sink_.data()), 64 * sizeof(float))) {
        return false;
    }
    q_norm_.resize(params_.q_lora_rank);
    kv_norm_.resize(params_.head_dim);
    if (!vol_mgr->read_tensor_bf16_to_fp32(prefix + "kv_norm.weight", kv_norm_.data(), params_.head_dim)) return false;
    if (!vol_mgr->read_tensor_bf16_to_fp32(prefix + "q_norm.weight", q_norm_.data(), params_.q_lora_rank)) return false;

    wq_a_f8_ = wq_a;
    wq_b_f8_ = wq_b;
    wkv_f8_ = wkv;
    wo_a_f8_ = wo_a;
    wo_b_f8_ = wo_b;
    return true;
}

bool M8MLAAttention::load_from_vol_mgr(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id) {
    if (!vol_mgr) return false;

    // Fast path: use the FP8 checkpoint bytes in place. The GPU MLA path needs FP32 uploads,
    // so it keeps the legacy dequantizing loader.
    const bool gpu_mla_active = gpu_mla_ && gpu_mla_->is_initialized();
    if (dense_fp8_direct_enabled() && !gpu_mla_active && bind_direct_fp8(vol_mgr, layer_id)) {
        if (!direct_fp8_) {
            release_fp32_projections();
            direct_fp8_ = true;
        }
        current_layer_id_ = layer_id;
        is_initialized_ = true;
        return true;
    }
    direct_fp8_ = false;

    init_buffers();
    current_layer_id_ = layer_id;
    std::string prefix = "layers." + std::to_string(layer_id) + ".attn.";

    // 1. attn_sink [64] FP32
    if (!vol_mgr->read_tensor(prefix + "attn_sink", reinterpret_cast<uint8_t*>(attn_sink_.data()), 64 * sizeof(float))) {
        return false;
    }

    // 2. kv_norm.weight [512] BF16 -> FP32
    if (!vol_mgr->read_tensor_bf16_to_fp32(prefix + "kv_norm.weight", kv_norm_.data(), 512)) {
        return false;
    }

    // 3. q_norm.weight [1280] BF16 -> FP32
    if (!vol_mgr->read_tensor_bf16_to_fp32(prefix + "q_norm.weight", q_norm_.data(), 1280)) {
        return false;
    }

    // Helper lambda for reading and dequantizing an FP8 matrix
    auto read_and_dequant = [&](const std::string& name_w, const std::string& name_s,
                                float* out, int rows, int cols) -> bool {
        size_t w_bytes = static_cast<size_t>(rows) * cols;
        size_t s_bytes = (static_cast<size_t>(rows) / 32) * (cols / 32);
        std::vector<uint8_t> w_buf(w_bytes);
        std::vector<uint8_t> s_buf(s_bytes);
        if (!vol_mgr->read_tensor(prefix + name_w, w_buf.data(), w_bytes)) return false;
        if (!vol_mgr->read_tensor(prefix + name_s, s_buf.data(), s_bytes)) return false;
        dequant_fp8_block(w_buf.data(), s_buf.data(), out, rows, cols);
        return true;
    };

    // 4. wq_a [1280, 5120]
    if (!read_and_dequant("wq_a.weight", "wq_a.scale", wq_a_.data(), 1280, 5120)) return false;

    // 5. wq_b [32768, 1280]
    if (!read_and_dequant("wq_b.weight", "wq_b.scale", wq_b_.data(), 32768, 1280)) return false;

    // 6. wkv [512, 5120]
    if (!read_and_dequant("wkv.weight", "wkv.scale", wkv_.data(), 512, 5120)) return false;

    // 7. wo_a [8192, 4096]
    if (!read_and_dequant("wo_a.weight", "wo_a.scale", wo_a_.data(), 8192, 4096)) return false;

    // 8. wo_b [5120, 8192]
    if (!read_and_dequant("wo_b.weight", "wo_b.scale", wo_b_.data(), 5120, 8192)) return false;

    is_initialized_ = true;

    if (gpu_mla_ && gpu_mla_->is_initialized()) {
        gpu_mla_->upload_layer_weights(wq_a_.data(), q_norm_.data(), wq_b_.data(),
                                       wkv_.data(), kv_norm_.data(),
                                       wo_a_.data(), wo_b_.data());
        gpu_mla_->upload_rope_tables(cos_table_.data(), sin_table_.data());
        gpu_mla_->upload_attn_sink(attn_sink_.data());
    }
    return true;
}

bool M8MLAAttention::load_from_buffers(const float* wq_a, const float* q_norm, const float* wq_b,
                                      const float* wkv, const float* kv_norm,
                                      const float* wo_a, const float* wo_b, const float* attn_sink) {
    if (!wq_a || !q_norm || !wq_b || !wkv || !kv_norm || !wo_a || !wo_b || !attn_sink) return false;
    init_buffers();
    std::memcpy(wq_a_.data(), wq_a, wq_a_.size() * sizeof(float));
    std::memcpy(q_norm_.data(), q_norm, q_norm_.size() * sizeof(float));
    std::memcpy(wq_b_.data(), wq_b, wq_b_.size() * sizeof(float));
    std::memcpy(wkv_.data(), wkv, wkv_.size() * sizeof(float));
    std::memcpy(kv_norm_.data(), kv_norm, kv_norm_.size() * sizeof(float));
    std::memcpy(wo_a_.data(), wo_a, wo_a_.size() * sizeof(float));
    std::memcpy(wo_b_.data(), wo_b, wo_b_.size() * sizeof(float));
    std::memcpy(attn_sink_.data(), attn_sink, attn_sink_.size() * sizeof(float));
    is_initialized_ = true;

    if (gpu_mla_ && gpu_mla_->is_initialized()) {
        gpu_mla_->upload_layer_weights(wq_a_.data(), q_norm_.data(), wq_b_.data(),
                                       wkv_.data(), kv_norm_.data(),
                                       wo_a_.data(), wo_b_.data());
        gpu_mla_->upload_rope_tables(cos_table_.data(), sin_table_.data());
        gpu_mla_->upload_attn_sink(attn_sink_.data());
    }
    return true;
}

void M8MLAAttention::set_active_layer(int layer_id) {
    if (layer_id >= 0 && layer_id < NUM_KV_LAYERS) {
        current_layer_id_ = layer_id;
    }
}

void M8MLAAttention::reset_kv_cache() {
    for (int l = 0; l < NUM_KV_LAYERS; ++l) {
        if (!kv_caches_.empty() && l < (int)kv_caches_.size()) {
            std::fill(kv_caches_[l].begin(), kv_caches_[l].end(), 0.0f);
        }
        if (!cached_tokens_per_layer_.empty() && l < (int)cached_tokens_per_layer_.size()) {
            cached_tokens_per_layer_[l] = 0;
        }
    }
    current_layer_id_ = 0;
    if (gpu_mla_ && gpu_mla_->is_initialized()) {
        gpu_mla_->reset_kv_cache();
    }
}

void M8MLAAttention::forward(const float* x, float* out, int start_pos) {
    if (!is_initialized_ || !x || !out) return;

    if (current_layer_id_ < 0 || current_layer_id_ >= (int)kv_caches_.size()) {
        current_layer_id_ = 0;
    }
    auto& cur_cache = kv_caches_[current_layer_id_];
    int& cur_tokens = cached_tokens_per_layer_[current_layer_id_];

    // 1. Q projection: wq_a -> q_norm -> wq_b -> RoPE on last 64 dims
    int total_q_dim = params_.n_heads * params_.head_dim; // 32768
    // Projection helper: in-place FP8 kernel when direct, legacy FP32 GEMV otherwise.
    auto project = [&](const Fp8MatView& f8, const std::vector<float>& w32, int rows, int cols,
                       const float* in, float* o) {
        if (direct_fp8_) fp8_gemv(f8, in, o);
        else gemv_dense(rows, cols, w32.data(), in, o);
    };

    std::vector<float> qr_raw(params_.q_lora_rank);
    project(wq_a_f8_, wq_a_, params_.q_lora_rank, params_.dim, x, qr_raw.data());

    std::vector<float> qr(params_.q_lora_rank);
    rms_norm(qr_raw.data(), q_norm_.data(), qr.data(), params_.q_lora_rank);

    std::vector<float> q(total_q_dim);
    project(wq_b_f8_, wq_b_, total_q_dim, params_.q_lora_rank, qr.data(), q.data());

    for (int h = 0; h < params_.n_heads; ++h) {
        apply_rope(q.data() + h * params_.head_dim, params_.head_dim, start_pos, false);
    }

    // 2. KV projection: wkv -> kv_norm -> RoPE on last 64 dims
    std::vector<float> kv_raw(params_.head_dim);
    project(wkv_f8_, wkv_, params_.head_dim, params_.dim, x, kv_raw.data());

    std::vector<float> kv(params_.head_dim);
    rms_norm(kv_raw.data(), kv_norm_.data(), kv.data(), params_.head_dim);
    apply_rope(kv.data(), params_.head_dim, start_pos, false);

    // 3. Store into KV cache ring buffer
    const auto t_kv0 = std::chrono::high_resolution_clock::now();
    int ring_slot = start_pos % params_.window_size;
    std::memcpy(&cur_cache[ring_slot * params_.head_dim], kv.data(), params_.head_dim * sizeof(float));
    cur_tokens = std::min(cur_tokens + 1, params_.window_size);
    const auto t_kv1 = std::chrono::high_resolution_clock::now();
    last_kv_update_ms_ = std::chrono::duration<double, std::milli>(t_kv1 - t_kv0).count();

    // 4. Attention with attn_sink
    const auto t_att0 = std::chrono::high_resolution_clock::now();
    std::vector<float> o(total_q_dim, 0.0f);
    std::vector<float> scores(cur_tokens);

    for (int h = 0; h < params_.n_heads; ++h) {
        const float* q_h = q.data() + h * params_.head_dim;
        float max_score = -1e9f;

        for (int t = 0; t < cur_tokens; ++t) {
            const float* kv_t = &cur_cache[t * params_.head_dim];
            float s = dot_avx2(q_h, kv_t, params_.head_dim) * params_.softmax_scale;
            scores[t] = s;
            if (s > max_score) max_score = s;
        }

        float max_val = std::max(max_score, attn_sink_[h]);
        float denom = 0.0f;
        for (int t = 0; t < cur_tokens; ++t) {
            denom += std::exp(scores[t] - max_val);
        }
        denom += std::exp(attn_sink_[h] - max_val);

        float inv_denom = 1.0f / (denom + 1e-20f);
        float* o_h = o.data() + h * params_.head_dim;

        for (int t = 0; t < cur_tokens; ++t) {
            float w = std::exp(scores[t] - max_val) * inv_denom;
            const float* kv_t = &cur_cache[t * params_.head_dim];
            for (int d = 0; d < params_.head_dim; ++d) {
                o_h[d] += w * kv_t[d];
            }
        }

        // Inverse RoPE on attention output
        apply_rope(o_h, params_.head_dim, start_pos, true);
    }

    last_kv_attend_ms_ = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_att0).count();

    // 5. Grouped Output Projection: wo_a (8 groups of 8 heads)
    int heads_per_group = params_.n_heads / params_.o_groups; // 8
    int group_in_dim = heads_per_group * params_.head_dim;    // 4096
    int total_lora_dim = params_.o_groups * params_.o_lora_rank; // 8192
    std::vector<float> o_lora(total_lora_dim);

    for (int g = 0; g < params_.o_groups; ++g) {
        const float* g_in = o.data() + g * group_in_dim;
        float* g_out = o_lora.data() + g * params_.o_lora_rank;
        if (direct_fp8_) {
            fp8_gemv(wo_a_f8_.slice_rows(g * params_.o_lora_rank, params_.o_lora_rank), g_in, g_out);
        } else {
            const float* W_g = wo_a_.data() + g * (params_.o_lora_rank * group_in_dim);
            gemv_dense(params_.o_lora_rank, group_in_dim, W_g, g_in, g_out);
        }
    }

    // 6. Final projection wo_b: 8192 -> 5120
    project(wo_b_f8_, wo_b_, params_.dim, total_lora_dim, o_lora.data(), out);
}

} // namespace m8
} // namespace asema
