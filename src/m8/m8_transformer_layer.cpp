#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_transformer_layer.hpp"
#include "asema/m8/m8_fp8_lut.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <immintrin.h>

#include <future>
#include <vector>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

namespace {

static void dequant_fp8_block_range(const uint8_t* w_raw, const uint8_t* s_raw, float* out,
                                    int r_start, int r_end, int cols) {
    const int s_cols = cols / 32;
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

static void dequant_fp8_block(const uint8_t* w_raw, const uint8_t* s_raw,
                              float* out, int rows, int cols) {
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

static float dot_product_f32_avx2(const float* a, const float* b, int n) {
    __m256 sum256 = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
    }
    __m128 low = _mm256_castps256_ps128(sum256);
    __m128 high = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

} // namespace

M8TransformerLayer::M8TransformerLayer(int layer_id, std::shared_ptr<M8ByteRangeLoader> byte_loader)
    : layer_id_(layer_id), byte_loader_(std::move(byte_loader)) {
    init();
}

bool M8TransformerLayer::load_layer(int layer_id) {
    layer_id_ = layer_id;
    self_attn_.set_active_layer(layer_id);

    if (loaded_layer_id_ == layer_id) {
        return true;
    }

    auto vol_mgr = byte_loader_ ? byte_loader_->volume_manager() : nullptr;
    if (vol_mgr) {
        std::string attn_norm_name = "layers." + std::to_string(layer_id) + ".attn_norm.weight";
        std::string ffn_norm_name = "layers." + std::to_string(layer_id) + ".ffn_norm.weight";

        attn_norm_.assign(5120, 1.0f);
        ffn_norm_.assign(5120, 1.0f);

        bool has_attn_norm = vol_mgr->read_tensor_bf16_to_fp32(attn_norm_name, attn_norm_.data(), 5120);
        bool has_ffn_norm = vol_mgr->read_tensor_bf16_to_fp32(ffn_norm_name, ffn_norm_.data(), 5120);

        if (has_attn_norm && has_ffn_norm) {
            // 1. Load attention weights via volume manager
            self_attn_.load_from_vol_mgr(vol_mgr, layer_id);

            // 2. Load router via volume manager
            router_.load_from_vol_mgr(vol_mgr, layer_id);

            // 3. Load shared expert via volume manager
            std::string shared_prefix = "layers." + std::to_string(layer_id) + ".ffn.shared_experts.";
            auto read_and_dequant_shared = [&](const std::string& name_w, const std::string& name_s,
                                               float* out, int rows, int cols) -> bool {
                size_t w_bytes = static_cast<size_t>(rows) * cols;
                size_t s_bytes = (static_cast<size_t>(rows) / 32) * (cols / 32);
                std::vector<uint8_t> w_buf(w_bytes);
                std::vector<uint8_t> s_buf(s_bytes);
                if (!vol_mgr->read_tensor(shared_prefix + name_w, w_buf.data(), w_bytes)) return false;
                if (!vol_mgr->read_tensor(shared_prefix + name_s, s_buf.data(), s_bytes)) return false;
                dequant_fp8_block(w_buf.data(), s_buf.data(), out, rows, cols);
                return true;
            };

            // Fast path: use the FP8 shared-expert bytes in place from the memory-mapped shards.
            auto bind_shared = [&](const char* w, const char* s, int rows, int cols, Fp8MatView& out) -> bool {
                uint64_t wb = 0, sb = 0;
                const uint8_t* wp = vol_mgr->safetensors_index().map_tensor(shared_prefix + w, &wb);
                const uint8_t* sp = vol_mgr->safetensors_index().map_tensor(shared_prefix + s, &sb);
                if (!wp || !sp) return false;
                if (wb != static_cast<uint64_t>(rows) * cols) return false;
                if (sb != static_cast<uint64_t>(rows / 32) * (cols / 32)) return false;
                out = Fp8MatView{wp, sp, rows, cols};
                return out.valid();
            };

            shared_direct_fp8_ = dense_fp8_direct_enabled() &&
                                 bind_shared("w1.weight", "w1.scale", 2304, 5120, shared_w1_f8_) &&
                                 bind_shared("w2.weight", "w2.scale", 5120, 2304, shared_w2_f8_) &&
                                 bind_shared("w3.weight", "w3.scale", 2304, 5120, shared_w3_f8_);

            if (shared_direct_fp8_) {
                std::vector<float>().swap(shared_w1_);
                std::vector<float>().swap(shared_w2_);
                std::vector<float>().swap(shared_w3_);
                has_shared_expert_ = true;
            } else {
                shared_w1_.resize(2304 * 5120);
                shared_w2_.resize(5120 * 2304);
                shared_w3_.resize(2304 * 5120);

                bool ok_w1 = read_and_dequant_shared("w1.weight", "w1.scale", shared_w1_.data(), 2304, 5120);
                bool ok_w2 = read_and_dequant_shared("w2.weight", "w2.scale", shared_w2_.data(), 5120, 2304);
                bool ok_w3 = read_and_dequant_shared("w3.weight", "w3.scale", shared_w3_.data(), 2304, 5120);
                has_shared_expert_ = (ok_w1 && ok_w2 && ok_w3);
            }

            // 4. Load Hyper-Connections weights
            std::string prefix = "layers." + std::to_string(layer_id) + ".";
            hc_attn_fn_.resize(24 * 20480);
            hc_ffn_fn_.resize(24 * 20480);
            hc_attn_scale_.resize(3);
            hc_ffn_scale_.resize(3);
            hc_attn_base_.resize(24);
            hc_ffn_base_.resize(24);

            bool ok_hc = true;
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_attn_fn", reinterpret_cast<uint8_t*>(hc_attn_fn_.data()), 24 * 20480 * sizeof(float));
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_attn_scale", reinterpret_cast<uint8_t*>(hc_attn_scale_.data()), 3 * sizeof(float));
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_attn_base", reinterpret_cast<uint8_t*>(hc_attn_base_.data()), 24 * sizeof(float));
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_ffn_fn", reinterpret_cast<uint8_t*>(hc_ffn_fn_.data()), 24 * 20480 * sizeof(float));
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_ffn_scale", reinterpret_cast<uint8_t*>(hc_ffn_scale_.data()), 3 * sizeof(float));
            ok_hc &= vol_mgr->read_tensor(prefix + "hc_ffn_base", reinterpret_cast<uint8_t*>(hc_ffn_base_.data()), 24 * sizeof(float));
            has_hc_ = ok_hc;

            loaded_layer_id_ = layer_id;
            return true;
        }
    }

    // Fallback for layer 0 legacy bin files if present
    if (layer_id == 0) {
        attn_norm_.assign(5120, 1.0f);
        ffn_norm_.assign(5120, 1.0f);
        self_attn_.load_mock_or_reference_weights();

        std::string c1 = (asema::m8::paths::primary_shards() + "/l0_chunk1_dense_norms_scales.bin");
        if (fs::exists(c1)) {
            std::ifstream f(c1, std::ios::binary);
            if (f.is_open()) {
                f.seekg(3939288);
                std::vector<uint16_t> raw_attn(5120);
                f.read(reinterpret_cast<char*>(raw_attn.data()), 5120 * sizeof(uint16_t));
                for (int i = 0; i < 5120; ++i) {
                    uint32_t u = static_cast<uint32_t>(raw_attn[i]) << 16;
                    attn_norm_[i] = *reinterpret_cast<float*>(&u);
                }

                f.seekg(7881688);
                std::vector<uint16_t> raw_ffn(5120);
                f.read(reinterpret_cast<char*>(raw_ffn.data()), 5120 * sizeof(uint16_t));
                for (int i = 0; i < 5120; ++i) {
                    uint32_t u = static_cast<uint32_t>(raw_ffn[i]) << 16;
                    ffn_norm_[i] = *reinterpret_cast<float*>(&u);
                }
            }
        }

        std::string c2 = (asema::m8::paths::primary_shards() + "/l0_chunk2_weights_shared.bin");
        if (fs::exists(c2)) {
            std::ifstream f2(c2, std::ios::binary);
            if (f2.is_open()) {
                f2.seekg(0, std::ios::end);
                size_t sz2 = f2.tellg();
                f2.seekg(0, std::ios::beg);
                std::vector<uint8_t> c2_data(sz2);
                f2.read(reinterpret_cast<char*>(c2_data.data()), sz2);

                const uint64_t base2 = 432688856ULL;
                const uint8_t* s_w1 = c2_data.data() + (432688856ULL - base2);
                const uint8_t* s_w2 = c2_data.data() + (432700376ULL - base2);
                const uint8_t* s_w3 = c2_data.data() + (432711896ULL - base2);

                const uint8_t* w_w1 = c2_data.data() + (559338968ULL - base2);
                const uint8_t* w_w2 = c2_data.data() + (571135448ULL - base2);
                const uint8_t* w_w3 = c2_data.data() + (582931928ULL - base2);

                shared_w1_.resize(2304 * 5120);
                shared_w2_.resize(5120 * 2304);
                shared_w3_.resize(2304 * 5120);

                dequant_fp8_block(w_w1, s_w1, shared_w1_.data(), 2304, 5120);
                dequant_fp8_block(w_w2, s_w2, shared_w2_.data(), 5120, 2304);
                dequant_fp8_block(w_w3, s_w3, shared_w3_.data(), 2304, 5120);
                has_shared_expert_ = true;
            }
        }
        loaded_layer_id_ = 0;
        return true;
    }

    return false;
}

void M8TransformerLayer::init() {
    load_layer(layer_id_);
}

bool M8TransformerLayer::load_router(const std::string& gate_weight_file, const std::string& gate_bias_file) {
    return router_.load_from_files(gate_weight_file, gate_bias_file);
}

void M8TransformerLayer::rms_norm(const float* in, const float* weight, float* out, int size, float eps) const {
    double sq = 0.0;
    for (int i = 0; i < size; ++i) sq += static_cast<double>(in[i]) * in[i];
    float scale = 1.0f / std::sqrt(static_cast<float>(sq / size) + eps);
    for (int i = 0; i < size; ++i) out[i] = in[i] * scale * weight[i];
}

void M8TransformerLayer::compute_shared_expert(const float* in, float* out_accum) const {
    if (!has_shared_expert_) return;
    const int D = 5120;
    const int M = 2304;
    std::vector<float> gate(M);
    std::vector<float> up(M);

    if (shared_direct_fp8_) {
        fp8_gemv(shared_w1_f8_, in, gate.data());
        fp8_gemv(shared_w3_f8_, in, up.data());
    } else {
        for (int r = 0; r < M; ++r) gate[r] = dot_product_f32_avx2(shared_w1_.data() + r * D, in, D);
        for (int r = 0; r < M; ++r) up[r] = dot_product_f32_avx2(shared_w3_.data() + r * D, in, D);
    }
    // Same clamping semantics as before: gate clamped above, up clamped both sides.
    for (int r = 0; r < M; ++r) {
        if (gate[r] > 10.0f) gate[r] = 10.0f;
        if (up[r] < -10.0f) up[r] = -10.0f;
        else if (up[r] > 10.0f) up[r] = 10.0f;
    }

    std::vector<float> act(M);
    for (int i = 0; i < M; ++i) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        act[i] = silu * up[i];
    }

    if (shared_direct_fp8_) {
        std::vector<float> down(D);
        fp8_gemv(shared_w2_f8_, act.data(), down.data());
        for (int r = 0; r < D; ++r) out_accum[r] += down[r];
    } else {
        for (int r = 0; r < D; ++r) {
            out_accum[r] += dot_product_f32_avx2(shared_w2_.data() + r * M, act.data(), M);
        }
    }
}

void M8TransformerLayer::forward(const float* in, float* out, int start_pos, LayerTelemetry& tel) {
    auto t_layer_start = std::chrono::high_resolution_clock::now();
    tel.layer_id = layer_id_;

    const int D = 5120;
    std::vector<float> norm_in(D);
    std::vector<float> attn_out(D);
    std::vector<float> residual_attn(D);
    std::vector<float> norm_ffn(D);
    std::vector<float> moe_out(D, 0.0f);

    // 1. Attention sublayer pre-norm
    rms_norm(in, attn_norm_.data(), norm_in.data(), D);

    auto t_attn_start = std::chrono::high_resolution_clock::now();
    self_attn_.forward(norm_in.data(), attn_out.data(), start_pos);
    auto t_attn_end = std::chrono::high_resolution_clock::now();
    tel.attn_time_ms = std::chrono::duration<double, std::milli>(t_attn_end - t_attn_start).count();

    // 2. First residual connection: in + attn_out
    for (int i = 0; i < D; ++i) {
        residual_attn[i] = in[i] + attn_out[i];
    }

    // 3. MoE FFN sublayer pre-norm
    rms_norm(residual_attn.data(), ffn_norm_.data(), norm_ffn.data(), D);

    auto t_router_start = std::chrono::high_resolution_clock::now();
    RouterSelection selection = router_.route(norm_ffn.data());
    auto t_router_end = std::chrono::high_resolution_clock::now();
    tel.router_time_ms = std::chrono::duration<double, std::milli>(t_router_end - t_router_start).count();

    tel.selected_experts = selection.expert_indices;
    tel.routing_weights = selection.expert_weights;

    // 4. Shared expert execution
    if (has_shared_expert_) {
        compute_shared_expert(norm_ffn.data(), moe_out.data());
    }

    // 5. Sparse Expert Loading & Compute
    double load_time_acc = 0.0;
    double comp_time_acc = 0.0;

    std::vector<std::shared_ptr<const ExpertPayload>> payloads;
    auto t_l0 = std::chrono::high_resolution_clock::now();
    size_t b_before = byte_loader_->telemetry().total_bytes_read;
    uint64_t h_before = byte_loader_->telemetry().cache_hits;
    uint64_t m_before = byte_loader_->telemetry().cache_misses;

    if (!byte_loader_->load_layer_experts_parallel(layer_id_, selection.expert_indices, payloads)) {
        throw std::runtime_error("M8TransformerLayer: Physical expert for layer " + std::to_string(layer_id_) + " not available on disk. Refusing synthetic execution.");
    }
    auto t_l1 = std::chrono::high_resolution_clock::now();
    load_time_acc = std::chrono::duration<double, std::milli>(t_l1 - t_l0).count();
    tel.expert_bytes_loaded += (byte_loader_->telemetry().total_bytes_read - b_before);
    tel.expert_cache_hits += static_cast<int>(byte_loader_->telemetry().cache_hits - h_before);
    tel.expert_cache_misses += static_cast<int>(byte_loader_->telemetry().cache_misses - m_before);

    // Check if GPU acceleration path is active
    if (gpu_kernel_ && gpu_kernel_->is_initialized() && selection.expert_indices.size() == 6) {
        tel.gpu_accelerated = true;
        std::vector<const uint8_t*> scales(6);
        std::vector<const uint8_t*> weights(6);
        for (int e = 0; e < 6; ++e) {
            scales[e] = payloads[e]->scales.data();
            weights[e] = payloads[e]->weights.data();
        }

        std::vector<float> routed_out(D);
        auto t_c0 = std::chrono::high_resolution_clock::now();
        gpu_kernel_->forward_top6_layer(layer_id_, selection.expert_indices, scales, weights, selection.expert_weights, norm_ffn.data(), routed_out.data());
        for (int i = 0; i < D; ++i) {
            moe_out[i] += routed_out[i];
        }
        auto t_c1 = std::chrono::high_resolution_clock::now();
        comp_time_acc = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
    } else {
        // CPU fallback path
        tel.gpu_accelerated = false;
        auto t_c0 = std::chrono::high_resolution_clock::now();
        for (size_t k = 0; k < selection.expert_indices.size(); ++k) {
            float w = selection.expert_weights[k];
            routed_kernel_.attach(payloads[k]->scales.data(), payloads[k]->weights.data());
            routed_kernel_.forward(norm_ffn.data(), moe_out.data(), w, /*accumulate=*/true);
        }
        auto t_c1 = std::chrono::high_resolution_clock::now();
        comp_time_acc = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
    }

    tel.expert_load_time_ms = load_time_acc;
    tel.expert_compute_time_ms = comp_time_acc;

    // 6. Second residual connection: residual_attn + moe_out
    for (int i = 0; i < D; ++i) {
        out[i] = residual_attn[i] + moe_out[i];
    }

    auto t_layer_end = std::chrono::high_resolution_clock::now();
    tel.total_layer_time_ms = std::chrono::duration<double, std::milli>(t_layer_end - t_layer_start).count();
}

void M8TransformerLayer::compute_hc_mixes(const float* x, const float* hc_fn, const float* hc_scale, const float* hc_base,
                                          float* pre, float* post, float* comb) const {
    const int HC = 4;
    const int HC_D = HC * 5120; // 20480
    const int MIX_HC = 24;

    // 1. Mean square of x (all 20480 elements)
    double mean_sq = 0.0;
    for (int i = 0; i < HC_D; ++i) {
        mean_sq += static_cast<double>(x[i]) * x[i];
    }
    mean_sq /= HC_D;
    float rsqrt_val = 1.0f / std::sqrt(static_cast<float>(mean_sq) + 1e-20f);

    // 2. Linear projection: mixes = (hc_fn @ x) * rsqrt
    float mixes[24];
    for (int r = 0; r < MIX_HC; ++r) {
        const float* fn_row = hc_fn + r * HC_D;
        mixes[r] = dot_product_f32_avx2(fn_row, x, HC_D) * rsqrt_val;
    }

    // 3. pre: sigmoid(mixes[0..3] * scale[0] + base[0..3]) + eps
    for (int j = 0; j < HC; ++j) {
        float v = mixes[j] * hc_scale[0] + hc_base[j];
        pre[j] = 1.0f / (1.0f + std::exp(-v)) + 1e-6f;
    }

    // 4. post: 2.0 * sigmoid(mixes[4..7] * scale[1] + base[4..7])
    for (int j = 0; j < HC; ++j) {
        float v = mixes[j + HC] * hc_scale[1] + hc_base[j + HC];
        post[j] = 2.0f / (1.0f + std::exp(-v));
    }

    // 5. comb: mixes[8..23] * scale[2] + base[8..23]
    float c[4][4];
    for (int j = 0; j < HC; ++j) {
        for (int k = 0; k < HC; ++k) {
            int idx = 8 + j * HC + k;
            c[j][k] = mixes[idx] * hc_scale[2] + hc_base[idx];
        }
    }

    // 6. Sinkhorn iterations
    // Row softmax + eps
    for (int j = 0; j < HC; ++j) {
        float max_v = c[j][0];
        for (int k = 1; k < HC; ++k) if (c[j][k] > max_v) max_v = c[j][k];
        float sum_exp = 0.0f;
        for (int k = 0; k < HC; ++k) {
            c[j][k] = std::exp(c[j][k] - max_v);
            sum_exp += c[j][k];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-20f);
        for (int k = 0; k < HC; ++k) {
            c[j][k] = c[j][k] * inv_sum + 1e-6f;
        }
    }

    // Col normalization
    for (int k = 0; k < HC; ++k) {
        float col_sum = 0.0f;
        for (int j = 0; j < HC; ++j) col_sum += c[j][k];
        float inv_col = 1.0f / (col_sum + 1e-6f);
        for (int j = 0; j < HC; ++j) c[j][k] *= inv_col;
    }

    // 19 more iterations
    for (int iter = 0; iter < 19; ++iter) {
        // Row norm
        for (int j = 0; j < HC; ++j) {
            float row_sum = 0.0f;
            for (int k = 0; k < HC; ++k) row_sum += c[j][k];
            float inv_row = 1.0f / (row_sum + 1e-6f);
            for (int k = 0; k < HC; ++k) c[j][k] *= inv_row;
        }
        // Col norm
        for (int k = 0; k < HC; ++k) {
            float col_sum = 0.0f;
            for (int j = 0; j < HC; ++j) col_sum += c[j][k];
            float inv_col = 1.0f / (col_sum + 1e-6f);
            for (int j = 0; j < HC; ++j) c[j][k] *= inv_col;
        }
    }

    // Copy to comb output
    for (int j = 0; j < HC; ++j) {
        for (int k = 0; k < HC; ++k) {
            comb[j * HC + k] = c[j][k];
        }
    }
}

void M8TransformerLayer::forward_hc(const float* in, float* out, const float* pre_mix_in, float* pre_mix_out, int start_pos, LayerTelemetry& tel) {
    auto t_layer_start = std::chrono::high_resolution_clock::now();
    tel.layer_id = layer_id_;

    const int D = 5120;
    const int HC = 4;
    const int HC_D = HC * D; // 20480

    // 1. Attention HC mixes
    std::vector<float> attn_pre(HC);
    std::vector<float> attn_post(HC);
    std::vector<float> attn_comb(HC * HC);
    compute_hc_mixes(in, hc_attn_fn_.data(), hc_attn_scale_.data(), hc_attn_base_.data(),
                     attn_pre.data(), attn_post.data(), attn_comb.data());

    // 2. Collapse input with pre_mix_in
    std::vector<float> x_attn(D, 0.0f);
    for (int j = 0; j < HC; ++j) {
        float m = pre_mix_in[j];
        const float* src = in + j * D;
        for (int d = 0; d < D; ++d) {
            x_attn[d] += m * src[d];
        }
    }

    // 3. Attn Norm + Attention Forward
    std::vector<float> norm_attn_in(D);
    rms_norm(x_attn.data(), attn_norm_.data(), norm_attn_in.data(), D);

    auto t_attn_start = std::chrono::high_resolution_clock::now();
    std::vector<float> attn_out(D);
    self_attn_.forward(norm_attn_in.data(), attn_out.data(), start_pos);
    tel.kv_update_time_ms = self_attn_.last_kv_update_ms();
    tel.kv_attend_time_ms = self_attn_.last_kv_attend_ms();
    auto t_attn_end = std::chrono::high_resolution_clock::now();
    tel.attn_time_ms = std::chrono::duration<double, std::milli>(t_attn_end - t_attn_start).count();

    // 4. Attention HC post: expand back to HC copies and mix residual
    std::vector<float> h_mid(HC_D);
    for (int j = 0; j < HC; ++j) {
        float post_j = attn_post[j];
        float* dst = h_mid.data() + j * D;
        for (int d = 0; d < D; ++d) {
            dst[d] = post_j * attn_out[d];
            for (int k = 0; k < HC; ++k) {
                dst[d] += attn_comb[k * HC + j] * in[k * D + d];
            }
        }
    }

    // 5. FFN HC mixes from h_mid
    std::vector<float> ffn_pre(HC);
    std::vector<float> ffn_post(HC);
    std::vector<float> ffn_comb(HC * HC);
    compute_hc_mixes(h_mid.data(), hc_ffn_fn_.data(), hc_ffn_scale_.data(), hc_ffn_base_.data(),
                     ffn_pre.data(), ffn_post.data(), ffn_comb.data());

    // 6. Collapse h_mid with attn_pre
    std::vector<float> x_ffn(D, 0.0f);
    for (int j = 0; j < HC; ++j) {
        float m = attn_pre[j];
        const float* src = h_mid.data() + j * D;
        for (int d = 0; d < D; ++d) {
            x_ffn[d] += m * src[d];
        }
    }

    // 7. FFN Norm + Router + Experts
    std::vector<float> norm_ffn(D);
    rms_norm(x_ffn.data(), ffn_norm_.data(), norm_ffn.data(), D);

    auto t_router_start = std::chrono::high_resolution_clock::now();
    RouterSelection selection = router_.route(norm_ffn.data());
    auto t_router_end = std::chrono::high_resolution_clock::now();
    tel.router_time_ms = std::chrono::duration<double, std::milli>(t_router_end - t_router_start).count();
    tel.selected_experts = selection.expert_indices;
    tel.routing_weights = selection.expert_weights;

    std::vector<float> moe_out(D, 0.0f);

    // Start reading the routed experts NOW so storage overlaps (a) the CPU shared-expert math below and
    // (b) on the GPU path, the uploads/compute of experts that have already arrived. Expert selection,
    // the order experts are accumulated in, and every number computed are unchanged.
    const bool use_gpu = gpu_kernel_ && gpu_kernel_->is_initialized() && selection.expert_indices.size() == 6;
    const ByteLoaderTelemetry lt_start = byte_loader_->telemetry();
    auto t_l0 = std::chrono::high_resolution_clock::now();
    // On the GPU path, VRAM is the expert cache: experts already resident there are not read at all.
    std::vector<char> in_vram(selection.expert_indices.size(), 0);
    // ASEMA_LEGACY_RAM_CACHE=1 restores the previous data path (every expert is read through the host
    // cache even when VRAM already holds it) so the two designs can be A/B-tested in one binary.
    static const bool legacy_ram_cache = std::getenv("ASEMA_LEGACY_RAM_CACHE") != nullptr;
    if (use_gpu && !legacy_ram_cache) {
        for (size_t k = 0; k < in_vram.size(); ++k) {
            in_vram[k] = gpu_kernel_->is_resident(layer_id_, selection.expert_indices[k]) ? 1 : 0;
        }
    }
    M8ByteRangeLoader::PendingExpertLoad pending = byte_loader_->begin_layer_experts(
        layer_id_, selection.expert_indices, use_gpu ? &in_vram : nullptr);

    // Once this layer's own reads have arrived, storage is idle while the GPU finishes and the next
    // layer's attention runs: use that time to start reading the experts the next layer will probably need.
    int reads_outstanding = 0;
    for (char r : in_vram) reads_outstanding += r ? 0 : 1;
    const int next_layer = layer_id_ + 1;
    auto start_prefetch = [&]() {
        const int k = byte_loader_->prefetch_k();
        if (!use_gpu || k <= 0 || next_layer >= 40) return;
        byte_loader_->prefetch_predicted(next_layer, selection.expert_indices, k,
                                         [&](int e) { return gpu_kernel_->is_resident(next_layer, e); });
    };
    if (use_gpu && reads_outstanding == 0) start_prefetch();
    double load_time_acc = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_l0).count();

    auto t_shared_0 = std::chrono::high_resolution_clock::now();
    if (has_shared_expert_) {
        compute_shared_expert(norm_ffn.data(), moe_out.data());
    }
    auto t_shared_1 = std::chrono::high_resolution_clock::now();
    tel.shared_expert_time_ms = std::chrono::duration<double, std::milli>(t_shared_1 - t_shared_0).count();

    // Compute routed experts (CPU or GPU)
    double comp_time_acc = 0.0;
    std::vector<std::shared_ptr<const ExpertPayload>> payloads;
    if (use_gpu) {
        tel.gpu_accelerated = true;
        std::vector<std::shared_ptr<const ExpertPayload>> held(6);
        double wait_ms = 0.0; // time blocked on storage inside the GPU stage
        std::vector<float> routed_out(D);
        auto gpu_before = gpu_kernel_->telemetry();
        auto t_c0 = std::chrono::high_resolution_clock::now();
        const bool gpu_ok = gpu_kernel_->forward_top6_layer_streamed(
            layer_id_, selection.expert_indices,
            [&](int e, const uint8_t*& scales, const uint8_t*& weights) {
                auto t_w0 = std::chrono::high_resolution_clock::now();
                held[e] = pending.get(e);
                wait_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_w0).count();
                if (!held[e]) return false;
                if (--reads_outstanding == 0) start_prefetch();
                scales = held[e]->scales.data();
                weights = held[e]->weights.data();
                return true;
            },
            selection.expert_weights, norm_ffn.data(), routed_out.data());
        auto t_c1 = std::chrono::high_resolution_clock::now();
        auto gpu_after = gpu_kernel_->telemetry();
        const bool published = byte_loader_->finish_layer_experts(pending, /*publish_to_cache=*/legacy_ram_cache);
        if (!gpu_ok || !published) {
            throw std::runtime_error("M8TransformerLayer: routed experts for layer " + std::to_string(layer_id_) +
                                     " could not be read from disk or executed on the GPU. Refusing synthetic execution.");
        }
        for (int i = 0; i < D; ++i) moe_out[i] += routed_out[i];
        load_time_acc += wait_ms;
        comp_time_acc = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count() - wait_ms;
        tel.gpu_upload_time_ms = gpu_after.host_to_device_time_ms - gpu_before.host_to_device_time_ms;
        tel.gpu_kernel_time_ms = gpu_after.kernel_compute_time_ms - gpu_before.kernel_compute_time_ms;
        tel.gpu_readback_time_ms = gpu_after.device_to_host_time_ms - gpu_before.device_to_host_time_ms;
        tel.gpu_busy_upload_ms = gpu_after.gpu_busy_upload_ms - gpu_before.gpu_busy_upload_ms;
        tel.gpu_busy_compute_ms = gpu_after.gpu_busy_compute_ms - gpu_before.gpu_busy_compute_ms;
    } else {
        tel.gpu_accelerated = false;
        auto t_w0 = std::chrono::high_resolution_clock::now();
        for (size_t k = 0; k < selection.expert_indices.size(); ++k) {
            payloads.push_back(pending.get(k));
        }
        const bool published = byte_loader_->finish_layer_experts(pending);
        load_time_acc += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_w0).count();
        bool all_present = published;
        for (const auto& p : payloads) all_present = all_present && (p != nullptr);
        if (!all_present) {
            throw std::runtime_error("M8TransformerLayer: Physical expert for layer " + std::to_string(layer_id_) + " not available on disk.");
        }
        auto t_c0 = std::chrono::high_resolution_clock::now();
        for (size_t k = 0; k < selection.expert_indices.size(); ++k) {
            routed_kernel_.attach(payloads[k]->scales.data(), payloads[k]->weights.data());
            routed_kernel_.forward(norm_ffn.data(), moe_out.data(), selection.expert_weights[k], /*accumulate=*/true);
        }
        auto t_c1 = std::chrono::high_resolution_clock::now();
        comp_time_acc = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
    }
    const ByteLoaderTelemetry lt_end = byte_loader_->telemetry();
    tel.expert_bytes_loaded += (lt_end.total_bytes_read - lt_start.total_bytes_read);
    tel.expert_cache_hits += static_cast<int>(lt_end.cache_hits - lt_start.cache_hits);
    tel.expert_cache_misses += static_cast<int>(lt_end.cache_misses - lt_start.cache_misses);
    tel.expert_load_time_ms = load_time_acc;
    tel.expert_compute_time_ms = comp_time_acc;

    // 8. FFN HC post: expand back to HC copies and mix residual from h_mid
    for (int j = 0; j < HC; ++j) {
        float post_j = ffn_post[j];
        float* dst = out + j * D;
        for (int d = 0; d < D; ++d) {
            dst[d] = post_j * moe_out[d];
            for (int k = 0; k < HC; ++k) {
                dst[d] += ffn_comb[k * HC + j] * h_mid[k * D + d];
            }
        }
    }

    // 9. Output next pre_mix (ffn_pre)
    std::memcpy(pre_mix_out, ffn_pre.data(), HC * sizeof(float));

    auto t_layer_end = std::chrono::high_resolution_clock::now();
    tel.total_layer_time_ms = std::chrono::duration<double, std::milli>(t_layer_end - t_layer_start).count();
}

} // namespace m8
} // namespace asema
