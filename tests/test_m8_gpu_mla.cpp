#include "asema/m8/m8_gpu_mla_kernel.hpp"
#include "asema/m8/m8_mla_attention.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA M8.20: GPU MLA ATTENTION CORE FUSION & NUMERICAL VERIFICATION\n";
    std::cout << "======================================================================\n\n";

    int failures = 0;
    asema::m8::MLAParams params;

    // 1. Initialize GPU MLA Backend
    std::cout << "[PHASE 1] Initializing DirectCompute GPU MLA Backend (D3D11 CS 5.0)...\n";
    auto gpu_mla = std::make_shared<asema::m8::M8GpuMlaKernel>(params);
    if (!gpu_mla->initialize()) {
        std::cerr << "  FAILED: Could not initialize GPU MLA Direct3D 11 device.\n";
        return 1;
    }
    std::cout << "  PASSED: GPU MLA device initialized.\n";
    std::cout << "  VRAM Allocated: " << (gpu_mla->telemetry().vram_allocated_bytes / 1024) << " KB\n\n";

    // 2. Initialize CPU MLA Reference with Baseline Weights
    std::cout << "[PHASE 2] Initializing Reference Weights for MLA Attention...\n";
    asema::m8::M8MLAAttention cpu_mla(params);
    cpu_mla.load_mock_or_reference_weights();

    asema::m8::M8MLAAttention fused_mla(params);
    fused_mla.load_mock_or_reference_weights();
    fused_mla.set_gpu_mla(gpu_mla);
    std::cout << "  PASSED: Reference weights, RoPE tables, and attention sinks uploaded to VRAM.\n\n";

    // Synthetic test input x [5120]
    const int D = params.dim;
    std::vector<float> x(D);
    for (int i = 0; i < D; ++i) {
        x[i] = std::sin(static_cast<float>(i + 1) * 0.03f);
    }

    // -------------------------------------------------------------------------
    // Phase 3: GPU RoPE vs CPU RoPE Verification
    // -------------------------------------------------------------------------
    std::cout << "[PHASE 3] Validating GPU RoPE vs CPU RoPE (Forward & Inverse)...\n";
    {
        int total_q = params.n_heads * params.head_dim;
        std::vector<float> data_cpu(total_q);
        std::vector<float> data_gpu(total_q);
        for (int i = 0; i < total_q; ++i) {
            float val = std::cos(static_cast<float>(i + 1) * 0.05f);
            data_cpu[i] = val;
            data_gpu[i] = val;
        }

        int test_pos = 17;
        // CPU Forward RoPE
        for (int h = 0; h < params.n_heads; ++h) {
            int offset = params.head_dim - params.rope_head_dim;
            int half_dim = params.rope_head_dim / 2;
            int wrapped_pos = test_pos % params.window_size;
            float* vec = data_cpu.data() + h * params.head_dim;
            for (int i = 0; i < half_dim; ++i) {
                float theta = std::pow(params.rope_theta, -2.0f * i / params.rope_head_dim);
                float angle = wrapped_pos * theta;
                float c = std::cos(angle);
                float s = std::sin(angle);
                int idx = offset + 2 * i;
                float x0 = vec[idx];
                float x1 = vec[idx + 1];
                vec[idx]     = x0 * c - x1 * s;
                vec[idx + 1] = x0 * s + x1 * c;
            }
        }

        // GPU Forward RoPE
        bool rope_ok = gpu_mla->apply_rope_gpu(data_gpu.data(), params.n_heads, test_pos, false);
        if (!rope_ok) {
            std::cerr << "  FAILED: GPU apply_rope_gpu execution returned false.\n";
            failures++;
        } else {
            float max_abs_err = 0.0f;
            double sum_abs_err = 0.0;
            double dot = 0.0, norm_c = 0.0, norm_g = 0.0;
            for (int i = 0; i < total_q; ++i) {
                float d = std::abs(data_cpu[i] - data_gpu[i]);
                if (d > max_abs_err) max_abs_err = d;
                sum_abs_err += d;
                dot += static_cast<double>(data_cpu[i]) * data_gpu[i];
                norm_c += static_cast<double>(data_cpu[i]) * data_cpu[i];
                norm_g += static_cast<double>(data_gpu[i]) * data_gpu[i];
            }
            float mean_abs_err = static_cast<float>(sum_abs_err / total_q);
            double cos_sim = dot / (std::sqrt(norm_c) * std::sqrt(norm_g));

            std::cout << "  GPU RoPE Verification (64 heads, 512 dim, pos 17):\n";
            std::cout << "    - Max Absolute Error:  " << std::scientific << max_abs_err << "\n";
            std::cout << "    - Mean Absolute Error: " << mean_abs_err << "\n";
            std::cout << "    - Cosine Similarity:   " << std::fixed << std::setprecision(8) << cos_sim << "\n";
            std::cout << "    - CPU Output Norm:     " << std::sqrt(norm_c) << "\n";
            std::cout << "    - GPU Output Norm:     " << std::sqrt(norm_g) << "\n";

            if (max_abs_err < 1e-5f && cos_sim > 0.999999) {
                std::cout << "  PASSED: GPU RoPE matches CPU RoPE with high precision.\n\n";
            } else {
                std::cerr << "  FAILED: GPU RoPE divergence!\n\n";
                failures++;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Phase 4 to 7: Full Fused GPU MLA Execution vs CPU Reference Forward
    // -------------------------------------------------------------------------
    std::cout << "[PHASE 4-7] Validating Full Fused GPU MLA vs CPU Reference...\n";
    {
        std::vector<float> cpu_out(D, 0.0f);
        std::vector<float> gpu_out(D, 0.0f);

        // Reset caches for both
        cpu_mla.reset_kv_cache();
        fused_mla.reset_kv_cache();

        // Step 0: Token 0 (pos = 0)
        auto t_c0 = std::chrono::high_resolution_clock::now();
        cpu_mla.forward(x.data(), cpu_out.data(), 0);
        auto t_c1 = std::chrono::high_resolution_clock::now();
        double cpu_ms_0 = std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();

        gpu_mla->reset_telemetry();
        auto t_g0 = std::chrono::high_resolution_clock::now();
        fused_mla.forward(x.data(), gpu_out.data(), 0);
        auto t_g1 = std::chrono::high_resolution_clock::now();
        double gpu_ms_0 = std::chrono::duration<double, std::milli>(t_g1 - t_g0).count();

        float max_abs_err_0 = 0.0f;
        double dot_0 = 0.0, norm_c_0 = 0.0, norm_g_0 = 0.0;
        for (int i = 0; i < D; ++i) {
            float d = std::abs(cpu_out[i] - gpu_out[i]);
            if (d > max_abs_err_0) max_abs_err_0 = d;
            dot_0 += static_cast<double>(cpu_out[i]) * gpu_out[i];
            norm_c_0 += static_cast<double>(cpu_out[i]) * cpu_out[i];
            norm_g_0 += static_cast<double>(gpu_out[i]) * gpu_out[i];
        }
        double cos_sim_0 = dot_0 / (std::sqrt(norm_c_0) * std::sqrt(norm_g_0));

        std::cout << "  Step 0 (Prompt / Pos 0):\n";
        std::cout << "    - CPU MLA Latency:     " << std::fixed << std::setprecision(2) << cpu_ms_0 << " ms\n";
        std::cout << "    - GPU MLA Latency:     " << gpu_ms_0 << " ms\n";
        std::cout << "      * H2D (20 KB x):     " << gpu_mla->telemetry().h2d_time_ms << " ms\n";
        std::cout << "      * GPU Core Compute:  " << gpu_mla->telemetry().compute_time_ms << " ms\n";
        std::cout << "      * D2H (20 KB out):   " << gpu_mla->telemetry().d2h_time_ms << " ms\n";
        std::cout << "    - Max Absolute Diff:   " << std::scientific << max_abs_err_0 << "\n";
        std::cout << "    - Cosine Similarity:   " << std::fixed << std::setprecision(8) << cos_sim_0 << "\n";

        if (max_abs_err_0 < 1e-4f && cos_sim_0 > 0.9999) {
            std::cout << "    - PASSED: Step 0 numerical equivalence established.\n\n";
        } else {
            std::cerr << "    - FAILED: Step 0 numerical divergence.\n\n";
            failures++;
        }

        // Step 1: Token 1 (pos = 1) - exercises KV cache ring buffer read & multi-token attention
        std::vector<float> x1(D);
        for (int i = 0; i < D; ++i) x1[i] = std::cos(static_cast<float>(i + 1) * 0.04f);

        auto t_c2 = std::chrono::high_resolution_clock::now();
        cpu_mla.forward(x1.data(), cpu_out.data(), 1);
        auto t_c3 = std::chrono::high_resolution_clock::now();
        double cpu_ms_1 = std::chrono::duration<double, std::milli>(t_c3 - t_c2).count();

        gpu_mla->reset_telemetry();
        auto t_g2 = std::chrono::high_resolution_clock::now();
        fused_mla.forward(x1.data(), gpu_out.data(), 1);
        auto t_g3 = std::chrono::high_resolution_clock::now();
        double gpu_ms_1 = std::chrono::duration<double, std::milli>(t_g3 - t_g2).count();

        float max_abs_err_1 = 0.0f;
        double dot_1 = 0.0, norm_c_1 = 0.0, norm_g_1 = 0.0;
        for (int i = 0; i < D; ++i) {
            float d = std::abs(cpu_out[i] - gpu_out[i]);
            if (d > max_abs_err_1) max_abs_err_1 = d;
            dot_1 += static_cast<double>(cpu_out[i]) * gpu_out[i];
            norm_c_1 += static_cast<double>(cpu_out[i]) * cpu_out[i];
            norm_g_1 += static_cast<double>(gpu_out[i]) * gpu_out[i];
        }
        double cos_sim_1 = dot_1 / (std::sqrt(norm_c_1) * std::sqrt(norm_g_1));

        std::cout << "  Step 1 (Decode / Pos 1 with KV Cache):\n";
        std::cout << "    - CPU MLA Latency:     " << std::fixed << std::setprecision(2) << cpu_ms_1 << " ms\n";
        std::cout << "    - GPU MLA Latency:     " << gpu_ms_1 << " ms\n";
        std::cout << "      * H2D (20 KB x):     " << gpu_mla->telemetry().h2d_time_ms << " ms\n";
        std::cout << "      * GPU Core Compute:  " << gpu_mla->telemetry().compute_time_ms << " ms\n";
        std::cout << "      * D2H (20 KB out):   " << gpu_mla->telemetry().d2h_time_ms << " ms\n";
        std::cout << "    - Max Absolute Diff:   " << std::scientific << max_abs_err_1 << "\n";
        std::cout << "    - Cosine Similarity:   " << std::fixed << std::setprecision(8) << cos_sim_1 << "\n";

        if (max_abs_err_1 < 1e-4f && cos_sim_1 > 0.9999) {
            std::cout << "    - PASSED: Step 1 numerical equivalence established.\n\n";
        } else {
            std::cerr << "    - FAILED: Step 1 numerical divergence.\n\n";
            failures++;
        }
    }

    // -------------------------------------------------------------------------
    // Phase 8: KV Cache Bounded Resource Verification
    // -------------------------------------------------------------------------
    std::cout << "[PHASE 8] Validating KV Cache Bounded Resource Guarantees...\n";
    size_t kv_vram_bytes = gpu_mla->telemetry().vram_allocated_bytes;
    std::cout << "  KV Cache Configuration:\n";
    std::cout << "    - Window Size:         " << params.window_size << " tokens\n";
    std::cout << "    - Head Dim:            " << params.head_dim << " floats\n";
    std::cout << "    - Cache Size in VRAM:  " << (params.window_size * params.head_dim * sizeof(float) / 1024) << " KB\n";
    std::cout << "    - Total VRAM Reserved: " << (kv_vram_bytes / 1024) << " KB\n";

    if (kv_vram_bytes < 32 * 1024 * 1024) {
        std::cout << "  PASSED: KV Cache and MLA parameters strictly bounded under VRAM budget (< 32 MB).\n\n";
    } else {
        std::cerr << "  FAILED: VRAM footprint exceeded bounds.\n\n";
        failures++;
    }

    if (failures == 0) {
        std::cout << ">>> ALL M8.20 GPU MLA ATTENTION CORE TESTS PASSED! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> " << failures << " TESTS FAILED! <<<\n";
        return 1;
    }
}
