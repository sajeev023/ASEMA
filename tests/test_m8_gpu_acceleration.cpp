#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_gpu_expert_kernel.hpp"
#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA M8.18: GPU FP4 SWIGLU ACCELERATION & NUMERICAL VERIFICATION  \n";
    std::cout << "======================================================================\n\n";

    int failures = 0;

    // 1. Initialize GPU Kernel
    std::cout << "[PHASE 3] GPU Backend Initialization (DirectCompute / Direct3D 11)...\n";
    asema::m8::M8GpuExpertKernel gpu_kernel;
    if (!gpu_kernel.initialize()) {
        std::cerr << "  FAILED: GPU kernel initialization failed.\n";
        return 1;
    }
    std::cout << "  PASSED: GPU DirectCompute device initialized successfully.\n\n";

    // 2. Load Real Expert 0 from Storage
    std::cout << "[PHASE 5] Loading Real Expert 0 from Dual-NVMe Storage...\n";
    auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
    vol_mgr->register_volume(asema::m8::paths::primary_shards());
    vol_mgr->register_volume(asema::m8::paths::secondary_shards());
    asema::m8::M8ByteRangeLoader loader(vol_mgr);

    asema::m8::ExpertPayload payload0;
    if (!loader.load_expert_payload(0, 0, payload0)) {
        std::cerr << "  FAILED: Could not load layer 0 expert 0.\n";
        return 1;
    }
    std::cout << "  Expert 0 loaded: " << payload0.total_bytes() << " bytes.\n\n";

    const int H = gpu_kernel.dims().hidden_dim;
    std::vector<float> x(H);
    for (int i = 0; i < H; ++i) {
        x[i] = std::sin(static_cast<float>(i + 1) * 0.02f);
    }

    // 3. CPU Reference SwiGLU Forward Pass
    std::cout << "[PHASE 4] Running CPU Reference SwiGLU Expert Forward...\n";
    std::vector<float> cpu_out(H, 0.0f);
    asema::m8::M8ExpertKernel cpu_kernel;
    cpu_kernel.attach(payload0.scales.data(), payload0.weights.data());

    auto t_cpu_start = std::chrono::high_resolution_clock::now();
    cpu_kernel.forward(x.data(), cpu_out.data(), 1.0f, false);
    auto t_cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start).count();
    std::cout << "  CPU Forward Latency: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n\n";

    // 4. GPU SwiGLU Forward Pass
    std::cout << "[PHASE 4] Running GPU FP4 SwiGLU Expert Forward...\n";
    std::vector<float> gpu_out(H, 0.0f);
    gpu_kernel.reset_telemetry();

    auto t_gpu_start = std::chrono::high_resolution_clock::now();
    bool gpu_ok = gpu_kernel.forward_expert(payload0.scales.data(), payload0.weights.data(), x.data(), gpu_out.data(), 1.0f, false);
    auto t_gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t_gpu_end - t_gpu_start).count();

    if (!gpu_ok) {
        std::cerr << "  FAILED: GPU forward_expert returned false.\n";
        failures++;
    } else {
        std::cout << "  GPU Forward Latency (End-to-End): " << std::fixed << std::setprecision(2) << gpu_ms << " ms\n";
        std::cout << "    - Host->Device Upload:          " << gpu_kernel.telemetry().host_to_device_time_ms << " ms\n";
        std::cout << "    - Kernel Compute Time:          " << gpu_kernel.telemetry().kernel_compute_time_ms << " ms\n";
        std::cout << "    - Device->Host Readback:        " << gpu_kernel.telemetry().device_to_host_time_ms << " ms\n\n";
    }

    // 5. Numerical Equivalence Comparison
    std::cout << "[PHASE 4] Numerical Comparison (CPU vs GPU)...\n";
    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    double dot_prod = 0.0, norm_cpu = 0.0, norm_gpu = 0.0;
    for (int i = 0; i < H; ++i) {
        float diff = std::abs(cpu_out[i] - gpu_out[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_abs_diff += diff;
        dot_prod += cpu_out[i] * gpu_out[i];
        norm_cpu += cpu_out[i] * cpu_out[i];
        norm_gpu += gpu_out[i] * gpu_out[i];
    }
    float mean_abs_diff = sum_abs_diff / H;
    double cosine_sim = dot_prod / (std::sqrt(norm_cpu) * std::sqrt(norm_gpu));

    std::cout << "  Max Absolute Difference:  " << std::scientific << max_abs_diff << "\n";
    std::cout << "  Mean Absolute Difference: " << mean_abs_diff << "\n";
    std::cout << "  Cosine Similarity:        " << std::fixed << std::setprecision(8) << cosine_sim << "\n";
    std::cout << "  CPU Output Norm:          " << std::sqrt(norm_cpu) << "\n";
    std::cout << "  GPU Output Norm:          " << std::sqrt(norm_gpu) << "\n";
    std::cout << "  Speedup vs CPU:           " << std::setprecision(2) << (cpu_ms / gpu_ms) << "x\n\n";

    if (max_abs_diff > 1e-4f || cosine_sim < 0.9999) {
        std::cerr << "  FAILED: Numerical divergence between CPU and GPU exceeds tolerance!\n";
        failures++;
    } else {
        std::cout << "  PASSED: GPU FP4 SwiGLU is numerically identical to CPU reference within tolerance.\n\n";
    }

    // 6. Test Top-6 Layer Forward on GPU vs CPU
    std::cout << "[PHASE 5] Testing Top-6 Layer Direct GPU Execution...\n";
    std::vector<int> expert_ids = { 0, 1, 4, 6, 7, 8 };
    std::vector<float> router_weights = { 0.25f, 0.20f, 0.18f, 0.15f, 0.12f, 0.10f };

    std::vector<asema::m8::ExpertPayload> top6_payloads(6);
    std::vector<const uint8_t*> top6_scales(6);
    std::vector<const uint8_t*> top6_weights(6);
    for (int e = 0; e < 6; ++e) {
        bool ok = loader.load_expert_payload(0, expert_ids[e], top6_payloads[e]);
        if (!ok) {
            std::cerr << "  FAILED to load expert " << expert_ids[e] << "!\n";
        }
        top6_scales[e] = top6_payloads[e].scales.data();
        top6_weights[e] = top6_payloads[e].weights.data();
    }

    // CPU Top-6
    std::vector<float> cpu_top6_out(H, 0.0f);
    auto t_cpu_top6_start = std::chrono::high_resolution_clock::now();
    for (int e = 0; e < 6; ++e) {
        asema::m8::M8ExpertKernel k;
        k.attach(top6_scales[e], top6_weights[e]);
        k.forward(x.data(), cpu_top6_out.data(), router_weights[e], e > 0);
    }
    auto t_cpu_top6_end = std::chrono::high_resolution_clock::now();
    double cpu_top6_ms = std::chrono::duration<double, std::milli>(t_cpu_top6_end - t_cpu_top6_start).count();

    // GPU Top-6
    std::vector<float> gpu_top6_out(H, 0.0f);
    gpu_kernel.reset_telemetry();
    auto t_gpu_top6_start = std::chrono::high_resolution_clock::now();
    bool top6_ok = gpu_kernel.forward_top6_layer(top6_scales, top6_weights, router_weights, x.data(), gpu_top6_out.data());
    auto t_gpu_top6_end = std::chrono::high_resolution_clock::now();
    double gpu_top6_ms = std::chrono::duration<double, std::milli>(t_gpu_top6_end - t_gpu_top6_start).count();

    if (!top6_ok) {
        std::cerr << "  FAILED: GPU forward_top6_layer returned false.\n";
        failures++;
    } else {
        std::cout << "  CPU Top-6 Execution Latency: " << std::fixed << std::setprecision(2) << cpu_top6_ms << " ms\n";
        std::cout << "  GPU Top-6 Execution Latency: " << std::fixed << std::setprecision(2) << gpu_top6_ms << " ms\n";
        std::cout << "  Top-6 Layer Compute Speedup: " << (cpu_top6_ms / gpu_top6_ms) << "x\n";

        float top6_max_diff = 0.0f;
        double top6_dot = 0.0, top6_nc = 0.0, top6_ng = 0.0;
        for (int i = 0; i < H; ++i) {
            float d = std::abs(cpu_top6_out[i] - gpu_top6_out[i]);
            if (d > top6_max_diff) top6_max_diff = d;
            top6_dot += cpu_top6_out[i] * gpu_top6_out[i];
            top6_nc += cpu_top6_out[i] * cpu_top6_out[i];
            top6_ng += gpu_top6_out[i] * gpu_top6_out[i];
        }
        double top6_cos = top6_dot / (std::sqrt(top6_nc) * std::sqrt(top6_ng));
        std::cout << "  CPU Top-6 Norm:                " << std::sqrt(top6_nc) << "\n";
        std::cout << "  GPU Top-6 Norm:                " << std::sqrt(top6_ng) << "\n";
        std::cout << "  Top-6 Max Absolute Difference: " << std::scientific << top6_max_diff << "\n";
        std::cout << "  Top-6 Cosine Similarity:       " << std::fixed << std::setprecision(8) << top6_cos << "\n";

        if (top6_max_diff < 1e-4f && top6_cos > 0.9999) {
            std::cout << "  PASSED: GPU Top-6 execution is numerically equivalent.\n\n";
        } else {
            std::cerr << "  FAILED: GPU Top-6 execution divergence!\n\n";
            failures++;
        }
    }

    if (failures == 0) {
        std::cout << ">>> ALL M8.18 GPU ACCELERATION TESTS PASSED! <<<\n";
        return 0;
    } else {
        std::cerr << ">>> " << failures << " TESTS FAILED! <<<\n";
        return 1;
    }
}
