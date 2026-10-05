#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace asema {
namespace m8 {

enum class ExecutionDevice {
    CPU_AVX2,
    GPU_DIRECT3D11
};

struct SublayerDecision {
    std::string sublayer_name;
    ExecutionDevice target_device;
    double estimated_compute_ms;
    double estimated_transfer_ms;
    std::string rationale;
};

class M8HybridScheduler {
public:
    M8HybridScheduler();

    // Determine optimal device for each transformer sublayer
    SublayerDecision schedule_mla(int hidden_dim, int seq_pos);
    SublayerDecision schedule_router(int hidden_dim, int num_experts);
    SublayerDecision schedule_experts(int num_active_experts, size_t total_payload_bytes);
    SublayerDecision schedule_rmsnorm(int hidden_dim);
    SublayerDecision schedule_residual(int hidden_dim);

    // Profile and update execution cost estimates based on live telemetry
    void update_profile(const std::string& sublayer, ExecutionDevice device, double measured_time_ms);

    // Get current allocation stats
    size_t persistent_vram_bytes() const { return persistent_vram_bytes_; }
    size_t persistent_ram_bytes() const { return persistent_ram_bytes_; }

private:
    size_t persistent_vram_bytes_{18 * 1024 * 1024}; // 18.28 MB
    size_t persistent_ram_bytes_{108 * 1024 * 1024};  // 107.58 MB
    double pcie_h2d_bandwidth_gbps_{11.2};
    double pcie_d2h_bandwidth_gbps_{12.4};
};

} // namespace m8
} // namespace asema
