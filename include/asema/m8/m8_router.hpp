#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace asema {
namespace m8 {

class M8MultiVolumeManager;

struct RouterConfig {
    int num_routed_experts{384};
    int hidden_dim{5120};
    int top_k{6};
    float gate_temp{1.0f};
    float routed_scaling_factor{1.5f};
    bool norm_topk_prob{true};
};

struct RouterSelection {
    std::vector<int> expert_indices;   // Selected top-k expert IDs (0..num_routed_experts-1)
    std::vector<float> expert_weights; // Normalized and scaled routing weights
};

class M8Router {
public:
    explicit M8Router(RouterConfig config = RouterConfig{});

    // Load from contiguous raw BF16 weights [384, 5120] and FP32 biases [384]
    bool load_from_buffers(const uint16_t* bf16_weights, const float* fp32_biases);

    // Load from disk files
    bool load_from_files(const std::string& weight_path, const std::string& bias_path);

    // Load from physical checkpoint volumes via multi-volume manager
    bool load_from_vol_mgr(std::shared_ptr<M8MultiVolumeManager> vol_mgr, int layer_id);

    // Perform real router forward pass on input vector x [hidden_dim]
    RouterSelection route(const float* x) const;

    const RouterConfig& config() const { return config_; }
    bool is_loaded() const { return is_loaded_; }

private:
    RouterConfig config_;
    bool is_loaded_{false};
    std::vector<float> weights_fp32_; // [384 * 5120] unpacked to FP32 for maximum throughput
    std::vector<float> biases_fp32_;  // [384]
};

} // namespace m8
} // namespace asema
