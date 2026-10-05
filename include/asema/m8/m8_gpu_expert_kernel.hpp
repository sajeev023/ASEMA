#pragma once

#include "asema/m8/m8_expert_kernel.hpp"
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11ComputeShader;
struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11UnorderedAccessView;
struct ID3D11Query;

namespace asema {
namespace m8 {

struct GpuKernelTelemetry {
    double host_to_device_time_ms{0.0};
    double kernel_compute_time_ms{0.0};
    double device_to_host_time_ms{0.0};
    double total_gpu_time_ms{0.0};
    uint64_t total_dispatches{0};
    uint64_t vram_allocated_bytes{0};
    uint64_t vram_cache_hits{0};
    uint64_t vram_cache_misses{0};
    // GPU-side execution time from D3D11 timestamp queries (what the GPU actually spent working,
    // as opposed to CPU wall time which also includes waiting).
    double gpu_busy_upload_ms{0.0};   // time the GPU spent executing expert uploads (copies)
    double gpu_busy_compute_ms{0.0};  // time the GPU spent executing expert compute kernels
};

class M8GpuExpertKernel {
public:
    explicit M8GpuExpertKernel(ExpertDimensions dims = ExpertDimensions{});
    ~M8GpuExpertKernel();

    // Non-copyable, movable
    M8GpuExpertKernel(const M8GpuExpertKernel&) = delete;
    M8GpuExpertKernel& operator=(const M8GpuExpertKernel&) = delete;

    // Initialize D3D11 device, compile compute shaders, and allocate bounded VRAM scratch buffers
    bool initialize();

    // Execute single expert SwiGLU pass on GPU
    bool forward_expert(const uint8_t* scales_ptr,
                        const uint8_t* weights_ptr,
                        const float* x,
                        float* out,
                        float router_weight = 1.0f,
                        bool accumulate = false);

    // Execute all 6 top-routed experts for a layer entirely on GPU with VRAM slot cache
    bool forward_top6_layer(int layer_id,
                            const std::vector<int>& expert_ids,
                            const std::vector<const uint8_t*>& expert_scales,
                            const std::vector<const uint8_t*>& expert_weights,
                            const std::vector<float>& router_weights,
                            const float* x,
                            float* out);

    // Streamed variant: `provider(e, scales, weights)` is called for e = 0..5 in order, just before
    // expert e is uploaded, and may block until that expert has been read from storage. Returning
    // false aborts the layer. Identical math and accumulation order to the vector overload.
    using ExpertProvider = std::function<bool(int e, const uint8_t*& scales, const uint8_t*& weights)>;
    bool forward_top6_layer_streamed(int layer_id,
                                     const std::vector<int>& expert_ids,
                                     const ExpertProvider& provider,
                                     const std::vector<float>& router_weights,
                                     const float* x,
                                     float* out);

    bool forward_top6_layer(const std::vector<const uint8_t*>& expert_scales,
                            const std::vector<const uint8_t*>& expert_weights,
                            const std::vector<float>& router_weights,
                            const float* x,
                            float* out);

    // True if this layer's expert is already resident in a VRAM slot, so its bytes need not be read
    // from storage at all.
    bool is_resident(int layer_id, int expert_id) const;

    // Optional: relative cost of re-reading an expert (1 = fast drive). Eviction then keeps experts that
    // are expensive to re-read longer. Without it all experts weigh the same.
    using ExpertCostFn = std::function<double(int layer_id, int expert_id)>;
    void set_expert_cost_function(ExpertCostFn fn) { cost_fn_ = std::move(fn); }

    bool is_initialized() const { return initialized_; }
    const GpuKernelTelemetry& telemetry() const { return telemetry_; }
    void reset_telemetry();

    const ExpertDimensions& dims() const { return dims_; }

    ID3D11Device* device() const { return device_; }
    ID3D11DeviceContext* context() const { return context_; }

private:
    ExpertDimensions dims_;
    bool initialized_{false};
    GpuKernelTelemetry telemetry_;

    // D3D11 Core Interfaces
    ID3D11Device* device_{nullptr};
    ID3D11DeviceContext* context_{nullptr};

    // Shaders
    ID3D11ComputeShader* shader_gemv_{nullptr};
    ID3D11ComputeShader* shader_swiglu_{nullptr};
    ID3D11ComputeShader* shader_accumulate_{nullptr};

    // Constant buffers
    ID3D11Buffer* cb_gemv_params_{nullptr};
    ID3D11Buffer* cb_swiglu_params_{nullptr};
    ID3D11Buffer* cb_accum_params_{nullptr};

    // VRAM Intermediate Buffers
    ID3D11Buffer* buf_x_{nullptr};
    ID3D11ShaderResourceView* srv_x_{nullptr};

    ID3D11Buffer* buf_gate_{nullptr};
    ID3D11ShaderResourceView* srv_gate_{nullptr};
    ID3D11UnorderedAccessView* uav_gate_{nullptr};

    ID3D11Buffer* buf_up_{nullptr};
    ID3D11ShaderResourceView* srv_up_{nullptr};
    ID3D11UnorderedAccessView* uav_up_{nullptr};

    ID3D11Buffer* buf_act_{nullptr};
    ID3D11ShaderResourceView* srv_act_{nullptr};
    ID3D11UnorderedAccessView* uav_act_{nullptr};

    ID3D11Buffer* buf_expert_out_{nullptr};
    ID3D11ShaderResourceView* srv_expert_out_{nullptr};
    ID3D11UnorderedAccessView* uav_expert_out_{nullptr};

    ID3D11Buffer* buf_accum_out_{nullptr};
    ID3D11ShaderResourceView* srv_accum_out_{nullptr};
    ID3D11UnorderedAccessView* uav_accum_out_{nullptr};

    // Staging buffer for fast CPU readback (20 KB)
    ID3D11Buffer* buf_staging_out_{nullptr};

    // GPU timestamp queries: per expert {before upload, after upload, after compute} x 6 experts
    static constexpr int kTimestampsPerLayer = 18;
    ID3D11Query* q_disjoint_{nullptr};
    ID3D11Query* q_ts_[kTimestampsPerLayer] = {};
    void collect_gpu_timestamps(int n_experts);

    // Multi-slot VRAM Expert Cache (bounded under 5 GB ceiling)
    struct VramExpertSlot {
        int layer_id{-1};
        int expert_id{-1};
        ID3D11Buffer* buf_scales{nullptr};
        ID3D11ShaderResourceView* srv_w1_scales{nullptr};
        ID3D11ShaderResourceView* srv_w2_scales{nullptr};
        ID3D11ShaderResourceView* srv_w3_scales{nullptr};

        ID3D11Buffer* buf_weights{nullptr};
        ID3D11ShaderResourceView* srv_w1_weights{nullptr};
        ID3D11ShaderResourceView* srv_w2_weights{nullptr};
        ID3D11ShaderResourceView* srv_w3_weights{nullptr};
    };

    // Number of expert slots in VRAM (192 slots ~ 3.6 GB). Override with ASEMA_VRAM_SLOTS.
    size_t num_vram_slots_{192};
    std::vector<VramExpertSlot> slots_;
    std::unordered_map<uint32_t, int> slot_map_;

    // Eviction policy: score = accesses / (1 + alpha * age), the same frequency+recency blend the host
    // cache uses (plain LRU gets no hits when the working set cycles through more experts than slots).
    std::unordered_map<uint32_t, uint32_t> access_freq_;   // per (layer,expert), whole history
    std::vector<uint64_t> slot_last_access_;                // per slot
    std::vector<double> slot_cost_;                         // per slot: relative re-read cost of its expert
    ExpertCostFn cost_fn_;
    std::vector<char> slot_pinned_;                         // in use by the layer being executed
    std::vector<int> pinned_this_layer_;
    uint64_t gpu_clock_{0};
    void note_access(uint32_t key);
    int pick_victim_slot() const;

    int get_or_upload_slot(int layer_id, int expert_id, const uint8_t* scales, const uint8_t* weights);

    void cleanup();
    bool compile_shaders();
    bool allocate_buffers();
    void update_constant_buffer(ID3D11Buffer* cb, const void* data, size_t size);
};

} // namespace m8
} // namespace asema
