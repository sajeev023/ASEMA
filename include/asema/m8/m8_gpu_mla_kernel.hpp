#pragma once

#include "asema/m8/m8_mla_attention.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11ComputeShader;
struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11UnorderedAccessView;

namespace asema {
namespace m8 {

struct GpuMlaTelemetry {
    double h2d_time_ms{0.0};
    double compute_time_ms{0.0};
    double d2h_time_ms{0.0};
    double total_mla_time_ms{0.0};
    uint64_t vram_allocated_bytes{0};
};

class M8GpuMlaKernel {
public:
    explicit M8GpuMlaKernel(MLAParams params = MLAParams{});
    ~M8GpuMlaKernel();

    M8GpuMlaKernel(const M8GpuMlaKernel&) = delete;
    M8GpuMlaKernel& operator=(const M8GpuMlaKernel&) = delete;

    // Initialize DirectCompute pipeline (optionally sharing existing D3D11 device/context)
    bool initialize(ID3D11Device* shared_device = nullptr, ID3D11DeviceContext* shared_context = nullptr);

    // Upload layer projection weights into bounded VRAM
    bool upload_layer_weights(const float* wq_a, const float* q_norm, const float* wq_b,
                             const float* wkv, const float* kv_norm,
                             const float* wo_a, const float* wo_b);

    // Upload RoPE trigonometric tables and attention sink
    bool upload_rope_tables(const float* cos_table, const float* sin_table);
    bool upload_attn_sink(const float* attn_sink);

    // Full GPU-Fused MLA Execution:
    // x -> W_QA -> Norm -> W_QB -> RoPE(Q)
    // x -> W_KV -> Norm -> RoPE(KV) -> Update KV Cache
    // -> Multi-Head Latent Attention -> Inverse RoPE(O)
    // -> Grouped W_OA -> W_OB -> out
    // ONLY reads back out (5120 floats = 20 KB). ZERO intermediate D2H/H2D.
    bool forward_full_mla(const float* x, int start_pos, float* out);

    // Standalone GPU RoPE verification
    bool apply_rope_gpu(float* data, int num_heads, int pos, bool inverse);

    // Accelerated Query & Key projection GEMV (M8.19 fallback/benchmarking)
    bool forward_qk_projections(const float* x, float* out_q, float* out_kv);

    // Accelerated Output projection GEMV (M8.19 fallback/benchmarking)
    bool forward_out_projection(const float* o, float* out);

    // Standalone dense GEMV on GPU: out = matrix * vec
    bool gemv_dense(int rows, int cols, const float* matrix, const float* vec, float* out);

    // Readback helpers for inspection and verification
    bool readback_kv_cache(float* out_cache, int num_tokens);
    bool readback_attn_o(float* out_o);

    void reset_kv_cache();
    int cached_tokens() const { return cached_tokens_; }

    bool is_initialized() const { return initialized_; }
    const GpuMlaTelemetry& telemetry() const { return telemetry_; }
    void reset_telemetry();

    ID3D11Device* device() const { return device_; }
    ID3D11DeviceContext* context() const { return context_; }

private:
    MLAParams params_;
    bool initialized_{false};
    bool owns_device_{true};
    GpuMlaTelemetry telemetry_;

    ID3D11Device* device_{nullptr};
    ID3D11DeviceContext* context_{nullptr};

    // Shaders
    ID3D11ComputeShader* shader_dense_gemv_{nullptr};
    ID3D11ComputeShader* shader_grouped_gemv_{nullptr};
    ID3D11ComputeShader* shader_rmsnorm_{nullptr};
    ID3D11ComputeShader* shader_rope_{nullptr};
    ID3D11ComputeShader* shader_update_kv_cache_{nullptr};
    ID3D11ComputeShader* shader_latent_attention_{nullptr};

    // Constant buffers
    ID3D11Buffer* cb_gemv_params_{nullptr};
    ID3D11Buffer* cb_grouped_gemv_params_{nullptr};
    ID3D11Buffer* cb_norm_params_{nullptr};
    ID3D11Buffer* cb_rope_params_{nullptr};
    ID3D11Buffer* cb_attn_params_{nullptr};
    ID3D11Buffer* cb_kv_params_{nullptr};

    // Weight Buffers in VRAM
    ID3D11Buffer* buf_wq_a_{nullptr};
    ID3D11ShaderResourceView* srv_wq_a_{nullptr};

    ID3D11Buffer* buf_q_norm_{nullptr};
    ID3D11ShaderResourceView* srv_q_norm_{nullptr};

    ID3D11Buffer* buf_wq_b_{nullptr};
    ID3D11ShaderResourceView* srv_wq_b_{nullptr};

    ID3D11Buffer* buf_wkv_{nullptr};
    ID3D11ShaderResourceView* srv_wkv_{nullptr};

    ID3D11Buffer* buf_kv_norm_{nullptr};
    ID3D11ShaderResourceView* srv_kv_norm_{nullptr};

    ID3D11Buffer* buf_wo_a_{nullptr};
    ID3D11ShaderResourceView* srv_wo_a_{nullptr};

    ID3D11Buffer* buf_wo_b_{nullptr};
    ID3D11ShaderResourceView* srv_wo_b_{nullptr};

    // KV Cache & Attention Constants in VRAM
    ID3D11Buffer* buf_kv_cache_{nullptr};
    ID3D11ShaderResourceView* srv_kv_cache_{nullptr};
    ID3D11UnorderedAccessView* uav_kv_cache_{nullptr};
    int cached_tokens_{0};

    ID3D11Buffer* buf_rope_cos_{nullptr};
    ID3D11ShaderResourceView* srv_rope_cos_{nullptr};
    ID3D11Buffer* buf_rope_sin_{nullptr};
    ID3D11ShaderResourceView* srv_rope_sin_{nullptr};
    ID3D11Buffer* buf_attn_sink_{nullptr};
    ID3D11ShaderResourceView* srv_attn_sink_{nullptr};

    // Intermediate activation buffers in VRAM
    ID3D11Buffer* buf_x_{nullptr};
    ID3D11ShaderResourceView* srv_x_{nullptr};

    ID3D11Buffer* buf_qr_raw_{nullptr};
    ID3D11ShaderResourceView* srv_qr_raw_{nullptr};
    ID3D11UnorderedAccessView* uav_qr_raw_{nullptr};

    ID3D11Buffer* buf_qr_{nullptr};
    ID3D11ShaderResourceView* srv_qr_{nullptr};
    ID3D11UnorderedAccessView* uav_qr_{nullptr};

    ID3D11Buffer* buf_q_{nullptr};
    ID3D11ShaderResourceView* srv_q_{nullptr};
    ID3D11UnorderedAccessView* uav_q_{nullptr};

    ID3D11Buffer* buf_kv_raw_{nullptr};
    ID3D11ShaderResourceView* srv_kv_raw_{nullptr};
    ID3D11UnorderedAccessView* uav_kv_raw_{nullptr};

    ID3D11Buffer* buf_kv_{nullptr};
    ID3D11ShaderResourceView* srv_kv_{nullptr};
    ID3D11UnorderedAccessView* uav_kv_{nullptr};

    ID3D11Buffer* buf_o_{nullptr};
    ID3D11ShaderResourceView* srv_o_{nullptr};
    ID3D11UnorderedAccessView* uav_o_{nullptr};

    ID3D11Buffer* buf_o_lora_{nullptr};
    ID3D11ShaderResourceView* srv_o_lora_{nullptr};
    ID3D11UnorderedAccessView* uav_o_lora_{nullptr};

    ID3D11Buffer* buf_out_{nullptr};
    ID3D11ShaderResourceView* srv_out_{nullptr};
    ID3D11UnorderedAccessView* uav_out_{nullptr};

    // Staging buffers for readback
    ID3D11Buffer* stage_q_{nullptr};
    ID3D11Buffer* stage_kv_{nullptr};
    ID3D11Buffer* stage_out_{nullptr};
    ID3D11Buffer* stage_rope_{nullptr};
    ID3D11Buffer* stage_kv_cache_{nullptr};

    // Ad-hoc GEMV scratch buffers
    ID3D11Buffer* adhoc_mat_{nullptr};
    ID3D11ShaderResourceView* adhoc_mat_srv_{nullptr};
    ID3D11Buffer* adhoc_vec_{nullptr};
    ID3D11ShaderResourceView* adhoc_vec_srv_{nullptr};
    ID3D11Buffer* adhoc_out_{nullptr};
    ID3D11UnorderedAccessView* adhoc_out_uav_{nullptr};
    ID3D11Buffer* adhoc_stage_{nullptr};
    size_t adhoc_mat_capacity_{0};
    size_t adhoc_vec_capacity_{0};
    size_t adhoc_out_capacity_{0};

    void cleanup();
    bool compile_shaders();
    bool allocate_buffers();
    void update_cb(ID3D11Buffer* cb, const void* data, size_t size);
    void dispatch_gemv(int rows, int cols, ID3D11ShaderResourceView* srv_mat, ID3D11ShaderResourceView* srv_vec, ID3D11UnorderedAccessView* uav_out);
    void dispatch_grouped_gemv(int group_rows, int group_cols, int num_groups, ID3D11ShaderResourceView* srv_mat, ID3D11ShaderResourceView* srv_vec, ID3D11UnorderedAccessView* uav_out);
    void dispatch_rmsnorm(int size, ID3D11ShaderResourceView* srv_in, ID3D11ShaderResourceView* srv_weight, ID3D11UnorderedAccessView* uav_out);
    void dispatch_rope(int num_heads, int pos, bool inverse, ID3D11UnorderedAccessView* uav_data);
    void dispatch_update_kv_cache(int ring_slot);
    void dispatch_latent_attention(int pos, int cached_tokens);
};

} // namespace m8
} // namespace asema
