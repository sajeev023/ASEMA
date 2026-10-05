#include "asema/m8/m8_gpu_mla_kernel.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace asema {
namespace m8 {

namespace {

static const char* s_hlsl_mla = R"(
cbuffer GemvParams : register(b0) {
    uint g_rows;
    uint g_cols;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float> g_matrix : register(t0);
StructuredBuffer<float> g_vector : register(t1);
RWStructuredBuffer<float> g_out : register(u0);

[numthreads(64, 1, 1)]
void CSDenseGemv(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint r = dispatchThreadId.x;
    if (r >= g_rows) return;

    uint row_offset = r * g_cols;
    float sum = 0.0f;
    uint cols4 = g_cols / 4;

    for (uint i = 0; i < cols4; ++i) {
        uint c = i * 4;
        sum += g_matrix[row_offset + c] * g_vector[c]
             + g_matrix[row_offset + c + 1] * g_vector[c + 1]
             + g_matrix[row_offset + c + 2] * g_vector[c + 2]
             + g_matrix[row_offset + c + 3] * g_vector[c + 3];
    }
    for (uint rem = cols4 * 4; rem < g_cols; ++rem) {
        sum += g_matrix[row_offset + rem] * g_vector[rem];
    }

    g_out[r] = sum;
}

cbuffer GroupedGemvParams : register(b0) {
    uint g_group_rows;
    uint g_group_cols;
    uint g_num_groups;
    uint g_grp_pad;
};

StructuredBuffer<float> g_grp_matrix : register(t0);
StructuredBuffer<float> g_grp_vector : register(t1);
RWStructuredBuffer<float> g_grp_out : register(u0);

[numthreads(64, 1, 1)]
void CSGroupedGemv(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint r = dispatchThreadId.x;
    uint total_rows = g_group_rows * g_num_groups;
    if (r >= total_rows) return;

    uint group_idx = r / g_group_rows;
    uint vec_base = group_idx * g_group_cols;
    uint row_offset = r * g_group_cols;

    float sum = 0.0f;
    uint cols4 = g_group_cols / 4;

    for (uint i = 0; i < cols4; ++i) {
        uint c = i * 4;
        sum += g_grp_matrix[row_offset + c]     * g_grp_vector[vec_base + c]
             + g_grp_matrix[row_offset + c + 1] * g_grp_vector[vec_base + c + 1]
             + g_grp_matrix[row_offset + c + 2] * g_grp_vector[vec_base + c + 2]
             + g_grp_matrix[row_offset + c + 3] * g_grp_vector[vec_base + c + 3];
    }
    for (uint rem = cols4 * 4; rem < g_group_cols; ++rem) {
        sum += g_grp_matrix[row_offset + rem] * g_grp_vector[vec_base + rem];
    }

    g_grp_out[r] = sum;
}

cbuffer NormParams : register(b0) {
    uint g_size;
    float g_eps;
    uint n_pad0;
    uint n_pad1;
};

StructuredBuffer<float> g_in : register(t0);
StructuredBuffer<float> g_weight : register(t1);
RWStructuredBuffer<float> g_norm_out : register(u0);

groupshared float s_sum[256];

[numthreads(256, 1, 1)]
void CSRmsNorm(uint3 threadId : SV_GroupThreadID) {
    uint tid = threadId.x;
    float local_sq = 0.0f;
    for (uint i = tid; i < g_size; i += 256) {
        float val = g_in[i];
        local_sq += val * val;
    }
    s_sum[tid] = local_sq;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float scale = 1.0f / sqrt(s_sum[0] / float(g_size) + g_eps);

    for (uint j = tid; j < g_size; j += 256) {
        g_norm_out[j] = g_in[j] * scale * g_weight[j];
    }
}

cbuffer RopeParams : register(b0) {
    uint g_head_dim;
    uint g_rope_head_dim;
    uint g_pos;
    uint g_inverse;
    uint g_num_heads;
    uint g_window_size;
    uint rope_pad0;
    uint rope_pad1;
};

StructuredBuffer<float> g_rope_cos : register(t0);
StructuredBuffer<float> g_rope_sin : register(t1);
RWStructuredBuffer<float> g_rope_data : register(u0);

[numthreads(64, 1, 1)]
void CSRoPE(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint gid = dispatchThreadId.x;
    uint half_dim = g_rope_head_dim / 2;
    uint total_pairs = g_num_heads * half_dim;
    if (gid >= total_pairs) return;

    uint head_idx = gid / half_dim;
    uint pair_idx = gid % half_dim;

    uint offset = g_head_dim - g_rope_head_dim;
    uint idx0 = head_idx * g_head_dim + offset + 2 * pair_idx;
    uint idx1 = idx0 + 1;

    uint wrapped_pos = g_pos % g_window_size;
    uint table_idx = wrapped_pos * half_dim + pair_idx;
    float c = g_rope_cos[table_idx];
    float s = g_rope_sin[table_idx];

    float x0 = g_rope_data[idx0];
    float x1 = g_rope_data[idx1];

    if (g_inverse != 0) {
        g_rope_data[idx0] =  x0 * c + x1 * s;
        g_rope_data[idx1] = -x0 * s + x1 * c;
    } else {
        g_rope_data[idx0] = x0 * c - x1 * s;
        g_rope_data[idx1] = x0 * s + x1 * c;
    }
}

cbuffer KVCacheParams : register(b0) {
    uint g_slot;
    uint g_kv_head_dim;
    uint kv_pad0;
    uint kv_pad1;
};

StructuredBuffer<float> g_kv_in : register(t0);
RWStructuredBuffer<float> g_kv_cache : register(u0);

[numthreads(64, 1, 1)]
void CSUpdateKVCache(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint i = dispatchThreadId.x;
    if (i >= g_kv_head_dim) return;

    g_kv_cache[g_slot * g_kv_head_dim + i] = g_kv_in[i];
}

cbuffer AttnParams : register(b0) {
    uint g_attn_num_heads;
    uint g_attn_head_dim;
    uint g_cached_tokens;
    uint g_attn_window_size;
    float g_softmax_scale;
    uint g_attn_pos;
    uint attn_pad0;
    uint attn_pad1;
};

StructuredBuffer<float> g_q : register(t0);
StructuredBuffer<float> g_kv_cache_in : register(t1);
StructuredBuffer<float> g_attn_sink : register(t2);
StructuredBuffer<float> g_rope_cos_a : register(t3);
StructuredBuffer<float> g_rope_sin_a : register(t4);
RWStructuredBuffer<float> g_attn_out : register(u0);

groupshared float s_scores[128];
groupshared float s_weights[128];
groupshared float s_max_val[128];
groupshared float s_sum_exp[128];

[numthreads(128, 1, 1)]
void CSLatentAttention(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID) {
    uint h = groupId.x;
    uint tid = threadId.x;

    if (h >= g_attn_num_heads) return;

    uint q_head_offset = h * g_attn_head_dim;

    // Step 1: Compute Q.K dot products in parallel across time steps
    float s = -1e30f;
    if (tid < g_cached_tokens) {
        uint kv_offset = tid * g_attn_head_dim;
        float dot = 0.0f;
        for (uint i = 0; i < 512; i += 4) {
            dot += g_q[q_head_offset + i]     * g_kv_cache_in[kv_offset + i]
                 + g_q[q_head_offset + i + 1] * g_kv_cache_in[kv_offset + i + 1]
                 + g_q[q_head_offset + i + 2] * g_kv_cache_in[kv_offset + i + 2]
                 + g_q[q_head_offset + i + 3] * g_kv_cache_in[kv_offset + i + 3];
        }
        s = dot * g_softmax_scale;
    }
    s_scores[tid] = s;
    s_max_val[tid] = (tid < g_cached_tokens) ? s : -1e30f;
    GroupMemoryBarrierWithGroupSync();

    // Step 2: Max Reduction for stable softmax
    for (uint stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max_val[tid] = max(s_max_val[tid], s_max_val[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float sink = g_attn_sink[h];
    float max_score = max(sink, s_max_val[0]);

    // Step 3: Shifted exponentials and sum reduction
    float exp_val = (tid < g_cached_tokens) ? exp(s_scores[tid] - max_score) : 0.0f;
    s_sum_exp[tid] = exp_val;
    GroupMemoryBarrierWithGroupSync();

    for (stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_exp[tid] += s_sum_exp[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float denom = exp(sink - max_score) + s_sum_exp[0];
    float inv_denom = 1.0f / (denom + 1e-20f);

    // Step 4: Normalized attention weights
    s_weights[tid] = (tid < g_cached_tokens) ? (exp_val * inv_denom) : 0.0f;
    GroupMemoryBarrierWithGroupSync();

    // Step 5: Weighted V aggregation
    uint d_base = tid * 4;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;

    for (uint t = 0; t < g_cached_tokens; ++t) {
        float w = s_weights[t];
        uint kv_offset = t * g_attn_head_dim + d_base;
        o0 += w * g_kv_cache_in[kv_offset];
        o1 += w * g_kv_cache_in[kv_offset + 1];
        o2 += w * g_kv_cache_in[kv_offset + 2];
        o3 += w * g_kv_cache_in[kv_offset + 3];
    }

    // Step 6: In-register inverse RoPE for head tail [448..511]
    uint offset = g_attn_head_dim - 64;
    uint wrapped_pos = g_attn_pos % g_attn_window_size;
    uint half_dim = 32;

    if (d_base >= offset) {
        uint pair0 = (d_base - offset) / 2;
        float c0 = g_rope_cos_a[wrapped_pos * half_dim + pair0];
        float s0 = g_rope_sin_a[wrapped_pos * half_dim + pair0];
        float new_o0 =  o0 * c0 + o1 * s0;
        float new_o1 = -o0 * s0 + o1 * c0;
        o0 = new_o0;
        o1 = new_o1;

        uint pair1 = pair0 + 1;
        float c1 = g_rope_cos_a[wrapped_pos * half_dim + pair1];
        float s1 = g_rope_sin_a[wrapped_pos * half_dim + pair1];
        float new_o2 =  o2 * c1 + o3 * s1;
        float new_o3 = -o2 * s1 + o3 * c1;
        o2 = new_o2;
        o3 = new_o3;
    }

    uint out_offset = q_head_offset + d_base;
    g_attn_out[out_offset]     = o0;
    g_attn_out[out_offset + 1] = o1;
    g_attn_out[out_offset + 2] = o2;
    g_attn_out[out_offset + 3] = o3;
}
)";

template <typename T>
void SafeRelease(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

} // namespace

M8GpuMlaKernel::M8GpuMlaKernel(MLAParams params)
    : params_(params) {}

M8GpuMlaKernel::~M8GpuMlaKernel() {
    cleanup();
}

void M8GpuMlaKernel::reset_telemetry() {
    telemetry_ = GpuMlaTelemetry{};
}

void M8GpuMlaKernel::cleanup() {
    SafeRelease(cb_gemv_params_);
    SafeRelease(cb_grouped_gemv_params_);
    SafeRelease(cb_norm_params_);
    SafeRelease(cb_rope_params_);
    SafeRelease(cb_attn_params_);
    SafeRelease(cb_kv_params_);

    SafeRelease(srv_wq_a_);
    SafeRelease(buf_wq_a_);
    SafeRelease(srv_q_norm_);
    SafeRelease(buf_q_norm_);
    SafeRelease(srv_wq_b_);
    SafeRelease(buf_wq_b_);
    SafeRelease(srv_wkv_);
    SafeRelease(buf_wkv_);
    SafeRelease(srv_kv_norm_);
    SafeRelease(buf_kv_norm_);
    SafeRelease(srv_wo_a_);
    SafeRelease(buf_wo_a_);
    SafeRelease(srv_wo_b_);
    SafeRelease(buf_wo_b_);

    SafeRelease(uav_kv_cache_);
    SafeRelease(srv_kv_cache_);
    SafeRelease(buf_kv_cache_);

    SafeRelease(srv_rope_cos_);
    SafeRelease(buf_rope_cos_);
    SafeRelease(srv_rope_sin_);
    SafeRelease(buf_rope_sin_);
    SafeRelease(srv_attn_sink_);
    SafeRelease(buf_attn_sink_);

    SafeRelease(srv_x_);
    SafeRelease(buf_x_);

    SafeRelease(uav_qr_raw_);
    SafeRelease(srv_qr_raw_);
    SafeRelease(buf_qr_raw_);

    SafeRelease(uav_qr_);
    SafeRelease(srv_qr_);
    SafeRelease(buf_qr_);

    SafeRelease(uav_q_);
    SafeRelease(srv_q_);
    SafeRelease(buf_q_);

    SafeRelease(uav_kv_raw_);
    SafeRelease(srv_kv_raw_);
    SafeRelease(buf_kv_raw_);

    SafeRelease(uav_kv_);
    SafeRelease(srv_kv_);
    SafeRelease(buf_kv_);

    SafeRelease(uav_o_);
    SafeRelease(srv_o_);
    SafeRelease(buf_o_);

    SafeRelease(uav_o_lora_);
    SafeRelease(srv_o_lora_);
    SafeRelease(buf_o_lora_);

    SafeRelease(uav_out_);
    SafeRelease(srv_out_);
    SafeRelease(buf_out_);

    SafeRelease(stage_q_);
    SafeRelease(stage_kv_);
    SafeRelease(stage_out_);
    SafeRelease(stage_rope_);
    SafeRelease(stage_kv_cache_);

    SafeRelease(adhoc_mat_srv_);
    SafeRelease(adhoc_mat_);
    SafeRelease(adhoc_vec_srv_);
    SafeRelease(adhoc_vec_);
    SafeRelease(adhoc_out_uav_);
    SafeRelease(adhoc_out_);
    SafeRelease(adhoc_stage_);

    SafeRelease(shader_dense_gemv_);
    SafeRelease(shader_grouped_gemv_);
    SafeRelease(shader_rmsnorm_);
    SafeRelease(shader_rope_);
    SafeRelease(shader_update_kv_cache_);
    SafeRelease(shader_latent_attention_);

    if (owns_device_) {
        SafeRelease(context_);
        SafeRelease(device_);
    } else {
        context_ = nullptr;
        device_ = nullptr;
    }

    initialized_ = false;
}

bool M8GpuMlaKernel::initialize(ID3D11Device* shared_device, ID3D11DeviceContext* shared_context) {
    if (initialized_) return true;

    if (shared_device && shared_context) {
        device_ = shared_device;
        context_ = shared_context;
        owns_device_ = false;
    } else {
        D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            0, featureLevels, 2, D3D11_SDK_VERSION,
            &device_, &featureLevel, &context_
        );
        if (FAILED(hr)) return false;
        owns_device_ = true;
    }

    if (!compile_shaders()) {
        cleanup();
        return false;
    }

    if (!allocate_buffers()) {
        cleanup();
        return false;
    }

    initialized_ = true;
    reset_telemetry();
    return true;
}

bool M8GpuMlaKernel::compile_shaders() {
    auto compile_one = [&](const char* entry, ID3D11ComputeShader*& shader) -> bool {
        ID3DBlob* blob = nullptr;
        ID3DBlob* error = nullptr;
        HRESULT hr = D3DCompile(s_hlsl_mla, strlen(s_hlsl_mla), "mla_shaders", nullptr, nullptr, entry, "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &error);
        if (FAILED(hr)) {
            if (error) {
                std::cerr << "Shader compile error (" << entry << "): " << (const char*)error->GetBufferPointer() << "\n";
            }
            SafeRelease(error);
            return false;
        }
        hr = device_->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader);
        SafeRelease(blob);
        return SUCCEEDED(hr);
    };

    if (!compile_one("CSDenseGemv", shader_dense_gemv_)) return false;
    if (!compile_one("CSGroupedGemv", shader_grouped_gemv_)) return false;
    if (!compile_one("CSRmsNorm", shader_rmsnorm_)) return false;
    if (!compile_one("CSRoPE", shader_rope_)) return false;
    if (!compile_one("CSUpdateKVCache", shader_update_kv_cache_)) return false;
    if (!compile_one("CSLatentAttention", shader_latent_attention_)) return false;

    return true;
}

bool M8GpuMlaKernel::allocate_buffers() {
    const int D = params_.dim;           // 5120
    const int Q_rank = params_.q_lora_rank; // 1280
    const int Total_Q = params_.n_heads * params_.head_dim; // 32768
    const int Head_dim = params_.head_dim; // 512
    const int Total_Lora = params_.o_groups * params_.o_lora_rank; // 8192
    const int Win = params_.window_size; // 128
    const int Half_Rope = params_.rope_head_dim / 2; // 32

    // Constant buffers
    struct { uint32_t r, c, p0, p1; } cb_gemv = {};
    struct { uint32_t gr, gc, ng, p; } cb_grp = {};
    struct { uint32_t s; float eps; uint32_t p0, p1; } cb_norm = {};
    struct { uint32_t hd, rhd, p, inv, nh, ws, p0, p1; } cb_rope = {};
    struct { uint32_t nh, hd, ct, ws; float scale; uint32_t pos, p0, p1; } cb_attn = {};
    struct { uint32_t s, hd, p0, p1; } cb_kv = {};

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    cbDesc.ByteWidth = sizeof(cb_gemv);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_gemv_params_);

    cbDesc.ByteWidth = sizeof(cb_grp);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_grouped_gemv_params_);

    cbDesc.ByteWidth = sizeof(cb_norm);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_norm_params_);

    cbDesc.ByteWidth = sizeof(cb_rope);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_rope_params_);

    cbDesc.ByteWidth = sizeof(cb_attn);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_attn_params_);

    cbDesc.ByteWidth = sizeof(cb_kv);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_kv_params_);

    auto make_rw = [&](int count, ID3D11Buffer*& buf, ID3D11ShaderResourceView*& srv, ID3D11UnorderedAccessView*& uav) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = count * sizeof(float);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &buf);

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = count;
        device_->CreateShaderResourceView(buf, &srvDesc, &srv);

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.NumElements = count;
        device_->CreateUnorderedAccessView(buf, &uavDesc, &uav);
    };

    auto make_ro = [&](int count, ID3D11Buffer*& buf, ID3D11ShaderResourceView*& srv) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = count * sizeof(float);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &buf);

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = count;
        device_->CreateShaderResourceView(buf, &srvDesc, &srv);
    };

    auto make_stage = [&](int count, ID3D11Buffer*& buf) {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = count * sizeof(float);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device_->CreateBuffer(&desc, nullptr, &buf);
    };

    // Activations
    make_ro(D, buf_x_, srv_x_);
    make_rw(Q_rank, buf_qr_raw_, srv_qr_raw_, uav_qr_raw_);
    make_rw(Q_rank, buf_qr_, srv_qr_, uav_qr_);
    make_rw(Total_Q, buf_q_, srv_q_, uav_q_);
    make_rw(Head_dim, buf_kv_raw_, srv_kv_raw_, uav_kv_raw_);
    make_rw(Head_dim, buf_kv_, srv_kv_, uav_kv_);
    make_rw(Total_Q, buf_o_, srv_o_, uav_o_);
    make_rw(Total_Lora, buf_o_lora_, srv_o_lora_, uav_o_lora_);
    make_rw(D, buf_out_, srv_out_, uav_out_);

    // KV Cache & RoPE
    make_rw(Win * Head_dim, buf_kv_cache_, srv_kv_cache_, uav_kv_cache_);
    make_ro(Win * Half_Rope, buf_rope_cos_, srv_rope_cos_);
    make_ro(Win * Half_Rope, buf_rope_sin_, srv_rope_sin_);
    make_ro(params_.n_heads, buf_attn_sink_, srv_attn_sink_);

    // Staging
    make_stage(Total_Q, stage_q_);
    make_stage(Head_dim, stage_kv_);
    make_stage(D, stage_out_);
    make_stage(Total_Q, stage_rope_);
    make_stage(Win * Head_dim, stage_kv_cache_);

    // Weights
    make_ro(Q_rank * D, buf_wq_a_, srv_wq_a_);
    make_ro(Q_rank, buf_q_norm_, srv_q_norm_);
    make_ro(Total_Q * Q_rank, buf_wq_b_, srv_wq_b_);
    make_ro(Head_dim * D, buf_wkv_, srv_wkv_);
    make_ro(Head_dim, buf_kv_norm_, srv_kv_norm_);
    make_ro(Total_Lora * (Total_Q / params_.o_groups), buf_wo_a_, srv_wo_a_);
    make_ro(D * Total_Lora, buf_wo_b_, srv_wo_b_);

    telemetry_.vram_allocated_bytes =
        (Q_rank * D + Q_rank + Total_Q * Q_rank + Head_dim * D + Head_dim +
         Total_Lora * (Total_Q / params_.o_groups) + D * Total_Lora +
         D + Q_rank * 2 + Total_Q * 2 + Head_dim * 2 + Total_Lora + D +
         Win * Head_dim + Win * Half_Rope * 2 + params_.n_heads) * sizeof(float);

    cached_tokens_ = 0;
    return true;
}

bool M8GpuMlaKernel::upload_layer_weights(const float* wq_a, const float* q_norm, const float* wq_b,
                                          const float* wkv, const float* kv_norm,
                                          const float* wo_a, const float* wo_b) {
    if (!initialized_ || !wq_a || !q_norm || !wq_b || !wkv || !kv_norm || !wo_a || !wo_b) return false;

    context_->UpdateSubresource(buf_wq_a_, 0, nullptr, wq_a, 0, 0);
    context_->UpdateSubresource(buf_q_norm_, 0, nullptr, q_norm, 0, 0);
    context_->UpdateSubresource(buf_wq_b_, 0, nullptr, wq_b, 0, 0);
    context_->UpdateSubresource(buf_wkv_, 0, nullptr, wkv, 0, 0);
    context_->UpdateSubresource(buf_kv_norm_, 0, nullptr, kv_norm, 0, 0);
    context_->UpdateSubresource(buf_wo_a_, 0, nullptr, wo_a, 0, 0);
    context_->UpdateSubresource(buf_wo_b_, 0, nullptr, wo_b, 0, 0);
    return true;
}

bool M8GpuMlaKernel::upload_rope_tables(const float* cos_table, const float* sin_table) {
    if (!initialized_ || !cos_table || !sin_table) return false;
    context_->UpdateSubresource(buf_rope_cos_, 0, nullptr, cos_table, 0, 0);
    context_->UpdateSubresource(buf_rope_sin_, 0, nullptr, sin_table, 0, 0);
    return true;
}

bool M8GpuMlaKernel::upload_attn_sink(const float* attn_sink) {
    if (!initialized_ || !attn_sink) return false;
    context_->UpdateSubresource(buf_attn_sink_, 0, nullptr, attn_sink, 0, 0);
    return true;
}

void M8GpuMlaKernel::reset_kv_cache() {
    if (!initialized_) return;
    std::vector<float> zeros(params_.window_size * params_.head_dim, 0.0f);
    context_->UpdateSubresource(buf_kv_cache_, 0, nullptr, zeros.data(), 0, 0);
    cached_tokens_ = 0;
}

void M8GpuMlaKernel::update_cb(ID3D11Buffer* cb, const void* data, size_t size) {
    context_->UpdateSubresource(cb, 0, nullptr, data, 0, 0);
}

void M8GpuMlaKernel::dispatch_gemv(int rows, int cols,
                                   ID3D11ShaderResourceView* srv_mat,
                                   ID3D11ShaderResourceView* srv_vec,
                                   ID3D11UnorderedAccessView* uav_out) {
    struct { uint32_t r, c, p0, p1; } cb = { (uint32_t)rows, (uint32_t)cols, 0, 0 };
    update_cb(cb_gemv_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_dense_gemv_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_gemv_params_);
    ID3D11ShaderResourceView* srvs[2] = { srv_mat, srv_vec };
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_out, nullptr);

    context_->Dispatch((rows + 63) / 64, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 2, null_srvs);
}

void M8GpuMlaKernel::dispatch_grouped_gemv(int group_rows, int group_cols, int num_groups,
                                           ID3D11ShaderResourceView* srv_mat,
                                           ID3D11ShaderResourceView* srv_vec,
                                           ID3D11UnorderedAccessView* uav_out) {
    struct { uint32_t gr, gc, ng, p; } cb = { (uint32_t)group_rows, (uint32_t)group_cols, (uint32_t)num_groups, 0 };
    update_cb(cb_grouped_gemv_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_grouped_gemv_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_grouped_gemv_params_);
    ID3D11ShaderResourceView* srvs[2] = { srv_mat, srv_vec };
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_out, nullptr);

    uint32_t total_rows = group_rows * num_groups;
    context_->Dispatch((total_rows + 63) / 64, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 2, null_srvs);
}

void M8GpuMlaKernel::dispatch_rmsnorm(int size,
                                      ID3D11ShaderResourceView* srv_in,
                                      ID3D11ShaderResourceView* srv_weight,
                                      ID3D11UnorderedAccessView* uav_out) {
    struct { uint32_t s; float eps; uint32_t p0, p1; } cb = { (uint32_t)size, params_.eps, 0, 0 };
    update_cb(cb_norm_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_rmsnorm_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_norm_params_);
    ID3D11ShaderResourceView* srvs[2] = { srv_in, srv_weight };
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_out, nullptr);

    context_->Dispatch(1, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 2, null_srvs);
}

void M8GpuMlaKernel::dispatch_rope(int num_heads, int pos, bool inverse, ID3D11UnorderedAccessView* uav_data) {
    struct {
        uint32_t head_dim;
        uint32_t rope_head_dim;
        uint32_t pos;
        uint32_t inverse;
        uint32_t num_heads;
        uint32_t window_size;
        uint32_t p0, p1;
    } cb = {
        (uint32_t)params_.head_dim,
        (uint32_t)params_.rope_head_dim,
        (uint32_t)pos,
        (uint32_t)(inverse ? 1 : 0),
        (uint32_t)num_heads,
        (uint32_t)params_.window_size,
        0, 0
    };
    update_cb(cb_rope_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_rope_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_rope_params_);
    ID3D11ShaderResourceView* srvs[2] = { srv_rope_cos_, srv_rope_sin_ };
    context_->CSSetShaderResources(0, 2, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_data, nullptr);

    uint32_t total_pairs = num_heads * (params_.rope_head_dim / 2);
    context_->Dispatch((total_pairs + 63) / 64, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 2, null_srvs);
}

void M8GpuMlaKernel::dispatch_update_kv_cache(int ring_slot) {
    struct {
        uint32_t slot;
        uint32_t head_dim;
        uint32_t p0, p1;
    } cb = { (uint32_t)ring_slot, (uint32_t)params_.head_dim, 0, 0 };
    update_cb(cb_kv_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_update_kv_cache_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_kv_params_);
    ID3D11ShaderResourceView* srv = srv_kv_;
    context_->CSSetShaderResources(0, 1, &srv);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_kv_cache_, nullptr);

    context_->Dispatch((params_.head_dim + 63) / 64, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srv = nullptr;
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 1, &null_srv);
}

void M8GpuMlaKernel::dispatch_latent_attention(int pos, int cached_tokens) {
    struct {
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t cached_tokens;
        uint32_t window_size;
        float softmax_scale;
        uint32_t pos;
        uint32_t p0, p1;
    } cb = {
        (uint32_t)params_.n_heads,
        (uint32_t)params_.head_dim,
        (uint32_t)cached_tokens,
        (uint32_t)params_.window_size,
        params_.softmax_scale,
        (uint32_t)pos,
        0, 0
    };
    update_cb(cb_attn_params_, &cb, sizeof(cb));

    context_->CSSetShader(shader_latent_attention_, nullptr, 0);
    context_->CSSetConstantBuffers(0, 1, &cb_attn_params_);
    ID3D11ShaderResourceView* srvs[5] = { srv_q_, srv_kv_cache_, srv_attn_sink_, srv_rope_cos_, srv_rope_sin_ };
    context_->CSSetShaderResources(0, 5, srvs);
    context_->CSSetUnorderedAccessViews(0, 1, &uav_o_, nullptr);

    context_->Dispatch(params_.n_heads, 1, 1);

    ID3D11UnorderedAccessView* null_uav = nullptr;
    ID3D11ShaderResourceView* null_srvs[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };
    context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    context_->CSSetShaderResources(0, 5, null_srvs);
}

bool M8GpuMlaKernel::apply_rope_gpu(float* data, int num_heads, int pos, bool inverse) {
    if (!initialized_ || !data) return false;
    int count = num_heads * params_.head_dim;
    context_->UpdateSubresource(buf_q_, 0, nullptr, data, 0, 0);
    dispatch_rope(num_heads, pos, inverse, uav_q_);
    context_->CopyResource(stage_rope_, buf_q_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(stage_rope_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(data, mapped.pData, count * sizeof(float));
        context_->Unmap(stage_rope_, 0);
    }
    return SUCCEEDED(hr);
}

bool M8GpuMlaKernel::forward_full_mla(const float* x, int start_pos, float* out) {
    if (!initialized_ || !x || !out) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Host -> Device Upload of x (5120 floats = 20 KB)
    auto t_h2d0 = std::chrono::high_resolution_clock::now();
    context_->UpdateSubresource(buf_x_, 0, nullptr, x, 0, 0);
    auto t_h2d1 = std::chrono::high_resolution_clock::now();
    telemetry_.h2d_time_ms += std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count();

    auto t_k0 = std::chrono::high_resolution_clock::now();

    // 2. Query Path (WQA -> Norm -> WQB)
    dispatch_gemv(params_.q_lora_rank, params_.dim, srv_wq_a_, srv_x_, uav_qr_raw_);
    dispatch_rmsnorm(params_.q_lora_rank, srv_qr_raw_, srv_q_norm_, uav_qr_);
    int total_q = params_.n_heads * params_.head_dim;
    dispatch_gemv(total_q, params_.q_lora_rank, srv_wq_b_, srv_qr_, uav_q_);

    // 3. Key Path (WKV -> Norm)
    dispatch_gemv(params_.head_dim, params_.dim, srv_wkv_, srv_x_, uav_kv_raw_);
    dispatch_rmsnorm(params_.head_dim, srv_kv_raw_, srv_kv_norm_, uav_kv_);

    // 4. Forward RoPE (Entirely in VRAM)
    dispatch_rope(params_.n_heads, start_pos, false, uav_q_);
    dispatch_rope(1, start_pos, false, uav_kv_);

    // 5. Update KV Cache ring buffer in VRAM
    int ring_slot = start_pos % params_.window_size;
    dispatch_update_kv_cache(ring_slot);
    cached_tokens_ = std::min(cached_tokens_ + 1, params_.window_size);

    // 6. Multi-Head Latent Attention Core (On-chip Score GEMV, Softmax with Sink, V-Aggregation, Inverse RoPE)
    // Result written directly to buf_o_ (32768 floats = 128 KB) in VRAM
    dispatch_latent_attention(start_pos, cached_tokens_);

    // 7. Grouped Output Projections (WOA -> WOB)
    int total_lora = params_.o_groups * params_.o_lora_rank;
    dispatch_grouped_gemv(params_.o_lora_rank, total_q / params_.o_groups, params_.o_groups, srv_wo_a_, srv_o_, uav_o_lora_);
    dispatch_gemv(params_.dim, total_lora, srv_wo_b_, srv_o_lora_, uav_out_);

    auto t_k1 = std::chrono::high_resolution_clock::now();
    telemetry_.compute_time_ms += std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

    // 8. Device -> Host Readback: ONLY out (5120 floats = 20 KB)
    auto t_d2h0 = std::chrono::high_resolution_clock::now();
    context_->CopyResource(stage_out_, buf_out_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(stage_out_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(out, mapped.pData, params_.dim * sizeof(float));
        context_->Unmap(stage_out_, 0);
    }
    auto t_d2h1 = std::chrono::high_resolution_clock::now();
    telemetry_.d2h_time_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count();

    auto t1 = std::chrono::high_resolution_clock::now();
    telemetry_.total_mla_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return SUCCEEDED(hr);
}

bool M8GpuMlaKernel::forward_qk_projections(const float* x, float* out_q, float* out_kv) {
    if (!initialized_ || !x || !out_q || !out_kv) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Host -> Device Upload of x (5120 floats = 20 KB)
    auto t_h2d0 = std::chrono::high_resolution_clock::now();
    context_->UpdateSubresource(buf_x_, 0, nullptr, x, 0, 0);
    auto t_h2d1 = std::chrono::high_resolution_clock::now();
    telemetry_.h2d_time_ms += std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count();

    auto t_k0 = std::chrono::high_resolution_clock::now();

    // 2. Query Path: WQA -> RMSNorm -> WQB
    dispatch_gemv(params_.q_lora_rank, params_.dim, srv_wq_a_, srv_x_, uav_qr_raw_);
    dispatch_rmsnorm(params_.q_lora_rank, srv_qr_raw_, srv_q_norm_, uav_qr_);
    int total_q = params_.n_heads * params_.head_dim;
    dispatch_gemv(total_q, params_.q_lora_rank, srv_wq_b_, srv_qr_, uav_q_);

    // 3. Key Path: WKV -> RMSNorm
    dispatch_gemv(params_.head_dim, params_.dim, srv_wkv_, srv_x_, uav_kv_raw_);
    dispatch_rmsnorm(params_.head_dim, srv_kv_raw_, srv_kv_norm_, uav_kv_);

    auto t_k1 = std::chrono::high_resolution_clock::now();
    telemetry_.compute_time_ms += std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

    // 4. Device -> Host Readback: q (128 KB) and kv (2 KB)
    auto t_d2h0 = std::chrono::high_resolution_clock::now();
    context_->CopyResource(stage_q_, buf_q_);
    context_->CopyResource(stage_kv_, buf_kv_);

    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr_q = context_->Map(stage_q_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr_q)) {
        std::memcpy(out_q, mapped.pData, total_q * sizeof(float));
        context_->Unmap(stage_q_, 0);
    }

    HRESULT hr_kv = context_->Map(stage_kv_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr_kv)) {
        std::memcpy(out_kv, mapped.pData, params_.head_dim * sizeof(float));
        context_->Unmap(stage_kv_, 0);
    }
    auto t_d2h1 = std::chrono::high_resolution_clock::now();
    telemetry_.d2h_time_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count();

    auto t1 = std::chrono::high_resolution_clock::now();
    telemetry_.total_mla_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return SUCCEEDED(hr_q) && SUCCEEDED(hr_kv);
}

bool M8GpuMlaKernel::forward_out_projection(const float* o, float* out) {
    if (!initialized_ || !o || !out) return false;

    auto t0 = std::chrono::high_resolution_clock::now();
    int total_q = params_.n_heads * params_.head_dim;              // 32768
    int total_lora = params_.o_groups * params_.o_lora_rank;        // 8192

    // 1. Host -> Device Upload of o (128 KB)
    auto t_h2d0 = std::chrono::high_resolution_clock::now();
    context_->UpdateSubresource(buf_o_, 0, nullptr, o, 0, 0);
    auto t_h2d1 = std::chrono::high_resolution_clock::now();
    telemetry_.h2d_time_ms += std::chrono::duration<double, std::milli>(t_h2d1 - t_h2d0).count();

    auto t_k0 = std::chrono::high_resolution_clock::now();

    // 2. WOA Projection (Grouped GEMV)
    dispatch_grouped_gemv(params_.o_lora_rank, total_q / params_.o_groups, params_.o_groups, srv_wo_a_, srv_o_, uav_o_lora_);

    // 3. WOB Projection: 8192 -> 5120
    dispatch_gemv(params_.dim, total_lora, srv_wo_b_, srv_o_lora_, uav_out_);

    auto t_k1 = std::chrono::high_resolution_clock::now();
    telemetry_.compute_time_ms += std::chrono::duration<double, std::milli>(t_k1 - t_k0).count();

    // 4. Device -> Host Readback: out (20 KB)
    auto t_d2h0 = std::chrono::high_resolution_clock::now();
    context_->CopyResource(stage_out_, buf_out_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(stage_out_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(out, mapped.pData, params_.dim * sizeof(float));
        context_->Unmap(stage_out_, 0);
    }
    auto t_d2h1 = std::chrono::high_resolution_clock::now();
    telemetry_.d2h_time_ms += std::chrono::duration<double, std::milli>(t_d2h1 - t_d2h0).count();

    auto t1 = std::chrono::high_resolution_clock::now();
    telemetry_.total_mla_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return SUCCEEDED(hr);
}

bool M8GpuMlaKernel::gemv_dense(int rows, int cols, const float* matrix, const float* vec, float* out) {
    if (!initialized_ || !matrix || !vec || !out) return false;

    size_t mat_size = (size_t)rows * cols * sizeof(float);
    size_t vec_size = (size_t)cols * sizeof(float);
    size_t out_size = (size_t)rows * sizeof(float);

    if (mat_size > adhoc_mat_capacity_) {
        SafeRelease(adhoc_mat_srv_);
        SafeRelease(adhoc_mat_);

        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = mat_size;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &adhoc_mat_);

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = rows * cols;
        device_->CreateShaderResourceView(adhoc_mat_, &srvDesc, &adhoc_mat_srv_);
        adhoc_mat_capacity_ = mat_size;
    }

    if (vec_size > adhoc_vec_capacity_) {
        SafeRelease(adhoc_vec_srv_);
        SafeRelease(adhoc_vec_);

        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = vec_size;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &adhoc_vec_);

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = cols;
        device_->CreateShaderResourceView(adhoc_vec_, &srvDesc, &adhoc_vec_srv_);
        adhoc_vec_capacity_ = vec_size;
    }

    if (out_size > adhoc_out_capacity_) {
        SafeRelease(adhoc_out_uav_);
        SafeRelease(adhoc_out_);
        SafeRelease(adhoc_stage_);

        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = out_size;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &adhoc_out_);

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_UNKNOWN;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uavDesc.Buffer.NumElements = rows;
        device_->CreateUnorderedAccessView(adhoc_out_, &uavDesc, &adhoc_out_uav_);

        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device_->CreateBuffer(&desc, nullptr, &adhoc_stage_);

        adhoc_out_capacity_ = out_size;
    }

    context_->UpdateSubresource(adhoc_mat_, 0, nullptr, matrix, 0, 0);
    context_->UpdateSubresource(adhoc_vec_, 0, nullptr, vec, 0, 0);

    dispatch_gemv(rows, cols, adhoc_mat_srv_, adhoc_vec_srv_, adhoc_out_uav_);

    context_->CopyResource(adhoc_stage_, adhoc_out_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(adhoc_stage_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(out, mapped.pData, rows * sizeof(float));
        context_->Unmap(adhoc_stage_, 0);
    }
    return SUCCEEDED(hr);
}

bool M8GpuMlaKernel::readback_kv_cache(float* out_cache, int num_tokens) {
    if (!initialized_ || !out_cache || !stage_kv_cache_) return false;
    int count = std::min(num_tokens, params_.window_size) * params_.head_dim;
    context_->CopyResource(stage_kv_cache_, buf_kv_cache_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(stage_kv_cache_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(out_cache, mapped.pData, count * sizeof(float));
        context_->Unmap(stage_kv_cache_, 0);
    }
    return SUCCEEDED(hr);
}

bool M8GpuMlaKernel::readback_attn_o(float* out_o) {
    if (!initialized_ || !out_o || !stage_q_) return false;
    int count = params_.n_heads * params_.head_dim;
    context_->CopyResource(stage_q_, buf_o_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(stage_q_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        std::memcpy(out_o, mapped.pData, count * sizeof(float));
        context_->Unmap(stage_q_, 0);
    }
    return SUCCEEDED(hr);
}

} // namespace m8
} // namespace asema
