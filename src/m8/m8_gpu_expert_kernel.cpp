#include "asema/m8/m8_gpu_expert_kernel.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <chrono>
#include <cstring>
#include <iostream>

namespace asema {
namespace m8 {

namespace {

// HLSL Compute Shader for Packed FP4 GEMV
static const char* s_hlsl_gemv = R"(
cbuffer GemvParams : register(b0) {
    uint g_rows;
    uint g_cols;
    uint g_num_blocks;
    uint g_row_weight_stride;
    uint g_row_scale_stride;
    uint pad0;
    uint pad1;
    uint pad2;
};

ByteAddressBuffer g_scales : register(t0);
ByteAddressBuffer g_weights : register(t1);
StructuredBuffer<float> g_x : register(t2);
RWStructuredBuffer<float> g_out : register(u0);

static const float k_fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

[numthreads(64, 1, 1)]
void CSGemv(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint r = dispatchThreadId.x;
    if (r >= g_rows) return;

    uint row_w_offset = r * g_row_weight_stride;
    uint row_s_offset = r * g_row_scale_stride;

    float row_sum = 0.0f;

    for (uint b = 0; b < g_num_blocks; ++b) {
        // Direct IEEE-754 exponent injection: 2^(s - 127) == asfloat(s << 23)
        uint s_byte_addr = row_s_offset + b;
        uint s_word = g_scales.Load(s_byte_addr & ~3);
        uint s_val = (s_word >> ((s_byte_addr & 3) * 8)) & 0xFF;
        float scale = asfloat(s_val << 23);

        // Load 16 bytes of weights = 4 uints = 32 nibbles
        uint w_byte_addr = row_w_offset + b * 16;
        uint4 w4 = g_weights.Load4(w_byte_addr);

        uint x_base = b * 32;
        float blk_dot = 0.0f;

        uint w_arr[4] = { w4.x, w4.y, w4.z, w4.w };

        [unroll]
        for (uint u = 0; u < 4; ++u) {
            uint val = w_arr[u];
            [unroll]
            for (uint i = 0; i < 4; ++i) {
                uint byte_val = (val >> (i * 8)) & 0xFF;
                float low  = k_fp4_lut[byte_val & 0x0F];
                float high = k_fp4_lut[(byte_val >> 4) & 0x0F];

                uint x_idx = x_base + u * 8 + i * 2;
                blk_dot += low * g_x[x_idx] + high * g_x[x_idx + 1];
            }
        }

        row_sum += scale * blk_dot;
    }

    g_out[r] = row_sum;
}
)";

// HLSL Compute Shader for Element-wise SwiGLU Activation
static const char* s_hlsl_swiglu = R"(
cbuffer SwigluParams : register(b0) {
    uint g_intermediate_dim;
    float g_swiglu_limit;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float> g_gate : register(t0);
StructuredBuffer<float> g_up   : register(t1);
RWStructuredBuffer<float> g_act : register(u0);

[numthreads(64, 1, 1)]
void CSSwiglu(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint i = dispatchThreadId.x;
    if (i >= g_intermediate_dim) return;

    float g = min(g_gate[i], g_swiglu_limit);
    float u = clamp(g_up[i], -g_swiglu_limit, g_swiglu_limit);
    float silu = g / (1.0f + exp(-g));
    g_act[i] = silu * u;
}
)";

// HLSL Compute Shader for Output Accumulation
static const char* s_hlsl_accum = R"(
cbuffer AccumParams : register(b0) {
    uint g_hidden_dim;
    float g_router_weight;
    uint g_accumulate;
    uint pad0;
};

StructuredBuffer<float> g_expert_out : register(t0);
RWStructuredBuffer<float> g_final_out : register(u0);

[numthreads(64, 1, 1)]
void CSAccum(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint i = dispatchThreadId.x;
    if (i >= g_hidden_dim) return;

    float val = g_expert_out[i] * g_router_weight;
    if (g_accumulate == 0) {
        g_final_out[i] = val;
    } else {
        g_final_out[i] += val;
    }
}
)";

struct GemvCBuffer {
    uint32_t rows;
    uint32_t cols;
    uint32_t num_blocks;
    uint32_t row_weight_stride;
    uint32_t row_scale_stride;
    uint32_t pad[3];
};

struct SwigluCBuffer {
    uint32_t intermediate_dim;
    float swiglu_limit;
    uint32_t pad[2];
};

struct AccumCBuffer {
    uint32_t hidden_dim;
    float router_weight;
    uint32_t accumulate;
    uint32_t pad;
};

template <typename T>
void SafeRelease(T*& ptr) {
    if (ptr) {
        ptr->Release();
        ptr = nullptr;
    }
}

} // namespace

M8GpuExpertKernel::M8GpuExpertKernel(ExpertDimensions dims)
    : dims_(dims) {}

M8GpuExpertKernel::~M8GpuExpertKernel() {
    cleanup();
}

void M8GpuExpertKernel::reset_telemetry() {
    telemetry_ = GpuKernelTelemetry{};
    // Report what is really resident: every allocated expert slot plus the scratch vectors.
    // (Previously this hardcoded a single expert, hiding the ~1.8 GB slot cache from telemetry.)
    const size_t slot_count = slots_.empty() ? 1 : slots_.size();
    telemetry_.vram_allocated_bytes = slot_count * ExpertDimensions::TOTAL_EXPERT_BYTES +
                                      (static_cast<size_t>(dims_.hidden_dim) * 4 * 4) +
                                      (static_cast<size_t>(dims_.intermediate_dim) * 4 * 3);
}

void M8GpuExpertKernel::cleanup() {
    SafeRelease(cb_gemv_params_);
    SafeRelease(cb_swiglu_params_);
    SafeRelease(cb_accum_params_);

    SafeRelease(srv_x_);
    SafeRelease(buf_x_);

    SafeRelease(uav_gate_);
    SafeRelease(srv_gate_);
    SafeRelease(buf_gate_);

    SafeRelease(uav_up_);
    SafeRelease(srv_up_);
    SafeRelease(buf_up_);

    SafeRelease(uav_act_);
    SafeRelease(srv_act_);
    SafeRelease(buf_act_);

    SafeRelease(uav_expert_out_);
    SafeRelease(srv_expert_out_);
    SafeRelease(buf_expert_out_);

    SafeRelease(uav_accum_out_);
    SafeRelease(srv_accum_out_);
    SafeRelease(buf_accum_out_);

    SafeRelease(buf_staging_out_);

    SafeRelease(q_disjoint_);
    for (auto& q : q_ts_) SafeRelease(q);

    for (auto& slot : slots_) {
        SafeRelease(slot.srv_w1_scales);
        SafeRelease(slot.srv_w2_scales);
        SafeRelease(slot.srv_w3_scales);
        SafeRelease(slot.buf_scales);

        SafeRelease(slot.srv_w1_weights);
        SafeRelease(slot.srv_w2_weights);
        SafeRelease(slot.srv_w3_weights);
        SafeRelease(slot.buf_weights);
    }
    slots_.clear();
    slot_last_access_.clear();
    slot_pinned_.clear();
    pinned_this_layer_.clear();
    access_freq_.clear();
    slot_map_.clear();

    SafeRelease(shader_gemv_);
    SafeRelease(shader_swiglu_);
    SafeRelease(shader_accumulate_);

    SafeRelease(context_);
    SafeRelease(device_);

    initialized_ = false;
}

bool M8GpuExpertKernel::initialize() {
    if (initialized_) return true;

    // 1. Create D3D11 Hardware Device
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, featureLevels, 2, D3D11_SDK_VERSION,
        &device_, &featureLevel, &context_
    );
    if (FAILED(hr)) {
        return false;
    }

    // 2. Compile Compute Shaders
    if (!compile_shaders()) {
        cleanup();
        return false;
    }

    // 3. Allocate Persistent Scratch Buffers
    if (!allocate_buffers()) {
        cleanup();
        return false;
    }

    initialized_ = true;
    reset_telemetry();
    return true;
}

bool M8GpuExpertKernel::compile_shaders() {
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;

    // Shader 1: GEMV
    HRESULT hr = D3DCompile(s_hlsl_gemv, strlen(s_hlsl_gemv), "gemv", nullptr, nullptr, "CSGemv", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &error);
    if (FAILED(hr)) {
        SafeRelease(error);
        return false;
    }
    hr = device_->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader_gemv_);
    SafeRelease(blob);
    if (FAILED(hr)) return false;

    // Shader 2: SwiGLU
    hr = D3DCompile(s_hlsl_swiglu, strlen(s_hlsl_swiglu), "swiglu", nullptr, nullptr, "CSSwiglu", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &error);
    if (FAILED(hr)) {
        SafeRelease(error);
        return false;
    }
    hr = device_->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader_swiglu_);
    SafeRelease(blob);
    if (FAILED(hr)) return false;

    // Shader 3: Accumulate
    hr = D3DCompile(s_hlsl_accum, strlen(s_hlsl_accum), "accum", nullptr, nullptr, "CSAccum", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &blob, &error);
    if (FAILED(hr)) {
        SafeRelease(error);
        return false;
    }
    hr = device_->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, &shader_accumulate_);
    SafeRelease(blob);
    if (FAILED(hr)) return false;

    return true;
}

bool M8GpuExpertKernel::allocate_buffers() {
    const int H = dims_.hidden_dim;
    const int I = dims_.intermediate_dim;

    // Constant Buffers
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;

    cbDesc.ByteWidth = sizeof(GemvCBuffer);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_gemv_params_);

    cbDesc.ByteWidth = sizeof(SwigluCBuffer);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_swiglu_params_);

    cbDesc.ByteWidth = sizeof(AccumCBuffer);
    device_->CreateBuffer(&cbDesc, nullptr, &cb_accum_params_);

    // Helper lambda for creating Structured float buffers with SRV and UAV
    auto make_float_rw_buffer = [&](int count, ID3D11Buffer*& buf, ID3D11ShaderResourceView*& srv, ID3D11UnorderedAccessView*& uav) {
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

    // Vector X (Input SRV)
    {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = H * sizeof(float);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(float);
        device_->CreateBuffer(&desc, nullptr, &buf_x_);

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.NumElements = H;
        device_->CreateShaderResourceView(buf_x_, &srvDesc, &srv_x_);
    }

    make_float_rw_buffer(I, buf_gate_, srv_gate_, uav_gate_);
    make_float_rw_buffer(I, buf_up_, srv_up_, uav_up_);
    make_float_rw_buffer(I, buf_act_, srv_act_, uav_act_);
    make_float_rw_buffer(H, buf_expert_out_, srv_expert_out_, uav_expert_out_);
    make_float_rw_buffer(H, buf_accum_out_, srv_accum_out_, uav_accum_out_);

    // Staging buffer to read back output (5120 floats = 20 KB)
    {
        D3D11_BUFFER_DESC desc = {};
        desc.ByteWidth = H * sizeof(float);
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device_->CreateBuffer(&desc, nullptr, &buf_staging_out_);
    }

    // Multi-slot Scales & Weights Buffers (64 slots = ~1.15 GB VRAM)
    if (const char* env = std::getenv("ASEMA_VRAM_SLOTS")) {
        const long n = std::atol(env);
        if (n >= 8 && n <= 512) num_vram_slots_ = static_cast<size_t>(n);
    }
    slots_.resize(num_vram_slots_);
    slot_last_access_.assign(num_vram_slots_, 0);
    slot_cost_.assign(num_vram_slots_, 1.0);
    slot_pinned_.assign(num_vram_slots_, 0);
    pinned_this_layer_.clear();
    slot_map_.clear();
    access_freq_.clear();
    gpu_clock_ = 0;

    for (size_t i = 0; i < num_vram_slots_; ++i) {
        auto& slot = slots_[i];
        slot.layer_id = -1;
        slot.expert_id = -1;

        // Scales Buffer (Total 1,105,920 bytes)
        D3D11_BUFFER_DESC sDescBuf = {};
        sDescBuf.ByteWidth = ExpertDimensions::TOTAL_SCALE_BYTES;
        sDescBuf.Usage = D3D11_USAGE_DEFAULT;
        sDescBuf.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        sDescBuf.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        HRESULT hr = device_->CreateBuffer(&sDescBuf, nullptr, &slot.buf_scales);
        if (FAILED(hr)) return false;

        auto make_scale_srv = [&](size_t offset_bytes, size_t size_bytes, ID3D11ShaderResourceView*& srv) {
            D3D11_SHADER_RESOURCE_VIEW_DESC sDesc = {};
            sDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            sDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            sDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            sDesc.BufferEx.FirstElement = static_cast<UINT>(offset_bytes / 4);
            sDesc.BufferEx.NumElements = static_cast<UINT>(size_bytes / 4);
            device_->CreateShaderResourceView(slot.buf_scales, &sDesc, &srv);
        };
        make_scale_srv(0, ExpertDimensions::W1_SCALE_BYTES, slot.srv_w1_scales);
        make_scale_srv(ExpertDimensions::W1_SCALE_BYTES, ExpertDimensions::W2_SCALE_BYTES, slot.srv_w2_scales);
        make_scale_srv(ExpertDimensions::W1_SCALE_BYTES + ExpertDimensions::W2_SCALE_BYTES, ExpertDimensions::W3_SCALE_BYTES, slot.srv_w3_scales);

        // Weights Buffer (Total 17,694,720 bytes)
        D3D11_BUFFER_DESC wDescBuf = {};
        wDescBuf.ByteWidth = ExpertDimensions::TOTAL_WEIGHT_BYTES;
        wDescBuf.Usage = D3D11_USAGE_DEFAULT;
        wDescBuf.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        wDescBuf.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        hr = device_->CreateBuffer(&wDescBuf, nullptr, &slot.buf_weights);
        if (FAILED(hr)) return false;

        auto make_weight_srv = [&](size_t offset_bytes, size_t size_bytes, ID3D11ShaderResourceView*& srv) {
            D3D11_SHADER_RESOURCE_VIEW_DESC sDesc = {};
            sDesc.Format = DXGI_FORMAT_R32_TYPELESS;
            sDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            sDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
            sDesc.BufferEx.FirstElement = static_cast<UINT>(offset_bytes / 4);
            sDesc.BufferEx.NumElements = static_cast<UINT>(size_bytes / 4);
            device_->CreateShaderResourceView(slot.buf_weights, &sDesc, &srv);
        };
        make_weight_srv(0, ExpertDimensions::W1_WEIGHT_BYTES, slot.srv_w1_weights);
        make_weight_srv(ExpertDimensions::W1_WEIGHT_BYTES, ExpertDimensions::W2_WEIGHT_BYTES, slot.srv_w2_weights);
        make_weight_srv(ExpertDimensions::W1_WEIGHT_BYTES + ExpertDimensions::W2_WEIGHT_BYTES, ExpertDimensions::W3_WEIGHT_BYTES, slot.srv_w3_weights);
    }

    telemetry_.vram_allocated_bytes = num_vram_slots_ * (ExpertDimensions::TOTAL_SCALE_BYTES + ExpertDimensions::TOTAL_WEIGHT_BYTES)
        + H * sizeof(float) * 3 + I * sizeof(float) * 3;

    // Timestamp queries (best effort: if unavailable, GPU-busy figures simply stay at zero).
    D3D11_QUERY_DESC qd = {};
    qd.Query = D3D11_QUERY_TIMESTAMP_DISJOINT;
    if (SUCCEEDED(device_->CreateQuery(&qd, &q_disjoint_))) {
        qd.Query = D3D11_QUERY_TIMESTAMP;
        for (auto& q : q_ts_) {
            if (FAILED(device_->CreateQuery(&qd, &q))) { q = nullptr; SafeRelease(q_disjoint_); break; }
        }
    }

    return true;
}

// Called after the result readback has completed, so every timestamp is already available.
void M8GpuExpertKernel::collect_gpu_timestamps(int n_experts) {
    D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
    if (context_->GetData(q_disjoint_, &dj, sizeof(dj), 0) != S_OK || dj.Disjoint || dj.Frequency == 0) return;
    const double to_ms = 1000.0 / static_cast<double>(dj.Frequency);
    for (int e = 0; e < n_experts && e < 6; ++e) {
        UINT64 t[3] = {0, 0, 0};
        bool ok = true;
        for (int k = 0; k < 3; ++k) {
            ok = ok && (context_->GetData(q_ts_[e * 3 + k], &t[k], sizeof(UINT64), 0) == S_OK);
        }
        if (!ok || t[1] < t[0] || t[2] < t[1]) continue;
        telemetry_.gpu_busy_upload_ms += static_cast<double>(t[1] - t[0]) * to_ms;
        telemetry_.gpu_busy_compute_ms += static_cast<double>(t[2] - t[1]) * to_ms;
    }
}

void M8GpuExpertKernel::update_constant_buffer(ID3D11Buffer* cb, const void* data, size_t size) {
    context_->UpdateSubresource(cb, 0, nullptr, data, 0, 0);
}

bool M8GpuExpertKernel::is_resident(int layer_id, int expert_id) const {
    if (layer_id < 0 || expert_id < 0 || slots_.empty()) return false;
    const uint32_t key = (static_cast<uint32_t>(layer_id) << 16) | (static_cast<uint32_t>(expert_id) & 0xFFFF);
    return slot_map_.find(key) != slot_map_.end();
}

void M8GpuExpertKernel::note_access(uint32_t key) {
    ++gpu_clock_;
    ++access_freq_[key];
}

// Lowest score = accesses / (1 + 0.25 * age / 240). Free slots go first; slots in use by the layer that
// is currently executing are never evicted.
int M8GpuExpertKernel::pick_victim_slot() const {
    int best = -1;
    double best_score = 0.0;
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slot_pinned_[i]) continue;
        if (slots_[i].layer_id < 0) return static_cast<int>(i);
        const uint32_t key = (static_cast<uint32_t>(slots_[i].layer_id) << 16) | (static_cast<uint32_t>(slots_[i].expert_id) & 0xFFFF);
        const auto f = access_freq_.find(key);
        const double freq = (f == access_freq_.end()) ? 1.0 : static_cast<double>(f->second);
        const double age = static_cast<double>(gpu_clock_ - slot_last_access_[i]);
        const double score = (freq / (1.0 + 0.25 * age / 240.0)) * slot_cost_[i];
        if (best < 0 || score < best_score) { best = static_cast<int>(i); best_score = score; }
    }
    return best;
}

int M8GpuExpertKernel::get_or_upload_slot(int layer_id, int expert_id, const uint8_t* scales, const uint8_t* weights) {
    if (layer_id < 0 || expert_id < 0 || slots_.empty()) {
        int slot_idx = 0;
        if (!slots_.empty()) {
            context_->UpdateSubresource(slots_[slot_idx].buf_scales, 0, nullptr, scales, 0, 0);
            context_->UpdateSubresource(slots_[slot_idx].buf_weights, 0, nullptr, weights, 0, 0);
        }
        return slot_idx;
    }

    uint32_t key = (static_cast<uint32_t>(layer_id) << 16) | (static_cast<uint32_t>(expert_id) & 0xFFFF);
    note_access(key);
    auto it = slot_map_.find(key);
    if (it != slot_map_.end()) {
        // Hit in VRAM: no upload (and, via is_resident(), no storage read either).
        const int slot_idx = it->second;
        slot_last_access_[slot_idx] = gpu_clock_;
        if (!slot_pinned_[slot_idx]) { slot_pinned_[slot_idx] = 1; pinned_this_layer_.push_back(slot_idx); }
        telemetry_.vram_cache_hits++;
        return slot_idx;
    }

    // Miss in VRAM: evict the lowest-scoring slot that is not in use by this layer.
    telemetry_.vram_cache_misses++;
    const int slot_idx = pick_victim_slot();
    if (slot_idx < 0) return -1;

    if (slots_[slot_idx].layer_id >= 0 && slots_[slot_idx].expert_id >= 0) {
        uint32_t old_key = (static_cast<uint32_t>(slots_[slot_idx].layer_id) << 16) | (static_cast<uint32_t>(slots_[slot_idx].expert_id) & 0xFFFF);
        slot_map_.erase(old_key);
    }

    slots_[slot_idx].layer_id = layer_id;
    slots_[slot_idx].expert_id = expert_id;
    slot_map_[key] = slot_idx;
    slot_last_access_[slot_idx] = gpu_clock_;
    slot_cost_[slot_idx] = cost_fn_ ? std::max(0.1, cost_fn_(layer_id, expert_id)) : 1.0;
    slot_pinned_[slot_idx] = 1;
    pinned_this_layer_.push_back(slot_idx);

    auto t0 = std::chrono::high_resolution_clock::now();
    context_->UpdateSubresource(slots_[slot_idx].buf_scales, 0, nullptr, scales, 0, 0);
    context_->UpdateSubresource(slots_[slot_idx].buf_weights, 0, nullptr, weights, 0, 0);
    auto t1 = std::chrono::high_resolution_clock::now();
    telemetry_.host_to_device_time_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    return slot_idx;
}

bool M8GpuExpertKernel::forward_expert(const uint8_t* scales_ptr,
                                      const uint8_t* weights_ptr,
                                      const float* x,
                                      float* out,
                                      float router_weight,
                                      bool accumulate) {
    if (!initialized_ || !scales_ptr || !weights_ptr || !x || !out) {
        return false;
    }

    const int H = dims_.hidden_dim;
    const int I = dims_.intermediate_dim;
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. Host -> Device Upload
    auto t_h2d_start = std::chrono::high_resolution_clock::now();
    context_->UpdateSubresource(buf_x_, 0, nullptr, x, 0, 0);
    int slot_idx = get_or_upload_slot(-1, -1, scales_ptr, weights_ptr);
    const auto& slot = slots_[slot_idx];
    auto t_h2d_end = std::chrono::high_resolution_clock::now();
    telemetry_.host_to_device_time_ms += std::chrono::duration<double, std::milli>(t_h2d_end - t_h2d_start).count();

    // 2. Kernel Execution Pipeline
    auto t_k_start = std::chrono::high_resolution_clock::now();

    // A. W1: gate = w1(x) [5120 -> 2304]
    {
        GemvCBuffer cb = { (uint32_t)I, (uint32_t)H, (uint32_t)(H / 32), (uint32_t)(H / 2), (uint32_t)(H / 32), {0,0,0} };
        update_constant_buffer(cb_gemv_params_, &cb, sizeof(cb));
        context_->CSSetShader(shader_gemv_, nullptr, 0);
        context_->CSSetConstantBuffers(0, 1, &cb_gemv_params_);
        ID3D11ShaderResourceView* srvs[3] = { slot.srv_w1_scales, slot.srv_w1_weights, srv_x_ };
        context_->CSSetShaderResources(0, 3, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, &uav_gate_, nullptr);
        context_->Dispatch((I + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* null_uav = nullptr;
        context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    }

    // B. W3: up = w3(x) [5120 -> 2304]
    {
        ID3D11ShaderResourceView* srvs[3] = { slot.srv_w3_scales, slot.srv_w3_weights, srv_x_ };
        context_->CSSetShaderResources(0, 3, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, &uav_up_, nullptr);
        context_->Dispatch((I + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* null_uav = nullptr;
        context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    }

    // C. SwiGLU: act = silu(min(gate, limit)) * clamp(up, -limit, limit)
    {
        SwigluCBuffer cb = { (uint32_t)I, dims_.swiglu_limit, {0,0} };
        update_constant_buffer(cb_swiglu_params_, &cb, sizeof(cb));
        context_->CSSetShader(shader_swiglu_, nullptr, 0);
        context_->CSSetConstantBuffers(0, 1, &cb_swiglu_params_);
        ID3D11ShaderResourceView* srvs[2] = { srv_gate_, srv_up_ };
        context_->CSSetShaderResources(0, 2, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, &uav_act_, nullptr);
        context_->Dispatch((I + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* null_uav = nullptr;
        ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
        context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        context_->CSSetShaderResources(0, 2, null_srvs);
    }

    // D. W2: expert_out = w2(act) [2304 -> 5120]
    {
        GemvCBuffer cb = { (uint32_t)H, (uint32_t)I, (uint32_t)(I / 32), (uint32_t)(I / 2), (uint32_t)(I / 32), {0,0,0} };
        update_constant_buffer(cb_gemv_params_, &cb, sizeof(cb));
        context_->CSSetShader(shader_gemv_, nullptr, 0);
        context_->CSSetConstantBuffers(0, 1, &cb_gemv_params_);
        ID3D11ShaderResourceView* srvs[3] = { slot.srv_w2_scales, slot.srv_w2_weights, srv_act_ };
        context_->CSSetShaderResources(0, 3, srvs);
        context_->CSSetUnorderedAccessViews(0, 1, &uav_expert_out_, nullptr);
        context_->Dispatch((H + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* null_uav = nullptr;
        ID3D11ShaderResourceView* null_srvs[3] = { nullptr, nullptr, nullptr };
        context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        context_->CSSetShaderResources(0, 3, null_srvs);
    }

    // E. Output Accumulation
    {
        AccumCBuffer cb = { (uint32_t)H, router_weight, accumulate ? 1u : 0u, 0 };
        update_constant_buffer(cb_accum_params_, &cb, sizeof(cb));
        context_->CSSetShader(shader_accumulate_, nullptr, 0);
        context_->CSSetConstantBuffers(0, 1, &cb_accum_params_);
        context_->CSSetShaderResources(0, 1, &srv_expert_out_);
        context_->CSSetUnorderedAccessViews(0, 1, &uav_accum_out_, nullptr);
        context_->Dispatch((H + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* null_uav = nullptr;
        ID3D11ShaderResourceView* null_srv = nullptr;
        context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        context_->CSSetShaderResources(0, 1, &null_srv);
    }
    auto t_k_end = std::chrono::high_resolution_clock::now();
    telemetry_.kernel_compute_time_ms += std::chrono::duration<double, std::milli>(t_k_end - t_k_start).count();

    // 3. Readback to host
    auto t_d2h_start = std::chrono::high_resolution_clock::now();
    context_->CopyResource(buf_staging_out_, buf_accum_out_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(buf_staging_out_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(out, mapped.pData, H * sizeof(float));
        context_->Unmap(buf_staging_out_, 0);
    }
    auto t_d2h_end = std::chrono::high_resolution_clock::now();
    telemetry_.device_to_host_time_ms += std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();

    auto t_end = std::chrono::high_resolution_clock::now();
    telemetry_.total_gpu_time_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    telemetry_.total_dispatches++;
    return SUCCEEDED(hr);
}

bool M8GpuExpertKernel::forward_top6_layer(int layer_id,
                                          const std::vector<int>& expert_ids,
                                          const std::vector<const uint8_t*>& expert_scales,
                                          const std::vector<const uint8_t*>& expert_weights,
                                          const std::vector<float>& router_weights,
                                          const float* x,
                                          float* out) {
    if (expert_scales.size() != 6 || expert_weights.size() != 6) return false;
    return forward_top6_layer_streamed(
        layer_id, expert_ids,
        [&](int e, const uint8_t*& scales, const uint8_t*& weights) {
            scales = expert_scales[e];
            weights = expert_weights[e];
            return true;
        },
        router_weights, x, out);
}

// Same math and the same expert order as before; the only difference is that each expert's bytes are
// requested from `provider` just before they are needed, so storage reads of later experts can overlap
// the uploads and compute of earlier ones.
bool M8GpuExpertKernel::forward_top6_layer_streamed(int layer_id,
                                                   const std::vector<int>& expert_ids,
                                                   const ExpertProvider& provider,
                                                   const std::vector<float>& router_weights,
                                                   const float* x,
                                                   float* out) {
    if (!initialized_ || router_weights.size() != 6) {
        return false;
    }

    const int H = dims_.hidden_dim;
    const int I = dims_.intermediate_dim;
    auto t_start = std::chrono::high_resolution_clock::now();

    double h2d_acc = 0.0;
    double comp_acc = 0.0;

    auto t_h2d_0 = std::chrono::high_resolution_clock::now();
    const bool use_ts = (q_disjoint_ != nullptr) && expert_ids.size() <= 6;
    if (use_ts) context_->Begin(q_disjoint_);

    // Pin every expert of this layer that is already resident, so uploads for the other experts of the
    // layer cannot evict a slot whose bytes were deliberately not re-read from storage.
    for (int id : expert_ids) {
        if (layer_id < 0 || id < 0 || slots_.empty()) break;
        const auto it = slot_map_.find((static_cast<uint32_t>(layer_id) << 16) | (static_cast<uint32_t>(id) & 0xFFFF));
        if (it != slot_map_.end() && !slot_pinned_[it->second]) {
            slot_pinned_[it->second] = 1;
            pinned_this_layer_.push_back(it->second);
        }
    }

    // Upload input activation x once for all 6 experts
    context_->UpdateSubresource(buf_x_, 0, nullptr, x, 0, 0);
    auto t_h2d_1 = std::chrono::high_resolution_clock::now();
    h2d_acc += std::chrono::duration<double, std::milli>(t_h2d_1 - t_h2d_0).count();

    for (int e = 0; e < 6; ++e) {
        bool is_first = (e == 0);
        int expert_id = (e < static_cast<int>(expert_ids.size())) ? expert_ids[e] : -1;
        // Wait for this expert's bytes (a cache hit returns immediately, a miss blocks on storage)
        // BEFORE the GPU timestamp, so the wait never counts as GPU busy time.
        const uint8_t* e_scales = nullptr;
        const uint8_t* e_weights = nullptr;
        // An expert already resident in VRAM needs no bytes from the host, so storage is not touched.
        if (!is_resident(layer_id, expert_id)) {
            if (!provider(e, e_scales, e_weights) || !e_scales || !e_weights) {
                if (use_ts) context_->End(q_disjoint_);
                for (int s : pinned_this_layer_) slot_pinned_[s] = 0;
                pinned_this_layer_.clear();
                return false;
            }
        }
        if (use_ts) context_->End(q_ts_[e * 3 + 0]);
        int slot_idx = get_or_upload_slot(layer_id, expert_id, e_scales, e_weights);
        if (slot_idx < 0) {
            if (use_ts) context_->End(q_disjoint_);
            for (int s : pinned_this_layer_) slot_pinned_[s] = 0;
            pinned_this_layer_.clear();
            return false;
        }
        const auto& slot = slots_[slot_idx];
        if (use_ts) context_->End(q_ts_[e * 3 + 1]);

        auto t_c0 = std::chrono::high_resolution_clock::now();
        // 1. W1: gate = w1(x)
        {
            GemvCBuffer cb = { (uint32_t)I, (uint32_t)H, (uint32_t)(H / 32), (uint32_t)(H / 2), (uint32_t)(H / 32), {0,0,0} };
            update_constant_buffer(cb_gemv_params_, &cb, sizeof(cb));
            context_->CSSetShader(shader_gemv_, nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, &cb_gemv_params_);
            ID3D11ShaderResourceView* srvs[3] = { slot.srv_w1_scales, slot.srv_w1_weights, srv_x_ };
            context_->CSSetShaderResources(0, 3, srvs);
            context_->CSSetUnorderedAccessViews(0, 1, &uav_gate_, nullptr);
            context_->Dispatch((I + 63) / 64, 1, 1);

            ID3D11UnorderedAccessView* null_uav = nullptr;
            context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        }

        // 2. W3: up = w3(x)
        {
            ID3D11ShaderResourceView* srvs[3] = { slot.srv_w3_scales, slot.srv_w3_weights, srv_x_ };
            context_->CSSetShaderResources(0, 3, srvs);
            context_->CSSetUnorderedAccessViews(0, 1, &uav_up_, nullptr);
            context_->Dispatch((I + 63) / 64, 1, 1);

            ID3D11UnorderedAccessView* null_uav = nullptr;
            context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        }

        // 3. SwiGLU
        {
            SwigluCBuffer cb = { (uint32_t)I, dims_.swiglu_limit, {0,0} };
            update_constant_buffer(cb_swiglu_params_, &cb, sizeof(cb));
            context_->CSSetShader(shader_swiglu_, nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, &cb_swiglu_params_);
            ID3D11ShaderResourceView* srvs[2] = { srv_gate_, srv_up_ };
            context_->CSSetShaderResources(0, 2, srvs);
            context_->CSSetUnorderedAccessViews(0, 1, &uav_act_, nullptr);
            context_->Dispatch((I + 63) / 64, 1, 1);

            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11ShaderResourceView* null_srvs[2] = { nullptr, nullptr };
            context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context_->CSSetShaderResources(0, 2, null_srvs);
        }

        // 4. W2: expert_out = w2(act)
        {
            GemvCBuffer cb = { (uint32_t)H, (uint32_t)I, (uint32_t)(I / 32), (uint32_t)(I / 2), (uint32_t)(I / 32), {0,0,0} };
            update_constant_buffer(cb_gemv_params_, &cb, sizeof(cb));
            context_->CSSetShader(shader_gemv_, nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, &cb_gemv_params_);
            ID3D11ShaderResourceView* srvs[3] = { slot.srv_w2_scales, slot.srv_w2_weights, srv_act_ };
            context_->CSSetShaderResources(0, 3, srvs);
            context_->CSSetUnorderedAccessViews(0, 1, &uav_expert_out_, nullptr);
            context_->Dispatch((H + 63) / 64, 1, 1);

            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11ShaderResourceView* null_srvs[3] = { nullptr, nullptr, nullptr };
            context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context_->CSSetShaderResources(0, 3, null_srvs);
        }

        // 5. Accumulate directly into buf_accum_out_ in VRAM
        {
            AccumCBuffer cb = { (uint32_t)H, router_weights[e], is_first ? 0u : 1u, 0 };
            update_constant_buffer(cb_accum_params_, &cb, sizeof(cb));
            context_->CSSetShader(shader_accumulate_, nullptr, 0);
            context_->CSSetConstantBuffers(0, 1, &cb_accum_params_);
            context_->CSSetShaderResources(0, 1, &srv_expert_out_);
            context_->CSSetUnorderedAccessViews(0, 1, &uav_accum_out_, nullptr);
            context_->Dispatch((H + 63) / 64, 1, 1);

            ID3D11UnorderedAccessView* null_uav = nullptr;
            ID3D11ShaderResourceView* null_srv = nullptr;
            context_->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
            context_->CSSetShaderResources(0, 1, &null_srv);
        }
        auto t_c1 = std::chrono::high_resolution_clock::now();
        comp_acc += std::chrono::duration<double, std::milli>(t_c1 - t_c0).count();
        if (use_ts) context_->End(q_ts_[e * 3 + 2]);
    }
    if (use_ts) context_->End(q_disjoint_);

    // Read back accumulated result ONCE for the entire top-6 set
    auto t_d2h_0 = std::chrono::high_resolution_clock::now();
    context_->CopyResource(buf_staging_out_, buf_accum_out_);
    D3D11_MAPPED_SUBRESOURCE mapped;
    HRESULT hr = context_->Map(buf_staging_out_, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        memcpy(out, mapped.pData, H * sizeof(float));
        context_->Unmap(buf_staging_out_, 0);
    }
    auto t_d2h_1 = std::chrono::high_resolution_clock::now();
    double d2h_time = std::chrono::duration<double, std::milli>(t_d2h_1 - t_d2h_0).count();
    if (use_ts && SUCCEEDED(hr)) collect_gpu_timestamps(static_cast<int>(expert_ids.size()));

    for (int s : pinned_this_layer_) slot_pinned_[s] = 0;
    pinned_this_layer_.clear();

    auto t_end = std::chrono::high_resolution_clock::now();
    telemetry_.host_to_device_time_ms += h2d_acc;
    telemetry_.kernel_compute_time_ms += comp_acc;
    telemetry_.device_to_host_time_ms += d2h_time;
    telemetry_.total_gpu_time_ms += std::chrono::duration<double, std::milli>(t_end - t_start).count();
    telemetry_.total_dispatches += 6;
    return SUCCEEDED(hr);
}

bool M8GpuExpertKernel::forward_top6_layer(const std::vector<const uint8_t*>& expert_scales,
                                          const std::vector<const uint8_t*>& expert_weights,
                                          const std::vector<float>& router_weights,
                                          const float* x,
                                          float* out) {
    std::vector<int> dummy_ids = {-1, -1, -1, -1, -1, -1};
    return forward_top6_layer(-1, dummy_ids, expert_scales, expert_weights, router_weights, x, out);
}

} // namespace m8
} // namespace asema
