#include "asema/m8/m8_paths.hpp"
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_multi_volume.hpp"

// HLSL Compute Shader source for packed FP4 GEMV
static const char* s_hlsl_gemv = R"(
cbuffer Params : register(b0) {
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

[numthreads(64, 1, 1)]
void CSMain(uint3 dispatchThreadId : SV_DispatchThreadID) {
    uint r = dispatchThreadId.x;
    if (r >= g_rows) return;

    uint row_w_offset = r * g_row_weight_stride;
    uint row_s_offset = r * g_row_scale_stride;

    float row_sum = 0.0f;

    for (uint b = 0; b < g_num_blocks; ++b) {
        // Load scale byte
        uint s_byte_addr = row_s_offset + b;
        uint s_word = g_scales.Load(s_byte_addr & ~3);
        uint s_val = (s_word >> ((s_byte_addr & 3) * 8)) & 0xFF;
        float scale = ldexp(1.0f, int(s_val) - 127);

        // Load 16 bytes of weights = 4 uints = 32 nibbles
        uint w_byte_addr = row_w_offset + b * 16;
        uint4 w4 = g_weights.Load4(w_byte_addr);

        uint x_base = b * 32;
        float blk_dot = 0.0f;

        // Process 4 uints (each uint has 4 bytes = 8 nibbles)
        [unroll]
        for (uint u = 0; u < 4; ++u) {
            uint val = (u == 0) ? w4.x : ((u == 1) ? w4.y : ((u == 2) ? w4.z : w4.w));
            [unroll]
            for (uint i = 0; i < 4; ++i) {
                uint byte_val = (val >> (i * 8)) & 0xFF;
                int low = int(byte_val & 0x0F);
                if (low >= 8) low -= 16;
                int high = int((byte_val >> 4) & 0x0F);
                if (high >= 8) high -= 16;

                uint x_idx = x_base + u * 8 + i * 2;
                blk_dot += float(low) * g_x[x_idx] + float(high) * g_x[x_idx + 1];
            }
        }

        row_sum += scale * blk_dot;
    }

    g_out[r] = row_sum;
}
)";

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "  ASEMA M8.18: GPU FP4 GEMV PROTOTYPE TEST (DIRECT3D 11)" << std::endl;
    std::cout << "======================================================" << std::endl;

    // 1. Initialize Direct3D 11
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;

    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        0, featureLevels, 2, D3D11_SDK_VERSION,
        &device, &featureLevel, &context
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device. hr=0x" << std::hex << hr << std::endl;
        return 1;
    }
    std::cout << "[1] D3D11 Hardware Device Initialized (Feature Level: 0x" << std::hex << featureLevel << std::dec << ")" << std::endl;

    // 2. Compile Compute Shader
    ID3DBlob* csBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    hr = D3DCompile(
        s_hlsl_gemv, strlen(s_hlsl_gemv), "gemv_packed4",
        nullptr, nullptr, "CSMain", "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
        &csBlob, &errorBlob
    );
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Shader compile error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return 1;
    }
    std::cout << "[2] HLSL Compute Shader compiled successfully to cs_5_0" << std::endl;

    ID3D11ComputeShader* computeShader = nullptr;
    hr = device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &computeShader);
    csBlob->Release();
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader: 0x" << std::hex << hr << std::endl;
        return 1;
    }

    // 3. Load Real Expert 0 Weights from NVMe via M8MultiVolumeManager + M8ByteRangeLoader
    auto vol_mgr = std::make_shared<asema::m8::M8MultiVolumeManager>();
    vol_mgr->register_volume(asema::m8::paths::primary_shards());
    vol_mgr->register_volume(asema::m8::paths::secondary_shards());
    asema::m8::M8ByteRangeLoader loader(vol_mgr);
    asema::m8::ExpertPayload payload;
    if (!loader.load_expert_payload(0, 0, payload)) {
        std::cerr << "Failed to load layer 0 expert 0." << std::endl;
        return 1;
    }
    std::cout << "[3] Real Expert 0 loaded from storage (" << payload.total_bytes() << " bytes)" << std::endl;

    const uint8_t* w1_scales = payload.scales.data();
    const uint8_t* w1_weights = payload.weights.data();
    const int rows = 2304;
    const int cols = 5120;
    const int num_blocks = cols / 32;
    const int row_weight_stride = cols / 2;
    const int row_scale_stride = num_blocks;

    // Test input x
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = std::sin(static_cast<float>(i + 1) * 0.05f);
    }

    // 4. Run CPU Reference GEMV
    std::vector<float> cpu_out(rows, 0.0f);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    {
        const int row_stride = cols / 2;
        for (int r = 0; r < rows; ++r) {
            const uint8_t* row_w = w1_weights + r * row_stride;
            const uint8_t* row_s = w1_scales + r * num_blocks;
            float row_sum = 0.0f;
            for (int b = 0; b < num_blocks; ++b) {
                float scale = std::ldexp(1.0f, row_s[b] - 127);
                const uint8_t* blk_w = row_w + b * 16;
                const float* blk_x = x.data() + b * 32;
                float blk_dot = 0.0f;
                for (int k = 0; k < 16; ++k) {
                    int low = blk_w[k] & 0x0F;
                    if (low >= 8) low -= 16;
                    int high = (blk_w[k] >> 4) & 0x0F;
                    if (high >= 8) high -= 16;
                    blk_dot += static_cast<float>(low) * blk_x[2 * k] + static_cast<float>(high) * blk_x[2 * k + 1];
                }
                row_sum += scale * blk_dot;
            }
            cpu_out[r] = row_sum;
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "[4] CPU GEMV executed in " << cpu_time_ms << " ms" << std::endl;

    // 5. Create D3D11 Buffers for GPU
    // Constant Buffer
    struct CBuffer {
        uint32_t rows;
        uint32_t cols;
        uint32_t num_blocks;
        uint32_t row_weight_stride;
        uint32_t row_scale_stride;
        uint32_t pad[3];
    } cb_data = { (uint32_t)rows, (uint32_t)cols, (uint32_t)num_blocks, (uint32_t)row_weight_stride, (uint32_t)row_scale_stride, {0,0,0} };

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(CBuffer);
    cbDesc.Usage = D3D11_USAGE_IMMUTABLE;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA cbInit = { &cb_data, 0, 0 };
    ID3D11Buffer* cbParams = nullptr;
    device->CreateBuffer(&cbDesc, &cbInit, &cbParams);

    // Scales Buffer (Raw ByteAddressBuffer)
    D3D11_BUFFER_DESC sDesc = {};
    sDesc.ByteWidth = rows * row_scale_stride;
    sDesc.Usage = D3D11_USAGE_IMMUTABLE;
    sDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    sDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    D3D11_SUBRESOURCE_DATA sInit = { w1_scales, 0, 0 };
    ID3D11Buffer* bufScales = nullptr;
    device->CreateBuffer(&sDesc, &sInit, &bufScales);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    srvDesc.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    srvDesc.BufferEx.NumElements = sDesc.ByteWidth / 4;
    ID3D11ShaderResourceView* srvScales = nullptr;
    device->CreateShaderResourceView(bufScales, &srvDesc, &srvScales);

    // Weights Buffer (Raw ByteAddressBuffer)
    D3D11_BUFFER_DESC wDesc = {};
    wDesc.ByteWidth = rows * row_weight_stride;
    wDesc.Usage = D3D11_USAGE_IMMUTABLE;
    wDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    wDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    D3D11_SUBRESOURCE_DATA wInit = { w1_weights, 0, 0 };
    ID3D11Buffer* bufWeights = nullptr;
    device->CreateBuffer(&wDesc, &wInit, &bufWeights);

    srvDesc.BufferEx.NumElements = wDesc.ByteWidth / 4;
    ID3D11ShaderResourceView* srvWeights = nullptr;
    device->CreateShaderResourceView(bufWeights, &srvDesc, &srvWeights);

    // Vector X (StructuredBuffer<float>)
    D3D11_BUFFER_DESC xDesc = {};
    xDesc.ByteWidth = cols * sizeof(float);
    xDesc.Usage = D3D11_USAGE_IMMUTABLE;
    xDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    xDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    xDesc.StructureByteStride = sizeof(float);
    D3D11_SUBRESOURCE_DATA xInit = { x.data(), 0, 0 };
    ID3D11Buffer* bufX = nullptr;
    device->CreateBuffer(&xDesc, &xInit, &bufX);

    D3D11_SHADER_RESOURCE_VIEW_DESC xSrvDesc = {};
    xSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    xSrvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    xSrvDesc.Buffer.NumElements = cols;
    ID3D11ShaderResourceView* srvX = nullptr;
    device->CreateShaderResourceView(bufX, &xSrvDesc, &srvX);

    // Output Buffer (RWStructuredBuffer<float>)
    D3D11_BUFFER_DESC outDesc = {};
    outDesc.ByteWidth = rows * sizeof(float);
    outDesc.Usage = D3D11_USAGE_DEFAULT;
    outDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    outDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    outDesc.StructureByteStride = sizeof(float);
    ID3D11Buffer* bufOut = nullptr;
    device->CreateBuffer(&outDesc, nullptr, &bufOut);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.NumElements = rows;
    ID3D11UnorderedAccessView* uavOut = nullptr;
    device->CreateUnorderedAccessView(bufOut, &uavDesc, &uavOut);

    // Staging buffer to read back output
    D3D11_BUFFER_DESC stageDesc = {};
    stageDesc.ByteWidth = rows * sizeof(float);
    stageDesc.Usage = D3D11_USAGE_STAGING;
    stageDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* bufStage = nullptr;
    device->CreateBuffer(&stageDesc, nullptr, &bufStage);

    // 6. Execute GPU Dispatch
    context->CSSetShader(computeShader, nullptr, 0);
    context->CSSetConstantBuffers(0, 1, &cbParams);
    ID3D11ShaderResourceView* srvs[3] = { srvScales, srvWeights, srvX };
    context->CSSetShaderResources(0, 3, srvs);
    context->CSSetUnorderedAccessViews(0, 1, &uavOut, nullptr);

    auto gpu_start = std::chrono::high_resolution_clock::now();
    // Dispatch: 2304 / 64 = 36 threadgroups
    context->Dispatch((rows + 63) / 64, 1, 1);
    
    // Copy to staging and map
    context->CopyResource(bufStage, bufOut);
    D3D11_MAPPED_SUBRESOURCE mapped;
    context->Map(bufStage, 0, D3D11_MAP_READ, 0, &mapped);
    std::vector<float> gpu_out(rows);
    memcpy(gpu_out.data(), mapped.pData, rows * sizeof(float));
    context->Unmap(bufStage, 0);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::cout << "[6] GPU GEMV + Readback executed in " << gpu_time_ms << " ms" << std::endl;

    // 7. Numerical Comparison
    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    double dot_prod = 0.0, norm_cpu = 0.0, norm_gpu = 0.0;
    for (int r = 0; r < rows; ++r) {
        float diff = std::abs(cpu_out[r] - gpu_out[r]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        sum_abs_diff += diff;
        dot_prod += cpu_out[r] * gpu_out[r];
        norm_cpu += cpu_out[r] * cpu_out[r];
        norm_gpu += gpu_out[r] * gpu_out[r];
    }
    float mean_abs_diff = sum_abs_diff / rows;
    double cosine_sim = dot_prod / (std::sqrt(norm_cpu) * std::sqrt(norm_gpu));

    std::cout << "\n================ NUMERICAL COMPARISON ================" << std::endl;
    std::cout << "  Max Absolute Difference:  " << max_abs_diff << std::endl;
    std::cout << "  Mean Absolute Difference: " << mean_abs_diff << std::endl;
    std::cout << "  Cosine Similarity:        " << cosine_sim << std::endl;
    std::cout << "  Speedup vs CPU GEMV:      " << (cpu_time_ms / gpu_time_ms) << "x" << std::endl;
    std::cout << "======================================================\n" << std::endl;

    // Cleanup
    bufStage->Release();
    uavOut->Release();
    bufOut->Release();
    srvX->Release();
    bufX->Release();
    srvWeights->Release();
    bufWeights->Release();
    srvScales->Release();
    bufScales->Release();
    cbParams->Release();
    computeShader->Release();
    context->Release();
    device->Release();

    if (max_abs_diff > 1e-4f || cosine_sim < 0.9999) {
        std::cerr << "FAIL: GPU output diverged from CPU reference!" << std::endl;
        return 1;
    }

    std::cout << "PASSED: GPU FP4 GEMV is numerically identical to CPU reference!" << std::endl;
    return 0;
}
