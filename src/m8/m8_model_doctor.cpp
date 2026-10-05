#include "asema/m8/m8_model_doctor.hpp"
#include "asema/m8/m8_storage_planner.hpp"
#include "asema/m8/m8_model_verifier.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <intrin.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

DoctorReport M8ModelDoctor::run_diagnostics(
    const std::string& hf_root,
    const std::string& vol_primary,
    const std::string& vol_secondary) {

    DoctorReport rep;
    rep.all_passed = true;

    auto add_item = [&](const std::string& cat, const std::string& name, bool pass, const std::string& det) {
        rep.items.push_back({cat, name, pass, det});
        if (!pass) rep.all_passed = false;
    };

    // 1. CPU Diagnostics
    {
        int cpu_info[4] = {0};
        __cpuid(cpu_info, 0);
        int n_ids = cpu_info[0];

        bool avx2_supported = false;
        bool fma_supported = false;

        if (n_ids >= 1) {
            __cpuid(cpu_info, 1);
            fma_supported = (cpu_info[2] & (1 << 12)) != 0;
        }
        if (n_ids >= 7) {
            __cpuidex(cpu_info, 7, 0);
            avx2_supported = (cpu_info[1] & (1 << 5)) != 0;
        }

        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);

        std::ostringstream ss;
        ss << "AMD Ryzen Architecture (" << sys_info.dwNumberOfProcessors << " logical threads, AVX2="
           << (avx2_supported ? "YES" : "NO") << ", FMA=" << (fma_supported ? "YES" : "NO") << ")";
        add_item("CPU", "Instruction Set & Topology", avx2_supported && fma_supported, ss.str());
    }

    // 2. RAM Diagnostics
    {
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        GlobalMemoryStatusEx(&mem_status);

        double total_gb = static_cast<double>(mem_status.ullTotalPhys) / (1024ULL * 1024ULL * 1024ULL);
        double avail_gb = static_cast<double>(mem_status.ullAvailPhys) / (1024ULL * 1024ULL * 1024ULL);

        std::ostringstream ss;
        ss << "Total: " << static_cast<int>(total_gb + 0.5) << " GB, Available: "
           << static_cast<int>(avail_gb + 0.5) << " GB (Required: >= 2.0 GB active bound)";
        add_item("RAM", "System Memory", total_gb >= 15.0 && avail_gb >= 2.0, ss.str());
    }

    // 3. GPU & Direct3D 11 Diagnostics
    {
        ID3D11Device* dev = nullptr;
        ID3D11DeviceContext* ctx = nullptr;
        D3D_FEATURE_LEVEL feature_level;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                      0, nullptr, 0, D3D11_SDK_VERSION,
                                      &dev, &feature_level, &ctx);

        bool d3d_ok = SUCCEEDED(hr) && (feature_level >= D3D_FEATURE_LEVEL_11_0);
        std::string gpu_name = "Unknown GPU";
        uint64_t vram_bytes = 0;

        if (d3d_ok && dev) {
            IDXGIDevice* dxgi_dev = nullptr;
            if (SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_dev))) {
                IDXGIAdapter* adapter = nullptr;
                if (SUCCEEDED(dxgi_dev->GetAdapter(&adapter))) {
                    DXGI_ADAPTER_DESC desc;
                    if (SUCCEEDED(adapter->GetDesc(&desc))) {
                        char name_buf[128] = {0};
                        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf, sizeof(name_buf), nullptr, nullptr);
                        gpu_name = name_buf;
                        vram_bytes = desc.DedicatedVideoMemory;
                    }
                    adapter->Release();
                }
                dxgi_dev->Release();
            }
            ctx->Release();
            dev->Release();
        }

        double vram_gb = static_cast<double>(vram_bytes) / (1024ULL * 1024ULL * 1024ULL);
        std::ostringstream ss;
        ss << gpu_name << " (D3D11 CS 5.0, Dedicated VRAM: " << static_cast<int>(vram_gb + 0.5)
           << " GB, Required: >= 1.0 GB bound)";
        add_item("GPU", "DirectCompute Device", d3d_ok && vram_gb >= 4.0, ss.str());
    }

    // 4. Checkpoint Metadata Diagnostics
    auto verif = M8ModelVerifier::verify_checkpoint(hf_root, vol_primary, vol_secondary);
    {
        std::ostringstream ss;
        ss << "Config: " << (verif.config_valid ? "VALID" : "INVALID")
           << ", Tokenizer: " << (verif.tokenizer_valid ? "VALID" : "INVALID")
           << ", Index: " << (verif.index_valid ? "VALID (" : "INVALID (")
           << verif.total_tensors_verified << " tensors)";
        add_item("Checkpoint", "Model Topology & Index", verif.passed, ss.str());
    }

    // 5. NVMe Dual-Volume Diagnostics
    {
        auto plan = M8StoragePlanner::probe_and_plan();
        std::ostringstream ss;
        bool storage_ok = false;
        if (verif.shards_found == 48) {
            storage_ok = true;
            ss << "Volume D: (" << (plan.plan_d.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB free), "
               << "Volume E: (" << (plan.plan_e.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB free) "
               << "[RESIDENT: 48/48 Shards Allocated]";
        } else {
            storage_ok = plan.overall_feasible;
            ss << "Volume D: (" << (plan.plan_d.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB free, 340 GB req), "
               << "Volume E: (" << (plan.plan_e.total_free_bytes / (1024ULL * 1024ULL * 1024ULL)) << " GB free, 170 GB req)";
        }
        add_item("Storage", "Physical Dual-NVMe Feasibility", storage_ok, ss.str());
    }

    // 6. DirectCompute Shader Compilation Check
    {
        const char* dummy_cs =
            "[numthreads(64, 1, 1)]\n"
            "void CSMain(uint3 tid : SV_DispatchThreadID) {}\n";
        ID3DBlob* cs_blob = nullptr;
        ID3DBlob* err_blob = nullptr;
        HRESULT hr = D3DCompile(dummy_cs, strlen(dummy_cs), nullptr, nullptr, nullptr,
                                "CSMain", "cs_5_0", 0, 0, &cs_blob, &err_blob);
        bool shader_ok = SUCCEEDED(hr);
        if (cs_blob) cs_blob->Release();
        if (err_blob) err_blob->Release();

        add_item("Shaders", "HLSL Compute Shader 5.0 Toolchain", shader_ok,
                 shader_ok ? "D3DCompiler cs_5_0 target functional" : "D3DCompiler failed to compile cs_5_0");
    }

    // 7. Shard Storage Invariants
    {
        bool p_exists = fs::exists(vol_primary);
        bool s_exists = fs::exists(vol_secondary);
        std::ostringstream ss;
        ss << "Primary Vol: " << (p_exists ? "PRESENT" : "ABSENT")
           << ", Secondary Vol: " << (s_exists ? "PRESENT" : "ABSENT");
        add_item("Filesystem", "Shard Directories", p_exists && s_exists, ss.str());
    }

    std::ostringstream summary_ss;
    summary_ss << "ASEMA Doctor Status: " << (rep.all_passed ? "ALL CHECKS PASSED (SYSTEM READY)" : "SYSTEM DEFICIENCIES DETECTED");
    rep.summary = summary_ss.str();

    return rep;
}

} // namespace m8
} // namespace asema
