#include "asema/m8/m8_model_doctor.hpp"
#include "asema/m8/m8_model_verifier.hpp"
#include "asema/m8/m8_drive_info.hpp"

#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <intrin.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

namespace {

std::string cpu_brand() {
    int regs[4] = {0};
    __cpuid(regs, 0x80000000);
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) return "unknown CPU";
    char brand[49] = {0};
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002 + i);
        std::memcpy(brand + i * 16, regs, 16);
    }
    std::string s(brand);
    const size_t b = s.find_first_not_of(' ');
    return b == std::string::npos ? "unknown CPU" : s.substr(b);
}

// A safetensors file starts with a little-endian u64 header length followed by that many bytes of JSON.
// This is a cheap structural check, not a hash: it catches truncated or wrong files, not bit rot.
bool shard_header_plausible(const fs::path& p) {
    std::error_code ec;
    const uint64_t size = fs::file_size(p, ec);
    if (ec || size < 16) return false;
    std::ifstream in(p, std::ios::binary);
    uint64_t hdr = 0;
    if (!in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr))) return false;
    return hdr > 1 && hdr < (200ULL << 20) && hdr + 8 <= size;
}

} // namespace

DoctorReport M8ModelDoctor::run_diagnostics(
    const std::string& hf_root,
    const std::string& vol_primary,
    const std::string& vol_secondary) {

    DoctorReport rep;
    rep.all_passed = true;

    auto add_item = [&](const std::string& cat, const std::string& name, bool pass, const std::string& det,
                        const std::string& hint = "") {
        rep.items.push_back({cat, name, pass, det, pass ? std::string() : hint});
        if (!pass) rep.all_passed = false;
    };

    // 1. CPU
    {
        int cpu_info[4] = {0};
        __cpuid(cpu_info, 0);
        const int n_ids = cpu_info[0];
        bool avx2 = false, fma = false;
        if (n_ids >= 1) { __cpuid(cpu_info, 1); fma = (cpu_info[2] & (1 << 12)) != 0; }
        if (n_ids >= 7) { __cpuidex(cpu_info, 7, 0); avx2 = (cpu_info[1] & (1 << 5)) != 0; }
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        std::ostringstream ss;
        ss << cpu_brand() << " (" << si.dwNumberOfProcessors << " logical threads, AVX2=" << (avx2 ? "yes" : "NO")
           << ", FMA=" << (fma ? "yes" : "NO") << ")";
        add_item("CPU", "Instruction set", avx2 && fma, ss.str(),
                 "ASEMA's CPU kernels need AVX2 and FMA (Intel Haswell / AMD Zen or newer).");
    }

    // 2. RAM
    {
        MEMORYSTATUSEX m;
        m.dwLength = sizeof(m);
        GlobalMemoryStatusEx(&m);
        const double total_gb = static_cast<double>(m.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
        const double avail_gb = static_cast<double>(m.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream ss;
        ss << static_cast<int>(total_gb + 0.5) << " GB total, " << static_cast<int>(avail_gb + 0.5)
           << " GB available (32 GB tested; the process used about 8-14 GB)";
        add_item("RAM", "System memory", total_gb >= 15.0 && avail_gb >= 4.0, ss.str(),
                 "Close other programs, or add RAM. Under 16 GB total is untested.");
    }

    // 3. GPU
    {
        ID3D11Device* dev = nullptr;
        ID3D11DeviceContext* ctx = nullptr;
        D3D_FEATURE_LEVEL fl;
        const HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0,
                                             D3D11_SDK_VERSION, &dev, &fl, &ctx);
        const bool d3d_ok = SUCCEEDED(hr) && fl >= D3D_FEATURE_LEVEL_11_0;
        std::string gpu_name = "no Direct3D 11 hardware device";
        uint64_t vram = 0;
        if (d3d_ok && dev) {
            IDXGIDevice* dxgi_dev = nullptr;
            if (SUCCEEDED(dev->QueryInterface(__uuidof(IDXGIDevice), reinterpret_cast<void**>(&dxgi_dev)))) {
                IDXGIAdapter* adapter = nullptr;
                if (SUCCEEDED(dxgi_dev->GetAdapter(&adapter))) {
                    DXGI_ADAPTER_DESC desc;
                    if (SUCCEEDED(adapter->GetDesc(&desc))) {
                        char buf[256] = {0};
                        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, buf, sizeof(buf) - 1, nullptr, nullptr);
                        gpu_name = buf;
                        vram = desc.DedicatedVideoMemory;
                    }
                    adapter->Release();
                }
                dxgi_dev->Release();
            }
        }
        if (ctx) ctx->Release();
        if (dev) dev->Release();
        const double vram_gb = static_cast<double>(vram) / (1024.0 * 1024.0 * 1024.0);
        std::ostringstream ss;
        ss << gpu_name << " (feature level 11+, " << static_cast<int>(vram_gb + 0.5) << " GB dedicated VRAM)";
        add_item("GPU", "DirectCompute device", d3d_ok && vram_gb >= 4.0, ss.str(),
                 d3d_ok ? "Under 4 GB VRAM: lower ASEMA_VRAM_SLOTS (each slot holds one 18.8 MB expert; default 192)."
                        : "Install/update the GPU driver; a Direct3D 11 feature-level 11.0 GPU is required.");
    }

    // 4. Checkpoint metadata
    const auto verif = M8ModelVerifier::verify_checkpoint(hf_root, vol_primary, vol_secondary);
    add_item("Model", "config.json", verif.config_valid,
             verif.config_valid ? "present, 40 layers" : "missing or not the expected DeepSeek-V4.1-Flash config",
             "Put the Hugging Face metadata (config.json, tokenizer.json) in the directory set by hf_dir / ASEMA_HF_DIR.");
    add_item("Model", "tokenizer", verif.tokenizer_valid,
             verif.tokenizer_valid ? "tokenizer.json present and parseable" : "tokenizer.json missing or unreadable",
             "Download tokenizer.json from the model repository into hf_dir. See MODEL_SETUP.md.");
    {
        std::ostringstream ss;
        ss << (verif.index_valid ? "model.safetensors.index.json valid, " : "model.safetensors.index.json missing or invalid, ")
           << verif.total_tensors_verified << " tensors";
        add_item("Model", "tensor index", verif.index_valid, ss.str(),
                 "Place model.safetensors.index.json next to the metadata (model_root / hf_dir). See MODEL_SETUP.md.");
    }

    // 5. Shards present and structurally plausible
    {
        int present = 0, bad = 0;
        std::string first_bad;
        for (const std::string& dir : {vol_primary, vol_secondary}) {
            if (dir.empty() || !fs::exists(dir)) continue;
            for (const auto& e : fs::directory_iterator(dir)) {
                const std::string n = e.path().filename().string();
                if (n.rfind("model-", 0) != 0 || e.path().extension() != ".safetensors") continue;
                ++present;
                if (!shard_header_plausible(e.path())) { ++bad; if (first_bad.empty()) first_bad = n; }
            }
        }
        std::ostringstream ss;
        ss << present << " of 48 shard files found";
        if (bad) ss << "; " << bad << " have an invalid header (first: " << first_bad << ")";
        ss << " (header check only, not a hash)";
        add_item("Model", "shards", present == 48 && bad == 0, ss.str(),
                 bad ? "Re-download the corrupt shard(s); scripts/plan_shard_move.py verifies SHA-256 when moving."
                     : "Fetch the missing shards (see MODEL_SETUP.md) and put them in shards_primary or shards_secondary.");
    }

    // 6. Storage: where the shards live matters more than anything else for speed
    for (const std::string& dir : {vol_primary, vol_secondary}) {
        if (dir.empty()) continue;
        const bool exists = fs::exists(dir);
        std::ostringstream ss;
        ss << dir << ": ";
        if (exists) {
            const DriveClass dc = classify_drive_of_path(dir);
            ss << drive_class_name(dc) << ", " << (free_bytes_of_path(dir) >> 30) << " GB free";
            if (dc != DriveClass::NVME) ss << " (decode speed is limited by the slowest drive that holds experts)";
        } else {
            ss << "directory does not exist";
        }
        add_item("Storage", "shard directory", exists, ss.str(),
                 "Fix shards_primary / shards_secondary in asema.config (or ASEMA_SHARDS_*).");
    }

    // 7. Permissions
    {
        bool can_read = false;
        for (const std::string& dir : {vol_primary, vol_secondary}) {
            if (dir.empty() || !fs::exists(dir)) continue;
            for (const auto& e : fs::directory_iterator(dir)) {
                if (e.path().extension() == ".safetensors") {
                    std::ifstream in(e.path(), std::ios::binary);
                    can_read = in.good();
                    break;
                }
            }
            if (can_read) break;
        }
        add_item("Permissions", "read shards", can_read, can_read ? "a shard file opened for reading" : "could not open any shard for reading",
                 "Check that the current user can read the shard directories (not locked by another program).");

        const fs::path probe = fs::current_path() / ".asema_write_probe";
        std::ofstream out(probe, std::ios::binary);
        const bool can_write = out.good();
        out.close();
        std::error_code ec;
        fs::remove(probe, ec);
        add_item("Permissions", "write working dir", can_write, can_write ? "can create files (reports, traces)" : "cannot create files here",
                 "Run from a directory you can write to.");
    }

    // 8. Shader toolchain and runtime DLLs
    {
        const char* cs = "[numthreads(64, 1, 1)]\nvoid CSMain(uint3 tid : SV_DispatchThreadID) {}\n";
        ID3DBlob* blob = nullptr;
        ID3DBlob* err = nullptr;
        const HRESULT hr = D3DCompile(cs, std::strlen(cs), nullptr, nullptr, nullptr, "CSMain", "cs_5_0", 0, 0, &blob, &err);
        const bool ok = SUCCEEDED(hr);
        if (blob) blob->Release();
        if (err) err->Release();
        add_item("Runtime", "HLSL cs_5_0 compiler", ok, ok ? "compiles a compute shader" : "D3DCompile failed",
                 "Install/repair Windows graphics components (d3dcompiler_47.dll).");

        std::string missing;
        for (const char* dll : {"d3d11.dll", "dxgi.dll", "d3dcompiler_47.dll"}) {
            HMODULE h = LoadLibraryA(dll);
            if (!h) { missing += std::string(missing.empty() ? "" : ", ") + dll; } else { FreeLibrary(h); }
        }
        add_item("Runtime", "system DLLs", missing.empty(), missing.empty() ? "d3d11, dxgi, d3dcompiler_47 load" : "missing: " + missing,
                 "Reinstall the GPU driver / Windows graphics components.");
    }

    rep.summary = rep.all_passed ? "READY" : "PROBLEM";
    return rep;
}

} // namespace m8
} // namespace asema
