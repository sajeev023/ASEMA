// D3D11 experiments (E4 round trip + upload throughput, E5 VRAM residency). Standalone, no production code.
// usage: d3d_bench rt | upload | vram
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static ID3D11Device* dev; static ID3D11DeviceContext* ctx;
static double now_us() { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return c.QuadPart * 1e6 / f.QuadPart; }

static ID3D11ComputeShader* make_cs(const char* src) {
    ID3DBlob *b = nullptr, *e = nullptr;
    if (FAILED(D3DCompile(src, strlen(src), nullptr, nullptr, nullptr, "CSMain", "cs_5_0", 0, 0, &b, &e))) {
        std::fprintf(stderr, "shader error: %s\n", e ? (char*)e->GetBufferPointer() : "?"); std::exit(1);
    }
    ID3D11ComputeShader* cs = nullptr; dev->CreateComputeShader(b->GetBufferPointer(), b->GetBufferSize(), nullptr, &cs); b->Release(); return cs;
}
static ID3D11Buffer* make_buf(UINT bytes, D3D11_USAGE usage, UINT bind, UINT cpu, UINT misc = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) {
    D3D11_BUFFER_DESC d{}; d.ByteWidth = bytes; d.Usage = usage; d.BindFlags = bind; d.CPUAccessFlags = cpu; d.MiscFlags = (bind ? misc : 0);
    ID3D11Buffer* b = nullptr; if (FAILED(dev->CreateBuffer(&d, nullptr, &b))) return nullptr; return b;
}
static ID3D11ShaderResourceView* srv_raw(ID3D11Buffer* b, UINT bytes) {
    D3D11_SHADER_RESOURCE_VIEW_DESC d{}; d.Format = DXGI_FORMAT_R32_TYPELESS; d.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
    d.BufferEx.NumElements = bytes / 4; d.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    ID3D11ShaderResourceView* v = nullptr; dev->CreateShaderResourceView(b, &d, &v); return v;
}
static ID3D11UnorderedAccessView* uav_raw(ID3D11Buffer* b, UINT bytes) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC d{}; d.Format = DXGI_FORMAT_R32_TYPELESS; d.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements = bytes / 4; d.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    ID3D11UnorderedAccessView* v = nullptr; dev->CreateUnorderedAccessView(b, &d, &v); return v;
}
static void wait_gpu() {  // block until all queued work is done
    static ID3D11Query* q = nullptr;
    if (!q) { D3D11_QUERY_DESC d{D3D11_QUERY_EVENT, 0}; dev->CreateQuery(&d, &q); }
    ctx->End(q); ctx->Flush(); BOOL done = FALSE;
    while (ctx->GetData(q, &done, sizeof(done), 0) != S_OK || !done) { }
}
static void stats(std::vector<double>& v, double& avg, double& p50, double& p99) {
    std::sort(v.begin(), v.end()); avg = 0; for (double x : v) avg += x; avg /= v.size(); p50 = v[v.size() / 2]; p99 = v[size_t(v.size() * 0.99)];
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "rt";
    D3D_FEATURE_LEVEL fl;
    if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, &fl, &ctx))) { std::puts("no device"); return 1; }
    IDXGIDevice* dd = nullptr; dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dd);
    IDXGIAdapter* ad = nullptr; dd->GetAdapter(&ad); DXGI_ADAPTER_DESC desc; ad->GetDesc(&desc);
    char name[128]; WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, 128, nullptr, nullptr);
    std::printf("device: %s, dedicated VRAM %.2f GB, feature level 0x%x\n", name, desc.DedicatedVideoMemory / 1073741824.0, fl);

    if (mode == "rt") {
        // copy kernel: out[i] = in[i]
        auto cs = make_cs("ByteAddressBuffer s:register(t0); RWByteAddressBuffer d:register(u0);"
                          "[numthreads(64,1,1)] void CSMain(uint3 id:SV_DispatchThreadID){ d.Store(id.x*4, s.Load(id.x*4)); }");
        std::printf("\nCPU->GPU->compute->GPU->CPU round trip, per iteration (microseconds), 400 iterations\n");
        std::printf("%-9s | %-30s | %-30s\n", "bytes", "DEFAULT + UpdateSubresource", "DYNAMIC + Map(WRITE_DISCARD)");
        std::printf("%-9s | %9s %9s %9s | %9s %9s %9s\n", "", "avg", "p50", "p99", "avg", "p50", "p99");
        for (UINT S : {1024u, 4096u, 16384u, 20480u, 65536u, 262144u, 1048576u}) {
            std::vector<char> host(S, 7);
            double res[2][3];
            for (int variant = 0; variant < 2; ++variant) {
                ID3D11Buffer* in = variant == 0 ? make_buf(S, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0)
                                                : make_buf(S, D3D11_USAGE_DYNAMIC, D3D11_BIND_SHADER_RESOURCE, D3D11_CPU_ACCESS_WRITE);
                ID3D11Buffer* out = make_buf(S, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0);
                ID3D11Buffer* stg = make_buf(S, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0);
                auto sv = srv_raw(in, S); auto uv = uav_raw(out, S);
                ctx->CSSetShader(cs, nullptr, 0); ctx->CSSetShaderResources(0, 1, &sv); ctx->CSSetUnorderedAccessViews(0, 1, &uv, nullptr);
                std::vector<double> t;
                for (int i = 0; i < 450; ++i) {
                    const double t0 = now_us();
                    if (variant == 0) ctx->UpdateSubresource(in, 0, nullptr, host.data(), 0, 0);
                    else { D3D11_MAPPED_SUBRESOURCE m; ctx->Map(in, 0, D3D11_MAP_WRITE_DISCARD, 0, &m); std::memcpy(m.pData, host.data(), S); ctx->Unmap(in, 0); }
                    ctx->Dispatch((S / 4 + 63) / 64, 1, 1);
                    ctx->CopyResource(stg, out);
                    D3D11_MAPPED_SUBRESOURCE m; ctx->Map(stg, 0, D3D11_MAP_READ, 0, &m); volatile char c = ((char*)m.pData)[0]; (void)c; ctx->Unmap(stg, 0);
                    if (i >= 50) t.push_back(now_us() - t0);
                }
                stats(t, res[variant][0], res[variant][1], res[variant][2]);
                ID3D11ShaderResourceView* nsv = nullptr; ID3D11UnorderedAccessView* nuv = nullptr;
                ctx->CSSetShaderResources(0, 1, &nsv); ctx->CSSetUnorderedAccessViews(0, 1, &nuv, nullptr);
                sv->Release(); uv->Release(); in->Release(); out->Release(); stg->Release();
            }
            std::printf("%-9u | %9.0f %9.0f %9.0f | %9.0f %9.0f %9.0f\n", S, res[0][0], res[0][1], res[0][2], res[1][0], res[1][1], res[1][2]);
        }
        std::puts("\n(the 1 KB row is effectively the fixed synchronisation floor of one CPU->GPU->CPU round trip)");
    }

    if (mode == "upload") {
        const UINT S = 18800640;  // one expert
        std::vector<char> src(S, 3);
        void* pinned = VirtualAlloc(nullptr, S, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); std::memset(pinned, 5, S); VirtualLock(pinned, S);
        std::printf("\nHost->VRAM upload of one 18.8 MB expert, GB/s (waits for the GPU to finish each batch of 32)\n");
        auto run = [&](const char* label, auto&& body, int batches = 12) {
            std::vector<double> t;
            for (int b = 0; b < batches; ++b) { const double t0 = now_us(); for (int i = 0; i < 32; ++i) body(i); wait_gpu(); if (b >= 2) t.push_back(now_us() - t0); }
            double avg, p50, p99; stats(t, avg, p50, p99);
            std::printf("  %-62s %6.2f GB/s  (%.2f ms/expert)\n", label, 32.0 * S / (p50 * 1e-6) / 1e9, p50 / 32 / 1000.0);
        };
        std::vector<ID3D11Buffer*> dst; for (int i = 0; i < 8; ++i) dst.push_back(make_buf(S, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0));
        run("DEFAULT <- UpdateSubresource(pageable heap)  [current]", [&](int i) { ctx->UpdateSubresource(dst[i % 8], 0, nullptr, src.data(), 0, 0); });
        run("DEFAULT <- UpdateSubresource(VirtualLock'ed memory)", [&](int i) { ctx->UpdateSubresource(dst[i % 8], 0, nullptr, pinned, 0, 0); });
        std::vector<ID3D11Buffer*> stg; for (int i = 0; i < 4; ++i) stg.push_back(make_buf(S, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_WRITE, 0));
        run("STAGING Map(WRITE)+memcpy+CopyResource, ring of 4", [&](int i) {
            D3D11_MAPPED_SUBRESOURCE m; ctx->Map(stg[i % 4], 0, D3D11_MAP_WRITE, 0, &m); std::memcpy(m.pData, src.data(), S); ctx->Unmap(stg[i % 4], 0);
            ctx->CopyResource(dst[i % 8], stg[i % 4]); });
        run("STAGING Map(WRITE)+memcpy only (the CPU side of the above)", [&](int i) {
            D3D11_MAPPED_SUBRESOURCE m; ctx->Map(stg[i % 4], 0, D3D11_MAP_WRITE, 0, &m); std::memcpy(m.pData, src.data(), S); ctx->Unmap(stg[i % 4], 0); });
        run("STAGING->DEFAULT CopyResource only (GPU DMA, data already staged)", [&](int i) { ctx->CopyResource(dst[i % 8], stg[i % 4]); });
        std::puts("  (PCIe 3.0 x16 theoretical 15.75 GB/s; the RX 580 may run x8 or x16 depending on the slot)");
    }

    if (mode == "vram") {
        IDXGIAdapter3* a3 = nullptr; ad->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3);
        auto mem = [&]() { DXGI_QUERY_VIDEO_MEMORY_INFO m{}; if (a3) a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &m);
                           std::printf("    DXGI local segment: budget %.2f GB, current usage %.2f GB\n", m.Budget / 1073741824.0, m.CurrentUsage / 1073741824.0); };
        mem();
        auto cs = make_cs("ByteAddressBuffer s:register(t0); RWByteAddressBuffer d:register(u0);"
                          "[numthreads(256,1,1)] void CSMain(uint3 id:SV_DispatchThreadID){ uint4 a=0; const uint T=1048576;"
                          " for(uint k=0;k<8;++k){ a ^= s.Load4(((id.x + k*T)*16)); } d.Store(id.x*4, a.x^a.y^a.z^a.w); }");
        const UINT CH = 128u << 20;   // 128 MiB buffers (8 uint4 loads x 1,048,576 threads x 16 B = 128 MiB)
        ID3D11Buffer* dst = make_buf(1048576 * 4, D3D11_USAGE_DEFAULT, D3D11_BIND_UNORDERED_ACCESS, 0); auto uv = uav_raw(dst, 1048576 * 4);
        std::printf("\nResident set vs sustained read bandwidth (compute kernel sweeping every buffer, GB/s)\n");
        std::printf("%8s %8s %10s %10s %10s\n", "target", "buffers", "alloc", "GB/s p50", "GB/s min");
        for (double gb : {1.0, 2.0, 4.0, 5.0, 6.0, 6.5, 7.0}) {
            const int n = (int)(gb * 1024 / 128 + 0.5);
            std::vector<ID3D11Buffer*> bufs; std::vector<ID3D11ShaderResourceView*> svs; bool ok = true;
            for (int i = 0; i < n; ++i) { auto b = make_buf(CH, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE, 0); if (!b) { ok = false; break; } bufs.push_back(b); svs.push_back(srv_raw(b, CH)); }
            if (!ok) { std::printf("%7.1fG %8zu   ALLOCATION FAILED\n", gb, bufs.size()); }
            else {
                std::vector<char> pat(1 << 20, 9);
                for (auto b : bufs) { D3D11_BOX box{0, 0, 0, 1u << 20, 1, 1}; ctx->UpdateSubresource(b, 0, &box, pat.data(), 0, 0); }  // touch
                wait_gpu();
                ctx->CSSetShader(cs, nullptr, 0); ctx->CSSetUnorderedAccessViews(0, 1, &uv, nullptr);
                std::vector<double> bw;
                for (int sweep = 0; sweep < 6; ++sweep) {
                    const double t0 = now_us();
                    for (int i = 0; i < n; ++i) { ctx->CSSetShaderResources(0, 1, &svs[i]); ctx->Dispatch(4096, 1, 1); }
                    wait_gpu();
                    if (sweep >= 1) bw.push_back((double)n * CH / ((now_us() - t0) * 1e-6) / 1e9);
                }
                std::sort(bw.begin(), bw.end());
                std::printf("%7.1fG %8d %10s %10.1f %10.1f\n", gb, n, "ok", bw[bw.size() / 2], bw.front());
                mem();
            }
            ID3D11ShaderResourceView* nsv = nullptr; ctx->CSSetShaderResources(0, 1, &nsv);
            for (auto v : svs) v->Release(); for (auto b : bufs) b->Release(); ctx->ClearState(); ctx->Flush();
            if (!ok) break;
        }
    }
    return 0;
}
