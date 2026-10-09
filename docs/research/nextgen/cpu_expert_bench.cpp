// Isolated CPU FP4 expert benchmark (E3). Production code is only READ (linked as a library).
// Format (from src/m8/m8_expert_kernel.cpp): FP4 E2M1, 2 weights/byte (low nibble = even column),
// blocks of 32 columns share one UE8M0 scale byte, scale = 2^(s-127). x stays FP32 (no new quantization).
#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <immintrin.h>
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace asema::m8;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// ---- AVX2 kernel ------------------------------------------------------------------------------
// out[r] = sum_b scale_b * sum_k wq[k] * x[k], wq = E2M1 value * 2 as int8, so scale is multiplied by 0.5.
static void gemv_fp4_avx2(int row0, int row1, int cols, const uint8_t* w, const uint8_t* s,
                          const float* xe, const float* xo, float* out) {
    const int blocks = cols / 32, wstride = cols / 2;
    const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask = _mm_set1_epi8(0x0F);
    for (int r = row0; r < row1; ++r) {
        const uint8_t* rw = w + (size_t)r * wstride;
        const uint8_t* rs = s + (size_t)r * blocks;
        __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < blocks; ++b) {
            const __m128i v = _mm_loadu_si128((const __m128i*)(rw + b * 16));
            const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(v, mask));
            const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(v, 4), mask));
            const float* pe = xe + b * 16;
            const float* po = xo + b * 16;
            __m256 d = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(pe));
            d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(pe + 8), d);
            d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_loadu_ps(po), d);
            d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(po + 8), d);
            const uint32_t bits = (uint32_t)rs[b] << 23;            // 2^(s-127); s=0 -> 0 (true value ~6e-39)
            float sc; std::memcpy(&sc, &bits, 4);
            acc = _mm256_fmadd_ps(d, _mm256_set1_ps(sc * 0.5f), acc);
        }
        __m128 t = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        t = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
        out[r] = _mm_cvtss_f32(t);
    }
}

static void deinterleave(const float* x, int n, float* xe, float* xo) {
    for (int i = 0; i < n / 2; ++i) { xe[i] = x[2 * i]; xo[i] = x[2 * i + 1]; }
}

struct Scratch { std::vector<float> xe, xo, gate, up, act, ae, ao, outv; };

// ---- minimal spinning pool (lowest dispatch latency; this is a benchmark) ----------------------
struct Pool {
    int n; std::vector<std::thread> th; std::atomic<int> gen{0}, done{0}; std::atomic<bool> stop{false};
    std::function<void(int)> job;
    explicit Pool(int n_) : n(n_) {
        for (int t = 1; t < n; ++t) th.emplace_back([this, t] {
            int seen = 0;
            while (true) {
                int g; int spins = 0;
                while ((g = gen.load(std::memory_order_acquire)) == seen && !stop.load()) { if (++spins > 2000) { std::this_thread::yield(); } else _mm_pause(); }
                if (stop.load()) return;
                seen = g; job(t); done.fetch_add(1, std::memory_order_release);
            }
        });
    }
    void run(const std::function<void(int)>& f) {
        job = f; done.store(0); gen.fetch_add(1, std::memory_order_release);
        f(0);
        while (done.load(std::memory_order_acquire) < n - 1) _mm_pause();
    }
    ~Pool() { stop = true; for (auto& t : th) t.join(); }
};

// One expert, rows split across `T` pool threads, 3 dependent stages with a barrier between them.
static void expert_split(Pool& pool, int T, const uint8_t* sc, const uint8_t* wt, const float* x, float* out, float rw, bool accumulate, Scratch& s) {
    const int H = 5120, I = 2304;
    deinterleave(x, H, s.xe.data(), s.xo.data());
    pool.run([&](int t) {
        if (t >= T) return;
        const int a = I * t / T, b = I * (t + 1) / T;
        gemv_fp4_avx2(a, b, H, wt, sc, s.xe.data(), s.xo.data(), s.gate.data());
        gemv_fp4_avx2(a, b, H, wt + ExpertDimensions::W1_WEIGHT_BYTES + ExpertDimensions::W2_WEIGHT_BYTES,
                      sc + ExpertDimensions::W1_SCALE_BYTES + ExpertDimensions::W2_SCALE_BYTES, s.xe.data(), s.xo.data(), s.up.data());
        for (int i = a; i < b; ++i) {
            const float g = std::min(s.gate[i], 10.0f), u = std::clamp(s.up[i], -10.0f, 10.0f);
            s.act[i] = g / (1.0f + std::exp(-g)) * u;
        }
    });
    deinterleave(s.act.data(), I, s.ae.data(), s.ao.data());
    pool.run([&](int t) {
        if (t >= T) return;
        const int a = H * t / T, b = H * (t + 1) / T;
        gemv_fp4_avx2(a, b, I, wt + ExpertDimensions::W1_WEIGHT_BYTES, sc + ExpertDimensions::W1_SCALE_BYTES, s.ae.data(), s.ao.data(), s.outv.data());
        for (int i = a; i < b; ++i) out[i] = accumulate ? out[i] + s.outv[i] * rw : s.outv[i] * rw;
    });
}

static void init_scratch(Scratch& s) {
    s.xe.assign(2560, 0); s.xo.assign(2560, 0); s.gate.assign(2304, 0); s.up.assign(2304, 0); s.act.assign(2304, 0);
    s.ae.assign(1152, 0); s.ao.assign(1152, 0); s.outv.assign(5120, 0);
}

int main(int argc, char** argv) {
    const int max_experts = argc > 1 ? std::atoi(argv[1]) : 240;
    // ---- load real experts (layer 30, experts 0..5) through the production loader -------------
    auto vol = std::make_shared<M8MultiVolumeManager>();
    vol->register_volume(paths::primary_shards());
    vol->register_volume(paths::secondary_shards());
    vol->load_index(paths::hf_dir() + "/model.safetensors.index.json");
    M8ByteRangeLoader loader(vol);
    std::vector<ExpertPayload> real(6);
    for (int e = 0; e < 6; ++e) if (!loader.load_expert_payload(30, e, real[e])) { std::printf("load failed\n"); return 1; }
    std::printf("loaded 6 real experts (layer 30), %zu bytes each\n", real[0].total_bytes());

    // ---- correctness vs the production scalar kernel ------------------------------------------
    std::mt19937 rng(7); std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(5120); for (auto& v : x) v = nd(rng);
    {
        M8ExpertKernel ref; ref.attach(real[0].scales.data(), real[0].weights.data());
        std::vector<float> o_ref(5120, 0.f), o_new(5120, 0.f);
        const auto t0 = Clock::now(); ref.forward(x.data(), o_ref.data(), 1.0f, false); const double t_ref = ms_since(t0);
        Pool p1(1); Scratch s; init_scratch(s);
        expert_split(p1, 1, real[0].scales.data(), real[0].weights.data(), x.data(), o_new.data(), 1.0f, false, s);
        double maxabs = 0, num = 0, den = 0, dot = 0, nn = 0;
        for (int i = 0; i < 5120; ++i) {
            maxabs = std::max(maxabs, (double)std::fabs(o_ref[i] - o_new[i]));
            num += (double)(o_ref[i] - o_new[i]) * (o_ref[i] - o_new[i]); den += (double)o_ref[i] * o_ref[i];
            dot += (double)o_ref[i] * o_new[i]; nn += (double)o_new[i] * o_new[i];
        }
        std::printf("[correctness] vs production scalar kernel: max|diff| %.3e, rel-L2 %.3e, cosine %.9f (scalar took %.1f ms)\n",
                    maxabs, std::sqrt(num / den), dot / std::sqrt(den * nn), t_ref);
    }

    // ---- allocate N distinct expert buffers (so DRAM, not the 32 MB L3, is measured) -----------
    std::vector<uint8_t*> sc(max_experts), wt(max_experts);
    for (int i = 0; i < max_experts; ++i) {
        sc[i] = (uint8_t*)VirtualAlloc(nullptr, ExpertDimensions::TOTAL_SCALE_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        wt[i] = (uint8_t*)VirtualAlloc(nullptr, ExpertDimensions::TOTAL_WEIGHT_BYTES, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        std::memcpy(sc[i], real[i % 6].scales.data(), ExpertDimensions::TOTAL_SCALE_BYTES);
        std::memcpy(wt[i], real[i % 6].weights.data(), ExpertDimensions::TOTAL_WEIGHT_BYTES);
    }
    std::printf("allocated %d distinct expert buffers (%.2f GB)\n", max_experts, max_experts * 18.8006e6 / 1e9);

    std::vector<float> out(5120, 0.f);
    // ---- single expert latency, rows split over T threads (cold-from-DRAM: rotate through 64+ buffers) ----
    std::printf("\n[1 expert, rows split across T threads]  (rotating through %d buffers)\n  T   ms/expert   GB/s\n", std::min(max_experts, 64));
    for (int T : {1, 2, 4, 8, 12, 16}) {
        Pool pool(T); Scratch s; init_scratch(s);
        const int reps = 60, rot = std::min(max_experts, 64);
        for (int i = 0; i < 6; ++i) expert_split(pool, T, sc[i % rot], wt[i % rot], x.data(), out.data(), 1.f, true, s);  // warm
        const auto t0 = Clock::now();
        for (int i = 0; i < reps; ++i) expert_split(pool, T, sc[i % rot], wt[i % rot], x.data(), out.data(), 1.f, true, s);
        const double ms = ms_since(t0) / reps;
        std::printf("  %-3d %9.3f  %7.1f\n", T, ms, 18.8006e6 / (ms * 1e-3) / 1e9);
    }
    // ---- N experts, each expert on ONE thread, experts distributed over T threads ----------------
    std::printf("\n[N experts, expert-parallel, T threads]  total ms / ms per expert / aggregate GB/s\n");
    for (int N : {1, 6, 24, 64, 128, 240}) {
        if (N > max_experts) continue;
        std::printf("  N=%-3d", N);
        for (int T : {1, 2, 4, 8, 16}) {
            Pool pool(T);
            std::vector<Scratch> sc_(T); for (auto& s : sc_) init_scratch(s);
            auto run_once = [&] {
                pool.run([&](int t) {
                    for (int e = t; e < N; e += T) {
                        auto& s = sc_[t];
                        deinterleave(x.data(), 5120, s.xe.data(), s.xo.data());
                        gemv_fp4_avx2(0, 2304, 5120, wt[e], sc[e], s.xe.data(), s.xo.data(), s.gate.data());
                        gemv_fp4_avx2(0, 2304, 5120, wt[e] + ExpertDimensions::W1_WEIGHT_BYTES + ExpertDimensions::W2_WEIGHT_BYTES,
                                      sc[e] + ExpertDimensions::W1_SCALE_BYTES + ExpertDimensions::W2_SCALE_BYTES, s.xe.data(), s.xo.data(), s.up.data());
                        for (int i = 0; i < 2304; ++i) { const float g = std::min(s.gate[i], 10.f), u = std::clamp(s.up[i], -10.f, 10.f); s.act[i] = g / (1.f + std::exp(-g)) * u; }
                        deinterleave(s.act.data(), 2304, s.ae.data(), s.ao.data());
                        gemv_fp4_avx2(0, 5120, 2304, wt[e] + ExpertDimensions::W1_WEIGHT_BYTES, sc[e] + ExpertDimensions::W1_SCALE_BYTES, s.ae.data(), s.ao.data(), s.outv.data());
                    }
                });
            };
            run_once();
            const int reps = N >= 64 ? 3 : 8;
            const auto t0 = Clock::now();
            for (int r = 0; r < reps; ++r) run_once();
            const double ms = ms_since(t0) / reps;
            std::printf("  T%-2d %7.2f/%5.3f/%4.1f", T, ms, ms / N, N * 18.8006e6 / (ms * 1e-3) / 1e9);
        }
        std::printf("\n");
    }
    // ---- a decode layer as the engine would run it: 6 experts, summed ---------------------------
    std::printf("\n[one layer = 6 experts, each split over T threads]  ms/layer  (x40 = ms/token if ALL experts ran on CPU)\n");
    for (int T : {4, 6, 8, 12, 16}) {
        Pool pool(T); Scratch s; init_scratch(s);
        const int reps = 20;
        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) for (int e = 0; e < 6; ++e) { const int i = (r * 6 + e) % std::min(max_experts, 240); expert_split(pool, T, sc[i], wt[i], x.data(), out.data(), 0.25f, true, s); }
        const double a = ms_since(t0) / reps;
        std::printf("  T=%-2d %7.2f ms/layer -> %7.0f ms/token\n", T, a, a * 40);
    }
    std::printf("\nnote: attention, dense layers and the LM head are NOT included above.\n");
    return 0;
}
