// Numerical parity test for the zero-copy FP8 GEMV against the legacy LUT dequantization path.
//  1. Every E4M3 code decodes identically to k_e4m3_lut (NaN codes 0x7F/0xFF excluded).
//  2. Random matrices match a double-precision LUT reference to FP32 rounding noise.
//  3. (If the checkpoint is present) real dense tensors contain no NaN codes and match the
//     LUT reference.
#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_fp8_gemv.hpp"
#include "asema/m8/m8_fp8_lut.hpp"
#include "asema/m8/m8_multi_volume.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <vector>

using namespace asema::m8;

static int g_failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++g_failures;                                 \
            std::printf("FAIL: " __VA_ARGS__);            \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

static double reference_row(const Fp8MatView& m, int r, const std::vector<float>& x) {
    double acc = 0.0;
    for (int c = 0; c < m.cols; ++c) {
        double w = k_e4m3_lut[m.weights[static_cast<size_t>(r) * m.cols + c]];
        double s = k_e8m0_lut[m.scales[(r / 32) * (m.cols / 32) + c / 32]];
        acc += w * s * x[c];
    }
    return acc;
}

static void test_all_codes_exact() {
    // 8 matrices of 32x32 cover every code; x = unit vector picks one column so out[r] == w*scale.
    for (int m = 0; m < 8; ++m) {
        std::vector<uint8_t> w(32 * 32), s(1, static_cast<uint8_t>(124 + m));
        for (int i = 0; i < 32 * 32; ++i) w[i] = static_cast<uint8_t>((m * 1024 + i) & 0xFF);
        Fp8MatView v{w.data(), s.data(), 32, 32};
        for (int col = 0; col < 32; ++col) {
            std::vector<float> x(32, 0.0f), out(32);
            x[col] = 1.0f;
            fp8_gemv(v, x.data(), out.data());
            for (int r = 0; r < 32; ++r) {
                uint8_t code = w[r * 32 + col];
                if (code == 0x7F || code == 0xFF) continue;
                float expect = k_e4m3_lut[code] * k_e8m0_lut[s[0]];
                CHECK(out[r] == expect || (out[r] == 0.0f && expect == 0.0f),
                      "code 0x%02X scale %d: got %g expected %g", code, s[0], out[r], expect);
            }
        }
    }
}

static void test_random_matrix() {
    std::mt19937 rng(12345);
    const int rows = 512, cols = 1280;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * cols), s((rows / 32) * (cols / 32));
    for (auto& b : w) {
        do { b = static_cast<uint8_t>(rng()); } while (b == 0x7F || b == 0xFF);
    }
    for (auto& b : s) b = static_cast<uint8_t>(115 + rng() % 10);
    std::vector<float> x(cols);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng);
    Fp8MatView m{w.data(), s.data(), rows, cols};
    std::vector<float> out(rows);
    fp8_gemv(m, x.data(), out.data());
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
        double ref = reference_row(m, r, x);
        double scale = 0.0;
        for (int c = 0; c < cols; ++c) {
            scale += std::fabs(static_cast<double>(k_e4m3_lut[w[static_cast<size_t>(r) * cols + c]]) *
                               k_e8m0_lut[s[(r / 32) * (cols / 32) + c / 32]] * x[c]);
        }
        worst = std::max(worst, std::fabs(out[r] - ref) / (scale + 1e-30));
    }
    std::printf("random 512x1280: worst |err| / sum|terms| = %.3e\n", worst);
    CHECK(worst < 1e-5, "random matrix error too large: %g", worst);
}

static void test_real_tensors() {
    const std::string root = asema::m8::paths::primary_shards();
    if (!std::filesystem::exists(root)) {
        std::printf("real checkpoint not found, skipping tensor scan\n");
        return;
    }
    M8MultiVolumeManager vm;
    vm.register_volume(root);
    const int layers[] = {0, 19, 39};
    const char* names[] = {"attn.wq_a", "attn.wkv", "ffn.shared_experts.w1", "ffn.shared_experts.w2"};
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int l : layers) {
        for (const char* n : names) {
            std::string base = "layers." + std::to_string(l) + "." + n;
            uint64_t wb = 0, sb = 0;
            const uint8_t* w = vm.safetensors_index().map_tensor(base + ".weight", &wb);
            const uint8_t* s = vm.safetensors_index().map_tensor(base + ".scale", &sb);
            if (!w || !s) {
                std::printf("  %s: not mapped, skipping\n", base.c_str());
                continue;
            }
            const auto* meta = vm.safetensors_index().get_tensor_meta(base + ".weight");
            const int rows = static_cast<int>(meta->shape[0]);
            const int cols = static_cast<int>(meta->shape[1]);
            size_t nan_codes = 0;
            for (uint64_t i = 0; i < wb; ++i) nan_codes += (w[i] == 0x7F || w[i] == 0xFF);
            std::vector<float> x(cols), out(rows);
            for (auto& v : x) v = nd(rng);
            Fp8MatView m{w, s, rows, cols};
            fp8_gemv(m, x.data(), out.data());
            double worst = 0.0;
            for (int r = 0; r < rows; r += 7) {
                double ref = reference_row(m, r, x);
                double mag = 0.0;
                for (int c = 0; c < cols; ++c) {
                    mag += std::fabs(static_cast<double>(k_e4m3_lut[w[static_cast<size_t>(r) * cols + c]]) *
                                     k_e8m0_lut[s[(r / 32) * (cols / 32) + c / 32]] * x[c]);
                }
                worst = std::max(worst, std::fabs(out[r] - ref) / (mag + 1e-30));
            }
            std::printf("  %-40s %5dx%-5d nan_codes=%zu worst_rel=%.3e\n", base.c_str(), rows, cols, nan_codes, worst);
            CHECK(nan_codes == 0, "%s contains E4M3 NaN codes", base.c_str());
            CHECK(worst < 1e-5, "%s parity error %g", base.c_str(), worst);
        }
    }
}

int main() {
    test_all_codes_exact();
    test_random_matrix();
    test_real_tensors();
    std::printf(g_failures == 0 ? "PASS\n" : "FAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
