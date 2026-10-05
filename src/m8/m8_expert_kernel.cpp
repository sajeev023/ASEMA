#include "asema/m8/m8_expert_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <vector>

namespace asema {
namespace m8 {

namespace {

struct NibblePair {
    float low;
    float high;
};

// Static lookup tables initialized once
class PrecomputedTables {
public:
    NibblePair nibble_lut[256];
    float scale_lut[256];

    PrecomputedTables() {
        // Exact IEEE FP4 E2M1 representation (torch.float4_e2m1fn):
        // Sign bit is bit 3 (val >= 8 is negative).
        // Positive: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        // Negative: [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
        static const float k_fp4_e2m1[16] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };

        for (int i = 0; i < 256; ++i) {
            int low = i & 0x0F;
            int high = (i >> 4) & 0x0F;
            nibble_lut[i].low = k_fp4_e2m1[low];
            nibble_lut[i].high = k_fp4_e2m1[high];

            // scale: 2^(val - 127)
            scale_lut[i] = std::ldexp(1.0f, i - 127);
        }
    }
};

const PrecomputedTables& get_tables() {
    static const PrecomputedTables tables;
    return tables;
}

} // namespace

M8ExpertKernel::M8ExpertKernel(ExpertDimensions dims)
    : dims_(dims) {}

bool M8ExpertKernel::attach(const uint8_t* scales_ptr, const uint8_t* weights_ptr) {
    if (!scales_ptr || !weights_ptr) {
        return false;
    }

    w1_scale_ = scales_ptr;
    w2_scale_ = w1_scale_ + ExpertDimensions::W1_SCALE_BYTES;
    w3_scale_ = w2_scale_ + ExpertDimensions::W2_SCALE_BYTES;

    w1_weight_ = weights_ptr;
    w2_weight_ = w1_weight_ + ExpertDimensions::W1_WEIGHT_BYTES;
    w3_weight_ = w2_weight_ + ExpertDimensions::W2_WEIGHT_BYTES;

    attached_ = true;
    return true;
}

bool M8ExpertKernel::load_from_files(const std::string& scales_file, const std::string& weights_file) {
    std::ifstream fs(scales_file, std::ios::binary);
    if (!fs.is_open()) {
        return false;
    }
    internal_scales_.resize(ExpertDimensions::TOTAL_SCALE_BYTES);
    fs.read(reinterpret_cast<char*>(internal_scales_.data()), ExpertDimensions::TOTAL_SCALE_BYTES);
    if (!fs) {
        return false;
    }

    std::ifstream fw(weights_file, std::ios::binary);
    if (!fw.is_open()) {
        return false;
    }
    internal_weights_.resize(ExpertDimensions::TOTAL_WEIGHT_BYTES);
    fw.read(reinterpret_cast<char*>(internal_weights_.data()), ExpertDimensions::TOTAL_WEIGHT_BYTES);
    if (!fw) {
        return false;
    }

    return attach(internal_scales_.data(), internal_weights_.data());
}

void M8ExpertKernel::gemv_packed4(int rows, int cols,
                                  const uint8_t* weights, const uint8_t* scales,
                                  const float* x, float* out) const {
    const auto& tables = get_tables();
    const int num_blocks = cols / 32;
    const int row_weight_stride = cols / 2;

    for (int r = 0; r < rows; ++r) {
        const uint8_t* row_w = weights + r * row_weight_stride;
        const uint8_t* row_s = scales + r * num_blocks;

        float row_sum = 0.0f;

        for (int b = 0; b < num_blocks; ++b) {
            float scale = tables.scale_lut[row_s[b]];
            const uint8_t* blk_w = row_w + b * 16;
            const float* blk_x = x + b * 32;

            float blk_dot = 0.0f;
            for (int k = 0; k < 16; ++k) {
                const auto& p = tables.nibble_lut[blk_w[k]];
                blk_dot += p.low * blk_x[2 * k] + p.high * blk_x[2 * k + 1];
            }
            row_sum += scale * blk_dot;
        }

        out[r] = row_sum;
    }
}

void M8ExpertKernel::forward(const float* x, float* out, float router_weight, bool accumulate) const {
    if (!attached_ || !x || !out) {
        return;
    }

    const int H = dims_.hidden_dim;
    const int I = dims_.intermediate_dim;

    // Scratch buffers for intermediate vectors
    std::vector<float> gate(I);
    std::vector<float> up(I);
    std::vector<float> act(I);
    std::vector<float> raw_out(H);

    // 1. gate = w1(x) [2304]
    gemv_packed4(I, H, w1_weight_, w1_scale_, x, gate.data());

    // 2. up = w3(x) [2304]
    gemv_packed4(I, H, w3_weight_, w3_scale_, x, up.data());

    // 3. SwiGLU activation with clamp
    for (int i = 0; i < I; ++i) {
        float g = std::min(gate[i], dims_.swiglu_limit);
        float u = std::clamp(up[i], -dims_.swiglu_limit, dims_.swiglu_limit);
        float silu = g / (1.0f + std::exp(-g));
        act[i] = silu * u;
    }

    // 4. out = w2(act) [5120]
    gemv_packed4(H, I, w2_weight_, w2_scale_, act.data(), raw_out.data());

    // 5. Output accumulation / scaling
    if (accumulate) {
        for (int i = 0; i < H; ++i) {
            out[i] += raw_out[i] * router_weight;
        }
    } else {
        for (int i = 0; i < H; ++i) {
            out[i] = raw_out[i] * router_weight;
        }
    }
}

} // namespace m8
} // namespace asema
