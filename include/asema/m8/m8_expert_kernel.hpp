#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace asema {
namespace m8 {

struct ExpertDimensions {
    int hidden_dim{5120};
    int intermediate_dim{2304};
    float swiglu_limit{10.0f};

    static constexpr size_t W1_SCALE_BYTES = 2304 * 160;     // 368,640 bytes
    static constexpr size_t W2_SCALE_BYTES = 5120 * 72;      // 368,640 bytes
    static constexpr size_t W3_SCALE_BYTES = 2304 * 160;     // 368,640 bytes
    static constexpr size_t TOTAL_SCALE_BYTES = W1_SCALE_BYTES + W2_SCALE_BYTES + W3_SCALE_BYTES; // 1,105,920

    static constexpr size_t W1_WEIGHT_BYTES = 2304 * 2560;   // 5,898,240 bytes
    static constexpr size_t W2_WEIGHT_BYTES = 5120 * 1152;   // 5,898,240 bytes
    static constexpr size_t W3_WEIGHT_BYTES = 2304 * 2560;   // 5,898,240 bytes
    static constexpr size_t TOTAL_WEIGHT_BYTES = W1_WEIGHT_BYTES + W2_WEIGHT_BYTES + W3_WEIGHT_BYTES; // 17,694,720

    static constexpr size_t TOTAL_EXPERT_BYTES = TOTAL_SCALE_BYTES + TOTAL_WEIGHT_BYTES; // 18,800,640 bytes
};

class M8ExpertKernel {
public:
    explicit M8ExpertKernel(ExpertDimensions dims = ExpertDimensions{});

    // Attach to contiguous memory buffer holding [scales (1.1MB) | weights (17.7MB)]
    // Zero-copy: operates directly on the paged memory buffer
    bool attach(const uint8_t* scales_ptr, const uint8_t* weights_ptr);

    // Convenience loader from files
    bool load_from_files(const std::string& scales_file, const std::string& weights_file);

    // Compute SwiGLU forward pass:
    // x: [hidden_dim]
    // out: [hidden_dim]
    // router_weight: scaling factor from router (e.g. 0.25)
    // accumulate: if true, adds (out * router_weight) to out; if false, overwrites out
    void forward(const float* x, float* out, float router_weight = 1.0f, bool accumulate = false) const;

    bool is_attached() const { return attached_; }
    const ExpertDimensions& dims() const { return dims_; }

private:
    ExpertDimensions dims_;
    bool attached_{false};

    const uint8_t* w1_scale_{nullptr};
    const uint8_t* w2_scale_{nullptr};
    const uint8_t* w3_scale_{nullptr};

    const uint8_t* w1_weight_{nullptr};
    const uint8_t* w2_weight_{nullptr};
    const uint8_t* w3_weight_{nullptr};

    // Internal buffer if loaded from files
    std::vector<uint8_t> internal_scales_;
    std::vector<uint8_t> internal_weights_;

    // Matrix-vector multiplication for packed 4-bit weights with block scales
    void gemv_packed4(int rows, int cols,
                      const uint8_t* weights, const uint8_t* scales,
                      const float* x, float* out) const;
};

} // namespace m8
} // namespace asema
