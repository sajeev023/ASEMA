#include "asema/m8/m8_reference_engine.hpp"
#include "asema/m8/m8_expert_kernel.hpp"
#include "asema/m8/m8_router.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA M8: INDEPENDENT CPU REFERENCE ENGINE VERIFICATION (SECTION 19) \n";
    std::cout << "======================================================================\n\n";

    // 1. RMSNorm Reference Verification
    std::cout << "[1/4] Verifying RMSNorm Scalar Reference...\n";
    const int dim = 5120;
    std::vector<float> x(dim);
    std::vector<float> w(dim, 1.5f);
    for (int i = 0; i < dim; ++i) {
        x[i] = std::sin(static_cast<float>(i) * 0.05f) * 0.2f;
    }

    std::vector<float> ref_norm(dim);
    asema::m8::M8ReferenceEngine::rms_norm(x.data(), w.data(), ref_norm.data(), dim);

    double sum_sq = 0.0;
    for (float v : x) sum_sq += static_cast<double>(v) * v;
    double expected_rms = std::sqrt(sum_sq / dim + 1e-20);
    float expected_first = static_cast<float>((x[0] / expected_rms) * w[0]);
    float expected_last = static_cast<float>((x[dim - 1] / expected_rms) * w[dim - 1]);

    assert(std::abs(ref_norm[0] - expected_first) < 1e-5f);
    assert(std::abs(ref_norm[dim - 1] - expected_last) < 1e-5f);
    std::cout << "  RMSNorm Reference exact match: PASS\n";

    // 2. SwiGLU Reference Verification
    std::cout << "\n[2/4] Verifying SwiGLU Reference with swiglu_limit = 10.0...\n";
    const int inter_dim = 2304;
    std::vector<float> gate(inter_dim);
    std::vector<float> up(inter_dim);
    std::vector<float> act(inter_dim);
    for (int i = 0; i < inter_dim; ++i) {
        gate[i] = (i % 2 == 0) ? 15.0f : -15.0f; // test clamping
        up[i] = (i % 2 == 0) ? 20.0f : -20.0f;   // test clamping
    }
    asema::m8::M8ReferenceEngine::swiglu(gate.data(), up.data(), act.data(), inter_dim, 10.0f);

    // Gate clamped to 10.0: silu(10) = 10 / (1 + exp(-10)) ~ 9.999546
    // Up clamped to 10.0: act = silu(10) * 10 ~ 99.99546
    float silu_10 = 10.0f / (1.0f + std::exp(-10.0f));
    float expected_even = silu_10 * 10.0f;
    assert(std::abs(act[0] - expected_even) < 1e-4f);
    std::cout << "  SwiGLU Clamped Reference exact match: PASS\n";

    // 3. Router Reference Verification
    std::cout << "\n[3/4] Verifying Router Reference (sqrtsoftplus + top-6)...\n";
    const int num_experts = 384;
    std::vector<float> gate_w(num_experts * dim, 0.001f);
    std::vector<float> gate_b(num_experts, 0.0f);
    // bias expert 42 and 99 higher
    gate_b[42] = 5.0f;
    gate_b[99] = 4.5f;

    auto res = asema::m8::M8ReferenceEngine::route(x.data(), gate_w.data(), gate_b.data(),
                                                   num_experts, dim, 6, 1.5f);
    assert(res.indices.size() == 6);
    assert(res.weights.size() == 6);
    assert(res.indices[0] == 42);
    assert(res.indices[1] == 99);
    float w_sum = 0.0f;
    for (float weight : res.weights) w_sum += weight;
    assert(std::abs(w_sum - 1.5f) < 1e-4f);
    std::cout << "  Router Reference Top-6 Selection & Scaling: PASS\n";

    // 4. Matrix-Vector Multiply Reference
    std::cout << "\n[4/4] Verifying Scalar MatVec Reference...\n";
    std::vector<float> A(64 * 128, 0.5f);
    std::vector<float> vec(128, 2.0f);
    std::vector<float> out(64, 0.0f);
    asema::m8::M8ReferenceEngine::matvec(A.data(), vec.data(), out.data(), 64, 128);
    for (int i = 0; i < 64; ++i) {
        assert(std::abs(out[i] - 128.0f) < 1e-5f);
    }
    std::cout << "  Scalar MatVec Reference exact match: PASS\n";

    std::cout << "\n======================================================================\n";
    std::cout << "  ALL INDEPENDENT REFERENCE ENGINE CHECKS PASSED                      \n";
    std::cout << "======================================================================\n";
    return 0;
}
