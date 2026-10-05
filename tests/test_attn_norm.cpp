#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_mla_attention.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    asema::m8::MLAParams params;
    asema::m8::M8MLAAttention attn(params);

    std::vector<float> x(params.dim);
    FILE* fp = fopen((asema::m8::paths::primary_shards() + "/model-00002-of-00048.safetensors").c_str(), "rb");
    if (fp) {
        _fseeki64(fp, 352 + 65106ULL * 5120 * 2, SEEK_SET);
        std::vector<uint16_t> b16(params.dim);
        fread(b16.data(), 2, params.dim, fp);
        fclose(fp);
        for (int i = 0; i < params.dim; ++i) {
            uint32_t u = static_cast<uint32_t>(b16[i]) << 16;
            x[i] = *reinterpret_cast<float*>(&u);
        }
    }

    std::vector<float> out(params.dim, 0.0f);
    attn.forward(x.data(), out.data(), 0);

    double sum = 0.0, l2 = 0.0;
    for (float v : out) { sum += v; l2 += v * v; }
    std::cout << "Real Attn Out L2: " << std::sqrt(l2) << ", Sum: " << sum << "\n";
    return 0;
}
