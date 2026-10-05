#include "asema/m8/m8_model_runner.hpp"
#include "asema/m8/m8_model_adapter.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA FORENSIC TRACE & DIAGNOSTICS                                  \n";
    std::cout << "======================================================================\n\n";

    asema::m8::M8ModelRunner runner;
    std::string hf_root = "examples/real_model/DeepSeek-V4.1-Flash/hf";
    if (!runner.init(hf_root)) {
        std::cerr << "FAIL: Could not init runner.\n";
        return 1;
    }

    std::cout << "Real Embeddings Mapped: " << (runner.has_real_embeddings() ? "YES" : "NO") << "\n";
    std::cout << "Real LM Head Mapped:    " << (runner.has_real_lm_head() ? "YES" : "NO") << "\n\n";

    // 1. Trace embedding for token 65106 ("Explain")
    std::vector<float> emb(5120);
    if (runner.lookup_token_embedding(65106, emb.data())) {
        double sum = 0.0, l2 = 0.0;
        for (float v : emb) { sum += v; l2 += v * v; }
        l2 = std::sqrt(l2);
        std::cout << "[TRACE EMBEDDING 65106 ('Explain')]\n";
        std::cout << "  L2 Norm:  " << l2 << "\n";
        std::cout << "  Sum:      " << sum << "\n";
        std::cout << "  First 16: ";
        for (int i = 0; i < 16; ++i) std::cout << emb[i] << " ";
        std::cout << "\n\n";
    }

    // 2. Trace full prompt sequence: "Explain what a neural network is in simple terms."
    std::string prompt = "Explain what a neural network is in simple terms.";
    auto prompt_ids = runner.tokenizer().encode(prompt);
    std::cout << "[ENCODED PROMPT]\n  \"" << prompt << "\"\n  Tokens (" << prompt_ids.size() << "): [ ";
    for (int id : prompt_ids) std::cout << id << " ";
    std::cout << "]\n\n";

    runner.reset_state();
    std::vector<float> logits;
    std::vector<asema::m8::LayerTelemetry> layer_tels;

    for (size_t p = 0; p < prompt_ids.size(); ++p) {
        bool is_last = (p + 1 == prompt_ids.size());
        runner.step(prompt_ids[p], static_cast<int>(p), logits, layer_tels, is_last);
    }

    std::cout << "[TRACE LOGITS AFTER FULL PROMPT (Pos " << (prompt_ids.size() - 1) << ")]\n";
    std::cout << "  Total Logits: " << logits.size() << "\n";

    // Find top-20 logits
    std::vector<std::pair<float, int>> top;
    for (size_t v = 0; v < logits.size(); ++v) {
        top.push_back({logits[v], static_cast<int>(v)});
    }
    std::partial_sort(top.begin(), top.begin() + 20, top.end(), std::greater<std::pair<float, int>>());

    std::cout << "  Top-20 Tokens:\n";
    for (int i = 0; i < 20; ++i) {
        int tid = top[i].second;
        float logit = top[i].first;
        std::string piece = runner.tokenizer().decode({tid});
        std::cout << "    #" << std::setw(2) << (i + 1)
                  << " | ID: " << std::setw(6) << tid
                  << " | Logit: " << std::fixed << std::setprecision(4) << std::setw(8) << logit
                  << " | Piece: \"" << piece << "\"\n";
    }

    return 0;
}
