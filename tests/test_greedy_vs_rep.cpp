#include "asema/m8/m8_model_runner.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <unordered_map>

void run_test(asema::m8::M8ModelRunner& runner, const std::string& prompt, bool use_rep_penalty) {
    runner.reset_state();
    auto prompt_ids = runner.tokenizer().encode(prompt);
    std::cout << "======================================================================\n";
    std::cout << "PROMPT: \"" << prompt << "\" (" << (use_rep_penalty ? "WITH REP PENALTY" : "PURE GREEDY ARGMAX") << ")\n";
    std::cout << "Output: ";

    std::vector<int> generated_ids = prompt_ids;
    std::vector<float> logits;
    std::vector<asema::m8::LayerTelemetry> layer_tels;

    // Prefill
    for (size_t p = 0; p + 1 < prompt_ids.size(); ++p) {
        runner.step(prompt_ids[p], static_cast<int>(p), logits, layer_tels, false);
    }
    int last_p = static_cast<int>(prompt_ids.size()) - 1;
    runner.step(prompt_ids[last_p], last_p, logits, layer_tels, true);

    int current_pos = static_cast<int>(prompt_ids.size());

    for (int step = 0; step < 16; ++step) {
        if (use_rep_penalty) {
            // Apply penalty ONLY on generated tokens (not prompt tokens!)
            std::unordered_map<int, int> gen_counts;
            for (size_t r = prompt_ids.size(); r < generated_ids.size(); ++r) {
                gen_counts[generated_ids[r]]++;
            }
            for (auto& pair : gen_counts) {
                int tok = pair.first;
                if (tok >= 0 && tok < (int)logits.size()) {
                    if (logits[tok] > 0) logits[tok] /= (1.0f + 0.35f * pair.second);
                    else logits[tok] *= (1.0f + 0.35f * pair.second);
                }
            }
        }

        // Pure argmax
        int next_tok = 0;
        float max_l = -1e9f;
        for (size_t v = 0; v < logits.size(); ++v) {
            if (logits[v] > max_l) {
                max_l = logits[v];
                next_tok = static_cast<int>(v);
            }
        }

        if (next_tok == runner.tokenizer().eos_id() || next_tok == 1 || next_tok == 129279) break;

        std::string piece = runner.tokenizer().decode({next_tok});
        std::cout << piece << std::flush;
        generated_ids.push_back(next_tok);

        // Next step
        runner.step(next_tok, current_pos++, logits, layer_tels, true);
    }
    std::cout << "\n";
}

int main() {
    asema::m8::M8ModelRunner runner;
    runner.set_gpu_acceleration(true);
    runner.set_gpu_mla_acceleration(false);
    if (!runner.init("examples/real_model/DeepSeek-V4.1-Flash/hf")) {
        std::cerr << "Init failed\n";
        return 1;
    }

    std::string prompt1 = "Explain what a neural network is in simple terms.";
    run_test(runner, prompt1, false);
    run_test(runner, prompt1, true);

    std::string prompt2 = "A neural network is";
    run_test(runner, prompt2, false);
    run_test(runner, prompt2, true);

    return 0;
}
