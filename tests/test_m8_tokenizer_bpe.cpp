#include "asema/m8/m8_model_adapter.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  ASEMA POST-M100: BYTE-LEVEL BPE TOKENIZER VERIFICATION TEST        \n";
    std::cout << "======================================================================\n\n";

    asema::m8::Tokenizer tok;
    std::string tok_path = "examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json";
    bool loaded = tok.load(tok_path);
    if (!loaded) {
        tok_path = "../" + tok_path;
        loaded = tok.load(tok_path);
    }
    std::cout << "Loaded tokenizer: " << (loaded ? "PASS" : "FAIL") << "\n";
    std::cout << "Vocab size: " << tok.vocab_size() << "\n";
    assert(loaded);
    assert(tok.vocab_size() > 120000);

    // Test Round-Trip on critical prompts
    std::vector<std::string> test_prompts = {
        "Hello",
        "Hello world",
        "DeepSeek",
        "Explain what a neural network is in simple terms.",
        "What is artificial intelligence?",
        "Write a Python function to add two numbers.",
        "123456789",
        "punctuation: !?,.-_",
        "multiple   spaces",
        "line 1\nline 2",
        "<｜User｜>Explain what a neural network is in simple terms.<｜Assistant｜>"
    };

    for (const auto& prompt : test_prompts) {
        auto ids = tok.encode(prompt);
        std::string dec = tok.decode(ids);

        std::cout << "Original:  \"" << prompt << "\"\n";
        std::cout << "Token IDs: [ ";
        for (size_t i = 0; i < std::min<size_t>(ids.size(), 10); ++i) {
            std::cout << ids[i] << " ";
        }
        if (ids.size() > 10) std::cout << "... ";
        std::cout << "] (" << ids.size() << " tokens)\n";
        std::cout << "Decoded:   \"" << dec << "\"\n";

        // Must match exactly or with standard trim
        std::cout << "Match:     " << (prompt == dec ? "EXACT MATCH (PASS)" : "MISMATCH") << "\n\n";
        assert(prompt == dec);
    }

    std::cout << "ALL TOKENIZER TESTS PASSED SUCCESSFULLY.\n";
    return 0;
}
