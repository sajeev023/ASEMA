// Debug program: print summary_json output to file.
#include "asema/m8/m8_paths.hpp"
#include "../include/asema/m8/m8_model_adapter.hpp"
#include <fstream>
#include <iostream>

int main() {
    auto adapter = asema::m8::make_deepseek_v41_adapter();
    const std::string hf = asema::m8::paths::hf_dir();
    if (!adapter->load(hf)) { std::cerr << "load failed\n"; return 1; }
    std::string s = adapter->summary_json();
    std::ofstream out("m8_summary_debug.json");
    out << s;
    out.close();
    std::cout << "wrote m8_summary_debug.json (" << s.size() << " bytes)\n";
    // Print the first 400 bytes for visual inspection.
    std::cout << "FIRST 400 BYTES:\n" << s.substr(0, 400) << "\n---\n";
    return 0;
}
