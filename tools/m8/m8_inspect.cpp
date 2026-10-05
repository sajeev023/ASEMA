// ASEMA v0.2 — M8 inspect tool.
// Prints architecture summary + tensor inventory from a local HF directory.
// Run: m8-inspect <hf_root> [--json]

#include "../include/asema/cli.hpp"
#include "../include/asema/m8/m8_model_adapter.hpp"

#include <iostream>

int main(int argc, char** argv) {
    using namespace asema;

    cli::Parser parser({
        {"m8-inspect", "0.1.0",
         "Print DeepSeek-V4.1-Flash architecture summary and tensor inventory.",
         "<hf_root> [--json]"},
        {
            {"json",   "j", cli::ArgType::Flag, "",
             "Emit machine-readable JSON instead of Markdown."},
        }
    });

    // First positional = hf_root.
    std::vector<std::string> argv_vec;
    for (int i = 0; i < argc; ++i) argv_vec.emplace_back(argv[i]);
    auto parsed = parser.parse(argc, argv);
    if (parsed.help_requested)    { std::cout << parser.render_help();    return cli::RC_SUCCESS; }
    if (parsed.version_requested) { std::cout << parser.render_version(); return cli::RC_SUCCESS; }
    if (!parsed.ok()) { std::cerr << "m8-inspect: " << parsed.error << "\n"; return cli::RC_INVALID_ARGUMENT; }
    if (argv_vec.size() < 2) {
        std::cerr << "m8-inspect: missing <hf_root> argument\n";
        return cli::RC_INVALID_ARGUMENT;
    }
    std::string hf_root = argv_vec[1];

    auto adapter = m8::make_deepseek_v41_adapter();
    if (!adapter->load(hf_root)) {
        std::cerr << "m8-inspect: failed to load " << hf_root
                  << "/config.json or /model.safetensors.index.json\n";
        return cli::RC_MODEL_ERROR;
    }

    if (parsed.has("json")) {
        std::cout << adapter->summary_json() << "\n";
    } else {
        std::cout << adapter->report_markdown() << "\n";
    }
    return cli::RC_SUCCESS;
}
