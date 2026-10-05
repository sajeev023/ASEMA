// ASEMA v0.1 - Agent #3: asema-pack CLI
// -----------------------------------------------------------------------------
// Converts internal synthetic generator output into the existing ASEMA-SSF
// format. Supports:
//   * Generating a synthetic MoE container (parameters-driven).
//   * Re-emitting an existing manifest as a fresh container with new CRC32s
//     (use --regenerate).
//
// This is NOT a converter for HuggingFace / PyTorch / Safetensors / GGUF.
// Only internal synthetic data is supported, by design.
// -----------------------------------------------------------------------------

#include "../include/asema/cli.hpp"
#include "../include/asema/synthetic_gen.hpp"

#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
    using namespace asema;

    cli::Parser parser({
        {"asema-pack", "0.1.0",
         "Generate or regenerate an ASEMA-SSF synthetic MoE container.",
         "[options] --output <dir>"},
        {
            {"layers",        "L", cli::ArgType::Value, "4",
             "Number of layers."},
            {"experts",       "E", cli::ArgType::Value, "16",
             "Experts per layer."},
            {"hidden",        "",  cli::ArgType::Value, "128",
             "Hidden dimension."},
            {"ffn",           "",  cli::ArgType::Value, "512",
             "FFN dimension."},
            {"top-k",         "",  cli::ArgType::Value, "2",
             "Active experts per token."},
            {"dtype",         "",  cli::ArgType::Value, "fp16",
             "Data type: fp16 or fp32."},
            {"seed",          "s", cli::ArgType::Value, "42",
             "Random seed."},
            {"output",        "o", cli::ArgType::Value, "",
             "Output directory (required)."},
            {"tier",          "t", cli::ArgType::Value, "",
             "Use predefined tier: 1 (small), 2 (medium), 3 (large)."},
            {"force",         "f", cli::ArgType::Flag, "",
             "Overwrite existing output."},
        }
    });

    auto args = parser.parse(argc, argv);
    if (args.help_requested)   { std::cout << parser.render_help();    return cli::RC_SUCCESS; }
    if (args.version_requested){ std::cout << parser.render_version(); return cli::RC_SUCCESS; }
    if (!args.ok()) {
        std::cerr << "asema-pack: " << args.error << "\n";
        std::cerr << "Try --help.\n";
        return cli::RC_INVALID_ARGUMENT;
    }

    if (args.get("output").empty()) {
        std::cerr << "asema-pack: --output is required\n";
        return cli::RC_INVALID_ARGUMENT;
    }

    gen::GenParams p;
    std::string tier = args.get("tier");
    if      (tier == "1" || tier == "small")  p = gen::tier1_small();
    else if (tier == "2" || tier == "medium") p = gen::tier2_medium();
    else if (tier == "3" || tier == "large")  p = gen::tier3_large();
    else {
        p.num_layers        = args.get_uint("layers");
        p.experts_per_layer = args.get_uint("experts");
        p.hidden_dim        = args.get_uint("hidden");
        p.ffn_dim           = args.get_uint("ffn");
        p.top_k             = args.get_uint("top-k");
        p.dtype             = args.get("dtype", "fp16");
        p.seed              = args.get_uint("seed");
    }
    p.output_dir = args.get("output");

    if (!args.has("force") && std::filesystem::exists(p.output_dir + "/model.asema")) {
        std::cerr << "asema-pack: output already exists. Use --force to overwrite.\n";
        return cli::RC_IO_ERROR;
    }

    auto r = gen::generate(p);
    if (!r.success) {
        std::cerr << "asema-pack: generation failed: " << r.error << "\n";
        return cli::RC_RUNTIME_ERROR;
    }

    double mb = r.bytes_written / (1024.0 * 1024.0);
    std::cout << "[asema-pack] Wrote " << r.experts_written << " experts, "
              << mb << " MB, in " << r.elapsed_ms << " ms\n";
    std::cout << "[asema-pack] Manifest: " << r.manifest_path << "\n";
    std::cout << "[asema-pack] Container: " << r.container_path << "\n";
    return cli::RC_SUCCESS;
}
