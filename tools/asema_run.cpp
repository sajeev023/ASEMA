// ASEMA v0.1 - Agent #3: asema-run CLI
// -----------------------------------------------------------------------------
// Thin CLI wrapper over asema::ASEMARuntime. Operates in the synthetic MoE
// mode - does NOT perform real neural-network text generation. This is the
// supported v0.1 capability; no fake LLM behavior is exposed.
// -----------------------------------------------------------------------------

#include "../include/asema/cli.hpp"
#include "../include/asema/runtime.hpp"
#include "../include/asema/manifest.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

std::string derive_manifest_path(const std::string& model_path) {
    namespace fs = std::filesystem;
    fs::path p(model_path);
    if (p.extension() == ".json") return p.string();
    if (fs::is_directory(p))      return (p / "manifest.json").string();
    return p.string() + "/manifest.json";
}

} // namespace

int main(int argc, char** argv) {
    using namespace asema;

    cli::Parser parser({
        {"asema-run", "0.1.0",
         "Run the ASEMA synthetic-runtime pipeline against a model container.",
         "[options] --model <model.asema|model_dir>"},
        {
            {"model",       "m", cli::ArgType::Value, "",
             "Path to model.asema or the model directory containing manifest.json."},
            {"tokens",      "n", cli::ArgType::Value, "32",
             "Number of synthetic tokens to execute."},
            {"cache",       "c", cli::ArgType::Value, "8388608",
             "RAM cache size in bytes (0 = disabled)."},
            {"lookahead",   "k", cli::ArgType::Value, "2",
             "Prefetch lookahead K (0 = disabled)."},
            {"workers",     "w", cli::ArgType::Value, "4",
             "Async loader worker count."},
            {"seed",        "s", cli::ArgType::Value, "42",
             "Router seed (deterministic routing)."},
            {"telemetry",   "",  cli::ArgType::Value, "",
             "If set, write JSONL telemetry to this path."},
            {"report",      "r", cli::ArgType::Value, "",
             "If set, write aggregate report text file to this path."},
            {"report-json", "",  cli::ArgType::Value, "",
             "If set, write aggregate report JSON file to this path."},
            {"quiet",       "q", cli::ArgType::Flag, "",
             "Suppress non-error output."},
        }
    });

    auto args = parser.parse(argc, argv);
    if (args.help_requested)    { std::cout << parser.render_help();    return cli::RC_SUCCESS; }
    if (args.version_requested) { std::cout << parser.render_version(); return cli::RC_SUCCESS; }
    if (!args.ok()) {
        std::cerr << "asema-run: " << args.error << "\n";
        return cli::RC_INVALID_ARGUMENT;
    }

    if (args.get("model").empty()) {
        std::cerr << "asema-run: --model is required\n";
        return cli::RC_INVALID_ARGUMENT;
    }

    RuntimeConfig cfg;
    cfg.container_path = args.get("model");
    cfg.manifest_path = derive_manifest_path(cfg.container_path);
    cfg.num_tokens      = static_cast<uint32_t>(args.get_uint("tokens"));
    cfg.ram_cache_bytes = args.get_uint("cache");
    cfg.prefetch_lookahead = static_cast<size_t>(args.get_uint("lookahead"));
    cfg.loader_workers  = static_cast<size_t>(args.get_uint("workers"));
    cfg.router_seed     = args.get_uint("seed");
    cfg.telemetry_jsonl = args.get("telemetry");
    cfg.report_text_path = args.get("report");
    cfg.report_json_path = args.get("report-json");

    namespace fs = std::filesystem;
    if (fs::is_directory(cfg.container_path)) {
        cfg.container_path = (fs::path(cfg.container_path) / "model.asema").string();
    }

    {
        std::ifstream f(cfg.manifest_path);
        if (!f.is_open()) {
            std::cerr << "asema-run: cannot open manifest: " << cfg.manifest_path << "\n";
            return cli::RC_MODEL_ERROR;
        }
        std::stringstream buf;
        buf << f.rdbuf();
        try {
            ModelManifest m = ModelManifest::from_json(buf.str());
            if (m.num_layers == 0 || m.experts_per_layer == 0) {
                std::cerr << "asema-run: manifest has no layers/experts\n";
                return cli::RC_MODEL_ERROR;
            }
        } catch (const std::exception& e) {
            std::cerr << "asema-run: manifest parse failed: " << e.what() << "\n";
            return cli::RC_MODEL_ERROR;
        }
    }

    if (!fs::exists(cfg.container_path)) {
        std::cerr << "asema-run: container not found: " << cfg.container_path << "\n";
        return cli::RC_MODEL_ERROR;
    }

    try {
        ASEMARuntime rt(cfg);
        auto res = rt.run();
        if (!res.completed) {
            std::cerr << "asema-run: run did not complete: " << res.error << "\n";
            return cli::RC_RUNTIME_ERROR;
        }

        if (!args.has("quiet")) {
            std::cout << "[asema-run] tokens=" << res.total_tokens
                      << "  wall_ms=" << std::fixed << std::setprecision(3)
                      << res.total_wall_time_ms
                      << "  ms/token=" << std::setprecision(3)
                      << (res.total_wall_time_ms / std::max<size_t>(res.total_tokens, 1))
                      << "  first_token_ms=" << std::setprecision(3)
                      << res.first_token_latency_ms << "\n";
            if (rt.cache()) {
                auto s = rt.cache()->stats();
                std::cout << "[asema-run] cache: hits=" << s.hits
                          << " misses=" << s.misses
                          << " evictions=" << s.evictions
                          << " peak=" << s.peak_bytes_used << " B\n";
            }
            if (rt.prefetch()) {
                auto s = rt.prefetch()->stats();
                std::cout << "[asema-run] prefetch: requests=" << s.prefetch_requests
                          << " useful=" << s.useful_prefetches
                          << " wasted=" << s.wasted_prefetches << "\n";
            }
        }
        return cli::RC_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "asema-run: exception: " << e.what() << "\n";
        return cli::RC_RUNTIME_ERROR;
    }
}
