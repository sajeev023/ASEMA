// ASEMA v0.1 - Agent #3: asema-bench CLI
// -----------------------------------------------------------------------------
// M7 experiment runner. Iterates over an experiment matrix, calls
// ASEMARuntime for each cell x iteration, and saves raw + summarized data.
//
// Outputs (under --output/<timestamp>/):
//   environment.json
//   configuration.json
//   raw/<experiment_id>_iter<n>.json
//   raw/all_measurements.csv
//   summaries/<experiment_id>.json
//   summaries/all_summaries.txt
// -----------------------------------------------------------------------------

#include "../include/asema/cli.hpp"
#include "../include/asema/experiments.hpp"
#include "../include/asema/runtime.hpp"
#include "../include/asema/manifest.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;
using namespace asema;

static exp::WorkloadConfig resolve_workload(const std::string& model_path) {
    exp::WorkloadConfig w;
    std::string manifest_path = model_path;
    if (fs::is_directory(model_path)) {
        manifest_path = (fs::path(model_path) / "manifest.json").string();
    } else if (fs::path(model_path).extension() != ".json") {
        manifest_path = (fs::path(model_path).parent_path() / "manifest.json").string();
    }
    w.manifest_path = manifest_path;
    if (fs::is_directory(model_path)) {
        w.container_path = (fs::path(model_path) / "model.asema").string();
    } else if (fs::path(model_path).extension() == ".asema") {
        w.container_path = model_path;
    } else {
        w.container_path = model_path;
    }

    std::ifstream f(manifest_path);
    if (f.is_open()) {
        std::stringstream buf;
        buf << f.rdbuf();
        try {
            auto m = ModelManifest::from_json(buf.str());
            w.num_layers = m.num_layers;
            w.experts_per_layer = m.experts_per_layer;
            w.top_k = m.active_experts_per_token;
            w.hidden_dim = m.hidden_dim;
            w.ffn_dim = m.ffn_dim;
            w.dtype = m.layers.empty() || m.layers[0].experts.empty() ? "fp16"
                       : m.layers[0].experts[0].dtype;
            w.name = m.model_name;
            w.total_storage_bytes = m.total_storage_bytes;
        } catch (...) {
            // Leave as zero.
        }
    }
    return w;
}

static std::vector<exp::ExperimentCell> build_matrix(const cli::ParsedArgs& args) {
    std::vector<exp::ExperimentCell> cells;

    std::vector<uint64_t> caches;
    if (args.has("cache-sweep")) {
        caches = {1ull << 20, 2ull << 20, 4ull << 20, 8ull << 20, 12ull << 20};
    } else {
        caches = {args.get_uint("cache", 8ull << 20)};
    }

    std::vector<size_t> Ks;
    if (args.has("lookahead-sweep")) {
        Ks = {0, 1, 2, 3, 4, 6};
    } else {
        Ks = {static_cast<size_t>(args.get_uint("lookahead", 2))};
    }

    std::vector<uint64_t> workers_list;
    if (args.has("workers-sweep")) {
        workers_list = {1, 2, 4, 8};
    } else {
        workers_list = {args.get_uint("workers", 4)};
    }

    std::vector<uint64_t> seeds;
    if (args.has("seed-sweep")) {
        seeds = {11, 42, 99};
    } else {
        seeds = {args.get_uint("seed", 42)};
    }

    uint32_t tokens = static_cast<uint32_t>(args.get_uint("tokens", 32));
    double locality = 0.5; // medium is default
    std::string pattern = args.get("pattern", "locality");
    if      (args.has("locality-low"))   { locality = 0.3; pattern = "low_locality"; }
    else if (args.has("locality-high"))  { locality = 0.9; pattern = "high_locality"; }
    else if (args.has("locality-medium")){ locality = 0.5; pattern = "medium_locality"; }
    else if (args.has("locality-random")){ locality = 0.0; pattern = "random"; }
    else if (args.has("locality-adversarial")){ locality = 0.0; pattern = "adversarial"; }

    for (auto cb : caches) {
        for (auto K : Ks) {
            for (auto w : workers_list) {
                for (auto s : seeds) {
                    exp::ExperimentCell c;
                    c.ram_cache_bytes = cb;
                    c.prefetch_lookahead = K;
                    c.loader_workers = static_cast<uint64_t>(w);
                    c.router_seed = s;
                    c.num_tokens = tokens;
                    c.locality = locality;
                    c.pattern = pattern;
                    c.cold_start = !args.has("warm");
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "cache_%lluMB_K%zu_w%llu_s%llu",
                                  (unsigned long long)(cb / (1024 * 1024)),
                                  K,
                                  (unsigned long long)w,
                                  (unsigned long long)s);
                    c.id = buf;
                    cells.push_back(c);
                }
            }
        }
    }

    return cells;
}

static exp::Measurement run_one(const exp::WorkloadConfig& w,
                                 const exp::ExperimentCell& cell,
                                 int iter) {
    exp::Measurement m;
    m.experiment_id = cell.id;
    m.iteration = iter;

    RuntimeConfig cfg;
    cfg.container_path = w.container_path;
    cfg.manifest_path = w.manifest_path;
    cfg.num_layers = w.num_layers;
    cfg.experts_per_layer = w.experts_per_layer;
    cfg.top_k = w.top_k;
    cfg.ram_cache_bytes = cell.ram_cache_bytes;
    cfg.prefetch_lookahead = cell.prefetch_lookahead;
    cfg.loader_workers = static_cast<size_t>(cell.loader_workers);
    cfg.router_seed = cell.router_seed;
    cfg.num_tokens = cell.num_tokens;

    try {
        ASEMARuntime rt(cfg);
        auto res = rt.run();
        if (!res.completed) {
            m.valid = false;
            m.invalid_reason = "runtime_did_not_complete: " + res.error;
            return m;
        }
        m.tokens_completed = res.total_tokens;
        m.wall_time_ms = res.total_wall_time_ms;
        m.first_token_ms = res.first_token_latency_ms;
        m.ms_per_token = res.total_tokens == 0 ? 0.0
            : res.total_wall_time_ms / static_cast<double>(res.total_tokens);
        m.tokens_per_sec = m.ms_per_token > 0.0 ? 1000.0 / m.ms_per_token : 0.0;

        uint64_t expert_bytes = 0;
        if (rt.cache()) {
            auto s = rt.cache()->stats();
            m.cache_hits = s.hits;
            m.cache_misses = s.misses;
            uint64_t total = m.cache_hits + m.cache_misses;
            m.cache_hit_rate = total ? double(m.cache_hits) / total : 0.0;
            m.cache_evictions = s.evictions;
            m.cache_insertions = s.insertions;
            m.peak_bytes_used = s.peak_bytes_used;
        }

        // Pull loader counters BEFORE the runtime's destructor touches them.
        uint64_t loader_submitted = 0, loader_coalesced = 0;
        uint64_t loader_ios = 0, loader_bytes = 0;
        if (rt.loader()) {
            loader_submitted = rt.loader()->total_requests_submitted();
            loader_coalesced = rt.loader()->total_coalesced_requests();
            loader_ios       = rt.loader()->total_io_dispatches();
            loader_bytes     = rt.loader()->total_bytes_loaded();
        }
        m.runtime_logical_requests = loader_submitted;
        m.coalesced_requests       = loader_coalesced;
        m.ssd_reads                = loader_ios;
        m.ssd_bytes_physical       = loader_bytes;
        // legacy alias (== physical)
        m.ssd_bytes_logical        = loader_bytes;

        // Infer expert bytes from a known-miss read.
        if (m.cache_misses > 0 && loader_bytes > 0) {
            expert_bytes = loader_bytes / m.cache_misses;
        } else if (w.total_storage_bytes > 0 &&
                   w.num_layers > 0 && w.experts_per_layer > 0) {
            expert_bytes = w.total_storage_bytes /
                           (uint64_t(w.num_layers) * w.experts_per_layer);
        }
        m.expert_bytes_per_request = expert_bytes;
        m.runtime_logical_bytes    = (m.cache_hits + m.cache_misses) * expert_bytes;
        m.cache_bytes_served       = m.cache_hits * expert_bytes;

        if (rt.prefetch()) {
            auto ps = rt.prefetch()->stats();
            m.prefetch_requests = ps.prefetch_requests;
            m.prefetch_useful = ps.useful_prefetches;
            m.prefetch_wasted = ps.wasted_prefetches;
            m.prefetch_cancelled = ps.cancelled_prefetches;
            m.prefetch_duplicate = ps.duplicate_prefetches;
            uint64_t denom = m.prefetch_useful + m.prefetch_wasted + m.prefetch_cancelled;
            m.prefetch_hit_rate = denom ? double(m.prefetch_useful) / denom : 0.0;
        }
        m.valid = true;
        return m;
    } catch (const std::exception& e) {
        m.valid = false;
        m.invalid_reason = std::string("exception: ") + e.what();
        return m;
    }
}

int main(int argc, char** argv) {
    cli::Parser parser({
        {"asema-bench", "0.1.0",
         "ASEMA research-grade benchmark runner.",
         "[options] --model <model_dir>"},
        {
            {"model",            "m", cli::ArgType::Value, "",
             "Path to model directory (containing manifest.json + model.asema)."},
            {"tokens",           "n", cli::ArgType::Value, "32",
             "Tokens per run."},
            {"warmup",           "",  cli::ArgType::Value, "5",
             "Warmup iterations (excluded from stats). 0 disables."},
            {"iterations",       "i", cli::ArgType::Value, "10",
             "Measured iterations per cell (research-grade >= 10)."},
            {"repetitions",      "r", cli::ArgType::Value, "1",
             "Run the entire matrix this many times (independent seeds)."},
            {"cache",            "c", cli::ArgType::Value, "8388608",
             "Default cache size (bytes) when --cache-sweep is off."},
            {"cache-sweep",      "",  cli::ArgType::Flag, "",
             "Sweep caches 1/2/4/8/12 MB."},
            {"lookahead",        "k", cli::ArgType::Value, "2",
             "Default lookahead K when --lookahead-sweep is off."},
            {"lookahead-sweep",  "",  cli::ArgType::Flag, "",
             "Sweep K = 0,1,2,3,4,6."},
            {"workers",          "w", cli::ArgType::Value, "4",
             "Default worker count."},
            {"workers-sweep",    "",  cli::ArgType::Flag, "",
             "Sweep workers = 1,2,4,8."},
            {"seed",             "s", cli::ArgType::Value, "42",
             "Router seed."},
            {"seed-sweep",       "",  cli::ArgType::Flag, "",
             "Use seeds {11, 42, 99}."},
            {"pattern",          "p", cli::ArgType::Value, "locality",
             "Request pattern label (recorded only)."},
            {"locality-low",     "",  cli::ArgType::Flag, "",
             "Use low-locality router (30% reuse)."},
            {"locality-high",    "",  cli::ArgType::Flag, "",
             "Use high-locality router (90% reuse)."},
            {"locality-medium",  "",  cli::ArgType::Flag, "",
             "Use medium-locality router (50% reuse, default)."},
            {"locality-random",  "",  cli::ArgType::Flag, "",
             "Pure random routing (no locality)."},
            {"locality-adversarial","", cli::ArgType::Flag, "",
             "Adversarial: hit each expert exactly once per round before reuse."},
            {"warm",             "",  cli::ArgType::Flag, "",
             "Mark runs as warm (informational; no OS cache drop)."},
            {"cold",             "",  cli::ArgType::Flag, "",
             "Mark runs as cold (informational; OS cache not flushed)."},
            {"output",           "o", cli::ArgType::Value, "reports/runs",
             "Root output directory. Each run goes to a timestamped subdir."},
            {"format",           "f", cli::ArgType::Value, "json",
             "Output format: json, csv, text."},
            {"quiet",            "q", cli::ArgType::Flag, "",
             "Suppress per-cell output."},
        }
    });

    auto args = parser.parse(argc, argv);
    if (args.help_requested)    { std::cout << parser.render_help();    return cli::RC_SUCCESS; }
    if (args.version_requested) { std::cout << parser.render_version(); return cli::RC_SUCCESS; }
    if (!args.ok()) { std::cerr << "asema-bench: " << args.error << "\n"; return cli::RC_INVALID_ARGUMENT; }

    if (args.get("model").empty()) {
        std::cerr << "asema-bench: --model is required\n";
        return cli::RC_INVALID_ARGUMENT;
    }

    auto workload = resolve_workload(args.get("model"));
    if (workload.num_layers == 0) {
        std::cerr << "asema-bench: could not resolve model from " << args.get("model") << "\n";
        return cli::RC_MODEL_ERROR;
    }

    auto cells = build_matrix(args);
    if (cells.empty()) {
        std::cerr << "asema-bench: no experiment cells defined\n";
        return cli::RC_BENCHMARK_ERROR;
    }

    int warmup = static_cast<int>(args.get_uint("warmup"));
    int iters  = static_cast<int>(args.get_uint("iterations", 3));

    std::string root = args.get("output", "reports/runs");
    std::string run_dir = root + "/" + exp::timestamp_iso8601_utc();
    if (!exp::ensure_directory(run_dir) ||
        !exp::ensure_directory(run_dir + "/raw") ||
        !exp::ensure_directory(run_dir + "/summaries")) {
        std::cerr << "asema-bench: cannot create output directory " << run_dir << "\n";
        return cli::RC_IO_ERROR;
    }

    auto env = exp::Environment::detect(workload);
    {
        std::ostringstream os;
        os << "{\n"
           << "  \"timestamp_utc\": \"" << exp::json_escape(env.timestamp_utc) << "\",\n"
           << "  \"os\": \"" << exp::json_escape(env.os) << "\",\n"
           << "  \"cpu\": \"" << exp::json_escape(env.cpu) << "\",\n"
           << "  \"total_ram_bytes\": " << env.total_ram_bytes << ",\n"
           << "  \"available_ram_bytes\": " << env.available_ram_bytes << ",\n"
           << "  \"gpu\": \"" << exp::json_escape(env.gpu) << "\",\n"
           << "  \"vram_bytes\": " << env.vram_bytes << ",\n"
           << "  \"compiler\": \"" << exp::json_escape(env.compiler) << "\",\n"
           << "  \"build_type\": \"" << exp::json_escape(env.build_type) << "\",\n"
           << "  \"device_class\": \"" << exp::json_escape(env.device_class) << "\",\n"
           << "  \"storage_path\": \"" << exp::json_escape(env.storage_path) << "\",\n"
           << "  \"workload\": {\n"
           << "    \"name\": \"" << exp::json_escape(workload.name) << "\",\n"
           << "    \"num_layers\": " << workload.num_layers << ",\n"
           << "    \"experts_per_layer\": " << workload.experts_per_layer << ",\n"
           << "    \"top_k\": " << workload.top_k << ",\n"
           << "    \"hidden_dim\": " << workload.hidden_dim << ",\n"
           << "    \"ffn_dim\": " << workload.ffn_dim << ",\n"
           << "    \"dtype\": \"" << exp::json_escape(workload.dtype) << "\",\n"
           << "    \"total_storage_bytes\": " << workload.total_storage_bytes << "\n"
           << "  }\n"
           << "}\n";
        exp::write_file(run_dir + "/environment.json", os.str());
    }

    {
        std::ostringstream os;
        os << "{\n"
           << "  \"cells\": [\n";
        for (size_t i = 0; i < cells.size(); ++i) {
            const auto& c = cells[i];
            if (i) os << ",\n";
            os << "    {"
               << "\"id\":\"" << exp::json_escape(c.id) << "\","
               << "\"ram_cache_bytes\":" << c.ram_cache_bytes << ","
               << "\"prefetch_lookahead\":" << c.prefetch_lookahead << ","
               << "\"loader_workers\":" << c.loader_workers << ","
               << "\"router_seed\":" << c.router_seed << ","
               << "\"num_tokens\":" << c.num_tokens << ","
               << "\"locality\":" << c.locality << ","
               << "\"pattern\":\"" << exp::json_escape(c.pattern) << "\","
               << "\"cold_start\":" << (c.cold_start ? "true" : "false")
               << "}";
        }
        os << "\n  ],\n"
           << "  \"iterations\": " << iters << ",\n"
           << "  \"warmup\": " << warmup << "\n"
           << "}\n";
        exp::write_file(run_dir + "/configuration.json", os.str());
    }

    std::ofstream csv(run_dir + "/raw/all_measurements.csv", std::ios::out | std::ios::trunc);
    csv << exp::measurement_to_csv_header() << "\n";

    std::vector<exp::CellSummary> summaries;
    int total_valid = 0, total_invalid = 0;
    auto wall_t0 = std::chrono::steady_clock::now();

    for (size_t cell_idx = 0; cell_idx < cells.size(); ++cell_idx) {
        const auto& cell = cells[cell_idx];
        if (!args.has("quiet")) {
            std::cout << "[bench " << (cell_idx + 1) << "/" << cells.size() << "] "
                      << cell.id << " ...\n";
        }

        std::vector<exp::Measurement> meas;

        for (int w = 0; w < warmup; ++w) {
            (void)run_one(workload, cell, -1 - w);
        }

        for (int it = 1; it <= iters; ++it) {
            auto m = run_one(workload, cell, it);
            if (m.valid) total_valid++; else total_invalid++;
            std::string raw_path = run_dir + "/raw/" + cell.id + "_iter" +
                                   std::to_string(it) + ".json";
            exp::write_file(raw_path, exp::measurement_to_json(m) + "\n");
            csv << exp::measurement_to_csv_row(m) << "\n";
            meas.push_back(std::move(m));
        }

        auto sum = exp::summarize(cell, meas);
        std::string sum_path = run_dir + "/summaries/" + cell.id + ".json";
        exp::write_file(sum_path, exp::summary_to_json(sum) + "\n");
        summaries.push_back(std::move(sum));
    }
    csv.close();

    auto wall_t1 = std::chrono::steady_clock::now();
    double total_wall = std::chrono::duration<double, std::milli>(
        wall_t1 - wall_t0).count();

    {
        std::ostringstream os;
        os << "ASEMA v0.1 Benchmark - " << exp::timestamp_iso8601_utc() << "\n";
        os << "Total wall: " << total_wall << " ms  |  Cells: " << summaries.size()
           << "  |  Valid measurements: " << total_valid
           << "  |  Invalid: " << total_invalid << "\n";
        os << "------------------------------------------------------------\n";
        os << std::left
           << std::setw(28) << "Cell"
           << std::setw(10) << "valid"
           << std::setw(14) << "mean_ms"
           << std::setw(14) << "stddev"
           << std::setw(10) << "min"
           << std::setw(10) << "max"
           << std::setw(10) << "median"
           << std::setw(10) << "p95"
           << std::setw(10) << "hit%"
           << std::setw(14) << "ssd_MB"
           << std::setw(10) << "pf_hit%"
           << "\n";
        for (const auto& s : summaries) {
            os << std::left
               << std::setw(28) << s.cell.id
               << std::setw(10) << s.valid_count
               << std::setw(14) << std::fixed << std::setprecision(3) << s.mean_ms_per_token
               << std::setw(14) << std::setprecision(3) << s.stddev_ms_per_token
               << std::setw(10) << std::setprecision(3) << s.min_ms_per_token
               << std::setw(10) << std::setprecision(3) << s.max_ms_per_token
               << std::setw(10) << std::setprecision(3) << s.median_ms_per_token
               << std::setw(10) << std::setprecision(3) << s.p95_ms_per_token
               << std::setw(10) << std::setprecision(1) << (s.mean_cache_hit_rate * 100.0)
               << std::setw(14) << std::setprecision(2)
                              << (s.mean_ssd_bytes_physical / (1024.0 * 1024.0))
               << std::setw(10) << std::setprecision(1) << (s.mean_prefetch_hit_rate * 100.0)
               << "\n";
        }
        exp::write_file(run_dir + "/summaries/all_summaries.txt", os.str());
        if (!args.has("quiet")) {
            std::cout << "\n" << os.str();
        }
    }

    std::cout << "[asema-bench] Outputs written to: " << run_dir << "\n";

    if (total_valid == 0) {
        std::cerr << "[asema-bench] WARNING: no valid measurements collected.\n";
        return cli::RC_BENCHMARK_ERROR;
    }
    return cli::RC_SUCCESS;
}
