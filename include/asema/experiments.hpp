#pragma once

// ASEMA v0.1 — Agent #3: experiment orchestration types
// -----------------------------------------------------------------------------
// The data model for M7's experiment runner. Experiments are described
// declaratively; results are written to JSON / CSV / text.
// -----------------------------------------------------------------------------

#include "expert.hpp"
#include "manifest.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace asema {
namespace exp {

// -----------------------------------------------------------------------------
// Workload description
// -----------------------------------------------------------------------------
struct WorkloadConfig {
    uint32_t num_layers{0};
    uint32_t experts_per_layer{0};
    uint32_t top_k{0};
    uint32_t hidden_dim{0};
    uint32_t ffn_dim{0};
    std::string dtype{"fp16"};
    std::string name{"unnamed"};
    std::string manifest_path;
    std::string container_path;
    uint64_t total_storage_bytes{0};
};

// -----------------------------------------------------------------------------
// One cell in the experiment matrix
// -----------------------------------------------------------------------------
struct ExperimentCell {
    std::string id;                       // human-readable label, e.g. "cache_8MB_K2_seed42"
    uint64_t ram_cache_bytes{0};
    size_t prefetch_lookahead{0};
    uint64_t loader_workers{4};
    uint32_t num_tokens{32};
    uint64_t router_seed{42};

    // Locality control: probability that a token reuses the previous token's
    // experts (1.0 = full locality, 0.0 = pure random).
    double locality{0.7};
    // Request-pattern: "sequential", "random", "locality", "adversarial"
    std::string pattern{"locality"};

    // Cold / warm flag — affects whether we clear the OS file cache before
    // the run (we don't actually do that on Windows, but we record intent).
    bool cold_start{true};
};

// -----------------------------------------------------------------------------
// One measurement (one run of one cell)
// -----------------------------------------------------------------------------
struct Measurement {
    std::string experiment_id;
    int iteration{0};                     // 1..N
    bool valid{false};                    // false = excluded from stats
    std::string invalid_reason;           // e.g. "runtime_error", "memory_budget_violated"

    uint32_t tokens_completed{0};
    double wall_time_ms{0.0};
    double first_token_ms{0.0};
    double ms_per_token{0.0};
    double tokens_per_sec{0.0};

    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    double cache_hit_rate{0.0};
    uint64_t cache_evictions{0};
    uint64_t cache_insertions{0};
    uint64_t peak_bytes_used{0};

    // ---- Storage metrics (physical vs logical, hardened in v0.1 / M7+) ----
    //
    // Definitions (see docs/windows_storage_measurement.md for the full
    // discussion):
    //
    //   runtime_logical_requests
    //     Number of expert requests the runtime emitted to the loader
    //     (cache misses + prefetch requests). Excludes cache hits.
    //     Source: loader->total_requests_submitted().
    //
    //   runtime_logical_bytes
    //     Bytes the runtime asked for, computed as
    //     runtime_logical_requests × expert_bytes_per_request.
    //     This is the upper bound on I/O if no coalescing occurred.
    //
    //   ssd_bytes_physical
    //     Bytes actually transferred from storage by the loader
    //     (after internal coalescing). Source: loader->total_bytes_loaded().
    //     On Windows this is the value the OS reports; OS file cache
    //     may reduce the hardware-level physical bytes actually read
    //     from the device.
    //
    //   ssd_reads
    //     Number of physical I/O operations the loader performed.
    //     Source: loader->total_io_dispatches().
    //
    //   coalesced_requests
    //     Number of runtime requests that the loader folded into an
    //     already-in-flight load. Source: loader->total_coalesced_requests().
    //
    //   cache_bytes_served
    //     Bytes served from the RAM cache without touching storage.
    //     = cache_hits × expert_bytes_per_request.
    //
    //   expert_bytes_per_request
    //     Average expert size inferred from cache misses + bytes read.
    //
    uint64_t runtime_logical_requests{0};
    uint64_t runtime_logical_bytes{0};
    uint64_t ssd_bytes_physical{0};
    uint64_t ssd_reads{0};
    uint64_t coalesced_requests{0};
    uint64_t cache_bytes_served{0};
    uint64_t expert_bytes_per_request{0};
    // legacy alias kept for backward compatibility with v0.1 raw JSON
    uint64_t ssd_bytes_logical{0};

    uint64_t prefetch_requests{0};
    uint64_t prefetch_useful{0};
    uint64_t prefetch_wasted{0};
    uint64_t prefetch_cancelled{0};
    uint64_t prefetch_duplicate{0};
    double prefetch_hit_rate{0.0};
};

// -----------------------------------------------------------------------------
// Aggregated stats across iterations of one cell
// -----------------------------------------------------------------------------
struct CellSummary {
    ExperimentCell cell;
    std::vector<Measurement> measurements;
    int valid_count{0};
    int invalid_count{0};

    double mean_ms_per_token{0.0};
    double stddev_ms_per_token{0.0};
    double min_ms_per_token{0.0};
    double max_ms_per_token{0.0};
    double median_ms_per_token{0.0};
    double p95_ms_per_token{0.0};

    double mean_cache_hit_rate{0.0};
    double mean_ssd_bytes_physical{0.0};
    double mean_runtime_logical_bytes{0.0};
    double mean_coalesced_requests{0.0};
    double mean_prefetch_hit_rate{0.0};
};

// -----------------------------------------------------------------------------
// Environment snapshot (detected at run time)
// -----------------------------------------------------------------------------
struct Environment {
    std::string timestamp_utc;
    std::string os{"windows"};
    std::string cpu;
    uint64_t total_ram_bytes{0};
    uint64_t available_ram_bytes{0};
    std::string gpu;
    uint64_t vram_bytes{0};
    std::string compiler;
    std::string build_type;
    WorkloadConfig workload;
    std::string storage_path;
    std::string device_class{"unknown"};     // SSD / NVMe / HDD / unknown

    static Environment detect(const WorkloadConfig& w);
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Aggregate measurements into a summary.
CellSummary summarize(const ExperimentCell& cell,
                      const std::vector<Measurement>& measurements);

// Serialize a measurement as JSON (single line).
std::string measurement_to_json(const Measurement& m);

// Serialize a cell summary as JSON (multi-line, indented).
std::string summary_to_json(const CellSummary& s);

// Serialize a measurement row as CSV (header must be the first call).
std::string measurement_to_csv_header();
std::string measurement_to_csv_row(const Measurement& m);

// JSON string escape (shared with asema-bench tool).
std::string json_escape(const std::string& s);

// Write a file atomically (write to .tmp then rename). Returns true on success.
bool write_file(const std::string& path, const std::string& content);

// Ensure directory exists (recursive mkdir).
bool ensure_directory(const std::string& path);

// Generate a timestamp string like "2026-09-26T15-30-00Z".
std::string timestamp_iso8601_utc();

// Human-readable byte size.
std::string human_bytes(uint64_t b);

} // namespace exp
} // namespace asema
