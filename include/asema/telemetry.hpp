#pragma once

// ASEMA v0.1 — Milestone 5: TelemetryEngine
// -----------------------------------------------------------------------------
// A first-class telemetry subsystem. Emits structured events to a JSONL sink
// (file or stdout). Provides high-resolution timing, percentile calculation,
// and aggregate report generation.
//
// Design properties:
//   * Events are emitted via lock-free enqueue into a bounded ring buffer;
//     a background thread (or the user-controlled flush()) drains the buffer
//     and writes JSON Lines.
//   * Timing uses std::chrono::steady_clock (monotonic, ns-resolution).
//   * Percentiles (P50, P90, P95, P99, max) are computed via a sorted-vector
//     snapshot. The method is documented and reproducible.
//   * Telemetry overhead is intentionally low: enqueue + a small atomic
//     bump. JSON serialization is done off the hot path.
// -----------------------------------------------------------------------------

#include "expert.hpp"
#include "async_loader.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

namespace asema {

// -----------------------------------------------------------------------------
// Event types
// -----------------------------------------------------------------------------
enum class EventType : uint16_t {
    TOKEN_BEGIN            = 1,
    TOKEN_END              = 2,
    EXPERT_REQUEST         = 3,
    EXPERT_CACHE_HIT       = 4,
    EXPERT_CACHE_MISS      = 5,
    EXPERT_LOAD_BEGIN      = 6,
    EXPERT_LOAD_END        = 7,
    PREFETCH_REQUEST       = 8,
    PREFETCH_USEFUL        = 9,
    PREFETCH_WASTED        = 10,
    PREFETCH_CANCELLED     = 11,
    CACHE_EVICTION         = 12,
    CACHE_INSERTION        = 13,
    EXECUTION_BEGIN        = 14,
    EXECUTION_END          = 15,
    ERROR                  = 16,
    QUEUE_PUSH             = 17,
    QUEUE_POP              = 18,
    QUEUE_COALESCE         = 19,
    RUNTIME_BEGIN          = 20,
    RUNTIME_END            = 21,
};

inline const char* event_name(EventType e) {
    switch (e) {
        case EventType::TOKEN_BEGIN:        return "TOKEN_BEGIN";
        case EventType::TOKEN_END:          return "TOKEN_END";
        case EventType::EXPERT_REQUEST:     return "EXPERT_REQUEST";
        case EventType::EXPERT_CACHE_HIT:   return "EXPERT_CACHE_HIT";
        case EventType::EXPERT_CACHE_MISS:  return "EXPERT_CACHE_MISS";
        case EventType::EXPERT_LOAD_BEGIN:  return "EXPERT_LOAD_BEGIN";
        case EventType::EXPERT_LOAD_END:    return "EXPERT_LOAD_END";
        case EventType::PREFETCH_REQUEST:   return "PREFETCH_REQUEST";
        case EventType::PREFETCH_USEFUL:    return "PREFETCH_USEFUL";
        case EventType::PREFETCH_WASTED:    return "PREFETCH_WASTED";
        case EventType::PREFETCH_CANCELLED: return "PREFETCH_CANCELLED";
        case EventType::CACHE_EVICTION:     return "CACHE_EVICTION";
        case EventType::CACHE_INSERTION:    return "CACHE_INSERTION";
        case EventType::EXECUTION_BEGIN:    return "EXECUTION_BEGIN";
        case EventType::EXECUTION_END:      return "EXECUTION_END";
        case EventType::ERROR:              return "ERROR";
        case EventType::QUEUE_PUSH:         return "QUEUE_PUSH";
        case EventType::QUEUE_POP:          return "QUEUE_POP";
        case EventType::QUEUE_COALESCE:     return "QUEUE_COALESCE";
        case EventType::RUNTIME_BEGIN:      return "RUNTIME_BEGIN";
        case EventType::RUNTIME_END:        return "RUNTIME_END";
        default:                            return "UNKNOWN";
    }
}

// -----------------------------------------------------------------------------
// Event payload (small, fixed-size, fast to construct)
// -----------------------------------------------------------------------------
struct TelemetryEvent {
    EventType type{EventType::ERROR};
    uint64_t timestamp_ns{0};

    // Token id (for TOKEN_*, EXPERT_*, EXECUTION_*)
    uint64_t token_id{0};

    // Layer / expert coordinate
    uint32_t layer_id{0};
    uint32_t expert_id{0};
    bool has_coord{false};

    // Latency (microseconds, integer)
    uint64_t latency_us{0};

    // Bytes
    uint64_t bytes{0};

    // Boolean / status
    bool success{true};
    bool is_cancelled{false};

    // Priority (matches LoadPriority uint8_t value, or 255 for non-applicable)
    uint8_t priority{255};

    // Free-form message (kept short to avoid per-event allocation)
    std::string message;
};

// -----------------------------------------------------------------------------
// Percentile statistics for a single timing series
// -----------------------------------------------------------------------------
struct LatencyStats {
    uint64_t count{0};
    double avg_us{0.0};
    double min_us{0.0};
    double max_us{0.0};
    double p50_us{0.0};
    double p90_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    double sum_us{0.0};

    static LatencyStats from(const std::vector<double>& values); // takes a copy internally
};

// -----------------------------------------------------------------------------
// Aggregate report — populated at runtime, dumped by write_report()
// -----------------------------------------------------------------------------
struct AggregateReport {
    uint64_t total_tokens{0};
    double total_wall_time_ms{0.0};
    double first_token_latency_ms{0.0};
    double tokens_per_sec{0.0};
    double ms_per_token{0.0};

    uint64_t cache_hits{0};
    uint64_t cache_misses{0};
    double cache_hit_rate{0.0};

    uint64_t cache_evictions{0};
    uint64_t cache_insertions{0};

    uint64_t ram_bytes_used{0};
    uint64_t ram_bytes_peak{0};
    uint64_t ram_capacity{0};

    uint64_t ssd_bytes{0};
    uint64_t ssd_reads{0};
    double ssd_throughput_mb_s{0.0};

    uint64_t prefetch_requests{0};
    uint64_t prefetch_useful{0};
    uint64_t prefetch_wasted{0};
    uint64_t prefetch_cancelled{0};
    double prefetch_hit_rate{0.0};

    uint64_t queue_depth{0};
    uint64_t queue_peak{0};

    double total_execution_ms{0.0};

    LatencyStats request_latency;
    LatencyStats load_latency;
    LatencyStats execution_latency;
    LatencyStats end_to_end_token_latency;
};

// -----------------------------------------------------------------------------
// Engine configuration
// -----------------------------------------------------------------------------
struct TelemetryConfig {
    // Path to JSONL output. Empty string = stdout.
    std::string jsonl_path;

    // Path to aggregate report text. Empty = disabled.
    std::string report_text_path;

    // Path to JSON report. Empty = disabled.
    std::string report_json_path;

    // Background flusher enabled?
    bool background_flush{true};

    // Buffer capacity (number of events). When full, oldest are dropped.
    size_t buffer_capacity{65536};

    // Flush every N events or every M ms.
    size_t flush_every_n_events{1024};
    uint64_t flush_every_ms{500};
};

// -----------------------------------------------------------------------------
// Telemetry engine
// -----------------------------------------------------------------------------
class TelemetryEngine {
public:
    explicit TelemetryEngine(const TelemetryConfig& cfg);
    ~TelemetryEngine();

    TelemetryEngine(const TelemetryEngine&) = delete;
    TelemetryEngine& operator=(const TelemetryEngine&) = delete;

    // -------------------------------------------------------------------------
    // Event emission (hot path)
    // -------------------------------------------------------------------------
    void emit(const TelemetryEvent& e);

    // Helpers — convenience for common events.
    void token_begin(uint64_t token_id);
    void token_end(uint64_t token_id, double latency_ms);

    void expert_request(uint64_t token_id, ExpertCoord c, LoadPriority prio);
    void expert_cache_hit(uint64_t token_id, ExpertCoord c, double lookup_us);
    void expert_cache_miss(uint64_t token_id, ExpertCoord c);
    void expert_load_begin(uint64_t token_id, ExpertCoord c);
    void expert_load_end(uint64_t token_id, ExpertCoord c, uint64_t bytes,
                         double latency_us, bool success);

    void prefetch_request(ExpertCoord c, uint64_t bytes);
    void prefetch_useful(ExpertCoord c);
    void prefetch_wasted(ExpertCoord c);
    void prefetch_cancelled(ExpertCoord c);

    void cache_eviction(ExpertCoord c, uint64_t bytes);
    void cache_insertion(ExpertCoord c, uint64_t bytes);

    void execution_begin(uint64_t token_id, ExpertCoord c);
    void execution_end(uint64_t token_id, ExpertCoord c, double latency_us);

    void queue_push(ExpertCoord c, LoadPriority prio);
    void queue_pop(ExpertCoord c, LoadPriority prio, uint32_t waiters);
    void queue_coalesce(ExpertCoord c);

    void runtime_begin();
    void runtime_end(double total_ms);

    void error(const std::string& message);

    // -------------------------------------------------------------------------
    // Timing helpers
    // -------------------------------------------------------------------------
    static uint64_t now_ns() noexcept;
    static uint64_t steady_ns_since(uint64_t start_ns) noexcept;

    // -------------------------------------------------------------------------
    // Latency record (used by the runtime for percentile reporting)
    // -------------------------------------------------------------------------
    void record_request_latency_us(double us);
    void record_load_latency_us(double us);
    void record_execution_latency_us(double us);
    void record_token_end_to_end_us(double us);
    void record_first_token_latency_ms(double ms);

    // -------------------------------------------------------------------------
    // Counters (used by the runtime for the aggregate report)
    // -------------------------------------------------------------------------
    void increment_cache_hit();
    void increment_cache_miss();
    void increment_cache_eviction(uint64_t bytes);
    void increment_cache_insertion(uint64_t bytes);
    void increment_ssd_read(uint64_t bytes);
    void increment_prefetch_request();
    void increment_prefetch_useful();
    void increment_prefetch_wasted();
    void increment_prefetch_cancelled();
    void set_queue_depth(uint64_t depth);
    void set_ram_bytes(uint64_t used, uint64_t peak, uint64_t capacity);
    void increment_token();
    void increment_execution_ms(double ms);

    // -------------------------------------------------------------------------
    // Lifecycle / output
    // -------------------------------------------------------------------------

    // Force-flush the event buffer to disk (or stdout).
    void flush();

    // Build an aggregate report from the current state.
    AggregateReport build_report() const;

    // Write the aggregate report (text + JSON). Returns true on success.
    bool write_report() const;

    // Disable emission (drain buffer). Safe to call at any time.
    void shutdown();

private:
    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------
    void background_loop();
    std::string serialize_event(const TelemetryEvent& e) const;

    TelemetryConfig cfg_;

    // Ring buffer (lock-free-ish; protected by mu_ for simplicity).
    std::vector<TelemetryEvent> buffer_;
    size_t buffer_capacity_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> tail_{0};
    std::atomic<size_t> count_{0};
    mutable std::mutex mu_;

    // Output sink
    std::ofstream out_file_;
    bool use_stdout_;

    // Background flusher
    std::thread flusher_thread_;
    std::atomic<bool> shutdown_{false};
    std::atomic<uint64_t> last_flush_ns_{0};

    // Latency tracking
    mutable std::mutex latency_mu_;
    std::vector<double> request_latencies_us_;
    std::vector<double> load_latencies_us_;
    std::vector<double> execution_latencies_us_;
    std::vector<double> token_end_to_end_us_;

    // Counters
    std::atomic<uint64_t> cache_hits_{0};
    std::atomic<uint64_t> cache_misses_{0};
    std::atomic<uint64_t> cache_evictions_{0};
    std::atomic<uint64_t> cache_insertions_{0};
    std::atomic<uint64_t> cache_evicted_bytes_{0};
    std::atomic<uint64_t> cache_inserted_bytes_{0};
    std::atomic<uint64_t> ssd_bytes_{0};
    std::atomic<uint64_t> ssd_reads_{0};
    std::atomic<uint64_t> prefetch_requests_{0};
    std::atomic<uint64_t> prefetch_useful_{0};
    std::atomic<uint64_t> prefetch_wasted_{0};
    std::atomic<uint64_t> prefetch_cancelled_{0};
    std::atomic<uint64_t> queue_depth_{0};
    std::atomic<uint64_t> queue_peak_{0};
    std::atomic<uint64_t> tokens_{0};
    std::atomic<uint64_t> first_token_latency_us_{0};
    std::atomic<double>   total_execution_ms_{0.0};

    std::atomic<uint64_t> ram_used_{0};
    std::atomic<uint64_t> ram_peak_{0};
    std::atomic<uint64_t> ram_capacity_{0};

    // Timing
    std::atomic<uint64_t> runtime_start_ns_{0};
    std::atomic<uint64_t> runtime_end_ns_{0};
};

} // namespace asema
