// ASEMA v0.1 — Milestone 5: TelemetryEngine implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/telemetry.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace asema {

// =============================================================================
// LatencyStats
// =============================================================================
LatencyStats LatencyStats::from(const std::vector<double>& values_in) {
    LatencyStats s;
    if (values_in.empty()) return s;
    std::vector<double> values = values_in; // copy for in-place sort
    std::sort(values.begin(), values.end());
    s.count = values.size();
    double sum = 0.0;
    for (double v : values) sum += v;
    s.sum_us = sum;
    s.avg_us = sum / static_cast<double>(values.size());
    s.min_us = values.front();
    s.max_us = values.back();

    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(values.size() * p);
        if (idx >= values.size()) idx = values.size() - 1;
        return values[idx];
    };

    s.p50_us = pct(0.50);
    s.p90_us = pct(0.90);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    return s;
}

// =============================================================================
// TelemetryEngine
// =============================================================================
TelemetryEngine::TelemetryEngine(const TelemetryConfig& cfg)
    : cfg_(cfg), buffer_capacity_(cfg.buffer_capacity), use_stdout_(cfg.jsonl_path.empty())
{
    buffer_.resize(buffer_capacity_);

    if (!use_stdout_) {
        out_file_.open(cfg.jsonl_path, std::ios::out | std::ios::trunc);
        if (!out_file_.is_open()) {
            // Fallback to stdout
            use_stdout_ = true;
        }
    }

    runtime_start_ns_.store(now_ns(), std::memory_order_relaxed);

    if (cfg.background_flush) {
        flusher_thread_ = std::thread(&TelemetryEngine::background_loop, this);
    }
}

TelemetryEngine::~TelemetryEngine() {
    shutdown();
}

uint64_t TelemetryEngine::now_ns() noexcept {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

uint64_t TelemetryEngine::steady_ns_since(uint64_t start_ns) noexcept {
    uint64_t now = now_ns();
    return now >= start_ns ? now - start_ns : 0;
}

// =============================================================================
// Event emission
// =============================================================================
void TelemetryEngine::emit(const TelemetryEvent& e) {
    // Hot path: lock-free enqueue into ring buffer.
    size_t idx = head_.load(std::memory_order_relaxed);
    size_t next = (idx + 1) % buffer_capacity_;

    // If buffer is full, drop the oldest event (advance tail).
    if (count_.load(std::memory_order_acquire) >= buffer_capacity_) {
        tail_.store((tail_.load(std::memory_order_relaxed) + 1) % buffer_capacity_,
                    std::memory_order_relaxed);
        count_.fetch_sub(1, std::memory_order_relaxed);
    }

    {
        std::lock_guard<std::mutex> lock(mu_);
        buffer_[idx] = e;
    }
    head_.store(next, std::memory_order_release);
    count_.fetch_add(1, std::memory_order_release);
}

std::string TelemetryEngine::serialize_event(const TelemetryEvent& e) const {
    // Build a compact JSON object manually to avoid heavy json.hpp in the hot
    // path. Format: one line, valid JSON.
    std::ostringstream os;
    os << "{\"event\":\"" << event_name(e.type) << "\""
       << ",\"ts\":" << e.timestamp_ns;
    if (e.token_id != 0) os << ",\"token\":" << e.token_id;
    if (e.has_coord) os << ",\"layer\":" << e.layer_id << ",\"expert\":" << e.expert_id;
    if (e.latency_us != 0) os << ",\"latency_us\":" << e.latency_us;
    if (e.bytes != 0) os << ",\"bytes\":" << e.bytes;
    os << ",\"success\":" << (e.success ? "true" : "false");
    if (e.is_cancelled) os << ",\"cancelled\":true";
    if (e.priority != 255) os << ",\"priority\":" << static_cast<int>(e.priority);
    if (!e.message.empty()) {
        // Escape minimal characters (quotes and backslashes)
        std::string msg;
        msg.reserve(e.message.size());
        for (char c : e.message) {
            if (c == '"') msg += "\\\"";
            else if (c == '\\') msg += "\\\\";
            else msg += c;
        }
        os << ",\"msg\":\"" << msg << "\"";
    }
    os << "}";
    return os.str();
}

// =============================================================================
// Convenience helpers
// =============================================================================
void TelemetryEngine::token_begin(uint64_t token_id) {
    TelemetryEvent e;
    e.type = EventType::TOKEN_BEGIN;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    emit(e);
}

void TelemetryEngine::token_end(uint64_t token_id, double latency_ms) {
    TelemetryEvent e;
    e.type = EventType::TOKEN_END;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.latency_us = static_cast<uint64_t>(latency_ms * 1000.0);
    emit(e);
    record_token_end_to_end_us(latency_ms * 1000.0);
    increment_token();
}

void TelemetryEngine::expert_request(uint64_t token_id, ExpertCoord c, LoadPriority prio) {
    TelemetryEvent e;
    e.type = EventType::EXPERT_REQUEST;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.priority = static_cast<uint8_t>(prio);
    emit(e);
}

void TelemetryEngine::expert_cache_hit(uint64_t token_id, ExpertCoord c, double lookup_us) {
    TelemetryEvent e;
    e.type = EventType::EXPERT_CACHE_HIT;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.latency_us = static_cast<uint64_t>(lookup_us);
    emit(e);
    increment_cache_hit();
}

void TelemetryEngine::expert_cache_miss(uint64_t token_id, ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::EXPERT_CACHE_MISS;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
    increment_cache_miss();
}

void TelemetryEngine::expert_load_begin(uint64_t token_id, ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::EXPERT_LOAD_BEGIN;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
}

void TelemetryEngine::expert_load_end(uint64_t token_id, ExpertCoord c, uint64_t bytes,
                                       double latency_us, bool success) {
    TelemetryEvent e;
    e.type = EventType::EXPERT_LOAD_END;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.latency_us = static_cast<uint64_t>(latency_us);
    e.bytes = bytes;
    e.success = success;
    emit(e);
    record_load_latency_us(latency_us);
    increment_ssd_read(bytes);
}

void TelemetryEngine::prefetch_request(ExpertCoord c, uint64_t bytes) {
    TelemetryEvent e;
    e.type = EventType::PREFETCH_REQUEST;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.bytes = bytes;
    emit(e);
    increment_prefetch_request();
}

void TelemetryEngine::prefetch_useful(ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::PREFETCH_USEFUL;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
    increment_prefetch_useful();
}

void TelemetryEngine::prefetch_wasted(ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::PREFETCH_WASTED;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
    increment_prefetch_wasted();
}

void TelemetryEngine::prefetch_cancelled(ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::PREFETCH_CANCELLED;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
    increment_prefetch_cancelled();
}

void TelemetryEngine::cache_eviction(ExpertCoord c, uint64_t bytes) {
    TelemetryEvent e;
    e.type = EventType::CACHE_EVICTION;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.bytes = bytes;
    emit(e);
    increment_cache_eviction(bytes);
}

void TelemetryEngine::cache_insertion(ExpertCoord c, uint64_t bytes) {
    TelemetryEvent e;
    e.type = EventType::CACHE_INSERTION;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.bytes = bytes;
    emit(e);
    increment_cache_insertion(bytes);
}

void TelemetryEngine::execution_begin(uint64_t token_id, ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::EXECUTION_BEGIN;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
}

void TelemetryEngine::execution_end(uint64_t token_id, ExpertCoord c, double latency_us) {
    TelemetryEvent e;
    e.type = EventType::EXECUTION_END;
    e.timestamp_ns = now_ns();
    e.token_id = token_id;
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.latency_us = static_cast<uint64_t>(latency_us);
    emit(e);
    record_execution_latency_us(latency_us);
}

void TelemetryEngine::queue_push(ExpertCoord c, LoadPriority prio) {
    TelemetryEvent e;
    e.type = EventType::QUEUE_PUSH;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.priority = static_cast<uint8_t>(prio);
    emit(e);
}

void TelemetryEngine::queue_pop(ExpertCoord c, LoadPriority prio, uint32_t waiters) {
    TelemetryEvent e;
    e.type = EventType::QUEUE_POP;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    e.priority = static_cast<uint8_t>(prio);
    e.bytes = waiters; // piggyback waiters
    emit(e);
}

void TelemetryEngine::queue_coalesce(ExpertCoord c) {
    TelemetryEvent e;
    e.type = EventType::QUEUE_COALESCE;
    e.timestamp_ns = now_ns();
    e.layer_id = c.layer_id;
    e.expert_id = c.expert_id;
    e.has_coord = true;
    emit(e);
}

void TelemetryEngine::runtime_begin() {
    TelemetryEvent e;
    e.type = EventType::RUNTIME_BEGIN;
    e.timestamp_ns = now_ns();
    runtime_start_ns_.store(e.timestamp_ns, std::memory_order_relaxed);
    emit(e);
}

void TelemetryEngine::runtime_end(double total_ms) {
    TelemetryEvent e;
    e.type = EventType::RUNTIME_END;
    e.timestamp_ns = now_ns();
    e.latency_us = static_cast<uint64_t>(total_ms * 1000.0);
    runtime_end_ns_.store(e.timestamp_ns, std::memory_order_relaxed);
    emit(e);
}

void TelemetryEngine::error(const std::string& message) {
    TelemetryEvent e;
    e.type = EventType::ERROR;
    e.timestamp_ns = now_ns();
    e.success = false;
    e.message = message;
    emit(e);
}

// =============================================================================
// Latency recording
// =============================================================================
void TelemetryEngine::record_request_latency_us(double us) {
    std::lock_guard<std::mutex> lock(latency_mu_);
    request_latencies_us_.push_back(us);
}

void TelemetryEngine::record_load_latency_us(double us) {
    std::lock_guard<std::mutex> lock(latency_mu_);
    load_latencies_us_.push_back(us);
}

void TelemetryEngine::record_execution_latency_us(double us) {
    std::lock_guard<std::mutex> lock(latency_mu_);
    execution_latencies_us_.push_back(us);
}

void TelemetryEngine::record_token_end_to_end_us(double us) {
    std::lock_guard<std::mutex> lock(latency_mu_);
    token_end_to_end_us_.push_back(us);
}

void TelemetryEngine::record_first_token_latency_ms(double ms) {
    first_token_latency_us_.store(static_cast<uint64_t>(ms * 1000.0), std::memory_order_relaxed);
}

// =============================================================================
// Counter increments
// =============================================================================
void TelemetryEngine::increment_cache_hit()              { cache_hits_.fetch_add(1, std::memory_order_relaxed); }
void TelemetryEngine::increment_cache_miss()             { cache_misses_.fetch_add(1, std::memory_order_relaxed); }
void TelemetryEngine::increment_cache_eviction(uint64_t b) {
    cache_evictions_.fetch_add(1, std::memory_order_relaxed);
    cache_evicted_bytes_.fetch_add(b, std::memory_order_relaxed);
}
void TelemetryEngine::increment_cache_insertion(uint64_t b) {
    cache_insertions_.fetch_add(1, std::memory_order_relaxed);
    cache_inserted_bytes_.fetch_add(b, std::memory_order_relaxed);
}
void TelemetryEngine::increment_ssd_read(uint64_t b) {
    ssd_bytes_.fetch_add(b, std::memory_order_relaxed);
    ssd_reads_.fetch_add(1, std::memory_order_relaxed);
}
void TelemetryEngine::increment_prefetch_request()       { prefetch_requests_.fetch_add(1, std::memory_order_relaxed); }
void TelemetryEngine::increment_prefetch_useful()        { prefetch_useful_.fetch_add(1, std::memory_order_relaxed); }
void TelemetryEngine::increment_prefetch_wasted()        { prefetch_wasted_.fetch_add(1, std::memory_order_relaxed); }
void TelemetryEngine::increment_prefetch_cancelled()     { prefetch_cancelled_.fetch_add(1, std::memory_order_relaxed); }

void TelemetryEngine::set_queue_depth(uint64_t depth) {
    queue_depth_.store(depth, std::memory_order_relaxed);
    uint64_t cur = depth;
    uint64_t peak = queue_peak_.load(std::memory_order_relaxed);
    while (cur > peak && !queue_peak_.compare_exchange_weak(
            peak, cur, std::memory_order_acq_rel, std::memory_order_relaxed)) {}
}

void TelemetryEngine::set_ram_bytes(uint64_t used, uint64_t peak, uint64_t capacity) {
    ram_used_.store(used, std::memory_order_relaxed);
    ram_peak_.store(peak, std::memory_order_relaxed);
    ram_capacity_.store(capacity, std::memory_order_relaxed);
}

void TelemetryEngine::increment_token() {
    tokens_.fetch_add(1, std::memory_order_relaxed);
}

void TelemetryEngine::increment_execution_ms(double ms) {
    double cur = total_execution_ms_.load(std::memory_order_relaxed);
    double desired;
    do {
        desired = cur + ms;
    } while (!total_execution_ms_.compare_exchange_weak(
            cur, desired, std::memory_order_acq_rel, std::memory_order_relaxed));
}

// =============================================================================
// Flush / report
// =============================================================================
void TelemetryEngine::flush() {
    std::vector<TelemetryEvent> drained;
    {
        std::lock_guard<std::mutex> lock(mu_);
        size_t n = count_.load(std::memory_order_acquire);
        if (n == 0) return;
        drained.reserve(n);
        size_t t = tail_.load(std::memory_order_relaxed);
        for (size_t i = 0; i < n; ++i) {
            drained.push_back(buffer_[t]);
            t = (t + 1) % buffer_capacity_;
        }
        tail_.store(t, std::memory_order_relaxed);
        count_.store(0, std::memory_order_release);
    }

    for (const auto& e : drained) {
        std::string line = serialize_event(e);
        line += '\n';
        if (use_stdout_) {
            std::cout << line;
        } else {
            out_file_ << line;
        }
    }
    if (!use_stdout_) out_file_.flush();
}

void TelemetryEngine::background_loop() {
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.flush_every_ms));
        flush();
    }
    flush();
}

void TelemetryEngine::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;
    if (flusher_thread_.joinable()) flusher_thread_.join();
    flush();
    if (out_file_.is_open()) out_file_.close();
}

AggregateReport TelemetryEngine::build_report() const {
    AggregateReport r;
    r.total_tokens = tokens_.load(std::memory_order_relaxed);
    r.cache_hits = cache_hits_.load(std::memory_order_relaxed);
    r.cache_misses = cache_misses_.load(std::memory_order_relaxed);
    r.cache_evictions = cache_evictions_.load(std::memory_order_relaxed);
    r.cache_insertions = cache_insertions_.load(std::memory_order_relaxed);
    r.ram_bytes_used = ram_used_.load(std::memory_order_relaxed);
    r.ram_bytes_peak = ram_peak_.load(std::memory_order_relaxed);
    r.ram_capacity = ram_capacity_.load(std::memory_order_relaxed);
    r.ssd_bytes = ssd_bytes_.load(std::memory_order_relaxed);
    r.ssd_reads = ssd_reads_.load(std::memory_order_relaxed);
    r.prefetch_requests = prefetch_requests_.load(std::memory_order_relaxed);
    r.prefetch_useful = prefetch_useful_.load(std::memory_order_relaxed);
    r.prefetch_wasted = prefetch_wasted_.load(std::memory_order_relaxed);
    r.prefetch_cancelled = prefetch_cancelled_.load(std::memory_order_relaxed);
    r.queue_depth = queue_depth_.load(std::memory_order_relaxed);
    r.queue_peak = queue_peak_.load(std::memory_order_relaxed);
    r.total_execution_ms = total_execution_ms_.load(std::memory_order_relaxed);
    r.first_token_latency_ms =
        first_token_latency_us_.load(std::memory_order_relaxed) / 1000.0;

    uint64_t total_hits_misses = r.cache_hits + r.cache_misses;
    r.cache_hit_rate = total_hits_misses == 0 ? 0.0 :
        static_cast<double>(r.cache_hits) / static_cast<double>(total_hits_misses);

    uint64_t prefetch_denom = r.prefetch_useful + r.prefetch_wasted + r.prefetch_cancelled;
    r.prefetch_hit_rate = prefetch_denom == 0 ? 0.0 :
        static_cast<double>(r.prefetch_useful) / static_cast<double>(prefetch_denom);

    uint64_t start = runtime_start_ns_.load(std::memory_order_relaxed);
    uint64_t end = runtime_end_ns_.load(std::memory_order_relaxed);
    if (end == 0) end = now_ns();
    if (start > 0 && end > start) {
        r.total_wall_time_ms = (end - start) / 1'000'000.0;
        r.ms_per_token = r.total_tokens == 0 ? 0.0 :
            r.total_wall_time_ms / static_cast<double>(r.total_tokens);
        r.tokens_per_sec = r.total_wall_time_ms == 0.0 ? 0.0 :
            static_cast<double>(r.total_tokens) / (r.total_wall_time_ms / 1000.0);
    }

    if (r.total_wall_time_ms > 0.0) {
        r.ssd_throughput_mb_s = (r.ssd_bytes / (1024.0 * 1024.0)) /
            (r.total_wall_time_ms / 1000.0);
    }

    {
        std::lock_guard<std::mutex> lock(latency_mu_);
        r.request_latency = LatencyStats::from(request_latencies_us_);
        r.load_latency = LatencyStats::from(load_latencies_us_);
        r.execution_latency = LatencyStats::from(execution_latencies_us_);
        r.end_to_end_token_latency = LatencyStats::from(token_end_to_end_us_);
    }

    return r;
}

bool TelemetryEngine::write_report() const {
    AggregateReport r = build_report();

    auto write_text = [&](const std::string& path) -> bool {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << "====================================================\n";
        f << "               ASEMA Aggregate Report               \n";
        f << "====================================================\n";
        f << "Total Tokens:               " << r.total_tokens << "\n";
        f << "Wall Time:                  " << std::fixed << std::setprecision(3)
          << r.total_wall_time_ms << " ms\n";
        f << "First-Token Latency:        " << std::setprecision(3)
          << r.first_token_latency_ms << " ms\n";
        f << "Tokens / Sec:               " << std::setprecision(2)
          << r.tokens_per_sec << "\n";
        f << "ms / Token:                 " << std::setprecision(3)
          << r.ms_per_token << "\n";
        f << "----------------------------------------------------\n";
        f << "Cache Hits / Misses:        " << r.cache_hits << " / "
          << r.cache_misses << "\n";
        f << "Cache Hit Rate:             " << std::setprecision(3)
          << (r.cache_hit_rate * 100.0) << "%\n";
        f << "Cache Evictions:            " << r.cache_evictions << "\n";
        f << "Cache Insertions:           " << r.cache_insertions << "\n";
        f << "RAM Bytes Used / Peak:      " << r.ram_bytes_used << " / "
          << r.ram_bytes_peak << " (capacity " << r.ram_capacity << ")\n";
        f << "SSD Bytes Read:             " << r.ssd_bytes
          << " (" << r.ssd_reads << " reads)\n";
        f << "SSD Throughput:             " << std::setprecision(2)
          << r.ssd_throughput_mb_s << " MB/s\n";
        f << "----------------------------------------------------\n";
        f << "Prefetch Requests:          " << r.prefetch_requests << "\n";
        f << "Prefetch Useful / Wasted:   " << r.prefetch_useful << " / "
          << r.prefetch_wasted << "\n";
        f << "Prefetch Cancelled:         " << r.prefetch_cancelled << "\n";
        f << "Prefetch Hit Rate:          " << std::setprecision(3)
          << (r.prefetch_hit_rate * 100.0) << "%\n";
        f << "Queue Depth (current/peak): " << r.queue_depth << " / "
          << r.queue_peak << "\n";
        f << "----------------------------------------------------\n";
        f << "Latency: Request (us)\n";
        f << "  count=" << r.request_latency.count
          << " avg=" << std::setprecision(2) << r.request_latency.avg_us
          << " p50=" << r.request_latency.p50_us
          << " p95=" << r.request_latency.p95_us
          << " p99=" << r.request_latency.p99_us << "\n";
        f << "Latency: Load (us)\n";
        f << "  count=" << r.load_latency.count
          << " avg=" << std::setprecision(2) << r.load_latency.avg_us
          << " p50=" << r.load_latency.p50_us
          << " p95=" << r.load_latency.p95_us
          << " p99=" << r.load_latency.p99_us << "\n";
        f << "Latency: Execution (us)\n";
        f << "  count=" << r.execution_latency.count
          << " avg=" << std::setprecision(2) << r.execution_latency.avg_us
          << " p50=" << r.execution_latency.p50_us
          << " p95=" << r.execution_latency.p95_us
          << " p99=" << r.execution_latency.p99_us << "\n";
        f << "Latency: End-to-End Token (us)\n";
        f << "  count=" << r.end_to_end_token_latency.count
          << " avg=" << std::setprecision(2) << r.end_to_end_token_latency.avg_us
          << " p50=" << r.end_to_end_token_latency.p50_us
          << " p95=" << r.end_to_end_token_latency.p95_us
          << " p99=" << r.end_to_end_token_latency.p99_us << "\n";
        f << "====================================================\n";
        return true;
    };

    auto write_json = [&](const std::string& path) -> bool {
        std::ofstream f(path);
        if (!f.is_open()) return false;
        f << "{\n"
          << "  \"total_tokens\": " << r.total_tokens << ",\n"
          << "  \"wall_time_ms\": " << r.total_wall_time_ms << ",\n"
          << "  \"first_token_latency_ms\": " << r.first_token_latency_ms << ",\n"
          << "  \"tokens_per_sec\": " << r.tokens_per_sec << ",\n"
          << "  \"ms_per_token\": " << r.ms_per_token << ",\n"
          << "  \"cache\": {\n"
          << "    \"hits\": " << r.cache_hits << ",\n"
          << "    \"misses\": " << r.cache_misses << ",\n"
          << "    \"hit_rate\": " << r.cache_hit_rate << ",\n"
          << "    \"evictions\": " << r.cache_evictions << ",\n"
          << "    \"insertions\": " << r.cache_insertions << "\n"
          << "  },\n"
          << "  \"ram\": {\n"
          << "    \"bytes_used\": " << r.ram_bytes_used << ",\n"
          << "    \"bytes_peak\": " << r.ram_bytes_peak << ",\n"
          << "    \"capacity\": " << r.ram_capacity << "\n"
          << "  },\n"
          << "  \"ssd\": {\n"
          << "    \"bytes\": " << r.ssd_bytes << ",\n"
          << "    \"reads\": " << r.ssd_reads << ",\n"
          << "    \"throughput_mb_s\": " << r.ssd_throughput_mb_s << "\n"
          << "  },\n"
          << "  \"prefetch\": {\n"
          << "    \"requests\": " << r.prefetch_requests << ",\n"
          << "    \"useful\": " << r.prefetch_useful << ",\n"
          << "    \"wasted\": " << r.prefetch_wasted << ",\n"
          << "    \"cancelled\": " << r.prefetch_cancelled << ",\n"
          << "    \"hit_rate\": " << r.prefetch_hit_rate << "\n"
          << "  },\n"
          << "  \"queue\": {\n"
          << "    \"depth\": " << r.queue_depth << ",\n"
          << "    \"peak\": " << r.queue_peak << "\n"
          << "  }\n"
          << "}\n";
        return true;
    };

    bool ok = true;
    if (!cfg_.report_text_path.empty()) ok &= write_text(cfg_.report_text_path);
    if (!cfg_.report_json_path.empty()) ok &= write_json(cfg_.report_json_path);
    return ok;
}

} // namespace asema
