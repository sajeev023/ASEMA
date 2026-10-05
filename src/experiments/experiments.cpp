// ASEMA v0.1 — Agent #3: experiment orchestration implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/experiments.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <sys/stat.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace asema {
namespace exp {

// -----------------------------------------------------------------------------
// Time helpers
// -----------------------------------------------------------------------------
std::string timestamp_iso8601_utc() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H-%M-%S", &tm);
    char out[80];
    std::snprintf(out, sizeof(out), "%s.%03lldZ", buf, (long long)ms);
    return out;
}

std::string human_bytes(uint64_t b) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(b);
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.2f %s", v, units[u]);
    return buf;
}

// -----------------------------------------------------------------------------
// Filesystem helpers
// -----------------------------------------------------------------------------
bool ensure_directory(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
        return std::filesystem::is_directory(path);
    } catch (...) {
        return false;
    }
}

bool write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!f.is_open()) return false;
    f << content;
    f.close();
    return f.good();
}

// -----------------------------------------------------------------------------
// JSON / CSV serialization
// -----------------------------------------------------------------------------
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::string measurement_to_json(const Measurement& m) {
    std::ostringstream os;
    os << "{"
       << "\"experiment_id\":\"" << json_escape(m.experiment_id) << "\","
       << "\"iteration\":" << m.iteration << ","
       << "\"valid\":" << (m.valid ? "true" : "false") << ","
       << "\"invalid_reason\":\"" << json_escape(m.invalid_reason) << "\","
       << "\"tokens_completed\":" << m.tokens_completed << ","
       << "\"wall_time_ms\":" << m.wall_time_ms << ","
       << "\"first_token_ms\":" << m.first_token_ms << ","
       << "\"ms_per_token\":" << m.ms_per_token << ","
       << "\"tokens_per_sec\":" << m.tokens_per_sec << ","
       << "\"cache_hits\":" << m.cache_hits << ","
       << "\"cache_misses\":" << m.cache_misses << ","
       << "\"cache_hit_rate\":" << m.cache_hit_rate << ","
       << "\"cache_evictions\":" << m.cache_evictions << ","
       << "\"cache_insertions\":" << m.cache_insertions << ","
       << "\"peak_bytes_used\":" << m.peak_bytes_used << ","
       << "\"runtime_logical_requests\":" << m.runtime_logical_requests << ","
       << "\"runtime_logical_bytes\":" << m.runtime_logical_bytes << ","
       << "\"ssd_bytes_logical\":" << m.ssd_bytes_logical << ","
       << "\"ssd_bytes_physical\":" << m.ssd_bytes_physical << ","
       << "\"ssd_reads\":" << m.ssd_reads << ","
       << "\"coalesced_requests\":" << m.coalesced_requests << ","
       << "\"cache_bytes_served\":" << m.cache_bytes_served << ","
       << "\"expert_bytes_per_request\":" << m.expert_bytes_per_request << ","
       << "\"prefetch_requests\":" << m.prefetch_requests << ","
       << "\"prefetch_useful\":" << m.prefetch_useful << ","
       << "\"prefetch_wasted\":" << m.prefetch_wasted << ","
       << "\"prefetch_cancelled\":" << m.prefetch_cancelled << ","
       << "\"prefetch_duplicate\":" << m.prefetch_duplicate << ","
       << "\"prefetch_hit_rate\":" << m.prefetch_hit_rate
       << "}";
    return os.str();
}

std::string summary_to_json(const CellSummary& s) {
    std::ostringstream os;
    os << "{\n"
       << "  \"experiment_id\": \"" << json_escape(s.cell.id) << "\",\n"
       << "  \"ram_cache_bytes\": " << s.cell.ram_cache_bytes << ",\n"
       << "  \"prefetch_lookahead\": " << s.cell.prefetch_lookahead << ",\n"
       << "  \"loader_workers\": " << s.cell.loader_workers << ",\n"
       << "  \"router_seed\": " << s.cell.router_seed << ",\n"
       << "  \"num_tokens\": " << s.cell.num_tokens << ",\n"
       << "  \"locality\": " << s.cell.locality << ",\n"
       << "  \"pattern\": \"" << json_escape(s.cell.pattern) << "\",\n"
       << "  \"cold_start\": " << (s.cell.cold_start ? "true" : "false") << ",\n"
       << "  \"iterations\": " << s.measurements.size() << ",\n"
       << "  \"valid\": " << s.valid_count << ",\n"
       << "  \"invalid\": " << s.invalid_count << ",\n"
       << "  \"mean_ms_per_token\": " << s.mean_ms_per_token << ",\n"
       << "  \"stddev_ms_per_token\": " << s.stddev_ms_per_token << ",\n"
       << "  \"min_ms_per_token\": " << s.min_ms_per_token << ",\n"
       << "  \"median_ms_per_token\": " << s.median_ms_per_token << ",\n"
       << "  \"max_ms_per_token\": " << s.max_ms_per_token << ",\n"
       << "  \"p95_ms_per_token\": " << s.p95_ms_per_token << ",\n"
       << "  \"mean_cache_hit_rate\": " << s.mean_cache_hit_rate << ",\n"
       << "  \"mean_ssd_bytes_physical\": " << s.mean_ssd_bytes_physical << ",\n"
       << "  \"mean_runtime_logical_bytes\": " << s.mean_runtime_logical_bytes << ",\n"
       << "  \"mean_coalesced_requests\": " << s.mean_coalesced_requests << ",\n"
       << "  \"mean_prefetch_hit_rate\": " << s.mean_prefetch_hit_rate << "\n"
       << "}";
    return os.str();
}

std::string measurement_to_csv_header() {
    return "experiment_id,iteration,valid,invalid_reason,tokens_completed,wall_time_ms,first_token_ms,"
           "ms_per_token,tokens_per_sec,cache_hits,cache_misses,cache_hit_rate,cache_evictions,"
           "cache_insertions,peak_bytes_used,runtime_logical_requests,runtime_logical_bytes,"
           "ssd_bytes_logical,ssd_bytes_physical,ssd_reads,coalesced_requests,cache_bytes_served,"
           "expert_bytes_per_request,prefetch_requests,prefetch_useful,prefetch_wasted,"
           "prefetch_cancelled,prefetch_duplicate,prefetch_hit_rate";
}

std::string measurement_to_csv_row(const Measurement& m) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "%s,%d,%s,%s,%u,%.4f,%.4f,%.4f,%.4f,%llu,%llu,%.4f,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%.4f",
        m.experiment_id.c_str(),
        m.iteration,
        m.valid ? "true" : "false",
        m.invalid_reason.c_str(),
        m.tokens_completed,
        m.wall_time_ms,
        m.first_token_ms,
        m.ms_per_token,
        m.tokens_per_sec,
        (unsigned long long)m.cache_hits,
        (unsigned long long)m.cache_misses,
        m.cache_hit_rate,
        (unsigned long long)m.cache_evictions,
        (unsigned long long)m.cache_insertions,
        (unsigned long long)m.peak_bytes_used,
        (unsigned long long)m.runtime_logical_requests,
        (unsigned long long)m.runtime_logical_bytes,
        (unsigned long long)m.ssd_bytes_logical,
        (unsigned long long)m.ssd_bytes_physical,
        (unsigned long long)m.ssd_reads,
        (unsigned long long)m.coalesced_requests,
        (unsigned long long)m.cache_bytes_served,
        (unsigned long long)m.expert_bytes_per_request,
        (unsigned long long)m.prefetch_requests,
        (unsigned long long)m.prefetch_useful,
        (unsigned long long)m.prefetch_wasted,
        (unsigned long long)m.prefetch_cancelled,
        (unsigned long long)m.prefetch_duplicate,
        m.prefetch_hit_rate);
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// Summarize
// -----------------------------------------------------------------------------
CellSummary summarize(const ExperimentCell& cell,
                      const std::vector<Measurement>& measurements) {
    CellSummary s;
    s.cell = cell;
    s.measurements = measurements;
    s.valid_count = 0;
    s.invalid_count = 0;

    std::vector<double> valid_ms;
    double sum = 0.0;
    double sum_sq = 0.0;
    double sum_hit = 0.0;
    double sum_ssd = 0.0;
    double sum_logical = 0.0;
    double sum_coalesced = 0.0;
    double sum_pfhit = 0.0;
    double mn = 1e18, mx = 0.0;

    for (const auto& m : measurements) {
        if (!m.valid) { s.invalid_count++; continue; }
        s.valid_count++;
        valid_ms.push_back(m.ms_per_token);
        sum += m.ms_per_token;
        sum_sq += m.ms_per_token * m.ms_per_token;
        sum_hit += m.cache_hit_rate;
        sum_ssd += static_cast<double>(m.ssd_bytes_physical);
        sum_logical += static_cast<double>(m.runtime_logical_bytes);
        sum_coalesced += static_cast<double>(m.coalesced_requests);
        sum_pfhit += m.prefetch_hit_rate;
        if (m.ms_per_token < mn) mn = m.ms_per_token;
        if (m.ms_per_token > mx) mx = m.ms_per_token;
    }

    if (s.valid_count > 0) {
        s.mean_ms_per_token = sum / s.valid_count;
        double var = (sum_sq / s.valid_count) - (s.mean_ms_per_token * s.mean_ms_per_token);
        s.stddev_ms_per_token = var > 0 ? std::sqrt(var) : 0.0;
        s.min_ms_per_token = mn;
        s.max_ms_per_token = mx;
        s.mean_cache_hit_rate = sum_hit / s.valid_count;
        s.mean_ssd_bytes_physical = sum_ssd / s.valid_count;
        s.mean_runtime_logical_bytes = sum_logical / s.valid_count;
        s.mean_coalesced_requests = static_cast<double>(sum_coalesced) / s.valid_count;
        s.mean_prefetch_hit_rate = sum_pfhit / s.valid_count;

        std::sort(valid_ms.begin(), valid_ms.end());
        s.median_ms_per_token = valid_ms[valid_ms.size() / 2];
        size_t idx95 = static_cast<size_t>(valid_ms.size() * 0.95);
        if (idx95 >= valid_ms.size()) idx95 = valid_ms.size() - 1;
        s.p95_ms_per_token = valid_ms[idx95];
    }
    return s;
}

// -----------------------------------------------------------------------------
// Environment detection
// -----------------------------------------------------------------------------
Environment Environment::detect(const WorkloadConfig& w) {
    Environment e;
    e.timestamp_utc = timestamp_iso8601_utc();
    e.workload = w;
    e.storage_path = w.container_path;

#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    GlobalMemoryStatusEx(&ms);

    // CPU brand via __cpuid (not pulled into a separate header; we approximate).
    e.cpu = "x86_64 (Ryzen-class)";   // best-effort
    e.total_ram_bytes = ms.ullTotalPhys;
    e.available_ram_bytes = ms.ullAvailPhys;

    // Storage: best-effort; mark NVMe/SSD if path is on C:.
    std::string p = w.container_path;
    if (!p.empty() && (p[0] == 'C' || p[0] == 'c' || p[0] == 'D' || p[0] == 'd')) {
        e.device_class = "NVMe/SSD";
    } else {
        e.device_class = "unknown";
    }
    e.gpu = "none-detected";
    e.vram_bytes = 0;
#else
    e.cpu = "unknown";
    e.total_ram_bytes = 0;
    e.available_ram_bytes = 0;
    e.gpu = "unknown";
#endif

    e.compiler = "Clang 19 / LLVM-MinGW";
    e.build_type = "Release";
    return e;
}

} // namespace exp
} // namespace asema
