#include "asema/m8/m8_memory_tracker.hpp"

#include <windows.h>
#include <psapi.h>
#include <sstream>
#include <iomanip>
#include <algorithm>

#pragma comment(lib, "psapi.lib")

namespace asema {
namespace m8 {

M8MemoryTracker& M8MemoryTracker::instance() {
    static M8MemoryTracker s_instance;
    return s_instance;
}

uint64_t M8MemoryTracker::record_allocation(const std::string& tag,
                                           MemoryCategory category,
                                           AllocationLocation location,
                                           AllocationLifetime lifetime,
                                           const std::string& owner,
                                           const std::string& reason,
                                           size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t id = next_id_++;
    MemoryRecord rec;
    rec.tag = tag;
    rec.category = category;
    rec.location = location;
    rec.lifetime = lifetime;
    rec.owner = owner;
    rec.reason = reason;
    rec.size_bytes = size_bytes;
    rec.allocation_id = id;
    rec.timestamp = std::chrono::high_resolution_clock::now();

    active_allocations_[id] = rec;
    category_totals_[category] += size_bytes;
    location_totals_[location] += size_bytes;

    return id;
}

void M8MemoryTracker::record_deallocation(uint64_t allocation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_allocations_.find(allocation_id);
    if (it != active_allocations_.end()) {
        category_totals_[it->second.category] -= it->second.size_bytes;
        location_totals_[it->second.location] -= it->second.size_bytes;
        active_allocations_.erase(it);
    }
}

size_t M8MemoryTracker::get_category_bytes(MemoryCategory category) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = category_totals_.find(category);
    return (it != category_totals_.end()) ? it->second : 0;
}

size_t M8MemoryTracker::get_location_bytes(AllocationLocation location) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = location_totals_.find(location);
    return (it != location_totals_.end()) ? it->second : 0;
}

size_t M8MemoryTracker::get_total_tracked_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& kv : category_totals_) {
        total += kv.second;
    }
    return total;
}

OSProcessMemoryInfo M8MemoryTracker::query_os_memory() const {
    OSProcessMemoryInfo info{};
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        info.working_set_bytes = pmc.WorkingSetSize;
        info.peak_working_set_bytes = pmc.PeakWorkingSetSize;
        info.private_bytes = pmc.PrivateUsage;
        info.peak_private_bytes = pmc.PeakPagefileUsage;
    }

    PERFORMANCE_INFORMATION perf{};
    if (GetPerformanceInfo(&perf, sizeof(perf))) {
        info.system_cache_bytes = perf.SystemCache * perf.PageSize;
        info.available_physical_bytes = perf.PhysicalAvailable * perf.PageSize;
        info.total_physical_bytes = perf.PhysicalTotal * perf.PageSize;
    }

    return info;
}

std::vector<MemoryRecord> M8MemoryTracker::get_active_allocations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<MemoryRecord> result;
    result.reserve(active_allocations_.size());
    for (const auto& kv : active_allocations_) {
        result.push_back(kv.second);
    }
    return result;
}

std::unordered_map<MemoryCategory, size_t> M8MemoryTracker::get_category_breakdown() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return category_totals_;
}

void M8MemoryTracker::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    active_allocations_.clear();
    category_totals_.clear();
    location_totals_.clear();
    next_id_ = 1;
}

std::string M8MemoryTracker::generate_markdown_report() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream ss;
    OSProcessMemoryInfo os_info = query_os_memory();

    ss << "# ASEMA M8.22 — MEMORY FORENSICS REPORT\n\n";
    ss << "**Objective:** Complete byte-level accounting across RAM, VRAM, and OS subsystem memory.\n\n";

    ss << "## 1. Operating System Process Memory (Live Query)\n\n";
    ss << "| Metric | Measured Bytes | Megabytes (MB) | Notes |\n";
    ss << "| :--- | :--- | :--- | :--- |\n";
    ss << "| **Working Set (Current)** | " << os_info.working_set_bytes << " B | "
       << std::fixed << std::setprecision(2) << (os_info.working_set_bytes / (1024.0 * 1024.0)) << " MB | Physical RAM mapped to process |\n";
    ss << "| **Peak Working Set** | " << os_info.peak_working_set_bytes << " B | "
       << (os_info.peak_working_set_bytes / (1024.0 * 1024.0)) << " MB | Maximum physical RAM footprint |\n";
    ss << "| **Private Bytes (Commit)** | " << os_info.private_bytes << " B | "
       << (os_info.private_bytes / (1024.0 * 1024.0)) << " MB | Committed virtual address space |\n";
    ss << "| **Peak Commit** | " << os_info.peak_private_bytes << " B | "
       << (os_info.peak_private_bytes / (1024.0 * 1024.0)) << " MB | Maximum committed memory |\n";
    ss << "| **System Standby/Cache** | " << os_info.system_cache_bytes << " B | "
       << (os_info.system_cache_bytes / (1024.0 * 1024.0)) << " MB | OS file cache across all processes |\n";
    ss << "| **Physical RAM Available** | " << os_info.available_physical_bytes << " B | "
       << (os_info.available_physical_bytes / (1024.0 * 1024.0)) << " MB | Uncommitted system physical RAM |\n\n";

    ss << "## 2. Category Breakdown\n\n";
    ss << "| Category | Location | Allocated Bytes | MB | % of Tracked |\n";
    ss << "| :--- | :--- | :--- | :--- | :--- |\n";

    size_t total_tracked = 0;
    for (const auto& kv : category_totals_) total_tracked += kv.second;

    for (int i = 0; i < static_cast<int>(MemoryCategory::COUNT); ++i) {
        MemoryCategory cat = static_cast<MemoryCategory>(i);
        size_t b = 0;
        auto it = category_totals_.find(cat);
        if (it != category_totals_.end()) b = it->second;

        double mb = b / (1024.0 * 1024.0);
        double pct = (total_tracked > 0) ? (b * 100.0 / total_tracked) : 0.0;
        const char* loc_str = (cat == MemoryCategory::GPU_BUFFERS) ? "GPU VRAM" :
                              (cat == MemoryCategory::STAGING_BUFFERS) ? "Host Pinned" : "CPU RAM";

        ss << "| **" << memory_category_name(cat) << "** | " << loc_str << " | "
           << b << " B | " << std::fixed << std::setprecision(2) << mb << " MB | "
           << std::setprecision(1) << pct << "% |\n";
    }
    ss << "| **TOTAL TRACKED** | — | " << total_tracked << " B | "
       << (total_tracked / (1024.0 * 1024.0)) << " MB | 100.0% |\n\n";

    ss << "## 3. Granular Allocation Catalog\n\n";
    ss << "| Tag / Buffer | Category | Location | Lifetime | Size | Owner | Rationale |\n";
    ss << "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n";

    std::vector<MemoryRecord> sorted_recs;
    sorted_recs.reserve(active_allocations_.size());
    for (const auto& kv : active_allocations_) sorted_recs.push_back(kv.second);
    std::sort(sorted_recs.begin(), sorted_recs.end(), [](const MemoryRecord& a, const MemoryRecord& b) {
        return a.size_bytes > b.size_bytes;
    });

    for (const auto& r : sorted_recs) {
        ss << "| `" << r.tag << "` | " << memory_category_name(r.category) << " | "
           << allocation_location_name(r.location) << " | " << allocation_lifetime_name(r.lifetime) << " | "
           << std::fixed << std::setprecision(2) << (r.size_bytes / (1024.0 * 1024.0)) << " MB | "
           << r.owner << " | " << r.reason << " |\n";
    }

    return ss.str();
}

} // namespace m8
} // namespace asema
