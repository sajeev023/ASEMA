#include "asema/m8/m8_resource_governor.hpp"
#include <windows.h>
#include <psapi.h>
#include <iostream>

namespace asema {
namespace m8 {

std::atomic<bool> M8ResourceGovernor::shutdown_requested_{false};

static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            std::cerr << "\n[ASEMA GOVERNOR] Clean shutdown signal received (Ctrl+C). Draining operations safely...\n";
            M8ResourceGovernor::request_shutdown();
            return TRUE;
        default:
            return FALSE;
    }
}

void M8ResourceGovernor::install_signal_handlers() {
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
}

bool M8ResourceGovernor::is_shutdown_requested() {
    return shutdown_requested_.load(std::memory_order_relaxed);
}

void M8ResourceGovernor::request_shutdown() {
    shutdown_requested_.store(true, std::memory_order_relaxed);
}

M8ResourceGovernor& M8ResourceGovernor::instance() {
    static M8ResourceGovernor inst;
    return inst;
}

SystemResourceSnapshot M8ResourceGovernor::probe(size_t vram_allocated_bytes) {
    SystemResourceSnapshot snap;
    snap.ram_ceiling_mb = ABSOLUTE_RAM_CEILING_MB;
    snap.vram_ceiling_mb = ABSOLUTE_VRAM_CEILING_MB;

    // 1. Physical system memory
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        snap.total_phys_ram_mb = mem.ullTotalPhys / (1024 * 1024);
        snap.avail_phys_ram_mb = mem.ullAvailPhys / (1024 * 1024);
    }

    // 2. ASEMA process working set
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        snap.asema_ram_working_set_mb = pmc.WorkingSetSize / (1024 * 1024);
        snap.asema_ram_peak_mb = pmc.PeakWorkingSetSize / (1024 * 1024);
    }

    // 3. VRAM
    snap.asema_vram_allocated_mb = vram_allocated_bytes / (1024 * 1024);

    // 4. Governor State Evaluation
    if (snap.avail_phys_ram_mb < MIN_SYSTEM_FREE_RAM_MB ||
        snap.asema_ram_working_set_mb >= ABSOLUTE_RAM_CEILING_MB ||
        snap.asema_vram_allocated_mb >= ABSOLUTE_VRAM_CEILING_MB) {
        snap.state = GovernorState::CRITICAL;
        snap.state_description = "CRITICAL: System available RAM < 10 GB or ceiling reached. Evict non-essential memory.";
    } else if (snap.avail_phys_ram_mb <= WARNING_SYSTEM_FREE_RAM_MB ||
               snap.asema_ram_working_set_mb >= 16384 ||
               snap.asema_vram_allocated_mb >= 4096) {
        snap.state = GovernorState::WARNING;
        snap.state_description = "WARNING: System memory pressure rising. Halt cache growth.";
    } else {
        snap.state = GovernorState::SAFE;
        snap.state_description = "SAFE: System resources healthy. Optimization allowed.";
    }

    return snap;
}

bool M8ResourceGovernor::can_grow_cache(size_t additional_mb) const {
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (!GlobalMemoryStatusEx(&mem)) return false;

    size_t avail_mb = mem.ullAvailPhys / (1024 * 1024);
    if (avail_mb <= WARNING_SYSTEM_FREE_RAM_MB + additional_mb) {
        return false;
    }

    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        size_t ws_mb = pmc.WorkingSetSize / (1024 * 1024);
        if (ws_mb + additional_mb >= ABSOLUTE_RAM_CEILING_MB) {
            return false;
        }
    }
    return true;
}

bool M8ResourceGovernor::is_critical() const {
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        if ((mem.ullAvailPhys / (1024 * 1024)) < MIN_SYSTEM_FREE_RAM_MB) {
            return true;
        }
    }
    return false;
}

bool M8ResourceGovernor::is_warning() const {
    MEMORYSTATUSEX mem{};
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        if ((mem.ullAvailPhys / (1024 * 1024)) <= WARNING_SYSTEM_FREE_RAM_MB) {
            return true;
        }
    }
    return false;
}

} // namespace m8
} // namespace asema
