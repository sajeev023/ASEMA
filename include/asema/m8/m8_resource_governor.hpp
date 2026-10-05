#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace asema {
namespace m8 {

enum class GovernorState {
    SAFE,       // Available RAM > 12 GB, Working set healthy
    WARNING,    // Available RAM <= 12 GB, throttle cache growth and concurrency
    CRITICAL    // Available RAM < 10 GB or approaching ceilings, evict and throttle
};

struct SystemResourceSnapshot {
    size_t total_phys_ram_mb{0};
    size_t avail_phys_ram_mb{0};
    size_t asema_ram_working_set_mb{0};
    size_t asema_ram_peak_mb{0};
    size_t ram_ceiling_mb{20480};

    size_t total_vram_mb{8192};
    size_t asema_vram_allocated_mb{0};
    size_t vram_ceiling_mb{5120};

    GovernorState state{GovernorState::SAFE};
    std::string state_description;
};

class M8ResourceGovernor {
public:
    static M8ResourceGovernor& instance();

    // Query live hardware telemetry and evaluate governor state
    SystemResourceSnapshot probe(size_t vram_allocated_bytes = 0);

    // Absolute ceilings (not targets)
    static constexpr size_t ABSOLUTE_RAM_CEILING_MB = 20480;
    static constexpr size_t ABSOLUTE_VRAM_CEILING_MB = 5120;

    // Safety margins for Windows 11, display driver, two monitors, and browser
    static constexpr size_t MIN_SYSTEM_FREE_RAM_MB = 10240;      // 10 GB absolute floor
    static constexpr size_t WARNING_SYSTEM_FREE_RAM_MB = 12288;  // 12 GB warning threshold

    bool can_grow_cache(size_t additional_mb) const;
    bool is_critical() const;
    bool is_warning() const;

    // Clean shutdown support (Ctrl+C handling)
    static void install_signal_handlers();
    static bool is_shutdown_requested();
    static void request_shutdown();

private:
    M8ResourceGovernor() = default;
    static std::atomic<bool> shutdown_requested_;
};

} // namespace m8
} // namespace asema
