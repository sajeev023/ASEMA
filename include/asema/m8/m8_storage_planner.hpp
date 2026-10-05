#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace asema {
namespace m8 {

struct VolumePlan {
    std::string drive_letter;
    size_t total_free_bytes{0};
    size_t required_bytes{0};
    size_t safety_margin_bytes{0};
    int start_shard{1};
    int end_shard{48};
    bool is_feasible{false};
};

struct StoragePlanResult {
    bool overall_feasible{false};
    std::string error_message;
    VolumePlan plan_d;
    VolumePlan plan_e;
    size_t total_required_bytes{0};
    size_t total_free_bytes{0};
};

class M8StoragePlanner {
public:
    static StoragePlanResult plan_dual_volume_allocation(
        size_t free_bytes_d,
        size_t free_bytes_e,
        size_t checkpoint_bytes = 510000000000ULL);

    static StoragePlanResult probe_and_plan();
};

} // namespace m8
} // namespace asema
