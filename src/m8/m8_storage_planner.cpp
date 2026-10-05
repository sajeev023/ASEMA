#include "asema/m8/m8_storage_planner.hpp"
#include <windows.h>

namespace asema {
namespace m8 {

StoragePlanResult M8StoragePlanner::plan_dual_volume_allocation(
    size_t free_bytes_d,
    size_t free_bytes_e,
    size_t checkpoint_bytes) {

    StoragePlanResult res;
    res.total_required_bytes = checkpoint_bytes;
    res.total_free_bytes = free_bytes_d + free_bytes_e;

    // Checkpoint partition: 32 shards on D:, 16 shards on E:
    // 32/48 = 66.67% of 510 GB = 340 GB
    // 16/48 = 33.33% of 510 GB = 170 GB
    size_t req_d = (checkpoint_bytes * 32) / 48;
    size_t req_e = checkpoint_bytes - req_d;

    res.plan_d.drive_letter = "D:";
    res.plan_d.total_free_bytes = free_bytes_d;
    res.plan_d.required_bytes = req_d;
    res.plan_d.start_shard = 1;
    res.plan_d.end_shard = 32;
    res.plan_d.is_feasible = (free_bytes_d > req_d + 10ULL * 1024 * 1024 * 1024); // 10 GB safety buffer
    res.plan_d.safety_margin_bytes = (free_bytes_d > req_d) ? (free_bytes_d - req_d) : 0;

    res.plan_e.drive_letter = "E:";
    res.plan_e.total_free_bytes = free_bytes_e;
    res.plan_e.required_bytes = req_e;
    res.plan_e.start_shard = 33;
    res.plan_e.end_shard = 48;
    res.plan_e.is_feasible = (free_bytes_e > req_e + 10ULL * 1024 * 1024 * 1024); // 10 GB safety buffer
    res.plan_e.safety_margin_bytes = (free_bytes_e > req_e) ? (free_bytes_e - req_e) : 0;

    if (res.plan_d.is_feasible && res.plan_e.is_feasible) {
        res.overall_feasible = true;
    } else {
        res.overall_feasible = false;
        res.error_message = "Insufficient free space on physical NVMe drives for complete checkpoint allocation.";
    }

    return res;
}

StoragePlanResult M8StoragePlanner::probe_and_plan() {
    ULARGE_INTEGER free_d{}, total_d{};
    ULARGE_INTEGER free_e{}, total_e{};

    GetDiskFreeSpaceExA("D:\\", &free_d, &total_d, nullptr);
    GetDiskFreeSpaceExA("E:\\", &free_e, &total_e, nullptr);

    return plan_dual_volume_allocation(free_d.QuadPart, free_e.QuadPart);
}

} // namespace m8
} // namespace asema
