#pragma once

#include "asema/m8/m8_model_runner.hpp"
#include <string>
#include <vector>

namespace asema {
namespace m8 {

struct FaultTestResult {
    std::string test_name;
    bool passed{false};
    std::string message;
};

class M8FaultTolerance {
public:
    static FaultTestResult test_missing_shard_handling();
    static FaultTestResult test_corrupted_shard_detection();
    static FaultTestResult test_volume_dropout_resilience();
    static FaultTestResult test_gpu_device_fallback(M8ModelRunner& runner);
    static FaultTestResult test_memory_allocation_limits();

    static std::vector<FaultTestResult> run_full_fault_battery(M8ModelRunner& runner);
};

} // namespace m8
} // namespace asema
