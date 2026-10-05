#pragma once

#include "asema/m8/m8_paths.hpp"
#include <string>
#include <vector>
#include <cstdint>

namespace asema {
namespace m8 {

struct DoctorCheckItem {
    std::string category;
    std::string name;
    bool passed{false};
    std::string details;
    std::string hint;   // what to do about it when passed == false
};

struct DoctorReport {
    bool all_passed{false};
    std::vector<DoctorCheckItem> items;
    std::string summary;
};

class M8ModelDoctor {
public:
    static DoctorReport run_diagnostics(
        const std::string& hf_root = asema::m8::paths::hf_dir(),
        const std::string& vol_primary = asema::m8::paths::primary_shards(),
        const std::string& vol_secondary = asema::m8::paths::secondary_shards());
};

} // namespace m8
} // namespace asema
