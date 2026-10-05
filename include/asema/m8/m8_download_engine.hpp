#pragma once

#include "asema/m8/m8_paths.hpp"
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

namespace asema {
namespace m8 {

struct DownloadProgress {
    std::string current_file;
    uint64_t bytes_downloaded{0};
    uint64_t total_bytes{0};
    double speed_mbps{0.0};
    double percent{0.0};
    int shards_completed{0};
    int total_shards{48};
};

class M8DownloadEngine {
public:
    explicit M8DownloadEngine(const std::string& repo_id = "deepseek-ai/DeepSeek-V4.1-Flash",
                              const std::string& revision = "main");

    // Plan download destinations based on dual-volume layout
    bool plan_destinations(const std::string& vol_d = asema::m8::paths::primary_shards(),
                           const std::string& vol_e = asema::m8::paths::secondary_shards());

    // Check existing files and compute resume offsets
    bool inspect_resume_state(int& out_completed, int& out_pending, size_t& out_bytes_present);

    // Download a single shard with resume support (mock/real transport)
    bool download_shard(int shard_id,
                        std::function<void(const DownloadProgress&)> on_progress = nullptr);

    const std::string& repo_id() const { return repo_id_; }
    const std::string& revision() const { return revision_; }

private:
    std::string repo_id_;
    std::string revision_;
    std::string vol_d_;
    std::string vol_e_;
};

} // namespace m8
} // namespace asema
