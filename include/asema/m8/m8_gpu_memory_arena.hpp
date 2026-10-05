#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

using Microsoft::WRL::ComPtr;

namespace asema {
namespace m8 {

struct ArenaBlock {
    size_t offset{0};
    size_t size{0};
    bool in_use{false};
    uint64_t allocation_id{0};
    std::string tag;
};

struct ArenaSuballocation {
    uint64_t allocation_id{0};
    ID3D11Buffer* buffer{nullptr};
    size_t byte_offset{0};
    size_t byte_size{0};
    ComPtr<ID3D11ShaderResourceView> srv;
    ComPtr<ID3D11UnorderedAccessView> uav;
};

struct ArenaTelemetry {
    size_t total_slab_bytes{0};
    size_t allocated_bytes{0};
    size_t peak_allocated_bytes{0};
    size_t free_bytes{0};
    size_t active_allocations{0};
    uint64_t allocation_requests{0};
    uint64_t suballoc_hits{0};
    double suballoc_latency_us{0.0};
};

class M8GPUMemoryArena {
public:
    explicit M8GPUMemoryArena(ID3D11Device* device, size_t slab_size_bytes = 32 * 1024 * 1024);
    ~M8GPUMemoryArena();

    bool initialize();

    // 256-byte aligned persistent suballocation
    ArenaSuballocation suballocate(size_t size_bytes, const std::string& tag = "");
    void free_suballocation(uint64_t allocation_id);

    // Staging buffer pool reuse (avoids repeated staging buffer creation)
    ComPtr<ID3D11Buffer> get_staging_buffer(size_t size_bytes);
    void release_staging_buffer(ComPtr<ID3D11Buffer> buf, size_t size_bytes);

    ArenaTelemetry get_telemetry() const;
    void reset();

private:
    ID3D11Device* device_{nullptr};
    size_t slab_size_bytes_{32 * 1024 * 1024}; // 32 MB default persistent slab
    mutable std::mutex mutex_;

    ComPtr<ID3D11Buffer> slab_buffer_;
    std::vector<ArenaBlock> blocks_;
    uint64_t next_alloc_id_{1};

    // Staging buffer cache: size -> list of free reusable staging buffers
    std::unordered_map<size_t, std::vector<ComPtr<ID3D11Buffer>>> staging_pool_;

    ArenaTelemetry telemetry_;
    void coalesce_free_blocks();
};

} // namespace m8
} // namespace asema
