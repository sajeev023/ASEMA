#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>

using Microsoft::WRL::ComPtr;

namespace asema {
namespace m8 {

enum class GPUNodeType : uint8_t {
    COMPUTE_DISPATCH,
    RESOURCE_COPY,
    STAGING_UPLOAD,
    STAGING_READBACK,
    BARRIER
};

struct GPUDAGNode {
    uint64_t node_id{0};
    std::string name;
    GPUNodeType type{GPUNodeType::COMPUTE_DISPATCH};
    std::vector<uint64_t> dependencies; // Node IDs that must execute prior
    std::function<void(ID3D11DeviceContext* ctx)> execute_fn;
    bool executed{false};
};

struct GPUSchedulerMetrics {
    uint64_t total_nodes_executed{0};
    uint64_t flushes_avoided{0};
    uint64_t maps_deferred{0};
    double cpu_wait_saved_ms{0.0};
    double total_gpu_batch_latency_ms{0.0};
};

class M8GPUCommandScheduler {
public:
    explicit M8GPUCommandScheduler(ID3D11DeviceContext* context);
    ~M8GPUCommandScheduler();

    uint64_t add_compute_node(const std::string& name,
                              const std::vector<uint64_t>& dependencies,
                              std::function<void(ID3D11DeviceContext* ctx)> dispatch_fn);

    uint64_t add_copy_node(const std::string& name,
                           const std::vector<uint64_t>& dependencies,
                           std::function<void(ID3D11DeviceContext* ctx)> copy_fn);

    uint64_t add_readback_node(const std::string& name,
                               const std::vector<uint64_t>& dependencies,
                               std::function<void(ID3D11DeviceContext* ctx)> readback_fn);

    // Executes the entire DAG in dependency topological order with a single pipeline submission
    void execute_dag();

    void clear_dag();
    GPUSchedulerMetrics get_metrics() const;
    void reset_metrics();

private:
    ID3D11DeviceContext* context_{nullptr};
    std::vector<GPUDAGNode> nodes_;
    uint64_t next_node_id_{1};
    GPUSchedulerMetrics metrics_;
};

} // namespace m8
} // namespace asema
