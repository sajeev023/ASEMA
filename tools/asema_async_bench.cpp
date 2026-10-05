#include "asema/m8/m8_paths.hpp"
#include "../include/asema/manifest.hpp"
#include "../include/asema/storage.hpp"
#include "../include/asema/async_loader.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <random>

struct LatencyStats {
    double min_ms{0.0};
    double max_ms{0.0};
    double avg_ms{0.0};
    double p50_ms{0.0};
    double p95_ms{0.0};
    double p99_ms{0.0};
    double total_time_ms{0.0};
    double throughput_mb_s{0.0};
    size_t total_requests{0};
    size_t total_bytes{0};

    static LatencyStats compute(std::vector<double>& latencies, size_t bytes, double wall_time_ms) {
        LatencyStats s;
        if (latencies.empty()) return s;
        std::sort(latencies.begin(), latencies.end());
        s.total_requests = latencies.size();
        s.total_bytes = bytes;
        s.total_time_ms = wall_time_ms;
        s.min_ms = latencies.front();
        s.max_ms = latencies.back();
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        s.avg_ms = sum / latencies.size();

        size_t idx50 = static_cast<size_t>(latencies.size() * 0.50);
        size_t idx95 = static_cast<size_t>(latencies.size() * 0.95);
        size_t idx99 = static_cast<size_t>(latencies.size() * 0.99);

        s.p50_ms = latencies[std::min(idx50, latencies.size() - 1)];
        s.p95_ms = latencies[std::min(idx95, latencies.size() - 1)];
        s.p99_ms = latencies[std::min(idx99, latencies.size() - 1)];
        s.throughput_mb_s = (bytes / (1024.0 * 1024.0)) / (wall_time_ms / 1000.0);
        return s;
    }

    void print(const std::string& title) const {
        std::cout << "----------------------------------------------------\n";
        std::cout << "  " << title << "\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "  Total Requests:     " << total_requests << "\n";
        std::cout << "  Total Data:         " << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Total Wall Time:    " << total_time_ms << " ms\n";
        std::cout << "  Throughput:         " << throughput_mb_s << " MB/s\n";
        std::cout << "  Avg Latency:        " << avg_ms << " ms\n";
        std::cout << "  Min / Max Latency:  " << min_ms << " / " << max_ms << " ms\n";
        std::cout << "  P50 Latency:        " << p50_ms << " ms\n";
        std::cout << "  P95 Latency:        " << p95_ms << " ms\n";
        std::cout << "  P99 Latency:        " << p99_ms << " ms\n";
    }
};

int main(int argc, char* argv[]) {
    std::string model_dir = asema::m8::paths::synthetic_dir();
    if (argc > 1) {
        model_dir = argv[1];
    }

    std::string manifest_path = model_dir + "/manifest.json";
    std::string model_bin = model_dir + "/model.asema";

    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        std::cerr << "Cannot open " << manifest_path << "\n";
        return 1;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    auto manifest = asema::ModelManifest::from_json(buf.str());

    std::cout << "====================================================\n";
    std::cout << "    ASEMA Milestone 2 Benchmark: Sync vs Async I/O   \n";
    std::cout << "====================================================\n";
    std::cout << "Model: " << manifest.model_name << " (" << manifest.num_layers 
              << " layers x " << manifest.experts_per_layer << " experts)\n";
    std::cout << "Container: " << model_bin << "\n";

    // Prepare workloads: 64 requests
    std::vector<asema::ExpertCoord> all_coords;
    for (uint32_t l = 0; l < manifest.num_layers; ++l) {
        for (uint32_t e = 0; e < manifest.experts_per_layer; ++e) {
            all_coords.push_back(asema::ExpertCoord{l, e});
        }
    }

    // Benchmark A: Synchronous Sequential Read
    {
        auto backend = asema::create_storage_backend(model_bin);
        std::vector<double> latencies;
        size_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        for (const auto& coord : all_coords) {
            const auto* meta = manifest.get_expert_metadata(coord.layer_id, coord.expert_id);
            asema::ExpertBuffer buf(meta->storage_length, meta->alignment);
            double lat = 0.0;
            backend->read_expert_sync(*meta, buf, lat);
            latencies.push_back(lat);
            bytes += meta->storage_length;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto stats = LatencyStats::compute(latencies, bytes, wall);
        stats.print("A. Synchronous Sequential Execution (Baseline)");
    }

    // Benchmark B: Asynchronous Concurrent Read (4 Workers)
    {
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);
        std::vector<double> latencies;
        size_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<asema::ExpertLoadResult>> futures;
        for (const auto& coord : all_coords) {
            futures.push_back(loader.submit_future(coord, asema::LoadPriority::REQUIRED_NOW));
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            latencies.push_back(res.total_latency_ms);
            bytes += res.bytes_read;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto stats = LatencyStats::compute(latencies, bytes, wall);
        stats.print("B. Asynchronous Concurrent Read (4 Workers / IOCP)");
    }

    // Benchmark C: Asynchronous Concurrent Read (8 Workers)
    {
        asema::AsyncExpertLoader loader(model_bin, manifest, 8);
        std::vector<double> latencies;
        size_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<asema::ExpertLoadResult>> futures;
        for (const auto& coord : all_coords) {
            futures.push_back(loader.submit_future(coord, asema::LoadPriority::REQUIRED_NOW));
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            latencies.push_back(res.total_latency_ms);
            bytes += res.bytes_read;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto stats = LatencyStats::compute(latencies, bytes, wall);
        stats.print("C. Asynchronous High-Concurrency Read (8 Workers / IOCP)");
    }

    // Benchmark D: Repeated Concurrent Access (Coalescing Test: 64 requests with 4 repeats)
    {
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);
        std::vector<double> latencies;
        size_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<asema::ExpertLoadResult>> futures;
        for (int rep = 0; rep < 4; ++rep) {
            for (uint32_t e = 0; e < 16; ++e) {
                futures.push_back(loader.submit_future(asema::ExpertCoord{0, e}, asema::LoadPriority::REQUIRED_NOW));
            }
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            latencies.push_back(res.total_latency_ms);
            bytes += res.bytes_read;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto stats = LatencyStats::compute(latencies, bytes, wall);
        stats.print("D. Repeated Concurrent Requests (Coalescing)");
        std::cout << "  Coalesced Request Count: " << loader.total_coalesced_requests() << " / " 
                  << loader.total_requests_submitted() << "\n";
        std::cout << "  Physical Disk Dispatches: " << loader.total_io_dispatches() << "\n";
    }

    // Benchmark E: Random Expert Access Pattern (128 random requests)
    {
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);
        std::vector<double> latencies;
        size_t bytes = 0;

        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> layer_dist(0, manifest.num_layers - 1);
        std::uniform_int_distribution<uint32_t> exp_dist(0, manifest.experts_per_layer - 1);

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<asema::ExpertLoadResult>> futures;
        for (int i = 0; i < 128; ++i) {
            asema::ExpertCoord c{layer_dist(rng), exp_dist(rng)};
            futures.push_back(loader.submit_future(c, asema::LoadPriority::REQUIRED_NOW));
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            latencies.push_back(res.total_latency_ms);
            bytes += res.bytes_read;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto stats = LatencyStats::compute(latencies, bytes, wall);
        stats.print("E. Random Expert Access Pattern (128 Requests)");
    }

    std::cout << "====================================================\n";
    std::cout << "               Benchmark Complete                   \n";
    std::cout << "====================================================\n";
    return 0;
}
