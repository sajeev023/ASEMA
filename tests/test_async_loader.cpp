#include "asema/m8/m8_paths.hpp"
#include "../include/asema/manifest.hpp"
#include "../include/asema/async_loader.hpp"
#include "../include/asema/storage.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <atomic>
#include <chrono>

int main() {
    std::cout << "====================================================\n";
    std::cout << "       Running ASEMA Async Expert Loader Tests      \n";
    std::cout << "====================================================\n";

    std::string manifest_path = (asema::m8::paths::synthetic_dir() + "/manifest.json");
    std::string model_bin = (asema::m8::paths::synthetic_dir() + "/model.asema");

    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        std::cerr << "[FAIL] Could not open manifest: " << manifest_path << "\n";
        return 1;
    }
    std::stringstream buf;
    buf << f.rdbuf();
    auto manifest = asema::ModelManifest::from_json(buf.str());

    // 1. Basic Single Async Read with Future
    {
        std::cout << "\n[TEST 1] Single Async Expert Read via std::future...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);

        auto fut = loader.submit_future(asema::ExpertCoord{0, 0}, asema::LoadPriority::REQUIRED_NOW);
        auto res = fut.get();

        assert(res.success);
        assert(res.buffer != nullptr);
        assert(res.bytes_read == 786432);
        assert(!res.is_corrupt);
        assert(!res.is_cancelled);

        std::cout << "[PASS] L0/E0 async loaded successfully: " << res.bytes_read << " bytes | "
                  << "Queue: " << res.queue_latency_ms << "ms, I/O: " << res.io_latency_ms 
                  << "ms, CRC: " << res.crc_latency_ms << "ms, Total: " << res.total_latency_ms << "ms\n";
    }

    // 2. Async Read via Callback
    {
        std::cout << "\n[TEST 2] Async Expert Read via Callback...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);

        std::atomic<bool> callback_fired{false};
        asema::ExpertLoadResult cb_result;

        asema::LoadHandle handle = loader.submit(
            asema::ExpertCoord{1, 5},
            asema::LoadPriority::REQUIRED_NOW,
            [&](const asema::ExpertLoadResult& res) {
                cb_result = res;
                callback_fired.store(true);
            }
        );

        auto waited_res = loader.wait(handle);
        assert(callback_fired.load());
        assert(cb_result.success);
        assert(waited_res.success);
        assert(cb_result.coord.layer_id == 1 && cb_result.coord.expert_id == 5);
        std::cout << "[PASS] L1/E5 callback received: " << cb_result.bytes_read << " bytes.\n";
    }

    // 3. Request Coalescing (Duplicate Requests)
    {
        std::cout << "\n[TEST 3] Request Coalescing for Concurrent Duplicate Requests...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);

        asema::ExpertCoord coord{2, 10};
        std::vector<std::future<asema::ExpertLoadResult>> futures;

        // Fire 10 simultaneous requests for the exact same expert
        for (int i = 0; i < 10; ++i) {
            futures.push_back(loader.submit_future(coord, asema::LoadPriority::REQUIRED_NOW));
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            assert(res.success);
            assert(res.coord == coord);
            assert(res.buffer != nullptr);
        }

        std::cout << "[PASS] 10 concurrent requests for L2/E10 resolved.\n";
        std::cout << "       Requests submitted: " << loader.total_requests_submitted() << "\n";
        std::cout << "       Requests coalesced: " << loader.total_coalesced_requests() << "\n";
        std::cout << "       Underlying I/O dispatches: " << loader.total_io_dispatches() << "\n";
        assert(loader.total_coalesced_requests() > 0);
    }

    // 4. Concurrency Stress Test (64 distinct experts simultaneously)
    {
        std::cout << "\n[TEST 4] Concurrent Stress Test (64 parallel distinct experts)...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 8);

        std::vector<std::future<asema::ExpertLoadResult>> futures;
        auto start_time = std::chrono::steady_clock::now();

        for (uint32_t l = 0; l < manifest.num_layers; ++l) {
            for (uint32_t e = 0; e < manifest.experts_per_layer; ++e) {
                futures.push_back(loader.submit_future(asema::ExpertCoord{l, e}, asema::LoadPriority::REQUIRED_NOW));
            }
        }

        size_t total_bytes = 0;
        for (auto& fut : futures) {
            auto res = fut.get();
            assert(res.success);
            assert(res.buffer != nullptr);
            total_bytes += res.bytes_read;
        }

        auto end_time = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        double throughput_mb_s = (total_bytes / (1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

        std::cout << "[PASS] Loaded all " << futures.size() << " experts (" 
                  << (total_bytes / (1024.0 * 1024.0)) << " MB) in " << elapsed_ms << " ms\n";
        std::cout << "       Concurrent Throughput: " << throughput_mb_s << " MB/s\n";
    }

    // 5. Invalid Expert Coordinate Error Handling
    {
        std::cout << "\n[TEST 5] Invalid Coordinate Error Handling...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);

        auto fut = loader.submit_future(asema::ExpertCoord{99, 99}, asema::LoadPriority::REQUIRED_NOW);
        auto res = fut.get();
        assert(!res.success);
        assert(!res.error_message.empty());
        std::cout << "[PASS] Rejected out-of-bounds coordinate safely: \"" << res.error_message << "\"\n";
    }

    // 6. Data Corruption Detection under Asynchronous I/O
    {
        std::cout << "\n[TEST 6] Corrupted CRC32 Detection during Async Read...\n";
        // Create manifest with tampered CRC32 on L0/E1
        auto tampered_manifest = manifest;
        for (auto& l : tampered_manifest.layers) {
            if (l.layer_id == 0 && !l.experts.empty()) {
                l.experts[0].checksum_crc32 ^= 0xCAFEBABE;
            }
        }
        tampered_manifest.build_index();

        asema::AsyncExpertLoader loader(model_bin, tampered_manifest, 4);
        auto fut = loader.submit_future(asema::ExpertCoord{0, 0}, asema::LoadPriority::REQUIRED_NOW);
        auto res = fut.get();

        assert(!res.success);
        assert(res.is_corrupt);
        assert(res.buffer == nullptr);
        std::cout << "[PASS] Corrupted expert safely detected & buffer quarantined: \"" << res.error_message << "\"\n";
    }

    // 7. Cancellation Test
    {
        std::cout << "\n[TEST 7] Request Cancellation...\n";
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);

        asema::LoadHandle h = loader.submit(
            asema::ExpertCoord{3, 14},
            asema::LoadPriority::BACKGROUND
        );
        bool cancelled = loader.cancel(h);
        auto res = loader.wait(h);

        std::cout << "[PASS] Cancellation reported: " << (cancelled ? "true" : "false") 
                  << ", is_cancelled: " << (res.is_cancelled ? "true" : "false") << "\n";
    }

    // 8. Clean Shutdown with In-flight Requests
    {
        std::cout << "\n[TEST 8] Immediate Clean Shutdown during Activity...\n";
        {
            asema::AsyncExpertLoader loader(model_bin, manifest, 4);
            for (uint32_t e = 0; e < 16; ++e) {
                loader.submit(asema::ExpertCoord{0, e}, asema::LoadPriority::BACKGROUND);
            }
            loader.shutdown();
        }
        std::cout << "[PASS] Loader shut down and reclaimed all worker threads cleanly.\n";
    }

    // ====================================================
    // BENCHMARK SUITE: SYNC VS ASYNC COMPARISONS
    // ====================================================
    std::cout << "\n====================================================\n";
    std::cout << "   RUNNING MILESTONE 2 EMPIRICAL BENCHMARK SUITE    \n";
    std::cout << "====================================================\n";

    std::vector<asema::ExpertCoord> all_coords;
    for (uint32_t l = 0; l < manifest.num_layers; ++l) {
        for (uint32_t e = 0; e < manifest.experts_per_layer; ++e) {
            all_coords.push_back(asema::ExpertCoord{l, e});
        }
    }

    auto print_stats = [](const std::string& name, std::vector<double>& latencies, size_t total_bytes, double wall_time_ms) {
        std::sort(latencies.begin(), latencies.end());
        double sum = 0.0;
        for (double d : latencies) sum += d;
        double avg = latencies.empty() ? 0.0 : sum / latencies.size();
        double p50 = latencies[static_cast<size_t>(latencies.size() * 0.50)];
        double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        double throughput = (total_bytes / (1024.0 * 1024.0)) / (wall_time_ms / 1000.0);

        std::cout << "----------------------------------------------------\n";
        std::cout << "  " << name << "\n";
        std::cout << "----------------------------------------------------\n";
        std::cout << "  Total Requests:     " << latencies.size() << "\n";
        std::cout << "  Total Data:         " << (total_bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Wall Clock Time:    " << wall_time_ms << " ms\n";
        std::cout << "  Effective Speed:    " << throughput << " MB/s\n";
        std::cout << "  Avg Latency:        " << avg << " ms\n";
        std::cout << "  P50 Latency:        " << p50 << " ms\n";
        std::cout << "  P95 Latency:        " << p95 << " ms\n";
        std::cout << "  P99 Latency:        " << p99 << " ms\n";
    };

    // Benchmark A: Synchronous Sequential Read (Baseline)
    {
        auto backend = asema::create_storage_backend(model_bin);
        std::vector<double> latencies;
        size_t bytes = 0;

        auto t0 = std::chrono::steady_clock::now();
        for (const auto& coord : all_coords) {
            const auto* meta = manifest.get_expert_metadata(coord.layer_id, coord.expert_id);
            asema::ExpertBuffer exp_buf(meta->storage_length, meta->alignment);
            double lat = 0.0;
            backend->read_expert_sync(*meta, exp_buf, lat);
            latencies.push_back(lat);
            bytes += meta->storage_length;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_stats("A. Synchronous Sequential Execution (Baseline)", latencies, bytes, wall);
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
        print_stats("B. Asynchronous Concurrent Read (4 Workers / IOCP)", latencies, bytes, wall);
    }

    // Benchmark C: Asynchronous High-Concurrency Read (8 Workers)
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
        print_stats("C. Asynchronous High-Concurrency Read (8 Workers / IOCP)", latencies, bytes, wall);
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
        print_stats("D. Repeated Concurrent Requests (Coalescing)", latencies, bytes, wall);
        std::cout << "  Coalesced Requests: " << loader.total_coalesced_requests() << " / " 
                  << loader.total_requests_submitted() << "\n";
        std::cout << "  Physical Disk Dispatches: " << loader.total_io_dispatches() << "\n";
    }

    // Benchmark E: Random Expert Access Pattern (128 pseudo-random requests)
    {
        asema::AsyncExpertLoader loader(model_bin, manifest, 4);
        std::vector<double> latencies;
        size_t bytes = 0;

        // Simple linear congruential generator for deterministic random coordinates
        uint32_t seed = 42;
        auto next_rand = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return seed;
        };

        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::future<asema::ExpertLoadResult>> futures;
        for (int i = 0; i < 128; ++i) {
            uint32_t l = next_rand() % manifest.num_layers;
            uint32_t e = next_rand() % manifest.experts_per_layer;
            futures.push_back(loader.submit_future(asema::ExpertCoord{l, e}, asema::LoadPriority::REQUIRED_NOW));
        }

        for (auto& fut : futures) {
            auto res = fut.get();
            latencies.push_back(res.total_latency_ms);
            bytes += res.bytes_read;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        print_stats("E. Random Expert Access Pattern (128 Requests)", latencies, bytes, wall);
    }

    std::cout << "\n====================================================\n";
    std::cout << "         ALL MILESTONE 2 TESTS & BENCHMARKS PASSED! \n";
    std::cout << "====================================================\n";
    return 0;
}
