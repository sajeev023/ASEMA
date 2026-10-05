// ASEMA v0.1 — Milestone 5: ASEMARuntime implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>

namespace asema {

namespace {

inline uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

// Deterministic token → per-layer expert choice.
// Locality-heavy: 70% chance of repeating the previous choice, 30% random.
class SyntheticRouter {
public:
    SyntheticRouter(uint64_t seed, uint32_t experts_per_layer, uint32_t top_k)
        : rng_(seed), experts_per_layer_(experts_per_layer), top_k_(top_k) {}

    std::vector<ExpertCoord> route(uint64_t token_id, uint32_t layer_id) {
        std::vector<ExpertCoord> out;
        std::uniform_int_distribution<uint32_t> dist(0, experts_per_layer_ - 1);
        std::uniform_real_distribution<double> real(0.0, 1.0);

        // 70% reuse last token's experts; 30% pick fresh
        if (!last_experts_.empty() && real(rng_) < 0.7) {
            out = last_experts_;
        } else {
            std::unordered_set<uint32_t> picked;
            while (out.size() < top_k_ && picked.size() < experts_per_layer_) {
                uint32_t e = dist(rng_);
                if (picked.insert(e).second) {
                    out.push_back(ExpertCoord{layer_id, e});
                }
            }
        }
        last_experts_ = out;
        return out;
    }

private:
    std::mt19937_64 rng_;
    uint32_t experts_per_layer_;
    uint32_t top_k_;
    std::vector<ExpertCoord> last_experts_;
};

ModelManifest load_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open manifest: " + path);
    }
    std::stringstream buf;
    buf << f.rdbuf();
    return ModelManifest::from_json(buf.str());
}

} // namespace

ASEMARuntime::ASEMARuntime(RuntimeConfig cfg)
    : cfg_(std::move(cfg))
{
    manifest_ = load_manifest(cfg_.manifest_path);

    // Use the actual expert count from the manifest if it disagrees with cfg.
    if (manifest_.num_layers > 0) cfg_.num_layers = manifest_.num_layers;
    if (manifest_.experts_per_layer > 0) cfg_.experts_per_layer = manifest_.experts_per_layer;
    if (manifest_.active_experts_per_token > 0) cfg_.top_k = manifest_.active_experts_per_token;

    loader_ = create_async_loader_adapter(cfg_.container_path, manifest_, cfg_.loader_workers);

    if (cfg_.ram_cache_bytes > 0) {
        cache_ = std::make_unique<ExpertRAMCache>(cfg_.ram_cache_bytes);
    }

    queue_ = std::make_unique<ExpertRequestQueue>(4096);

    if (cfg_.prefetch_lookahead > 0 && cache_) {
        prefetch_ = std::make_unique<PrefetchEngine>(
            *queue_, *cache_, cfg_.prefetch_lookahead,
            /*max_experts_per_layer=*/8,
            /*history_depth=*/8,
            /*max_prefetch_bytes_per_tick=*/64ull * 1024 * 1024);
    }

    TelemetryConfig tcfg;
    tcfg.jsonl_path = cfg_.telemetry_jsonl;
    tcfg.report_text_path = cfg_.report_text_path;
    tcfg.report_json_path = cfg_.report_json_path;
    tcfg.background_flush = !cfg_.telemetry_jsonl.empty() ||
                            !cfg_.report_text_path.empty() ||
                            !cfg_.report_json_path.empty();
    telemetry_ = std::make_unique<TelemetryEngine>(tcfg);
}

ASEMARuntime::~ASEMARuntime() {
    if (prefetch_) prefetch_->shutdown();
    if (loader_)   loader_->shutdown();
    if (queue_)    queue_->shutdown();
    if (cache_)    cache_->clear(true);
    if (telemetry_) telemetry_->shutdown();
}

std::vector<ExpertCoord> ASEMARuntime::route_token(uint64_t token_id, uint32_t layer_id) {
    // For runtime we synthesize a small per-token vector by consulting a
    // per-layer sequence. The actual per-layer routing is generated inside
    // run() — here we only generate for one layer. (run() handles all layers.)
    static thread_local SyntheticRouter router(cfg_.router_seed,
                                               cfg_.experts_per_layer,
                                               cfg_.top_k);
    (void)token_id;
    return router.route(token_id, layer_id);
}

void ASEMARuntime::execute_synthetic(uint64_t token_id, ExpertCoord c,
                                       std::shared_ptr<ExpertBuffer> buf) {
    auto t0 = now_ns();
    telemetry_->execution_begin(token_id, c);
    // Touch every 4 KB page to simulate read access pattern.
    if (buf && buf->data()) {
        volatile uint64_t sink = 0;
        for (size_t i = 0; i < buf->size(); i += 4096) {
            sink += buf->data()[i];
        }
        (void)sink;
    }
    auto t1 = now_ns();
    double us = static_cast<double>(t1 - t0) / 1000.0;
    telemetry_->execution_end(token_id, c, us);
    telemetry_->increment_execution_ms(us / 1000.0);
}

void ASEMARuntime::load_and_execute(uint64_t token_id, ExpertCoord c, LoadPriority prio) {
    // 1. Cache lookup
    auto entry = cache_ ? cache_->get(c) : nullptr;
    if (entry) {
        auto t0 = now_ns();
        telemetry_->expert_cache_hit(token_id, c, static_cast<double>(now_ns() - t0) / 1000.0);
        if (prefetch_) prefetch_->record_useful(c);
        execute_synthetic(token_id, c, entry->buffer);
        return;
    }

    if (cache_) telemetry_->expert_cache_miss(token_id, c);

    // 2. Submit load via the loader adapter. Use a future for synchronous wait.
    telemetry_->expert_load_begin(token_id, c);
    auto load_t0 = now_ns();
    auto fut = loader_->submit_future(c, prio);
    auto res = fut.get();
    auto load_t1 = now_ns();
    double load_us = static_cast<double>(load_t1 - load_t0) / 1000.0;
    telemetry_->expert_load_end(token_id, c, res.bytes_read, load_us, res.success);

    if (!res.success) {
        telemetry_->error("load failed for L" + std::to_string(c.layer_id) +
                          " E" + std::to_string(c.expert_id));
        return;
    }

    // 3. Insert into cache.
    if (cache_) {
        const auto* meta = manifest_.get_expert_metadata(c.layer_id, c.expert_id);
        if (meta) {
            auto result = cache_->put(c, *meta, res.buffer);
            if (result == CacheResult::OK || result == CacheResult::EVICTED) {
                telemetry_->cache_insertion(c, res.bytes_read);
            } else {
                telemetry_->error("cache insertion failed (EXHAUSTED)");
            }
        }
    }

    // 4. Execute
    execute_synthetic(token_id, c, res.buffer);
}

RuntimeResult ASEMARuntime::run() {
    RuntimeResult result;
    if (shutdown_.load()) return result;

    telemetry_->runtime_begin();

    auto wall_t0 = now_ns();

    SyntheticRouter router(cfg_.router_seed, cfg_.experts_per_layer, cfg_.top_k);

    bool first_token = true;

    for (uint64_t tok = 1; tok <= cfg_.num_tokens; ++tok) {
        if (shutdown_.load()) break;

        auto tok_t0 = now_ns();
        telemetry_->token_begin(tok);

        // Route and execute each layer's chosen experts.
        std::vector<std::vector<ExpertCoord>> chosen_per_layer;
        for (uint32_t l = 0; l < cfg_.num_layers; ++l) {
            auto experts = router.route(tok, l);
            chosen_per_layer.push_back(experts);
            for (const auto& c : experts) {
                telemetry_->expert_request(tok, c, LoadPriority::REQUIRED_NOW);
                telemetry_->queue_push(c, LoadPriority::REQUIRED_NOW);
                if (cache_) queue_->mark_in_flight(c);
                queue_->push(c, LoadPriority::REQUIRED_NOW, RequestSource::RUNTIME);
                load_and_execute(tok, c, LoadPriority::REQUIRED_NOW);
                if (cache_) queue_->clear_in_flight(c);
            }
        }

        // Prefetch tick
        if (prefetch_ && cfg_.prefetch_lookahead > 0) {
            uint32_t cur_layer = static_cast<uint32_t>(chosen_per_layer.size() - 1);
            prefetch_->on_token_complete(cur_layer, chosen_per_layer);
            prefetch_->tick(cur_layer);
        }

        telemetry_->set_queue_depth(queue_->size());
        if (cache_) {
            auto stats = cache_->stats();
            telemetry_->set_ram_bytes(stats.bytes_used, stats.peak_bytes_used, stats.capacity_bytes);
        }

        telemetry_->token_end(tok, static_cast<double>(now_ns() - tok_t0) / 1'000'000.0);

        if (first_token) {
            double ms = static_cast<double>(now_ns() - tok_t0) / 1'000'000.0;
            telemetry_->record_first_token_latency_ms(ms);
            result.first_token_latency_ms = ms;
            first_token = false;
        }

        // Inject a small synthetic delay to keep numbers realistic for short
        // workloads (otherwise everything completes in <1 ms).
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    auto wall_t1 = now_ns();
    result.total_wall_time_ms = static_cast<double>(wall_t1 - wall_t0) / 1'000'000.0;
    result.total_tokens = cfg_.num_tokens;
    result.completed = true;

    telemetry_->runtime_end(result.total_wall_time_ms);
    telemetry_->flush();

    if (!cfg_.report_text_path.empty() || !cfg_.report_json_path.empty()) {
        telemetry_->write_report();
    }

    return result;
}

} // namespace asema
