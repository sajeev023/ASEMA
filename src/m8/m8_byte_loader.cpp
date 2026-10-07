#include "asema/m8/m8_paths.hpp"
#include "asema/m8/m8_byte_loader.hpp"
#include "asema/m8/m8_fp8_gemv.hpp" // M8WorkerPool (bounded worker threads)

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <windows.h>
#include <cctype>
#include "asema/m8/m8_drive_info.hpp"

namespace fs = std::filesystem;

namespace asema {
namespace m8 {

namespace {
// Optional access trace for offline cache-policy replay: ASEMA_TRACE_EXPERTS=<file>.
// One line per expert lookup ("layer expert"); a "T" line marks the start of each token (layer 0).
std::ofstream* expert_trace_stream() {
    static std::ofstream* stream = [] () -> std::ofstream* {
        const char* path = std::getenv("ASEMA_TRACE_EXPERTS");
        if (!path || !*path) return nullptr;
        auto* s = new std::ofstream(path, std::ios::out | std::ios::trunc);
        return s->is_open() ? s : nullptr;
    }();
    return stream;
}
} // namespace

M8ByteRangeLoader::M8ByteRangeLoader(std::shared_ptr<M8MultiVolumeManager> vol_mgr)
    : vol_mgr_(std::move(vol_mgr)) {
    if (const char* p = std::getenv("ASEMA_CACHE_POLICY")) {
        if (std::string(p) == "lru") policy_ = Policy::LRU;
    }
    if (const char* a = std::getenv("ASEMA_CACHE_ALPHA")) {
        const double v = std::atof(a);
        if (v > 0.0) blend_alpha_ = v;
    }
    if (const char* k = std::getenv("ASEMA_PREFETCH_K")) set_prefetch_k(std::atoi(k));
    ensure_default_expert();
}

M8ByteRangeLoader::~M8ByteRangeLoader() {
    wait_for_prefetches();
    std::lock_guard<std::mutex> lock(handle_mutex_);
    for (auto& pair : handle_pool_) {
        if (pair.second && pair.second != INVALID_HANDLE_VALUE) {
            CloseHandle(static_cast<HANDLE>(pair.second));
        }
    }
    handle_pool_.clear();
}

// ---------------------------------------------------------------------------
// Cache management (true LRU: front of lru_ is most recently used)
// ---------------------------------------------------------------------------

void M8ByteRangeLoader::touch_locked(CacheEntry& e, const CacheKey& key) {
    lru_.erase(e.lru_it);
    lru_.push_front(key);
    e.lru_it = lru_.begin();
}

void M8ByteRangeLoader::note_access_locked(const CacheKey& key) {
    ++access_clock_;
    ++freq_[key];
}

void M8ByteRangeLoader::evict_one_locked() {
    if (cache_.empty()) return;
    auto victim = cache_.end();
    if (policy_ == Policy::LRU) {
        if (lru_.empty()) return;
        victim = cache_.find(lru_.back());
    } else {
        // Lowest (access count) / (1 + alpha * age) goes first. Age is measured in decode steps
        // (~240 expert lookups per step). The cache is small (hundreds of entries), so a scan is cheap.
        constexpr double kLookupsPerStep = 240.0;
        double best = 0.0;
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            auto f = freq_.find(it->first);
            const double count = (f != freq_.end()) ? static_cast<double>(f->second) : 1.0;
            const double age = static_cast<double>(access_clock_ - it->second.last_access) / kLookupsPerStep;
            const double score = count / (1.0 + blend_alpha_ * age);
            if (victim == cache_.end() || score < best) {
                victim = it;
                best = score;
            }
        }
    }
    if (victim == cache_.end()) return;
    if (victim->second.prefetched) telemetry_.wasted_prefetches++;
    lru_.erase(victim->second.lru_it);
    cache_.erase(victim);
    telemetry_.evictions++;
}

void M8ByteRangeLoader::evict_to_capacity_locked(size_t capacity) {
    while (cache_.size() > capacity && !cache_.empty()) {
        evict_one_locked();
    }
}

void M8ByteRangeLoader::insert_locked(const CacheKey& key, std::shared_ptr<ExpertPayload> payload, bool prefetched) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        touch_locked(it->second, key);
        return;
    }
    evict_to_capacity_locked(max_cached_experts_ > 0 ? max_cached_experts_ - 1 : 0);
    lru_.push_front(key);
    CacheEntry e;
    e.payload = std::move(payload);
    e.lru_it = lru_.begin();
    e.prefetched = prefetched;
    e.last_access = access_clock_;
    cache_.emplace(key, std::move(e));
}

void M8ByteRangeLoader::set_cache_capacity_experts(size_t max_experts) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    max_cached_experts_ = std::max<size_t>(1, max_experts);
    evict_to_capacity_locked(max_cached_experts_);
}

void M8ByteRangeLoader::set_cache_capacity_mb(size_t mb) {
    size_t experts = (mb * 1024ULL * 1024ULL) / ExpertDimensions::TOTAL_EXPERT_BYTES;
    set_cache_capacity_experts(experts);
}

size_t M8ByteRangeLoader::cache_capacity_experts() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return max_cached_experts_;
}

void M8ByteRangeLoader::clear_cache() {
    wait_for_prefetches();
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
    lru_.clear();
}

size_t M8ByteRangeLoader::cache_size_experts() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

size_t M8ByteRangeLoader::cache_size_bytes() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size() * ExpertDimensions::TOTAL_EXPERT_BYTES;
}

void M8ByteRangeLoader::set_max_parallel_reads(int n) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    max_parallel_reads_ = std::clamp(n, 1, 16);
}

int M8ByteRangeLoader::max_parallel_reads() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return max_parallel_reads_;
}

ByteLoaderTelemetry M8ByteRangeLoader::telemetry() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return telemetry_;
}

void M8ByteRangeLoader::reset_telemetry() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    telemetry_ = ByteLoaderTelemetry{};
}

void M8ByteRangeLoader::wait_for_prefetches() {
    std::vector<std::future<void>> futures;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        futures.swap(active_prefetches_);
    }
    for (auto& fut : futures) {
        if (fut.valid()) {
            fut.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// File access
// ---------------------------------------------------------------------------

void* M8ByteRangeLoader::get_or_open_handle(const std::string& path) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_pool_.find(path);
    if (it != handle_pool_.end() && it->second != INVALID_HANDLE_VALUE && it->second != nullptr) {
        return it->second;
    }

    std::wstring wpath(path.begin(), path.end());
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (hFile != INVALID_HANDLE_VALUE) {
        handle_pool_[path] = static_cast<void*>(hFile);
    }
    return static_cast<void*>(hFile);
}

bool M8ByteRangeLoader::read_file_range(const std::string& path, uint64_t offset, size_t length, uint8_t* dst) {
    if (!dst || length == 0) return false;

    HANDLE hFile = static_cast<HANDLE>(get_or_open_handle(path));
    if (hFile == INVALID_HANDLE_VALUE || hFile == nullptr) {
        return false;
    }

    // Positioned read: the offset travels with the request, so concurrent reads on one handle
    // cannot disturb each other (SetFilePointerEx + ReadFile would race).
    size_t done = 0;
    while (done < length) {
        const uint64_t pos = offset + done;
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(pos & 0xFFFFFFFFULL);
        ov.OffsetHigh = static_cast<DWORD>(pos >> 32);
        const DWORD chunk = static_cast<DWORD>(std::min<size_t>(length - done, 64u * 1024u * 1024u));
        DWORD got = 0;
        if (!ReadFile(hFile, dst + done, chunk, &got, &ov) || got == 0) {
            return false;
        }
        done += got;
    }
    return true;
}

void M8ByteRangeLoader::ensure_default_expert() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (has_default_payload_) return;
    std::string s_file = (asema::m8::paths::primary_shards() + "/e0_scales.bin");
    std::string w_file = (asema::m8::paths::primary_shards() + "/e0_weights.bin");
    if (fs::exists(s_file) && fs::exists(w_file)) {
        default_payload_.scales.resize(ExpertDimensions::TOTAL_SCALE_BYTES);
        default_payload_.weights.resize(ExpertDimensions::TOTAL_WEIGHT_BYTES);
        if (read_file_range(s_file, 0, ExpertDimensions::TOTAL_SCALE_BYTES, default_payload_.scales.data()) &&
            read_file_range(w_file, 0, ExpertDimensions::TOTAL_WEIGHT_BYTES, default_payload_.weights.data())) {
            has_default_payload_ = true;
        }
    }
}

bool M8ByteRangeLoader::locate_expert_cached(int layer_id, int expert_id, ExpertFileLoc& out) const {
    const uint32_t key = (static_cast<uint32_t>(layer_id) << 16) | (static_cast<uint32_t>(expert_id) & 0xFFFF);
    {
        std::lock_guard<std::mutex> lock(loc_mutex_);
        auto it = loc_cache_.find(key);
        if (it != loc_cache_.end()) {
            out = it->second;
            return true;
        }
    }
    if (!vol_mgr_) return false;
    ExpertLocation loc = vol_mgr_->locate_expert(layer_id, expert_id);
    if (!(loc.all_tensors_present && loc.w1_scale.offset_in_file > 0 && loc.w1_weight.offset_in_file > 0)) {
        return false;
    }
    out.scale_path = loc.w1_scale.physical_path;
    out.weight_path = loc.w1_weight.physical_path;
    out.scale_offset = loc.w1_scale.offset_in_file;
    out.weight_offset = loc.w1_weight.offset_in_file;
    std::lock_guard<std::mutex> lock(loc_mutex_);
    loc_cache_[key] = out;
    return true;
}

double M8ByteRangeLoader::expert_read_cost(int layer_id, int expert_id) const {
    ExpertFileLoc loc;
    if (!locate_expert_cached(layer_id, expert_id, loc) || loc.weight_path.size() < 2 || loc.weight_path[1] != ':') return 1.0;
    const int letter = std::toupper(static_cast<unsigned char>(loc.weight_path[0]));
    {
        std::lock_guard<std::mutex> lock(loc_mutex_);
        auto it = drive_cost_.find(letter);
        if (it != drive_cost_.end()) return it->second;
    }
    const double cost = drive_class_read_cost(classify_drive_letter(static_cast<char>(letter)));
    std::lock_guard<std::mutex> lock(loc_mutex_);
    drive_cost_[letter] = cost;
    return cost;
}

bool M8ByteRangeLoader::internal_read_expert(int layer_id, int expert_id, ExpertPayload& out_payload) {
    out_payload.layer_id = layer_id;
    out_payload.expert_id = expert_id;

    // 1. Direct byte-range read from the physical .safetensors shard
    ExpertFileLoc loc;
    if (locate_expert_cached(layer_id, expert_id, loc)) {
        out_payload.scales.resize(ExpertDimensions::TOTAL_SCALE_BYTES);
        out_payload.weights.resize(ExpertDimensions::TOTAL_WEIGHT_BYTES);

        bool ok_s = read_file_range(loc.scale_path, loc.scale_offset,
                                    ExpertDimensions::TOTAL_SCALE_BYTES, out_payload.scales.data());
        bool ok_w = read_file_range(loc.weight_path, loc.weight_offset,
                                    ExpertDimensions::TOTAL_WEIGHT_BYTES, out_payload.weights.data());
        if (ok_s && ok_w) {
            return true;
        }
    }

    // 2. Fall back to extracted layer expert binary files if physically present
    std::string s_file = (asema::m8::paths::primary_shards() + "/experts/l") + std::to_string(layer_id) + "_e" + std::to_string(expert_id) + "_scales.bin";
    std::string w_file = (asema::m8::paths::primary_shards() + "/experts/l") + std::to_string(layer_id) + "_e" + std::to_string(expert_id) + "_weights.bin";

    if (!fs::exists(s_file) || !fs::exists(w_file)) {
        s_file = (asema::m8::paths::primary_shards() + "/e") + std::to_string(expert_id) + "_scales.bin";
        w_file = (asema::m8::paths::primary_shards() + "/e") + std::to_string(expert_id) + "_weights.bin";
    }

    if (fs::exists(s_file) && fs::exists(w_file)) {
        out_payload.scales.resize(ExpertDimensions::TOTAL_SCALE_BYTES);
        out_payload.weights.resize(ExpertDimensions::TOTAL_WEIGHT_BYTES);
        return read_file_range(s_file, 0, ExpertDimensions::TOTAL_SCALE_BYTES, out_payload.scales.data()) &&
               read_file_range(w_file, 0, ExpertDimensions::TOTAL_WEIGHT_BYTES, out_payload.weights.data());
    }

    // Strictly no synthetic or expert-0 fallback
    return false;
}

// ---------------------------------------------------------------------------
// Prefetch (bounded: at most max_parallel_reads_ outstanding)
// ---------------------------------------------------------------------------

void M8ByteRangeLoader::prefetch_expert(int layer_id, int expert_id) {
    CacheKey key{layer_id, expert_id};
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.find(key) != cache_.end()) {
            telemetry_.duplicate_requests++;
            return;
        }
        // Reap finished prefetches and enforce the hard bound on outstanding work.
        active_prefetches_.erase(
            std::remove_if(active_prefetches_.begin(), active_prefetches_.end(), [](std::future<void>& f) {
                return !f.valid() || f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
            }),
            active_prefetches_.end());
        if (static_cast<int>(active_prefetches_.size()) >= max_parallel_reads_) {
            return; // queue full: drop speculative request rather than grow without bound
        }
        telemetry_.prefetches_issued++;
    }

    auto fut = std::async(std::launch::async, [this, layer_id, expert_id, key]() {
        auto p = std::make_shared<ExpertPayload>();
        if (!internal_read_expert(layer_id, expert_id, *p)) return;
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.find(key) == cache_.end()) {
            telemetry_.total_bytes_read += p->total_bytes();
            telemetry_.storage_read_ops += 2;
            insert_locked(key, std::move(p), /*prefetched=*/true);
        }
    });

    std::lock_guard<std::mutex> lock(cache_mutex_);
    active_prefetches_.push_back(std::move(fut));
}

// Learns which experts of `layer` follow which experts of layer-1 (same token), from real selections only.
void M8ByteRangeLoader::observe_selection_locked(int layer, const std::vector<int>& expert_ids) {
    if (layer < 0 || layer >= kPredLayers) return;
    if (cooc_.empty()) {
        cooc_.assign(static_cast<size_t>(kPredLayers) * kPredExperts * kPredExperts, 0);
        last_selection_.assign(kPredLayers, {});
    }
    if (layer > 0) {
        for (int f : last_selection_[layer - 1]) {
            if (f < 0 || f >= kPredExperts) continue;
            const size_t base = (static_cast<size_t>(layer) * kPredExperts + f) * kPredExperts;
            for (int e : expert_ids) {
                if (e < 0 || e >= kPredExperts) continue;
                if (cooc_[base + e] < 65535) ++cooc_[base + e];
            }
        }
    }
    last_selection_[layer] = expert_ids;
}

void M8ByteRangeLoader::prefetch_predicted(int target_layer, const std::vector<int>& current_selection, int k,
                                           const std::function<bool(int)>& is_resident) {
    if (k <= 0 || target_layer < 1 || target_layer >= kPredLayers) return;
    std::vector<int> chosen;
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        // Whatever the previous prefetch left unconsumed was a wrong guess: account for it, and park the
        // (possibly still running) read so nobody blocks on it.
        for (auto& kv : inflight_) {
            telemetry_.wasted_prefetches++;
            graveyard_.push_back(std::move(kv.second));
        }
        inflight_.clear();
        graveyard_.erase(std::remove_if(graveyard_.begin(), graveyard_.end(), [](const auto& f) {
            return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }), graveyard_.end());
        if (cooc_.empty()) return;

        std::vector<uint32_t> score(kPredExperts, 0);
        for (int f : current_selection) {
            if (f < 0 || f >= kPredExperts) continue;
            const size_t base = (static_cast<size_t>(target_layer) * kPredExperts + f) * kPredExperts;
            for (int e = 0; e < kPredExperts; ++e) score[e] += cooc_[base + e];
        }
        std::vector<std::pair<uint32_t, int>> cand;
        for (int e = 0; e < kPredExperts; ++e) {
            if (score[e] == 0) continue;
            if (cache_.find(CacheKey{target_layer, e}) != cache_.end()) continue;
            cand.emplace_back(score[e], e);
        }
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        // Take the best `k` that are not already resident in a faster tier; those are the ones worth reading.
        for (const auto& c : cand) {
            if (static_cast<int>(chosen.size()) >= k) break;
            if (is_resident && is_resident(c.second)) continue;
            chosen.push_back(c.second);
        }
        for (int e : chosen) {
            telemetry_.prefetches_issued++;
            telemetry_.total_bytes_read += ExpertDimensions::TOTAL_EXPERT_BYTES;
            telemetry_.storage_read_ops += 2;
            inflight_.emplace(CacheKey{target_layer, e},
                std::async(std::launch::async, [this, target_layer, e]() -> std::shared_ptr<ExpertPayload> {
                    acquire_io_slot();
                    auto payload = std::make_shared<ExpertPayload>();
                    const bool ok = internal_read_expert(target_layer, e, *payload);
                    release_io_slot();
                    return ok ? payload : nullptr;
                }).share());
        }
    }
}

void M8ByteRangeLoader::prefetch_layer_experts(int layer_id, const std::vector<int>& expert_ids) {
    for (int id : expert_ids) {
        prefetch_expert(layer_id, id);
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

bool M8ByteRangeLoader::load_expert_payload(int layer_id, int expert_id, ExpertPayload& out_payload) {
    CacheKey key{layer_id, expert_id};

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        telemetry_.total_requests++;
        note_access_locked(key);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            telemetry_.cache_hits++;
            if (it->second.prefetched) {
                telemetry_.useful_prefetches++;
                it->second.prefetched = false;
            }
            touch_locked(it->second, key);
            it->second.last_access = access_clock_;
            out_payload = *(it->second.payload);
            return true;
        }
        telemetry_.cache_misses++;
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    auto payload = std::make_shared<ExpertPayload>();
    bool ok = internal_read_expert(layer_id, expert_id, *payload);
    double io_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
    if (!ok) {
        return false;
    }

    out_payload = *payload;
    std::lock_guard<std::mutex> lock(cache_mutex_);
    telemetry_.total_io_time_ms += io_ms;
    telemetry_.total_bytes_read += payload->total_bytes();
    telemetry_.storage_read_ops += 2;
    insert_locked(key, std::move(payload), /*prefetched=*/false);
    return true;
}

bool M8ByteRangeLoader::load_layer_experts_parallel(int layer_id,
                                                   const std::vector<int>& expert_ids,
                                                   std::vector<std::shared_ptr<const ExpertPayload>>& out_payloads) {
    out_payloads.assign(expert_ids.size(), nullptr);
    PendingExpertLoad pending = begin_layer_experts(layer_id, expert_ids);
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        out_payloads[i] = pending.get(i); // blocks until expert i has been read (or is a cache hit)
    }
    bool ok = finish_layer_experts(pending);
    for (const auto& p : out_payloads) ok = ok && (p != nullptr);
    return ok;
}

void M8ByteRangeLoader::acquire_io_slot() {
    std::unique_lock<std::mutex> lk(io_mutex_);
    io_cv_.wait(lk, [this] { return io_in_flight_ < max_parallel_reads(); });
    ++io_in_flight_;
}

void M8ByteRangeLoader::release_io_slot() {
    {
        std::lock_guard<std::mutex> lk(io_mutex_);
        --io_in_flight_;
    }
    io_cv_.notify_all();
}

std::shared_ptr<const ExpertPayload> M8ByteRangeLoader::PendingExpertLoad::get(size_t i) const {
    if (i >= slots_.size()) return nullptr;
    const Slot& s = slots_[i];
    if (!s.is_miss) return s.cached;
    return s.pending.get(); // shared_future: safe to call repeatedly
}

M8ByteRangeLoader::PendingExpertLoad M8ByteRangeLoader::begin_layer_experts(
        int layer_id, const std::vector<int>& expert_ids, const std::vector<char>* resident_elsewhere) {
    PendingExpertLoad pending;
    pending.layer_id_ = layer_id;
    pending.slots_.resize(expert_ids.size());

    // Resolve cache hits now (refreshing recency/frequency) and note which experts must be read.
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (std::ofstream* trace = expert_trace_stream()) {
            if (layer_id == 0) *trace << "T\n";
            for (int id : expert_ids) *trace << layer_id << ' ' << id << '\n';
            trace->flush();
        }
        observe_selection_locked(layer_id, expert_ids);
        for (size_t i = 0; i < expert_ids.size(); ++i) {
            CacheKey key{layer_id, expert_ids[i]};
            auto& slot = pending.slots_[i];
            slot.expert_id = expert_ids[i];
            telemetry_.total_requests++;
            if (resident_elsewhere && i < resident_elsewhere->size() && (*resident_elsewhere)[i]) {
                telemetry_.cache_hits++;   // served from VRAM: no storage read
                continue;
            }
            note_access_locked(key);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                telemetry_.cache_hits++;
                if (it->second.prefetched) {
                    telemetry_.useful_prefetches++;
                    it->second.prefetched = false;
                }
                touch_locked(it->second, key);
                it->second.last_access = access_clock_;
                slot.cached = it->second.payload;
            } else {
                telemetry_.cache_misses++;
                slot.is_miss = true;
                // A read for this expert may already be running (started by prefetch_predicted): share it.
                auto fit = inflight_.find(key);
                if (fit != inflight_.end()) {
                    slot.pending = std::move(fit->second);
                    slot.from_prefetch = true;
                    telemetry_.useful_prefetches++;
                    inflight_.erase(fit);
                }
            }
        }
    }

    // Start reading every miss. At most max_parallel_reads() reads run at once, loader-wide
    // (extra tasks wait in acquire_io_slot), and a layer has at most 6 experts, so the number of
    // helper threads is bounded.
    for (auto& slot : pending.slots_) {
        if (!slot.is_miss || slot.from_prefetch) continue;
        const int expert = slot.expert_id;
        slot.pending = std::async(std::launch::async, [this, layer_id, expert]() -> std::shared_ptr<ExpertPayload> {
            acquire_io_slot();
            auto payload = std::make_shared<ExpertPayload>();
            const bool ok = internal_read_expert(layer_id, expert, *payload);
            release_io_slot();
            return ok ? payload : nullptr;
        }).share();
    }
    return pending;
}

bool M8ByteRangeLoader::finish_layer_experts(PendingExpertLoad& pending, bool publish_to_cache) {
    bool all_ok = true;
    bool any_miss = false;
    std::vector<std::shared_ptr<ExpertPayload>> loaded(pending.slots_.size());
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < pending.slots_.size(); ++i) {
        if (!pending.slots_[i].is_miss) continue;
        any_miss = true;
        loaded[i] = pending.slots_[i].pending.get();
        if (!loaded[i]) all_ok = false;
    }
    if (!any_miss) return all_ok;
    const double wait_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    std::lock_guard<std::mutex> lock(cache_mutex_);
    telemetry_.total_io_time_ms += wait_ms;
    telemetry_.parallel_batches++;
    for (size_t i = 0; i < loaded.size(); ++i) {
        if (!loaded[i]) continue;
        if (!pending.slots_[i].from_prefetch) {   // prefetched reads were counted when they were issued
            telemetry_.total_bytes_read += loaded[i]->total_bytes();
            telemetry_.storage_read_ops += 2;
        }
        if (publish_to_cache) {
            insert_locked(CacheKey{pending.layer_id_, pending.slots_[i].expert_id}, loaded[i], /*prefetched=*/false);
        }
    }
    return all_ok;
}

} // namespace m8
} // namespace asema
