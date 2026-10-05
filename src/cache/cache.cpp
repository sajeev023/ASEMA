// ASEMA v0.1 — Milestone 3: ExpertRAMCache implementation
// -----------------------------------------------------------------------------
// See include/asema/cache.hpp for the design contract.
// -----------------------------------------------------------------------------

#include "../../include/asema/cache.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace asema {

// =============================================================================
// CachePin (RAII)
// =============================================================================
CachePin::CachePin(ExpertRAMCache* cache, ExpertCoord coord, bool active)
    : cache_(cache), coord_(coord), active_(active) {}

CachePin::CachePin(CachePin&& other) noexcept
    : cache_(other.cache_), coord_(other.coord_), active_(other.active_) {
    other.cache_ = nullptr;
    other.active_ = false;
}

CachePin& CachePin::operator=(CachePin&& other) noexcept {
    if (this != &other) {
        release();
        cache_ = other.cache_;
        coord_ = other.coord_;
        active_ = other.active_;
        other.cache_ = nullptr;
        other.active_ = false;
    }
    return *this;
}

CachePin::~CachePin() {
    release();
}

void CachePin::release() noexcept {
    if (active_ && cache_) {
        cache_->unpin(coord_);
        cache_ = nullptr;
        active_ = false;
    }
}

// =============================================================================
// ExpertRAMCache
// =============================================================================
ExpertRAMCache::ExpertRAMCache(uint64_t capacity_bytes)
    : capacity_(capacity_bytes) {
    if (capacity_bytes == 0) {
        // A zero-capacity cache is technically valid but useless. We allow it
        // so the unit tests can verify EXHAUSTED behavior.
    }
}

ExpertRAMCache::~ExpertRAMCache() {
    // Lock-free teardown: detach everything; std::unique_ptr<ListNode>
    // destructors will free buffers.
    std::unique_lock<std::shared_mutex> lock(mu_);
    index_.clear();
    head_ = nullptr;
    tail_ = nullptr;
    resident_count_ = 0;
    bytes_used_.store(0, std::memory_order_relaxed);
}

uint64_t ExpertRAMCache::capacity() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return capacity_;
}

void ExpertRAMCache::set_capacity(uint64_t new_capacity) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    capacity_ = new_capacity;

    // If shrinking below current usage, evict.
    if (bytes_used_.load(std::memory_order_relaxed) > capacity_) {
        lock.unlock();
        evict_until(capacity_);
        lock.lock();
    }
}

// -----------------------------------------------------------------------------
// Reservation primitives
// -----------------------------------------------------------------------------
bool ExpertRAMCache::try_reserve_bytes(uint64_t n) noexcept {
    // Reserve-before-allocate. Loop with relaxed ordering; the budget is the
    // single ground truth.
    if (n > capacity_) {
        failed_reservations_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    uint64_t cur = bytes_used_.load(std::memory_order_relaxed);
    while (true) {
        if (cur + n > capacity_) {
            failed_reservations_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (bytes_used_.compare_exchange_weak(
                cur, cur + n,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
            // Update peak
            uint64_t peak = peak_bytes_used_.load(std::memory_order_relaxed);
            while (cur + n > peak) {
                if (peak_bytes_used_.compare_exchange_weak(
                        peak, cur + n,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    break;
                }
            }
            return true;
        }
        // cur was reloaded by CAS; loop
    }
}

void ExpertRAMCache::commit_reserved_bytes(uint64_t /*n*/) noexcept {
    // No-op: reservation and commit are atomic in the same step (try_reserve
    // already increments bytes_used_). Hook retained for clarity / future
    // deferred commit semantics.
}

void ExpertRAMCache::release_reserved_bytes(uint64_t n) noexcept {
    if (n == 0) return;
    uint64_t cur = bytes_used_.load(std::memory_order_relaxed);
    while (cur >= n && !bytes_used_.compare_exchange_weak(
            cur, cur - n,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
        // cur reloaded
    }
}

// -----------------------------------------------------------------------------
// LRU manipulation
// -----------------------------------------------------------------------------
void ExpertRAMCache::detach_node(ListNode* node) noexcept {
    if (!node) return;
    if (node->prev) node->prev->next = node->next;
    else            head_           = node->next;
    if (node->next) node->next->prev = node->prev;
    else            tail_           = node->prev;
    node->prev = nullptr;
    node->next = nullptr;
}

void ExpertRAMCache::attach_mru(ListNode* node) noexcept {
    if (!node) return;
    node->prev = nullptr;
    node->next = head_;
    if (head_) head_->prev = node;
    head_ = node;
    if (!tail_) tail_ = node;
}

void ExpertRAMCache::touch(ListNode* node) noexcept {
    if (!node || node->prev == nullptr) return; // already MRU
    detach_node(node);
    attach_mru(node);
}

ExpertRAMCache::ListNode* ExpertRAMCache::find_eviction_candidate() const {
    // Walk from LRU end.
    ListNode* cur = tail_;
    while (cur) {
        if (cur->entry && cur->entry->pin_count == 0) {
            return cur;
        }
        cur = cur->prev;
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
// Peek / get / contains
// -----------------------------------------------------------------------------
std::shared_ptr<CacheEntry> ExpertRAMCache::peek(ExpertCoord coord) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) return nullptr;
    return it->second->entry;
}

std::shared_ptr<CacheEntry> ExpertRAMCache::get(ExpertCoord coord) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        // Update layer stats
        std::lock_guard<std::mutex> ll(layer_mu_);
        layer_stats_[coord.layer_id].misses++;
        return nullptr;
    }

    hits_.fetch_add(1, std::memory_order_relaxed);
    auto entry = it->second->entry;
    entry->access_count.fetch_add(1, std::memory_order_relaxed);
    entry->last_access_timestamp_ns.store(now_ns(), std::memory_order_relaxed);

    touch(it->second.get());

    {
        std::lock_guard<std::mutex> ll(layer_mu_);
        layer_stats_[coord.layer_id].hits++;
    }
    return entry;
}

bool ExpertRAMCache::contains(ExpertCoord coord) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return index_.find(key_of(coord)) != index_.end();
}

// -----------------------------------------------------------------------------
// put
// -----------------------------------------------------------------------------
CacheResult ExpertRAMCache::put(
    ExpertCoord coord,
    const ExpertMetadata& metadata,
    std::shared_ptr<ExpertBuffer> buffer,
    bool replace)
{
    if (!buffer || buffer->size() == 0) {
        return CacheResult::INVALID_ARG;
    }

    const uint64_t sz = buffer->size();

    // Fast-path: try reservation first without taking the write lock.
    bool reserved = try_reserve_bytes(sz);

    std::unique_lock<std::shared_mutex> lock(mu_);

    if (!reserved) {
        // Need to evict to make room. Compute the post-eviction target so
        // that bytes_used + sz <= capacity after eviction.
        if (sz > capacity_) {
            // Single entry larger than entire cache: impossible to fit.
            return CacheResult::EXHAUSTED;
        }
        const uint64_t target = capacity_ - sz;

        // Eviction requires the write lock; release and reacquire.
        lock.unlock();
        evict_until(target);
        lock.lock();

        // Re-attempt reservation. Another thread may have grabbed the room;
        // retry a few times.
        for (int attempt = 0; attempt < 16; ++attempt) {
            if (try_reserve_bytes(sz)) { reserved = true; break; }
            lock.unlock();
            evict_until(capacity_ - sz);
            lock.lock();
        }
        if (!reserved) {
            return CacheResult::EXHAUSTED;
        }
    }

    auto key = key_of(coord);
    auto it = index_.find(key);

    if (it != index_.end()) {
        if (!replace) {
            release_reserved_bytes(sz);
            return CacheResult::ALREADY_PRESENT;
        }
        // Replace existing entry. Release old bytes first.
        ListNode* old_node = it->second.get();
        uint64_t old_sz = old_node->entry->metadata.storage_length;
        detach_node(old_node);
        index_.erase(it);
        resident_count_--;
        release_reserved_bytes(old_sz);

        // Reserve new bytes (already done above; we already hold them)
        // Re-create entry and node.
        auto entry = std::make_shared<CacheEntry>();
        entry->coord = coord;
        entry->metadata = metadata;
        entry->buffer = std::move(buffer);
        entry->state = ExpertState::RAM_RESIDENT;
        entry->creation_timestamp_ns.store(now_ns(), std::memory_order_relaxed);
        entry->last_access_timestamp_ns.store(entry->creation_timestamp_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
        entry->access_count.store(1, std::memory_order_relaxed);
        entry->pin_count.store(old_node->entry->pin_count.load(std::memory_order_relaxed), std::memory_order_relaxed);
        entry->generation.store(next_generation_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);

        auto node = std::make_unique<ListNode>(entry);
        attach_mru(node.get());
        index_[key] = std::move(node);
        resident_count_++;
        insertions_.fetch_add(1, std::memory_order_relaxed);
        return CacheResult::OK;
    }

    // New entry.
    auto entry = std::make_shared<CacheEntry>();
    entry->coord = coord;
    entry->metadata = metadata;
    entry->buffer = std::move(buffer);
    entry->state = ExpertState::RAM_RESIDENT;
    entry->creation_timestamp_ns.store(now_ns(), std::memory_order_relaxed);
    entry->last_access_timestamp_ns.store(entry->creation_timestamp_ns.load(std::memory_order_relaxed), std::memory_order_relaxed);
    entry->access_count.store(1, std::memory_order_relaxed);
    entry->pin_count.store(0, std::memory_order_relaxed);
    entry->generation.store(next_generation_.fetch_add(1, std::memory_order_relaxed), std::memory_order_relaxed);

    auto node = std::make_unique<ListNode>(entry);
    attach_mru(node.get());
    index_[key] = std::move(node);
    resident_count_++;
    insertions_.fetch_add(1, std::memory_order_relaxed);
    return CacheResult::OK;
}

// -----------------------------------------------------------------------------
// remove / clear
// -----------------------------------------------------------------------------
CacheResult ExpertRAMCache::remove(ExpertCoord coord) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) return CacheResult::NOT_FOUND;

    ListNode* node = it->second.get();
    uint64_t sz = node->entry->metadata.storage_length;
    detach_node(node);
    index_.erase(it);
    resident_count_--;
    release_reserved_bytes(sz);
    return CacheResult::OK;
}

void ExpertRAMCache::clear(bool force) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (force) {
        index_.clear();
        head_ = nullptr;
        tail_ = nullptr;
        resident_count_ = 0;
        bytes_used_.store(0, std::memory_order_relaxed);
        return;
    }

    // Non-forced: only drop unpinned entries.
    for (auto it = index_.begin(); it != index_.end();) {
        if (it->second->entry->pin_count == 0) {
            uint64_t sz = it->second->entry->metadata.storage_length;
            detach_node(it->second.get());
            it = index_.erase(it);
            resident_count_--;
            release_reserved_bytes(sz);
        } else {
            ++it;
        }
    }
}

// -----------------------------------------------------------------------------
// Pinning
// -----------------------------------------------------------------------------
CacheResult ExpertRAMCache::pin(ExpertCoord coord) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) return CacheResult::NOT_FOUND;
    it->second->entry->pin_count.fetch_add(1, std::memory_order_relaxed);
    return CacheResult::OK;
}

CacheResult ExpertRAMCache::unpin(ExpertCoord coord) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) {
        // Already evicted. Nothing to do.
        return CacheResult::NOT_FOUND;
    }
    if (it->second->entry->pin_count.load() > 0) {
        it->second->entry->pin_count.fetch_sub(1, std::memory_order_relaxed);
    }
    return CacheResult::OK;
}

CachePin ExpertRAMCache::pin_handle(ExpertCoord coord) {
    CacheResult r = pin(coord);
    return CachePin(this, coord, r == CacheResult::OK);
}

// -----------------------------------------------------------------------------
// Eviction
// -----------------------------------------------------------------------------
bool ExpertRAMCache::evict_lru() {
    std::unique_lock<std::shared_mutex> lock(mu_);
    ListNode* victim = find_eviction_candidate();
    if (!victim) return false;

    ExpertCoord vc = victim->entry->coord;
    uint64_t sz = victim->entry->metadata.storage_length;
    detach_node(victim);
    index_.erase(key_of(vc));
    resident_count_--;
    release_reserved_bytes(sz);
    evictions_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

size_t ExpertRAMCache::evict_until(uint64_t target_bytes) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    size_t evicted = 0;
    while (bytes_used_.load(std::memory_order_relaxed) > target_bytes) {
        ListNode* victim = find_eviction_candidate();
        if (!victim) break;
        ExpertCoord vc = victim->entry->coord;
        uint64_t sz = victim->entry->metadata.storage_length;
        detach_node(victim);
        index_.erase(key_of(vc));
        resident_count_--;
        release_reserved_bytes(sz);
        evictions_.fetch_add(1, std::memory_order_relaxed);
        evicted++;
    }
    return evicted;
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
CacheResult ExpertRAMCache::set_state(ExpertCoord coord, ExpertState new_state) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) return CacheResult::NOT_FOUND;
    it->second->entry->state = new_state;
    return CacheResult::OK;
}

ExpertState ExpertRAMCache::get_state(ExpertCoord coord) const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto it = index_.find(key_of(coord));
    if (it == index_.end()) return ExpertState::NOT_RESIDENT;
    return it->second->entry->state;
}

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------
CacheStats ExpertRAMCache::stats() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    CacheStats s;
    s.hits = hits_.load(std::memory_order_relaxed);
    s.misses = misses_.load(std::memory_order_relaxed);
    s.insertions = insertions_.load(std::memory_order_relaxed);
    s.evictions = evictions_.load(std::memory_order_relaxed);
    s.resident_entries = resident_count_;
    s.pinned_entries = 0;
    for (const auto& kv : index_) {
        if (kv.second->entry->pin_count > 0) s.pinned_entries++;
    }
    s.bytes_used = bytes_used_.load(std::memory_order_relaxed);
    s.bytes_reserved = s.bytes_used;
    s.peak_bytes_used = peak_bytes_used_.load(std::memory_order_relaxed);
    s.capacity_bytes = capacity_;
    s.failed_reservations = failed_reservations_.load(std::memory_order_relaxed);
    return s;
}

LayerCacheStats ExpertRAMCache::layer_stats(uint32_t layer_id) const {
    std::lock_guard<std::mutex> lock(layer_mu_);
    auto it = layer_stats_.find(layer_id);
    if (it == layer_stats_.end()) return LayerCacheStats{};
    return it->second;
}

std::vector<LayerCacheStats> ExpertRAMCache::all_layer_stats() const {
    std::lock_guard<std::mutex> lock(layer_mu_);
    std::vector<LayerCacheStats> out;
    out.reserve(layer_stats_.size());
    for (const auto& kv : layer_stats_) out.push_back(kv.second);
    return out;
}

uint64_t ExpertRAMCache::hits() const noexcept {
    return hits_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::misses() const noexcept {
    return misses_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::insertions() const noexcept {
    return insertions_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::evictions() const noexcept {
    return evictions_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::bytes_used() const noexcept {
    return bytes_used_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::bytes_reserved() const noexcept {
    return bytes_used_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::resident_count() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return resident_count_;
}
uint64_t ExpertRAMCache::pinned_count() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    uint64_t n = 0;
    for (const auto& kv : index_) {
        if (kv.second->entry->pin_count > 0) n++;
    }
    return n;
}
uint64_t ExpertRAMCache::peak_bytes_used() const noexcept {
    return peak_bytes_used_.load(std::memory_order_relaxed);
}
uint64_t ExpertRAMCache::free_bytes() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    uint64_t used = bytes_used_.load(std::memory_order_relaxed);
    return capacity_ >= used ? capacity_ - used : 0;
}
double ExpertRAMCache::utilization() const noexcept {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (capacity_ == 0) return 0.0;
    return static_cast<double>(bytes_used_.load(std::memory_order_relaxed)) / static_cast<double>(capacity_);
}

uint64_t ExpertRAMCache::now_ns() noexcept {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count());
}

} // namespace asema
