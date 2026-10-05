// ASEMA v0.1 — Milestone 4: ExpertRequestQueue implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/request_queue.hpp"

#include <algorithm>
#include <chrono>

namespace asema {

ExpertRequestQueue::ExpertRequestQueue(size_t max_capacity)
    : max_capacity_(max_capacity) {}

ExpertRequestQueue::~ExpertRequestQueue() {
    shutdown();
}

uint64_t ExpertRequestQueue::key_of(ExpertCoord c) noexcept {
    return (static_cast<uint64_t>(c.layer_id) << 32) | static_cast<uint64_t>(c.expert_id);
}

uint64_t ExpertRequestQueue::push(
    ExpertCoord coord,
    LoadPriority priority,
    RequestSource source)
{
    if (shutdown_.load(std::memory_order_acquire)) return 0;

    std::unique_lock<std::mutex> lock(mu_);

    const uint64_t key = key_of(coord);

    // Coalesce if already queued or in-flight.
    auto it = coord_to_request_.find(key);
    if (it != coord_to_request_.end()) {
        auto node_it = by_id_.find(it->second);
        if (node_it != by_id_.end()) {
            // The same logical node already represents this coord.
            node_it->second->waiter_count++;
            // Priority boost: if the new request is higher priority, upgrade.
            if (static_cast<uint8_t>(priority) <
                static_cast<uint8_t>(node_it->second->req.priority)) {
                node_it->second->req.priority = priority;
                sort_maintain();
                cv_.notify_all();
            }
            coalesced_.fetch_add(1, std::memory_order_relaxed);
            return it->second;
        }
    }

    if (by_id_.size() >= max_capacity_) {
        // Queue full. Apply back-pressure: refuse the new request.
        // Caller can retry later or downgrade to BACKGROUND.
        return 0;
    }

    const uint64_t id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
    const uint64_t created_ts =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    auto node = std::make_shared<Node>(coord, priority, source, id, created_ts);
    by_id_[id] = node;
    coord_to_request_[key] = id;
    sorted_nodes_.push_back(node);
    sort_maintain();

    // Update stats
    pushes_.fetch_add(1, std::memory_order_relaxed);
    if (priority == LoadPriority::REQUIRED_NOW) pushes_required_.fetch_add(1, std::memory_order_relaxed);
    else if (priority == LoadPriority::PREFETCH) pushes_prefetch_.fetch_add(1, std::memory_order_relaxed);
    else                                          pushes_background_.fetch_add(1, std::memory_order_relaxed);
    if (source == RequestSource::RUNTIME)        pushes_runtime_.fetch_add(1, std::memory_order_relaxed);
    else if (source == RequestSource::PREFETCH)  pushes_prefetch_src_.fetch_add(1, std::memory_order_relaxed);
    else                                          pushes_background_src_.fetch_add(1, std::memory_order_relaxed);

    uint64_t cur = by_id_.size();
    uint64_t peak = peak_size_.load(std::memory_order_relaxed);
    while (cur > peak && !peak_size_.compare_exchange_weak(
            peak, cur, std::memory_order_acq_rel, std::memory_order_relaxed)) {}

    lock.unlock();
    cv_.notify_one();
    return id;
}

uint64_t ExpertRequestQueue::push_prefetch(ExpertCoord coord) {
    return push(coord, LoadPriority::PREFETCH, RequestSource::PREFETCH);
}

uint64_t ExpertRequestQueue::push_background(ExpertCoord coord) {
    return push(coord, LoadPriority::BACKGROUND, RequestSource::BACKGROUND);
}

void ExpertRequestQueue::sort_maintain() {
    std::sort(sorted_nodes_.begin(), sorted_nodes_.end(),
        [](const std::shared_ptr<Node>& a, const std::shared_ptr<Node>& b) {
            const uint8_t pa = static_cast<uint8_t>(a->req.priority);
            const uint8_t pb = static_cast<uint8_t>(b->req.priority);
            if (pa != pb) return pa < pb;
            return a->req.created_timestamp_ns < b->req.created_timestamp_ns;
        });
}

std::optional<PoppedRequest> ExpertRequestQueue::try_pop() {
    std::lock_guard<std::mutex> lock(mu_);

    // Skip cancelled nodes
    while (!sorted_nodes_.empty()) {
        auto node = sorted_nodes_.front();
        sorted_nodes_.erase(sorted_nodes_.begin());

        if (node->req.cancelled.load(std::memory_order_acquire)) {
            // Already cancelled; clean up.
            by_id_.erase(node->req.request_id);
            coord_to_request_.erase(key_of(node->req.coord));
            continue;
        }

        PoppedRequest p;
        p.request_id = node->req.request_id;
        p.coord = node->req.coord;
        p.priority = node->req.priority;
        p.source = node->req.source;
        p.waiter_count = node->waiter_count;

        // Note: we do NOT remove from by_id_ / coord_to_request_ here — the
        // request transitions to "in-flight" via mark_in_flight() by the
        // caller. If the caller does not mark it, the next push() will
        // coalesce against it (which is also valid).

        pops_.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    return std::nullopt;
}

std::optional<PoppedRequest> ExpertRequestQueue::wait_pop() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [this]() {
        return shutdown_.load(std::memory_order_acquire) || !sorted_nodes_.empty();
    });
    if (shutdown_.load(std::memory_order_acquire) && sorted_nodes_.empty()) {
        return std::nullopt;
    }
    lock.unlock();
    return try_pop();
}

bool ExpertRequestQueue::mark_in_flight(ExpertCoord coord) {
    std::lock_guard<std::mutex> lock(mu_);
    auto [it, inserted] = inflight_.insert(key_of(coord));
    (void)it;
    return inserted;
}

void ExpertRequestQueue::clear_in_flight(ExpertCoord coord) {
    std::lock_guard<std::mutex> lock(mu_);
    inflight_.erase(key_of(coord));
    // Also clear coord_to_request_ mapping so future pushes go through the
    // queue again instead of coalescing forever.
    coord_to_request_.erase(key_of(coord));
    auto it = by_id_.begin();
    while (it != by_id_.end()) {
        if (it->second->req.coord == coord) {
            it = by_id_.erase(it);
        } else {
            ++it;
        }
    }
    // Drop cancelled or popped nodes from sorted list
    sorted_nodes_.erase(
        std::remove_if(sorted_nodes_.begin(), sorted_nodes_.end(),
            [&](const std::shared_ptr<Node>& n) {
                return n->req.coord == coord || n->req.cancelled.load();
            }),
        sorted_nodes_.end());
}

bool ExpertRequestQueue::is_in_flight(ExpertCoord coord) const {
    std::lock_guard<std::mutex> lock(mu_);
    return inflight_.find(key_of(coord)) != inflight_.end();
}

bool ExpertRequestQueue::is_queued(ExpertCoord coord) const {
    std::lock_guard<std::mutex> lock(mu_);
    return coord_to_request_.find(key_of(coord)) != coord_to_request_.end();
}

bool ExpertRequestQueue::cancel(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_id_.find(request_id);
    if (it == by_id_.end()) return false;
    it->second->req.cancelled.store(true, std::memory_order_release);
    cancelled_.fetch_add(1, std::memory_order_relaxed);
    // Remove from sorted list immediately so it's not popped.
    sorted_nodes_.erase(
        std::remove_if(sorted_nodes_.begin(), sorted_nodes_.end(),
            [&](const std::shared_ptr<Node>& n) {
                return n->req.request_id == request_id;
            }),
        sorted_nodes_.end());
    cv_.notify_all();
    return true;
}

size_t ExpertRequestQueue::cancel_coord(ExpertCoord coord) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t count = 0;
    auto it = coord_to_request_.find(key_of(coord));
    if (it == coord_to_request_.end()) return 0;
    auto node_it = by_id_.find(it->second);
    if (node_it != by_id_.end()) {
        node_it->second->req.cancelled.store(true, std::memory_order_release);
        cancelled_.fetch_add(1, std::memory_order_relaxed);
        count = node_it->second->waiter_count;
    }
    sorted_nodes_.erase(
        std::remove_if(sorted_nodes_.begin(), sorted_nodes_.end(),
            [&](const std::shared_ptr<Node>& n) {
                return n->req.coord == coord;
            }),
        sorted_nodes_.end());
    cv_.notify_all();
    return count;
}

bool ExpertRequestQueue::deprioritize(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_id_.find(request_id);
    if (it == by_id_.end()) return false;
    auto& prio = it->second->req.priority;
    if (prio == LoadPriority::BACKGROUND) return false;
    if (prio == LoadPriority::PREFETCH) {
        prio = LoadPriority::BACKGROUND;
    } else { // REQUIRED_NOW -> PREFETCH (rarely useful, but supported)
        prio = LoadPriority::PREFETCH;
    }
    sort_maintain();
    deprioritized_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_all();
    return true;
}

size_t ExpertRequestQueue::cancel_all_prefetches() {
    std::lock_guard<std::mutex> lock(mu_);
    size_t count = 0;
    for (auto& kv : by_id_) {
        if (kv.second->req.priority == LoadPriority::PREFETCH) {
            kv.second->req.cancelled.store(true, std::memory_order_release);
            count += kv.second->waiter_count;
            cancelled_.fetch_add(kv.second->waiter_count, std::memory_order_relaxed);
        }
    }
    sorted_nodes_.erase(
        std::remove_if(sorted_nodes_.begin(), sorted_nodes_.end(),
            [](const std::shared_ptr<Node>& n) {
                return n->req.priority == LoadPriority::PREFETCH;
            }),
        sorted_nodes_.end());
    cv_.notify_all();
    return count;
}

void ExpertRequestQueue::shutdown() {
    bool expected = false;
    if (!shutdown_.compare_exchange_strong(expected, true)) return;
    cv_.notify_all();
}

size_t ExpertRequestQueue::size() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return by_id_.size();
}

bool ExpertRequestQueue::empty() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return by_id_.empty();
}

QueueStats ExpertRequestQueue::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    QueueStats s;
    s.pushes = pushes_.load(std::memory_order_relaxed);
    s.pops = pops_.load(std::memory_order_relaxed);
    s.coalesced = coalesced_.load(std::memory_order_relaxed);
    s.cancelled = cancelled_.load(std::memory_order_relaxed);
    s.deprioritized = deprioritized_.load(std::memory_order_relaxed);
    s.current_size = by_id_.size();
    s.peak_size = peak_size_.load(std::memory_order_relaxed);

    s.pushes_required = pushes_required_.load(std::memory_order_relaxed);
    s.pushes_prefetch = pushes_prefetch_.load(std::memory_order_relaxed);
    s.pushes_background = pushes_background_.load(std::memory_order_relaxed);
    s.pushes_runtime = pushes_runtime_.load(std::memory_order_relaxed);
    s.pushes_prefetch_src = pushes_prefetch_src_.load(std::memory_order_relaxed);
    s.pushes_background_src = pushes_background_src_.load(std::memory_order_relaxed);
    return s;
}

} // namespace asema
