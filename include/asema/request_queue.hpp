#pragma once

// ASEMA v0.1 — Milestone 4: ExpertRequestQueue
// -----------------------------------------------------------------------------
// A thread-safe, prioritized request queue with request coalescing and
// cancellation. Holds metadata-only request objects — never owns ExpertBuffers.
//
// The queue is the *logical* front-end; it does not perform I/O itself.
// Consumers (typically the runtime) call pop() to dequeue and submit to the
// underlying loader. The queue provides:
//
//   * Three priority tiers: REQUIRED_NOW > PREFETCH > BACKGROUND.
//   * FIFO within each tier, ordered by creation timestamp.
//   * Request coalescing — duplicate (coord, priority-class) requests share a
//     single underlying request and are tracked as additional waiters.
//   * Cancellation — a request can be cancelled/deprioritized before it has
//     been physically submitted.
//
// The queue does NOT depend on Agent #1's loader; it only depends on the
// ExpertCoord and LoadPriority types from the existing core headers.
// -----------------------------------------------------------------------------

#include "async_loader.hpp"   // for LoadPriority + ExpertCoord (re-exported)
#include "expert.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace asema {

// -----------------------------------------------------------------------------
// Request source classification
// -----------------------------------------------------------------------------
enum class RequestSource : uint8_t {
    RUNTIME = 0, // Issued by the inference runtime (REQUIRED_NOW)
    PREFETCH = 1, // Issued by the prefetch engine
    BACKGROUND = 2 // Issued by background warmup
};

// -----------------------------------------------------------------------------
// Request structure (metadata only — never holds an ExpertBuffer)
// -----------------------------------------------------------------------------
struct ExpertRequest {
    uint64_t request_id{0};        // Globally unique within queue lifetime
    ExpertCoord coord{0, 0};
    LoadPriority priority{LoadPriority::REQUIRED_NOW};
    RequestSource source{RequestSource::RUNTIME};

    uint64_t created_timestamp_ns{0};
    uint64_t deadline_ns{0};       // 0 = no deadline

    // Cancellation
    std::atomic<bool> cancelled{false};

    // Waiters — list of logical "consumers" blocked on this request. Each
    // waiter registers a callback function it wants fired on completion.
    // We do not store ExpertBuffers here; the runtime owns the buffer
    // insertion into the cache.

    ExpertRequest() = default;
    ExpertRequest(uint64_t id, ExpertCoord c, LoadPriority p, RequestSource s)
        : request_id(id), coord(c), priority(p), source(s) {}
};

// -----------------------------------------------------------------------------
// Pop result — what the runtime receives
// -----------------------------------------------------------------------------
struct PoppedRequest {
    uint64_t request_id{0};
    ExpertCoord coord{0, 0};
    LoadPriority priority{LoadPriority::REQUIRED_NOW};
    RequestSource source{RequestSource::RUNTIME};
    uint32_t waiter_count{1};  // How many logical callers wanted this coord
};

// -----------------------------------------------------------------------------
// Queue statistics
// -----------------------------------------------------------------------------
struct QueueStats {
    uint64_t pushes{0};
    uint64_t pops{0};
    uint64_t coalesced{0};        // pushed but merged into existing in-flight
    uint64_t cancelled{0};
    uint64_t deprioritized{0};    // PREFETCH -> BACKGROUND or similar downgrades
    uint64_t current_size{0};
    uint64_t peak_size{0};

    // Per-priority counts
    uint64_t pushes_required{0};
    uint64_t pushes_prefetch{0};
    uint64_t pushes_background{0};

    // Per-source counts
    uint64_t pushes_runtime{0};
    uint64_t pushes_prefetch_src{0};
    uint64_t pushes_background_src{0};
};

// -----------------------------------------------------------------------------
// The queue
// -----------------------------------------------------------------------------
class ExpertRequestQueue {
public:
    explicit ExpertRequestQueue(size_t max_capacity = 4096);
    ~ExpertRequestQueue();

    ExpertRequestQueue(const ExpertRequestQueue&) = delete;
    ExpertRequestQueue& operator=(const ExpertRequestQueue&) = delete;

    // -------------------------------------------------------------------------
    // Producer
    // -------------------------------------------------------------------------

    // Submit a request. Returns the assigned request_id. If the same coord is
    // already queued (or in-flight via mark_in_flight), the new request is
    // coalesced — the underlying queue node is reused and the waiter's count
    // is incremented. The returned request_id is a *logical* handle; the
    // caller uses it to cancel its own waiter.
    uint64_t push(
        ExpertCoord coord,
        LoadPriority priority = LoadPriority::REQUIRED_NOW,
        RequestSource source = RequestSource::RUNTIME);

    // Convenience: build a push() for a prefetch.
    uint64_t push_prefetch(ExpertCoord coord);
    uint64_t push_background(ExpertCoord coord);

    // -------------------------------------------------------------------------
    // Consumer
    // -------------------------------------------------------------------------

    // Non-blocking dequeue. Returns nullopt if queue is empty.
    std::optional<PoppedRequest> try_pop();

    // Blocking dequeue. Returns nullopt if shutdown.
    std::optional<PoppedRequest> wait_pop();

    // -------------------------------------------------------------------------
    // Coalescing & inflight tracking
    // -------------------------------------------------------------------------

    // Mark a coord as in-flight (about to be physically loaded). Subsequent
    // pushes for the same coord will be coalesced against this inflight entry
    // instead of being added to the queue. Returns true if the inflight state
    // was newly set; false if it was already inflight.
    bool mark_in_flight(ExpertCoord coord);

    // Clear the in-flight flag for a coord (called once load finishes).
    void clear_in_flight(ExpertCoord coord);

    // Returns true if the coord is currently in-flight (regardless of
    // whether it's also still in the queue).
    bool is_in_flight(ExpertCoord coord) const;

    // Returns true if the coord is currently queued (not yet popped).
    bool is_queued(ExpertCoord coord) const;

    // -------------------------------------------------------------------------
    // Cancellation / deprioritization
    // -------------------------------------------------------------------------

    // Cancel a request by its logical id (returned from push()). If the
    // request has already been popped, this is a no-op for the queue — the
    // caller is then responsible for cancelling the underlying loader handle.
    bool cancel(uint64_t request_id);

    // Cancel all queued requests for a given coord.
    size_t cancel_coord(ExpertCoord coord);

    // Deprioritize a queued request (e.g., PREFETCH -> BACKGROUND). No-op
    // if the request is already at BACKGROUND.
    bool deprioritize(uint64_t request_id);

    // Cancel all queued prefetch requests (used when prefetch lookahead moves
    // out of relevance).
    size_t cancel_all_prefetches();

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------
    void shutdown();
    bool is_shutdown() const noexcept { return shutdown_.load(std::memory_order_acquire); }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    size_t size() const noexcept;
    size_t capacity() const noexcept { return max_capacity_; }
    bool empty() const noexcept;

    QueueStats stats() const;

private:
    // Internal queue node
    struct Node {
        ExpertRequest req;
        uint32_t waiter_count{1};
        // We use a multiset for ordering by (priority, created_timestamp).
        // However, std::multiset doesn't easily support priority change.
        // For simplicity we use a vector sorted on insert + lazy remove.

        Node() = default;
        Node(ExpertCoord c, LoadPriority p, RequestSource s, uint64_t id, uint64_t created_ts)
            : req(id, c, p, s) {
            req.created_timestamp_ns = created_ts;
        }
    };

    // Lookup by request_id (logical handle).
    std::unordered_map<uint64_t, std::shared_ptr<Node>> by_id_;

    // Lookup by coord -> request_id (so we can coalesce / cancel).
    std::unordered_map<uint64_t, uint64_t> coord_to_request_;

    // In-flight registry
    std::unordered_set<uint64_t> inflight_;

    // Sorted container: pair<priority_uint8, created_ts_ns, node_id>
    std::vector<std::shared_ptr<Node>> sorted_nodes_; // sorted ascending by (priority, ts)

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<uint64_t> next_request_id_{1};
    size_t max_capacity_;

    // Stats
    std::atomic<uint64_t> pushes_{0};
    std::atomic<uint64_t> pops_{0};
    std::atomic<uint64_t> coalesced_{0};
    std::atomic<uint64_t> cancelled_{0};
    std::atomic<uint64_t> deprioritized_{0};
    std::atomic<uint64_t> peak_size_{0};
    std::atomic<uint64_t> pushes_required_{0};
    std::atomic<uint64_t> pushes_prefetch_{0};
    std::atomic<uint64_t> pushes_background_{0};
    std::atomic<uint64_t> pushes_runtime_{0};
    std::atomic<uint64_t> pushes_prefetch_src_{0};
    std::atomic<uint64_t> pushes_background_src_{0};

    // Helpers
    static uint64_t key_of(ExpertCoord c) noexcept;
    void sort_maintain();
};

} // namespace asema
