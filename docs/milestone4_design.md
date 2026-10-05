# ASEMA v0.1 — Milestone 4 Design Document
## Prioritized Expert Request Queue + Deterministic Prefetch Engine

---

## 1. Goals

1. **ExpertRequestQueue** — a thread-safe, prioritized queue with:
   - Three priority tiers: `REQUIRED_NOW` > `PREFETCH` > `BACKGROUND`.
   - FIFO within a tier, ordered by creation timestamp.
   - **Request coalescing**: duplicate `(coord)` requests share one underlying node.
   - **Cancellation / deprioritization**: prefetch can be cancelled or downgraded.

2. **PrefetchEngine** — a deterministic, pluggable lookahead prefetcher:
   - Configurable lookahead K ∈ {0, 1, 2, 3, 4, 6}.
   - Pluggable `IRoutingOracle` (default: `LocalityRoutingOracle`).
   - Honours queue depth, cache capacity, and per-tick byte budget.
   - **Never starves REQUIRED_NOW** — preempts or cancels prefetch as needed.
   - Tracks useful / wasted / duplicate / cancelled prefetches.

---

## 2. ExpertRequestQueue Architecture

```
                  push() ─────► by_id_ (logical handles)
                                  │
                                  ├─► sorted_nodes_ (priority, ts) ascending
                                  │
                                  └─► coord_to_request_ (dedup)
                  
                  pop() ──────► sorted_nodes_.front() ─► PoppedRequest
```

Internal node:
```cpp
struct Node {
    ExpertCoord coord;
    LoadPriority priority;
    RequestSource source;
    uint64_t request_id;
    uint64_t created_timestamp_ns;
    uint32_t waiter_count;
    std::atomic<bool> cancelled;
};
```

Coalescing: a second `push(coord, ...)` while `coord_to_request_[coord]` exists increments the existing node's `waiter_count` instead of creating a new node. If the second push has higher priority, the existing node is boosted and the sorted container is re-sorted.

Cancellation: `cancel(request_id)` sets the atomic `cancelled` flag and removes the node from `sorted_nodes_` so it is never popped. The runtime is responsible for cancelling any loader handle that was already submitted.

---

## 3. Operation Semantics

| Op | Behavior |
|----|----------|
| `push(coord, prio, source)` | Coalesce or enqueue. Returns logical id. |
| `try_pop()` | Pop head (highest priority, oldest). Skip cancelled. |
| `wait_pop()` | Block on `cv_` until queue non-empty or shutdown. |
| `cancel(id)` | Mark cancelled + remove from sorted list. |
| `cancel_coord(coord)` | Cancel all waiters for a coord. |
| `cancel_all_prefetches()` | Cancel all PREFETCH priority entries. |
| `deprioritize(id)` | Move one tier down (REQUIRED→PREFETCH, PREFETCH→BACKGROUND). |
| `mark_in_flight(coord)` | Add to inflight set; pushes for this coord coalesce against it. |
| `clear_in_flight(coord)` | Remove from inflight set; future pushes will re-enqueue. |

---

## 4. PrefetchEngine Design

```
on_token_complete(layer, chosen_experts_per_layer)
   │
   ▼
push to recent_history_ (bounded ring, depth = 8)
   │
tick(current_layer)
   │
   ▼
for k = 1..K:
   │
   target_layer = current_layer + k
   │
   predictions = oracle_->predict(target_layer, history)
   │
   for each predicted coord c:
       if cache.contains(c): skip
       if queue.is_in_flight(c) || queue.is_queued(c): duplicate++
       if queue.size() >= capacity/4:        gated_by_queue++; return
       if per_tick_bytes >= budget:          gated_by_budget++; return
       queue.push_prefetch(c)
       prefetch_requests++
       tracking[c]++
```

The oracle is pluggable. The default `LocalityRoutingOracle` is locality-aware:
- Counts occurrences of each `(layer, expert)` pair in the last `history_depth_` tokens.
- Returns the top-N experts per layer.

### Starvation protection

The prefetch engine never holds the queue lock; the queue is the scheduler. `REQUIRED_NOW` always wins because:

1. The queue pops the lowest-priority-value (REQUIRED_NOW=0) first.
2. The prefetcher's `tick()` only enqueues PREFETCH entries.
3. `cancel_all_prefetches()` is called on `shutdown()` and may be called by the runtime when the lookahead horizon moves.

---

## 5. Metrics

### QueueStats
- pushes, pops, coalesced, cancelled, deprioritized, current_size, peak_size
- per-priority counters, per-source counters

### PrefetchStats
- prefetch_requests, useful_prefetches, wasted_prefetches, cancelled_prefetches
- duplicate_prefetches, prefetch_bytes, prefetch_hit_rate
- avg_prefetch_latency_us (placeholder)
- gated_by_queue, gated_by_cache, gated_by_budget

### Prefetch hit rate formula

```
prefetch_hit_rate = useful_prefetches /
                    (useful_prefetches + wasted_prefetches + cancelled_prefetches)
```

Duplicate prefetches are intentionally excluded from the denominator (they were never "wasted" — they were suppressed).

---

## 6. Tests

### Queue (14 tests, all pass)
- basic_push_pop, priority_ordering, fifo_within_priority
- request_coalescing, priority_boost, cancel, cancel_coord
- cancel_all_prefetches, deprioritize, inflight_tracking
- concurrent_push_pop (1 producer + 1 consumer, 1000 each)
- capacity_limit, wait_pop, required_now_starvation

### Prefetch (9 tests, all pass)
- lookahead_0 / 1 / 2
- prefetch_priority (REQUIRED_NOW wins over PREFETCH)
- prefetch_duplicate_coalescing (two ticks, same coord)
- prefetch_cancellation
- prefetch_cache_interaction (cache hit suppresses prefetch)
- required_starvation
- useful_wasted_accounting

---

## 7. Known Limitations

- The queue uses a sorted vector, so `try_pop()` is O(log n) (sort_maintain on insert). Acceptable up to capacity 4096.
- The prefetch engine is single-threaded; tick() is called sequentially by the runtime.
- The default oracle is intentionally simple (no ML); it predicts by historical locality only.
- `cancel_coord` reports a single count; for waiters > 1 the cancel count may be 1 even though multiple logical callers were affected. This is documented behavior.
