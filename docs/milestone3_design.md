# ASEMA v0.1 — Milestone 3 Design Document
## ExpertRAMCache

---

## 1. Goals

A production-quality, thread-safe RAM cache for expert weight buffers that:

- Provides **O(1) lookup** by `(layer_id, expert_id)`.
- Implements **O(1) LRU** eviction via doubly-linked list + `unordered_map`.
- Enforces a **hard memory budget** with reservation-before-allocation semantics.
- Supports **pinning** so that entries in active use are never evicted.
- Integrates with the existing `ExpertState` lifecycle model.
- Records **per-layer** hit/miss statistics.

---

## 2. Architecture

```
                     ┌──────────────────────────────┐
                     │ ExpertRAMCache               │
                     │                              │
   get(coord) ───────►  unordered_map<key,Node*>    │
                     │       + doubly-linked list   │
                     │                              │
   put(...) ────────►   atomic bytes_used_          │
                     │       + eviction loop        │
                     │                              │
   pin(coord) ──────►   entry.pin_count++           │
   unpin(coord) ────►   entry.pin_count--           │
                     │                              │
   stats() ─────────►   shared snapshot under lock  │
                     └──────────────────────────────┘
```

Key types (see `include/asema/cache.hpp`):

- `CacheEntry` — holds buffer, metadata, state, timestamps (atomic), pin count (atomic), generation.
- `CacheStats` / `LayerCacheStats` — read-only snapshots.
- `CachePin` — RAII guard returned by `pin_handle()`.
- `CacheResult` — outcome enum: `OK`, `MISS`, `EVICTED`, `EXHAUSTED`, `ALREADY_PRESENT`, `NOT_FOUND`, `INVALID_ARG`.

---

## 3. Hard Memory Budget

The naive implementation that checks `bytes_used < limit` *before* allocating is unsafe under concurrency. We implement reservation-first:

```
request bytes n
   │
   ▼
try_reserve_bytes(n)         ← atomic CAS on bytes_used_
   │
   ├─ OK  → allocate + commit
   │
   └─ FAIL
        │
        ▼
        evict LRU non-pinned entries until bytes_used <= capacity - n
        │
        ▼
        try_reserve_bytes(n) again (capped retry count)
        │
        ├─ OK → allocate + commit
        └─ FAIL → return EXHAUSTED
```

`try_reserve_bytes` uses a compare-and-swap loop on the `std::atomic<uint64_t> bytes_used_` so that concurrent puts across threads cannot collectively exceed the budget. `peak_bytes_used_` is updated opportunistically.

### Invariant

> `0 <= bytes_used_ <= capacity_` at all times, even under arbitrary contention.

This invariant is exercised by `test_reservation_atomic` (8 threads × 1000 ops) and `test_thrashing_10k` (10 000 entries into a 256 KB cache).

---

## 4. LRU Eviction

```
head (MRU) ◄───► ... ◄───► tail (LRU)
```

`unordered_map<key, unique_ptr<ListNode>>` for O(1) lookup; `head_/tail_` pointers for the doubly-linked list. Operations:

- `touch(node)` — O(1) detach + attach to head.
- `evict_lru()` — O(1) detach tail (if `pin_count == 0`).
- `evict_until(target)` — repeats evict_lru while `bytes_used_ > target`.

Skipping pinned entries is implemented in `find_eviction_candidate()` — the LRU walk stops at the first non-pinned entry. If every entry is pinned, eviction returns false and `put()` ultimately reports `EXHAUSTED`.

---

## 5. Pinning

`pin(coord)` increments `entry.pin_count`. `unpin` decrements. RAII helper `CachePin` automatically unpins in its destructor. While `pin_count > 0`, the entry is invisible to eviction.

If memory is exhausted and every resident entry is pinned, `put()` returns `EXHAUSTED` — it does NOT deadlock, block, or evict a pinned entry.

---

## 6. State Integration

The cache stores `ExpertState` per entry. Allowed transitions are owned by the runtime:

```
NOT_RESIDENT ─► LOADING ─► RAM_RESIDENT ─► IN_USE ─► RAM_RESIDENT ─► EVICTING ─► NOT_RESIDENT
```

The cache itself never moves a state; it only stores and returns the current value via `get_state()` / `set_state()`. This avoids duplicating the lifecycle system defined in `expert.hpp`.

---

## 7. Statistics

- **Global**: `hits`, `misses`, `insertions`, `evictions`, `bytes_used`, `bytes_reserved`, `peak_bytes_used`, `capacity_bytes`, `pinned_entries`, `failed_reservations`, `resident_entries`.
- **Per-layer**: `hits`, `misses`, `hit_rate`. Lazily grown.
- **Hit rate formula**: `hits / (hits + misses)`.

All counters are lock-free atomics except `resident_count_` and the per-layer stats map, which are protected by short critical sections.

---

## 8. Thread Safety

- `std::shared_mutex mu_` — multiple readers OR one writer.
- `bytes_used_` and other counters — atomic.
- Per-layer map — protected by a separate `std::mutex layer_mu_`.

The LRU list manipulation happens only under the write lock. `get()` and `peek()` take the read lock.

---

## 9. Tests

| Test | Property |
|------|----------|
| `test_cache_insert_and_hit` | basic put/get/hit-count |
| `test_cache_miss` | miss increments counter |
| `test_lru_order` | MRU promotion; oldest is evicted first |
| `test_memory_budget` | budget never exceeded; eviction under pressure |
| `test_pinned_eviction_protection` | pinned entries survive; full-pin ⇒ EXHAUSTED |
| `test_concurrent_insert_lookup` | 8 threads, 500 ops/thread — no errors |
| `test_thrashing_1k` | 1 000 entries into 64 KB cache — evictions > 0 |
| `test_thrashing_10k` | 10 000 entries into 256 KB cache — invariant holds |
| `test_cache_clear` | clear drops resident_count to 0 |
| `test_reservation_atomic` | max-seen bytes ≤ budget under contention |
| `test_pin_handle` | RAII semantics |
| `test_state_transitions` | state stored/retrieved correctly |
| `test_layer_stats` | per-layer counters isolated |
| `test_already_present` | `replace=false` returns ALREADY_PRESENT |
| `test_remove` | remove drops bytes from budget |

**Result**: 15 / 15 pass.

---

## 10. Known Limitations

- `set_capacity()` to a smaller value evicts but does not cancel in-flight loads.
- `clear(false)` (non-forced) leaves pinned entries but still consumes their bytes from the budget.
- The cache does not dedup equal buffers (only equal coords). Future work could share buffers across overlapping model shards.
