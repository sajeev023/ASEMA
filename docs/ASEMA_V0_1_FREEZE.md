# ASEMA v0.1 Freeze

**Date**: 2026-09-26
**Status**: M1–M7 frozen. M8 may begin.

---

## 1. Frozen state

The following deliverables are frozen as ASEMA v0.1. **No further changes** to these subsystems are permitted from this commit forward, except for explicit freeze-bug fixes (must include a reproducible test that demonstrates the bug and the fix).

### Headers (`include/asema/`)

- `expert.hpp` — M1 core types (ExpertCoord, ExpertMetadata, ExpertBuffer, ManagedExpert, ExpertState)
- `manifest.hpp` — M1 manifest schema (ModelManifest, JSON in/out)
- `storage.hpp` — M1 storage backend abstraction
- `async_loader.hpp` — M2 async loader (Agent #1)
- `cache.hpp` — M3 RAM cache (Agent #2)
- `request_queue.hpp` — M4 request queue (Agent #2)
- `prefetch.hpp` — M4 prefetch engine (Agent #2)
- `telemetry.hpp` — M5 telemetry engine (Agent #2)
- `runtime.hpp` — M5 ASEMARuntime (Agent #2)
- `loader_adapter.hpp` — M5 loader adapter (Agent #2)

### Sources (`src/`)

- `storage/storage_ssf.cpp` — M1
- `loader/async_loader.cpp` — M2 (Agent #1)
- `cache/cache.cpp` — M3 (Agent #2)
- `queue/request_queue.cpp` — M4 (Agent #2)
- `prefetch/prefetch.cpp` — M4 (Agent #2)
- `telemetry/telemetry.cpp` — M5 (Agent #2)
- `runtime/runtime.cpp` — M5 (Agent #2)
- `runtime/loader_adapter.cpp` — M5 (Agent #2)

### Tools (`tools/`)

- `asema_inspect.cpp` — M1
- `asema_async_bench.cpp` — M2 (Agent #1)
- `asema-pack.cpp` — M6 (Agent #3)
- `asema-run.cpp` — M6 (Agent #3)
- `asema-bench.cpp` — M6/M7 (Agent #3, hardened M7 in this session)
- `asema_full_bench.cpp` — M5 (Agent #2)

### Tests (`tests/`)

- `test_storage.cpp` — M1
- `test_async_loader.cpp` — M2 (Agent #1)
- `test_cache.cpp` — M3 (Agent #2)
- `test_queue.cpp` — M4 (Agent #2)
- `test_prefetch.cpp` — M4 (Agent #2)
- `test_telemetry.cpp` — M5 (Agent #2)
- `test_runtime.cpp` — M5 (Agent #2)
- `test_cli.cpp` — M6/M7 (Agent #3)
- `test_synthetic_gen.cpp` — M6/M7 (Agent #3)

### Documentation (`docs/`)

- All M1–M7 design docs, forensics, and methodology documents.

### Benchmark artifacts (`reports/`)

- `runs/` — all M7 raw runs (preserved, not to be deleted)
- `v0.1_performance_report.md` — initial M7 report (preserved)
- `v0.1_hardened_performance_report.md` — hardened M7 report (this session)

---

## 2. Hardened M7 freeze-gate checklist

| Gate | Status |
|------|:------:|
| Tier 3 generated safely (288 MB, disk-checked, CRC-validated) | ✅ |
| Logical/physical bytes distinguished in telemetry + benchmark output | ✅ |
| Coalesced counter captured (`coalesced_requests` field) | ✅ |
| I/O dispatch counter captured (`ssd_reads` field) | ✅ |
| Iterations configurable; N=10 research runs completed | ✅ |
| Locality sweep executed (high / medium / low / random / adversarial) | ✅ |
| Prefetch sweep executed (K=0..6) with N=10 | ✅ |
| Worker sweep executed (1/2/4/8) with N=10 | ✅ |
| Seed sweep executed (11/42/99) with N=10 | ✅ |
| Warm/cold distinction recorded per run | ✅ (informational only; Windows portable) |
| Tier 2 anomaly investigated (access-pattern analyzer) | ✅ |
| Hardened report generated (`reports/v0.1_hardened_performance_report.md`) | ✅ |
| Raw data preserved | ✅ (320 hardened measurements + 111 prior measurements) |
| All existing M1–M7 tests preserved | ✅ |

**All freeze-gate items pass.**

---

## 3. Public API contract (for M8 / future use)

### `IExpertLoader` (`include/asema/loader_adapter.hpp`)

```
LoadHandle submit(coord, priority, callback?) → LoadHandle
future<ExpertLoadResult> submit_future(coord, priority)
bool cancel(LoadHandle)
optional<ExpertLoadResult> try_get_result(LoadHandle)
ExpertLoadResult wait(LoadHandle)
shutdown()
uint64_t total_requests_submitted()
uint64_t total_coalesced_requests()
uint64_t total_io_dispatches()
uint64_t total_bytes_loaded()
uint64_t total_checksum_failures()
uint64_t active_in_flight()
```

### `ExpertRAMCache` (`include/asema/cache.hpp`)

```
ExpertRAMCache(capacity_bytes)
get(coord) → shared_ptr<CacheEntry>
put(coord, meta, buffer, replace=true) → CacheResult
contains(coord) → bool
remove(coord) → CacheResult
clear(force=false)
pin(coord) / unpin(coord) / pin_handle(coord) → CachePin (RAII)
evict_lru() → bool
evict_until(target_bytes) → size_t
set_state(coord, ExpertState) → CacheResult
get_state(coord) → ExpertState
stats() → CacheStats
layer_stats(layer_id) → LayerCacheStats
```

### `ExpertRequestQueue` (`include/asema/request_queue.hpp`)

```
ExpertRequestQueue(capacity)
push(coord, prio, source) → request_id (logical)
try_pop() → optional<PoppedRequest>
wait_pop() → optional<PoppedRequest>
mark_in_flight(coord) / clear_in_flight(coord)
is_in_flight(coord) / is_queued(coord)
cancel(request_id)
cancel_coord(coord)
deprioritize(request_id)
cancel_all_prefetches()
shutdown()
size() / capacity() / empty()
stats() → QueueStats
```

### `PrefetchEngine` (`include/asema/prefetch.hpp`)

```
PrefetchEngine(queue, cache, K, max_experts_per_layer, history_depth, max_prefetch_bytes_per_tick)
set_lookahead(K)
set_oracle(oracle)
on_token_complete(layer, chosen_per_layer)
tick(current_layer)
record_useful(coord) / record_wasted(coord) / record_cancelled(coord)
stats() → PrefetchStats
shutdown()
```

### `TelemetryEngine` (`include/asema/telemetry.hpp`)

```
TelemetryEngine(cfg)
emit(event)
[convenience helpers for each EventType]
record_*_latency_us(double)
record_first_token_latency_ms(double)
[increment_* helpers]
set_queue_depth / set_ram_bytes
flush() / build_report() / write_report()
shutdown()
```

### `ASEMARuntime` (`include/asema/runtime.hpp`)

```
ASEMARuntime(config)
run() → RuntimeResult
cache() / queue() / prefetch() / telemetry() / loader() accessors
```

---

## 4. Storage measurement contract

The layered model in `docs/windows_storage_measurement.md` is now part of v0.1.

| Layer | Measurable from user space? | ASEMA v0.1 measures? |
|-------|------------------------------|---------------------|
| Runtime application requests | ✅ | ✅ |
| AsyncExpertLoader (post-coalescing) | ✅ | ✅ |
| OS file cache | ❌ portable | ❌ declared as limitation |
| Hardware device | ❌ portable | ❌ declared as limitation |

**Any v0.1 result claiming hardware-level SSD performance requires ETW or vendor SMART, neither of which is integrated.**

---

## 5. M8 may now begin

The M8 real-model integration work (per `docs/ASEMA v0.1 v0.2 M8 MASTER PROMPT`) may begin, subject to:

- No modification of the frozen M1–M7 sources (except for explicit freeze-bug fixes).
- New M8 code lives under:
  - `include/asema/m8/`
  - `src/m8/`
  - `tools/m8_*`
  - `tests/test_m8_*`
  - `reports/m8_*`
  - `docs/m8_*`
- M8 correctness is the FIRST priority (router equivalence, expert equivalence, logit equivalence).
- Performance benchmarks only after correctness is verified.

---

## 6. Stop conditions (for any v0.1+ change)

- STOP if a router output differs from reference beyond documented tolerance.
- STOP if expert weights are corrupted.
- STOP if tokenizer differs.
- STOP if reference vs ASEMA numerical outputs diverge unexpectedly.
- STOP if memory safety is violated.
- STOP if benchmark data is incomplete.

Failures are research results; they are not hidden.

---

## 7. Reproduction

To regenerate every measurement in the hardened report:

```
cmake -B build -G Ninja ...
cmake --build build

asema-pack --tier 1 --output examples/synthetic_moe/model_synthetic
asema-pack --tier 2 --output examples/synthetic_moe/model_tier2
asema-pack --tier 3 --output examples/synthetic_moe/model_tier3

scripts/run_hardened_experiments.bat
```

Wall time: ~6 minutes on the test host.
