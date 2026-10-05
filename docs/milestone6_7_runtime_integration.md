# ASEMA v0.1 — M6/M7 Runtime Integration

This document maps Agent #3's CLI / benchmark surface to the existing M1–M5 public APIs.

---

## 1. Concrete Class Names (from repository)

| Subsystem | Class / Function | Header |
|-----------|------------------|--------|
| Storage backend factory | `asema::create_storage_backend(path)` | `storage.hpp` |
| CRC32 | `asema::ChecksumUtil::compute_crc32(data, len)` | `storage.hpp` |
| Manifest | `asema::ModelManifest::from_json(str)` / `to_json()` | `manifest.hpp` |
| Async loader | `asema::AsyncExpertLoader(path, manifest, workers)` | `async_loader.hpp` |
| Cache | `asema::ExpertRAMCache(capacity_bytes)` | `cache.hpp` |
| Queue | `asema::ExpertRequestQueue(capacity)` | `request_queue.hpp` |
| Prefetch | `asema::PrefetchEngine(queue, cache, K, ...)` | `prefetch.hpp` |
| Telemetry | `asema::TelemetryEngine(config)` | `telemetry.hpp` |
| Runtime | `asema::ASEMARuntime(config)` | `runtime.hpp` |
| Loader adapter | `asema::create_async_loader_adapter(...)` | `loader_adapter.hpp` |

---

## 2. Request Lifecycle

```
CLI / benchmark
  │
  ▼
ASEMARuntime::run()
  │
  │  for each token t in 1..N
  │  │  telemetry.token_begin(t)
  │  │  for each layer L in 0..num_layers-1
  │  │  │  experts = router.route(t, L)   [deterministic, locality-controlled]
  │  │  │  for each expert e in experts
  │  │  │  │  telemetry.expert_request(t, e, REQUIRED_NOW)
  │  │  │  │  entry = cache.get(e)        [if cache enabled]
  │  │  │  │  if entry: telemetry.expert_cache_hit(...); execute(entry.buffer); continue
  │  │  │  │  else:     telemetry.expert_cache_miss(...)
  │  │  │  │  │  telemetry.expert_load_begin(t, e)
  │  │  │  │  │  fut = loader->submit_future(e, REQUIRED_NOW)
  │  │  │  │  │  res = fut.get()             [IOCP + CRC inside Agent #1]
  │  │  │  │  │  telemetry.expert_load_end(t, e, res.bytes_read, lat_us, res.success)
  │  │  │  │  │  cache.put(e, meta, res.buffer) [eviction if necessary]
  │  │  │  │  │  telemetry.cache_insertion(e, res.bytes_read)
  │  │  │  │  │  execute(res.buffer)
  │  │  │  │  ▼
  │  │  │  prefetch.on_token_complete(L, chosen_per_layer)
  │  │  │  prefetch.tick(L)               [issues PREFETCH queue entries for L+1..L+K]
  │  │  telemetry.token_end(t, ...)
  ▼
telemetry.runtime_end(...)
telemetry.flush()
telemetry.write_report()   [aggregate_report.txt + .json]
```

---

## 3. Cache Lifecycle

```
constructor(capacity)
  │
  ▼
get(coord) → entry | nullptr      [promotes to MRU on hit]
put(coord, meta, buffer) → OK | EVICTED | EXHAUSTED | ALREADY_PRESENT | NOT_FOUND | INVALID_ARG
  │  atomic reservation BEFORE allocation
  │  on failure: evict_lru (non-pinned) until bytes_used + sz <= capacity
  │  retry reservation
  ▼
state changes via set_state(coord, state)  [runtime is sole owner]
pin / unpin via pin_handle / unpin / CachePin RAII
clear(force) at shutdown
```

---

## 4. Loader Lifecycle

```
AsyncExpertLoader(path, manifest, workers=4)
  │
  ▼
submit_future(coord, REQUIRED_NOW) → future<ExpertLoadResult>
  │   internal priority queue (REQUIRED_NOW > PREFETCH > BACKGROUND)
  │   internal request coalescing
  │   ReadFile(OVERLAPPED) → IOCP → worker thread → CRC32 verify
  ▼
ExpertLoadResult { buffer, bytes_read, queue_latency_ms, io_latency_ms,
                   crc_latency_ms, total_latency_ms, success, is_corrupt, ... }
shutdown() at end of run
```

---

## 5. Telemetry Lifecycle

```
TelemetryEngine(cfg)
  │
  │  ring buffer (lock-protected)
  ▼
emit(event) → enqueue into ring buffer
  │  background flusher (every 500 ms) drains → JSONL serialize → file/stdout
  ▼
flush()   [also called by shutdown]
build_report() → AggregateReport
write_report() → text + JSON files
```

---

## 6. Benchmark Lifecycle

```
asema-bench --model ... --tokens ... --warmup ... --iterations ...
  │
  ▼
ExperimentConfig { model, tokens, warmup, iterations, cache, lookahead, workers, seed, output, format }
  │
  ▼
for each (cache, lookahead, seed) in matrix:
  for iter in 1..iterations:
    ASEMARuntime rt(cfg)
    rt.run()   → RuntimeResult
    collect telemetry snapshot
  aggregate stats (mean, stddev, percentiles)
  │
  ▼
write raw/measurement_<expid>_<iter>.json
write summaries/<expid>_summary.txt
write raw/measurement_<expid>_<iter>.csv
```

---

## 7. CLI → Runtime Mapping

| CLI flag | RuntimeConfig field | Default |
|----------|---------------------|---------|
| `--model <path>` | `container_path` + `manifest_path` | required |
| `--tokens <n>` | `num_tokens` | 32 |
| `--cache <bytes>` | `ram_cache_bytes` | 0 (disabled) |
| `--lookahead <K>` | `prefetch_lookahead` | 0 |
| `--workers <n>` | `loader_workers` | 4 |
| `--seed <s>` | `router_seed` | 42 |
| `--telemetry <path>` | `telemetry_jsonl` | "" |
| `--report <path>` | `report_text_path` / `report_json_path` | "" |

---

## 8. What the CLI Does NOT Touch

- The Python generator (extended via CLI args, not modified in source).
- The storage binary layout (untouched).
- The async loader internals (used through IExpertLoader only).
- The cache state machine (writes RAM_RESIDENT and IN_USE only via set_state).
- The telemetry serialization (consumed via build_report() / write_report()).

---

## 9. Error Handling

All CLIs return one of:

| Exit code | Meaning |
|-----------|---------|
| 0 | SUCCESS |
| 2 | INVALID_ARGUMENT (bad flag, missing required arg) |
| 3 | MODEL_ERROR (manifest parse fail, missing file) |
| 4 | IO_ERROR (could not open / write file) |
| 5 | RUNTIME_ERROR (loader / cache construction fail) |
| 6 | BENCHMARK_ERROR (no valid measurements) |
| 7 | VERIFICATION_ERROR (CRC mismatch during --verify) |

These are documented in `docs/cli_reference.md`.
