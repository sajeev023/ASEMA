# ASEMA v0.1 — Milestone 6 & 7 Forensics Report

**Date**: 2026-09-26
**Author**: Agent #3
**Co-existing agents**: Agent #1 (M2), Agent #2 (M3–M5)

---

## 1. Current Repository State

The repository carries forward Agent #1's Milestone 2 and Agent #2's Milestones 3–5 unchanged. No Agent #3 file has been written prior to this report.

### Files inventoried

#### Headers (`include/asema/`)
| File | Bytes | Owner | Public API |
|------|------:|-------|------------|
| `expert.hpp` | 4098 | M1 | `ExpertCoord`, `ExpertMetadata`, `ExpertBuffer`, `ManagedExpert`, `ExpertState` |
| `manifest.hpp` | 6254 | M1 | `ModelManifest`, `ManifestLayer`, JSON in/out |
| `storage.hpp` | 1472 | M1 | `StorageBackend`, `ChecksumUtil`, `create_storage_backend()` |
| `async_loader.hpp` | 3135 | M2 (Agent #1) | `AsyncExpertLoader`, `ExpertLoadResult`, `LoadHandle`, `LoadPriority` |
| `cache.hpp` | 12305 | M3 (Agent #2) | `ExpertRAMCache`, `CacheEntry`, `CachePin`, `CacheResult`, `CacheStats` |
| `request_queue.hpp` | 10132 | M4 (Agent #2) | `ExpertRequestQueue`, `ExpertRequest`, `PoppedRequest`, `QueueStats` |
| `prefetch.hpp` | 7371 | M4 (Agent #2) | `PrefetchEngine`, `IRoutingOracle`, `LocalityRoutingOracle`, `PrefetchStats` |
| `telemetry.hpp` | 13253 | M5 (Agent #2) | `TelemetryEngine`, `TelemetryEvent`, `EventType`, `AggregateReport`, `LatencyStats` |
| `runtime.hpp` | 4146 | M5 (Agent #2) | `ASEMARuntime`, `RuntimeConfig`, `RuntimeResult` |
| `loader_adapter.hpp` | 3393 | M5 (Agent #2) | `IExpertLoader`, `AsyncLoaderAdapter` |

#### Implementations (`src/`)
- `storage/storage_ssf.cpp` — M1
- `loader/async_loader.cpp` — M2 (Agent #1)
- `cache/cache.cpp` — M3 (Agent #2)
- `queue/request_queue.cpp` — M4 (Agent #2)
- `prefetch/prefetch.cpp` — M4 (Agent #2)
- `telemetry/telemetry.cpp` — M5 (Agent #2)
- `runtime/runtime.cpp` — M5 (Agent #2)
- `runtime/loader_adapter.cpp` — M5 (Agent #2)

#### Tests (`tests/`)
- `test_storage.cpp`, `test_async_loader.cpp` — M1/M2
- `test_cache.cpp`, `test_queue.cpp`, `test_prefetch.cpp`, `test_telemetry.cpp`, `test_runtime.cpp` — M3–M5

#### Tools (`tools/`)
- `asema_inspect.cpp` — M1 (basic manifest dump, no JSON mode, no expert query, no verification)
- `asema_async_bench.cpp` — M2 (sync-vs-async I/O benchmark)
- `asema_full_bench.cpp` — M5 (full pipeline benchmark)

#### Existing synthetic data
- `examples/synthetic_moe/generate_synthetic_moe.py` — Python generator (hardcoded 4L × 16E × 128 hidden × 512 ffn = 48 MB)
- `examples/synthetic_moe/model_synthetic/manifest.json` + `model.asema` (48 MB container)

---

## 2. Public APIs I Will Consume

I will **not** modify any header. Agent #3 builds adapters/clients on top of:

- `ModelManifest::from_json()` / `to_json()` — M1
- `ChecksumUtil::compute_crc32()` — M1
- `create_storage_backend()` — M1
- `StorageBackend::read_expert_sync()` — M1
- `AsyncExpertLoader` — M2
- `ExpertRAMCache` — M3
- `ExpertRequestQueue` — M4
- `PrefetchEngine` — M4
- `TelemetryEngine` — M5
- `ASEMARuntime` — M5 (this is the primary entry point for benchmarks)

---

## 3. Files Owned by Agent #3

I am creating **all** new files in Agent #3 territory:

### Headers (`include/asema/`)
- `cli.hpp` — CLI argument parser (minimal, no external deps)
- `experiments.hpp` — experiment configuration & result types
- `synthetic_gen.hpp` — C++ synthetic MoE generator (replaces/extends Python)

### Implementations (`src/`)
- `cli/cli.cpp`
- `experiments/experiments.cpp`
- `experiments/synthetic_gen.cpp`

### CLI tool rewrites (`tools/`)
- `asema_pack.cpp` — NEW
- `asema_run.cpp` — NEW
- `asema_bench.cpp` — REPLACES `asema-full-bench` (the old one stays)

### Tests (`tests/`)
- `test_cli.cpp` — CLI parser tests
- `test_synthetic_gen.cpp` — synthetic generator tests

### Benchmark scripts (`scripts/`)
- `run_experiments.bat` — orchestration

### Configs (`configs/`)
- `m7_default.json` — declarative experiment configuration

### Reports (`reports/`)
- `runs/<timestamp>/environment.json`
- `runs/<timestamp>/configuration.json`
- `runs/<timestamp>/raw/*.json` and `*.csv`
- `runs/<timestamp>/summaries/*.txt`
- `v0.1_performance_report.md`

### Documentation (`docs/`)
- `milestone6_7_forensics.md` (this file)
- `milestone6_7_runtime_integration.md`
- `cli_reference.md`
- `benchmarking_methodology.md`
- `experiment_matrix.md`
- `scalability_methodology.md`
- `milestone6_completion.md`
- `milestone7_completion.md`
- `milestone6_7_final_report.md`

---

## 4. Files Intentionally Left Untouched

I will NOT modify:

- `include/asema/*.hpp` (all M1–M5 headers)
- `src/storage/storage_ssf.cpp` (M1)
- `src/loader/async_loader.cpp` (Agent #1 / M2)
- `src/cache/cache.cpp` (Agent #2 / M3)
- `src/queue/request_queue.cpp` (Agent #2 / M4)
- `src/prefetch/prefetch.cpp` (Agent #2 / M4)
- `src/telemetry/telemetry.cpp` (Agent #2 / M5)
- `src/runtime/runtime.cpp` (Agent #2 / M5)
- `src/runtime/loader_adapter.cpp` (Agent #2 / M5)
- `tests/test_*.cpp` (any existing test)
- `tools/asema_inspect.cpp` (M1; I will **extend** behavior by adding a new flag-aware tool)
- `tools/asema_async_bench.cpp` (M2)
- `tools/asema_full_bench.cpp` (M5; kept for backward compat, but my `asema_bench` is the new M7 entry)

I will make **additive** changes only to `CMakeLists.txt`.

---

## 5. Build System

Current `libasema_core.a` aggregates all six subsystem sources. New M6/M7 sources are added in a fenced Agent #3 block. New CLI binaries (`asema-pack`, `asema-run`, `asema-bench`) and tests are registered as new `add_executable` / `add_test` lines.

I will not change compile flags, optimization level, or compiler selection.

---

## 6. Existing Tools (what they do)

| Tool | Built by | Purpose |
|------|----------|---------|
| `asema-inspect` | M1 | Print manifest summary (text only, no JSON, no verification) |
| `asema-async-bench` | M2 | Sync vs async I/O microbenchmark |
| `asema-full-bench` | M5 | Cache × lookahead × stress sweep, writes `benchmark_report.txt` |

M6 will extend `asema-inspect` functionality (by adding `--json`, `--expert`, `--verify` flags) into a new `asema-inspect-v2` tool. The original tool stays for backward compatibility.

---

## 7. Device Guard / Build Environment Note

In this host, newly-built test binaries are intermittently blocked by Windows Device Guard. The `asema-full-bench.exe` binary is approved (built earlier) and runs successfully, producing real benchmark numbers. I will rely on it as the integration verifier when individual test binaries are blocked.

I will:
- Build all M6/M7 binaries cleanly.
- Run `asema-bench` (the new M7 tool) end-to-end on at least one synthetic workload.
- Document every blocked unit test honestly in the final report.
- Not fabricate numbers for blocked experiments.

---

## 8. Synthetic MoE Limitation

The Python generator only supports 4L × 16E × (128 hidden, 512 ffn) with fp32 dtype (786 432 B per expert, 48 MB total). It hardcodes the output path.

For M7's scaling experiments I need multiple tiers. I will:

1. Extend the Python generator to accept CLI parameters (no behavior change, just parameterization).
2. Add a C++ synthetic generator (`src/experiments/synthetic_gen.cpp`) that can build additional tiers (Tier 2: 8L × 32E, Tier 3: 12L × 64E) **on demand** when free disk + memory allow.
3. The C++ generator produces a manifest + `.asema` container that the existing M1 storage layer and M2 loader can consume directly. No new code paths in M1/M2 are needed.

---

## 9. Summary

Agent #3 territory is well-defined: build the CLI surface on top of the existing subsystems, build the experiment orchestration and synthetic generator, then run the experiments honestly. No Agent #1 or Agent #2 source is to be modified.
