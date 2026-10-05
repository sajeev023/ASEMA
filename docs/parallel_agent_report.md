# ASEMA v0.1 — Parallel Agent Forensics Report

**Date**: 2026-09-25
**Author**: Agent #2 (this submission)
**Co-existing Agent**: Agent #1 — Milestone 2 (Async Expert Loader)

---

## 1. Repository State Audit

The repository was inspected **without git history** (no `.git` directory is present in this checkout). Forensic analysis was performed by:

1. Walking every directory under the workspace root.
2. Reading every header and source file.
3. Building the project and running all existing tests to confirm Agent #1's work compiles and passes.
4. Cross-referencing file timestamps to identify which files were modified most recently.

### 1.1 Workspace Inventory

| Path | Type | Bytes | Status |
|------|------|------:|--------|
| `CMakeLists.txt` | Build script | 1230 | Co-editable (minimal change strategy) |
| `include/asema/expert.hpp` | Header | 4098 | **Milestone 1 — DO NOT MODIFY** |
| `include/asema/manifest.hpp` | Header | 6254 | **Milestone 1 — DO NOT MODIFY** |
| `include/asema/storage.hpp` | Header | 1472 | **Milestone 1 — DO NOT MODIFY** |
| `include/asema/async_loader.hpp` | Header | 3135 | **Agent #1 — DO NOT MODIFY** |
| `src/storage/storage_ssf.cpp` | Impl | 7646 | **Milestone 1 — DO NOT MODIFY** |
| `src/loader/async_loader.cpp` | Impl | 20749 | **Agent #1 — DO NOT MODIFY** |
| `tests/test_storage.cpp` | Test | 2750 | **Milestone 1 — DO NOT MODIFY** |
| `tests/test_async_loader.cpp` | Test | 8079 | **Agent #1 — DO NOT MODIFY** |
| `tools/asema_inspect.cpp` | Tool | 3020 | **Milestone 1 — DO NOT MODIFY** |
| `tools/asema_async_bench.cpp` | Tool | 9406 | **Agent #1 — DO NOT MODIFY** |
| `examples/synthetic_moe/...` | Data | — | Generator + container |
| `include/nlohmann/json.hpp` | Third-party | 919975 | DO NOT MODIFY |
| `src/cache/` | (empty) | — | **Agent #2 territory** |
| `src/queue/` | (empty) | — | **Agent #2 territory** |
| `src/prefetch/` | (empty) | — | **Agent #2 territory** |
| `src/telemetry/` | (empty) | — | **Agent #2 territory** |
| `src/runtime/` | (empty) | — | **Agent #2 territory** |
| `src/gpu/` | (empty) | — | Deferred (CPU-only path) |

### 1.2 Build Verification (Agent #1's Work)

Performed `ninja -C build` and ran all existing test binaries. Results:

```
[OK] test_storage.exe        — Milestone 1 storage/manifest tests pass
[OK] test_async_loader.exe   — All 8 Milestone 2 tests + 5 benchmarks pass
```

Agent #1's `AsyncExpertLoader` is fully functional with:
- IOCP-based overlapped Windows I/O
- Priority dispatch queue (REQUIRED_NOW > PREFETCH > BACKGROUND)
- Request coalescing (duplicate suppression)
- Callback + future + sync wait APIs
- Per-request sub-millisecond timing breakdown
- Cancellation support
- Built-in metrics counters

No rework needed. I will integrate through an adapter layer.

---

## 2. Files Owned / Modified by Agent #1

Agent #1 is the source of truth for:

- `include/asema/async_loader.hpp`
- `src/loader/async_loader.cpp`
- `tests/test_async_loader.cpp`
- `tools/asema_async_bench.cpp`
- `docs/milestone2_design.md`

These will not be touched by Agent #2 except for the absolute minimum required to add new test targets / tools to `CMakeLists.txt`.

---

## 3. Files Safe for Agent #2

I am creating **all** new files under my designated directories:

- `include/asema/cache.hpp`
- `include/asema/request_queue.hpp`
- `include/asema/prefetch.hpp`
- `include/asema/telemetry.hpp`
- `include/asema/runtime.hpp`
- `include/asema/loader_adapter.hpp`
- `src/cache/cache.cpp`
- `src/queue/request_queue.cpp`
- `src/prefetch/prefetch.cpp`
- `src/telemetry/telemetry.cpp`
- `src/runtime/runtime.cpp`
- `src/runtime/loader_adapter.cpp`
- `tests/test_cache.cpp`
- `tests/test_queue.cpp`
- `tests/test_prefetch.cpp`
- `tests/test_telemetry.cpp`
- `tests/test_runtime.cpp`
- `tools/asema_full_bench.cpp`
- `docs/milestone3_design.md`
- `docs/milestone4_design.md`
- `docs/milestone5_design.md`
- `docs/integration_contract.md`
- `docs/parallel_agent_report.md` (this file)

I will make the **smallest possible additive change** to `CMakeLists.txt` to register my new sources and tests. No existing source will be renamed, deleted, or rewritten.

---

## 4. Interfaces Already Available (Reusable)

Agent #1 has already defined and implemented:

### 4.1 `LoadPriority` enum (`include/asema/async_loader.hpp:19`)

```cpp
enum class LoadPriority : uint8_t {
    REQUIRED_NOW = 0,
    PREFETCH     = 1,
    BACKGROUND   = 2
};
```

I will reuse this enum rather than define a parallel one.

### 4.2 `ExpertState` enum (`include/asema/expert.hpp:13`)

```cpp
enum class ExpertState : uint8_t {
    NOT_RESIDENT = 0, LOADING = 1, RAM_RESIDENT = 2, GPU_RESIDENT = 3,
    IN_USE = 4, WARM = 5, EVICTING = 6
};
```

The RAM cache will move entries through these states and **must not** create a parallel lifecycle system.

### 4.3 `ExpertCoord`, `ExpertMetadata`, `ExpertBuffer`, `ManagedExpert`

All four are defined in `include/asema/expert.hpp` and are stable. The cache will store `ManagedExpert`-style records. (Note: I will not reuse the atomic `pin_count` from `ManagedExpert` directly because cache pinning is a different concern from execution-pinning, but the cache will maintain its own pin counter on a wrapper.)

### 4.4 `ExpertLoadResult`, `LoadHandle`, `LoadCallback`

Already defined in `async_loader.hpp`. My adapter will consume these.

### 4.5 `AsyncExpertLoader` API

- `submit(coord, priority, callback) -> LoadHandle`
- `submit_future(coord, priority) -> std::future<ExpertLoadResult>`
- `cancel(handle) -> bool`
- `try_get_result(handle) -> std::optional<ExpertLoadResult>`
- `wait(handle) -> ExpertLoadResult`
- `shutdown()`
- `total_*()` metric getters

---

## 5. Potential Integration Conflicts

| Risk | Mitigation |
|------|------------|
| Async loader already has its own internal priority queue | My `ExpertRequestQueue` is a **logical front-end**. The loader's internal queue is the **physical back-end**. The queue's `try_pop()` returns request metadata that the runtime then submits to the loader. The queue does **not** bypass the loader's coalescing. |
| Async loader does request coalescing by coordinate | My queue adds an additional layer of coalescing at the **front of the pipeline** so we never call `submit()` for something already in `LOADING` or `RAM_RESIDENT`. This is idempotent — the loader's own coalescing remains the source of truth for in-flight dedup. |
| Two systems trying to move `ExpertState` | Only the cache writes `RAM_RESIDENT` / `EVICTING`. The runtime is the single owner of state transitions and emits telemetry. The loader never sets `ExpertState` directly. |
| Build system collision on `CMakeLists.txt` | I make **only additive** changes to `CMakeLists.txt`: append new sources to `ASEMA_SOURCES`, append new executables, append `add_test()`. I do not modify any existing target or source line. |

---

## 6. Isolation Strategy

1. **Boundary discipline** — every component depends only on:
   - The existing core types in `expert.hpp`, `manifest.hpp`, `async_loader.hpp`, `storage.hpp`.
   - Its own header(s).
   - The `IExpertLoader` interface in `loader_adapter.hpp`.

2. **Adapter pattern** — the runtime and prefetch engine never call `AsyncExpertLoader` directly. They call `IExpertLoader`. A concrete adapter `AsyncLoaderAdapter` wraps `AsyncExpertLoader` and is swappable.

3. **No shared globals** — cache, queue, prefetch, and telemetry are owned by the runtime. They do not touch each other's internals.

4. **Telemetry is one-way** — subsystems emit events; the telemetry engine is the sole writer of JSONL. Subsystems never call into telemetry from inside critical sections.

---

## 7. Agent #1 Assumptions Verified by Build

- The synthetic MoE container has 4 layers × 16 experts × 786 432 bytes per expert.
- The async loader produces correct CRC32-validated buffers at ~6 GB/s aggregate with 4 IOCP workers (in cache hot path).
- Built-in coalescing saves ~75 % of physical I/O in the repeated-access benchmark.
- `cancel(handle)` works for both queued and in-flight operations.

These will be used as the baseline numbers for comparison in Milestone 5 experiments.

---

## 8. Conclusion

The repository is in a clean, buildable state with Agent #1's Milestone 2 complete. Agent #2 will now implement Milestones 3, 4, and 5 strictly in its own directories, using only additive `CMakeLists.txt` changes and an explicit adapter for the loader interface.
