# ASEMA v0.1 — M7 Hardening Audit

**Date**: 2026-09-26
**Author**: Agent #4 (M7 hardening + M8 real-model)
**Scope**: Phase A — close the validation gaps identified in the v0.1 report.

---

## 1. Repository state at session start

```
examples/synthetic_moe/
    generate_synthetic_moe.py     (extended with CLI args in Agent #3 session)
    model_synthetic/              (Tier 1, fp32, 48 MB)
    model_tier2/                  (Tier 2, fp32, 96 MB)         <- NEW in Agent #3 session
    model_tier3/                  (NOT GENERATED)                 <- THIS SESSION

reports/runs/                  (preserved from prior session)
    2026-09-26T09-57-59.876Z/    (initial smoke)
    m7_tier1_baseline/...        (3 cells, 9 measurements)
    m7_tier1_full_sweep/...      (30 cells, 90 measurements)
    m7_tier2_baseline/...        (3 cells, 9 measurements)
    m7_tier2_scaling/...         (5 cells, 15 measurements)
    TOTAL preserved: 111 raw measurements + 5 configuration.json + 5 environment.json

docs/                          (M1-M7 docs preserved)
include/asema/                  (M1-M7 headers preserved)
src/                            (M1-M7 sources preserved)
tests/                          (M1-M7 tests preserved)
tools/                          (M1-M7 tools preserved)
```

All prior Agent #1, #2, #3 work is intact. Nothing is deleted, overwritten, or rewritten.

---

## 2. What is actually measured today (from raw JSON inspection)

Per `Measurement` fields currently written by `asema_bench.cpp` (see `reports/runs/.../raw/cache_8MB_K2_w4_s42_iter1.json`):

```
valid                                  - bool
tokens_completed                       - uint (== num_tokens for completed runs)
wall_time_ms                           - double (runtime end - start)
first_token_ms                         - double
ms_per_token                           - double (wall / tokens)
tokens_per_sec                         - double (1000 / ms_per_token)
cache_hits                             - uint (from cache.stats())
cache_misses                           - uint
cache_hit_rate                         - double
cache_evictions                        - uint
cache_insertions                       - uint
peak_bytes_used                        - uint
ssd_bytes_logical                      - uint   <-- WRONG: == physical
ssd_bytes_physical                     - uint   <-- WRONG: == logical
ssd_reads                              - uint   <-- from telemetry.ssd_reads
coalesced_requests                     - uint   <-- ALWAYS 0 (not captured)
prefetch_requests                      - uint
prefetch_useful                        - uint
prefetch_wasted                        - uint
prefetch_cancelled                     - uint
prefetch_duplicate                     - uint
prefetch_hit_rate                      - double
```

### 2.1 Diagnosis of metric confusion

`asema_bench.cpp` (lines 182-184) does:

```cpp
auto tr = rt.telemetry()->build_report();
m.ssd_bytes_logical = tr.ssd_bytes;
m.ssd_bytes_physical = tr.ssd_bytes;
m.ssd_reads = tr.ssd_reads;
```

`tr.ssd_bytes` comes from `TelemetryEngine::ssd_bytes_` which is incremented by `increment_ssd_read(uint64_t b)` from `runtime.cpp::load_and_execute(...)` where `b = res.bytes_read` — i.e., the bytes the loader actually read from disk (after internal coalescing).

Therefore **`tr.ssd_bytes` is the PHYSICAL bytes read** (post-coalescing), not logical. The M7 report conflated the two. The `coalesced_requests` field is also always 0 because `asema_bench.cpp` never reads `loader->total_coalesced_requests()`.

### 2.2 What the loader already exposes (from `include/asema/async_loader.hpp`)

```cpp
uint64_t total_requests_submitted() const noexcept;
uint64_t total_coalesced_requests() const noexcept;
uint64_t total_io_dispatches() const noexcept;
uint64_t total_bytes_loaded() const noexcept;
uint64_t total_checksum_failures() const noexcept;
uint64_t active_in_flight() const noexcept;
```

These are all already implemented. The M7 runner simply doesn't read `total_coalesced_requests()` or distinguish `total_io_dispatches()` from logical submission count.

### 2.3 What the IExpertLoader adapter exposes (`include/asema/loader_adapter.hpp`)

Currently exposes only the 6 methods from `AsyncExpertLoader` directly. To get coalesced + dispatch info into `asema-bench`, we read them from `loader_->total_coalesced_requests()` and `loader_->total_io_dispatches()` — both are already on `IExpertLoader`.

---

## 3. What is inferred but not measured

| Claim in M7 report | Reality |
|--------------------|---------|
| "ssd_bytes_logical == ssd_bytes_physical" | both fields point to the same underlying value (physical) |
| "coalesced_requests = 0" | not queried; the loader does coalesce, but the counter was never read |
| "first_token_latency" | measured from runtime wall-time of token 1 only |
| "OS file cache may influence storage measurements" | acknowledged but no workaround implemented |
| "Tier 3 deferred" | disk budget, NOT a measurement limitation |
| "Tier 2 anomaly: ~72% hit at 1 MB cache" | hypothesis recorded but not investigated |
| "Prefetch 0% useful" | measured and recorded honestly |

---

## 4. What remains missing for the M7 hardening gate

| Gap | Fix |
|-----|-----|
| Logical/physical byte distinction | Add `logical_bytes_requested` to telemetry; read `total_bytes_loaded` for physical |
| Coalesced counter missing | Read `total_coalesced_requests()` from loader in `asema-bench` |
| I/O dispatch count missing | Read `total_io_dispatches()` from loader in `asema-bench` |
| Per-cell N=3 too small | New flag `--repetitions` + `--iterations`; default research config 10 iters + 5 warmup |
| Tier 3 not generated | Run `asema-pack --tier 3` after disk-space check |
| Locality sweep not executed | Run with `--locality-low / --locality-high` flags |
| Worker sweep not executed | Run with `--workers-sweep` |
| Seed sweep not executed | Run with `--seed-sweep` |
| Warm/cold not characterized | Add `--cold` flag (informational) + run fresh vs re-run |
| Tier 2 anomaly not investigated | Add access-frequency logging in runtime |
| Hardened report not generated | New `reports/v0.1_hardened_performance_report.md` |

---

## 5. Architectural rules for this session

- Do NOT modify M1, M2, M3, M4, M5, M6, M7 implementations unless required to fix the missing counter.
- All M7 raw measurements from the previous session are **preserved**. New runs go to new timestamped directories under `reports/runs/`.
- No fabricated numbers.
- Real-model selection (M8) is BLOCKED until the M7 freeze gate passes.

---

## 6. Freeze gate checklist

- [ ] Tier 3 generated safely (288 MB, disk-checked, CRC-validated)
- [ ] Logical/physical bytes distinguished in telemetry + benchmark output
- [ ] Coalesced counter captured
- [ ] I/O dispatch counter captured
- [ ] Iterations configurable; research run with N ≥ 10 per cell
- [ ] Locality sweep executed
- [ ] Prefetch sweep executed (already in `m7_tier1_full_sweep`)
- [ ] Worker sweep executed
- [ ] Seed sweep executed
- [ ] Warm/cold distinction recorded per run
- [ ] Tier 2 anomaly investigated
- [ ] Hardened report generated
- [ ] `docs/ASEMA_V0_1_FREEZE.md` written
