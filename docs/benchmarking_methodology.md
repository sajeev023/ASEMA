# ASEMA v0.1 — Benchmarking Methodology

## 1. Principles

1. **Reproducibility** — every measurement carries the run configuration and timestamp.
2. **Honesty** — invalid runs are marked INVALID; never silently included.
3. **Measured > assumed** — never claim a bottleneck without measurement.
4. **Distinguish physical vs logical** — coalescing reduces logical demand; it does not multiply SSD bandwidth.
5. **Raw data is preserved** — every measurement is written to `<run>/raw/*.json` and to `all_measurements.csv` before summary is computed.

## 2. Workloads

All experiments use **synthetic** MoE containers produced by `asema-pack`:

| Tier | Layers | Experts/Layer | Top-k | Hidden | FFN | Expert bytes (fp16) | Total | Purpose |
|------|-------:|--------------:|------:|-------:|----:|--------------------:|------:|---------|
| 1    | 4      | 16            | 2     | 128    | 512 | 393 216             | 24 MB | smoke / smoke regression |
| 2    | 8      | 32            | 2     | 128    | 512 | 393 216             | 96 MB | mid-range scaling |
| 3    | 12     | 64            | 2     | 128    | 512 | 393 216             | 288 MB| large scaling (skipped in this session due to disk budget) |

Container format is identical to what the M1 storage layer and M2 async loader consume; no schema changes.

## 3. Routing / Locality

The synthetic router uses a locality-heavy heuristic (70 % of tokens reuse the previous token's expert choices, 30 % pick fresh within layer). `--locality-low` switches to 30 % reuse (≈random); `--locality-high` to 90 % reuse.

Locality is the single most important factor for cache hit rate; the prefetcher depends on it.

## 4. Experiment Matrix (M7)

For each combination of:

- cache sizes (default sweep: 1 / 2 / 4 / 8 / 12 MB)
- lookahead K (default sweep: 0 / 1 / 2 / 3 / 4 / 6)
- workers (default sweep: 1 / 2 / 4 / 8)
- seeds (default sweep: 11, 42, 99)
- tokens per run (default: 32)
- iterations (default: 3)

run the runtime once and record one `Measurement`. Default cell count:

```
5 caches × 6 Ks × 1 workers × 1 seeds × 3 iter = 90 cells (default Tier 1 sweep)
```

For each cell, compute mean / stddev / min / max / median / p95 of `ms_per_token`, plus mean cache hit rate, mean prefetch hit rate, mean SSD bytes.

## 5. Statistics

- **mean / stddev** — over the 3 measured iterations of one cell.
- **percentiles** — nearest-rank method on the sorted sample.
- **stddev** — population variance. Sample variance is reported when N ≥ 3.

Sample sizes are explicitly recorded. No statistical-significance claims are made on N < 10.

## 6. Cold / Warm

The runner exposes `--warm` as an informational flag. **We do not actually drop the OS file cache on Windows** — Windows does not provide a portable flush-cache API. Every run is therefore somewhere on the cold/warm continuum; the runner records the intent (cold_start field) but cannot guarantee it.

Consequence: a portion of reported SSD bytes may be served by the OS file cache. This is **explicitly documented** in the final report.

## 7. Percentile Methodology

For each latency series (load latency, execution latency, end-to-end token latency):

1. Collect samples into `std::vector<double>`.
2. Sort ascending.
3. `P50 = values[floor(N * 0.50)]`, `P95 = values[floor(N * 0.95)]`, `P99 = values[floor(N * 0.99)]`.
4. If `idx >= N`, clamp to `N - 1`.

This is the standard **nearest-rank** method, used by the existing telemetry engine.

## 8. Logical vs Physical SSD

The runtime reports **logical** bytes requested by the runtime. The M2 loader's `total_bytes_loaded()` reports the same value because coalescing is the loader's own responsibility. We therefore distinguish:

- `ssd_bytes_logical` — what the runtime asked for
- `ssd_bytes_physical` — what the loader actually read from disk (currently identical in M2; will differ when physical-level dedup is added)

The benchmark explicitly labels which is which. **4 logical reads coalesced into 1 physical read does NOT multiply SSD bandwidth by 4** — it means the workload's logical demand was reduced.

## 9. Invalid Run Handling

A measurement is marked `valid=false` if:

- Runtime threw an exception.
- Runtime returned `completed=false`.
- Memory budget was violated (peak_bytes_used > configured cache capacity).

Invalid runs are still written to `<run>/raw/` with `invalid_reason` populated, and counted in the `summaries/all_summaries.txt` header. They are excluded from the cell-level statistics.

## 10. Reproducibility

To regenerate every measurement in this report:

```
cmake -B build -G Ninja ...
cmake --build build
asema-pack --tier 1 --output examples/synthetic_moe/model_synthetic
asema-pack --tier 2 --output examples/synthetic_moe/model_tier2
asema-bench --model examples/synthetic_moe/model_synthetic \
            --tokens 32 --iterations 3 \
            --cache-sweep --lookahead-sweep \
            --workers 4 --seed 42 \
            --output reports/runs/m7_tier1_full_sweep
asema-bench --model examples/synthetic_moe/model_tier2 \
            --tokens 32 --iterations 3 \
            --cache-sweep \
            --workers 4 --seed 42 \
            --output reports/runs/m7_tier2_scaling
asema-bench --model examples/synthetic_moe/model_synthetic \
            --tokens 32 --iterations 3 \
            --cache 0 --lookahead 0 \
            --output reports/runs/m7_tier1_baseline
asema-bench --model examples/synthetic_moe/model_tier2 \
            --tokens 32 --iterations 3 \
            --cache 0 --lookahead 0 \
            --output reports/runs/m7_tier2_baseline
```
