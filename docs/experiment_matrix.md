# ASEMA v0.1 — Experiment Matrix

The M7 experiment matrix is configurable but defaults to:

```
cache   × { 1 MB, 2 MB, 4 MB, 8 MB, 12 MB }   (5 levels)
lookahead × { 0, 1, 2, 3, 4, 6 }             (6 levels)
workers × { 1, 2, 4, 8 }                     (4 levels)
seeds × { 11, 42, 99 }                       (3 levels)
tokens = 32
iterations = 3
warmup = 0
```

## Run configurations actually executed (this session)

| Run | Workload | Cells | Iterations | Total measurements |
|-----|----------|------:|-----------:|-------------------:|
| `m7_tier1_baseline`   | Tier 1 (4L × 16E, 24 MB)  | 1  (cache=0, K=0)            | 3 | 3 |
| `m7_tier1_full_sweep` | Tier 1                   | 30 (5 cache × 6 K)           | 3 | 90 |
| `m7_tier2_baseline`   | Tier 2 (8L × 32E, 96 MB)  | 1  (cache=0, K=0)            | 3 | 3 |
| `m7_tier2_scaling`    | Tier 2                   | 5  (5 cache sizes, K=2)      | 3 | 15 |
| **Total**             |                          | **37**                       |   | **111** |

All runs are under `reports/runs/<run>/<timestamp>/`.

## Per-iteration structure

For each cell × iteration:

1. Construct a fresh `ASEMARuntime`.
2. Call `rt.run()` — runs `num_tokens` synthetic tokens.
3. Snapshot: cache stats, prefetch stats, telemetry report.
4. Build one `Measurement` row.
5. Write to `raw/<cell>_iter<n>.json` and append to `raw/all_measurements.csv`.

After all iterations, write one `CellSummary` to `summaries/<cell>.json` and update `summaries/all_summaries.txt`.

## Matrix construction

`asema_bench.cpp::build_matrix` produces the Cartesian product of (caches × Ks × workers × seeds). The `--cache-sweep`, `--lookahead-sweep`, `--workers-sweep`, `--seed-sweep` flags toggle each dimension.

Example full-sweep invocation:

```
asema-bench --model <dir> --cache-sweep --lookahead-sweep --workers-sweep --seed-sweep
```

Default matrix size = 5 × 6 × 4 × 3 = 360 cells × 3 iterations = 1080 measurements. Recommended only when ample wall time and disk space are available.

## Per-cell configuration is recorded

Each cell's configuration (cache, K, workers, seed, tokens, locality, pattern, cold/warm intent) is recorded in:

- `configuration.json` (global matrix summary)
- `summaries/<cell>.json` (per-cell summary includes the same fields)
- The `experiment_id` string encodes cache, K, workers, seed for grep-friendly filenames.

## Why a 30-cell Tier 1 sweep + 5-cell Tier 2 sweep (not 360)

The Tier 1 sweep was chosen to expose:

- the **cache-size knee** (1 MB thrashing vs 2 MB+ saturation)
- the **K-flatness** for the synthetic locality pattern
- reproducibility across iterations

The Tier 2 sweep is intentionally smaller (5 cells × 3 iter = 15 measurements) so it completes quickly while still demonstrating the **working-set ratio** phenomenon. A full 360-cell Tier 2 sweep was skipped in this session due to wall-time budget.

## Future expansions (not executed in this session)

- Tier 3 (12L × 64E, 288 MB) sweep with `--workers-sweep` and `--seed-sweep` enabled — would test cache-thrashing under realistic working sets.
- Locality sweep (low / medium / high) — would isolate the prefetch contribution.
- `--warm` flag interpretation — the runner records intent but cannot drop Windows OS file cache.
