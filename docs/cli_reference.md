# ASEMA v0.1 — CLI Reference

## Overview

ASEMA v0.1 ships four CLI tools built on a common argument parser (`include/asema/cli.hpp`):

| Tool | Purpose | v0.1 Status |
|------|---------|-------------|
| `asema-inspect` | Display model metadata (M1) | Existing, text only |
| `asema-pack` | Generate synthetic ASEMA-SSF containers | **NEW (Agent #3)** |
| `asema-run` | Run the synthetic-runtime pipeline | **NEW (Agent #3)** |
| `asema-bench` | Research-grade experiment runner | **NEW (Agent #3)** |
| `asema-async-bench` | Sync-vs-async I/O microbenchmark (M2) | Existing |
| `asema-full-bench` | End-to-end pipeline microbenchmark (M5) | Existing |

---

## Exit Codes (all CLIs)

| Code | Name | Meaning |
|-----:|------|---------|
| 0 | `RC_SUCCESS` | Operation succeeded |
| 2 | `RC_INVALID_ARGUMENT` | Bad flag, missing required arg |
| 3 | `RC_MODEL_ERROR` | Manifest parse fail, missing file |
| 4 | `RC_IO_ERROR` | Cannot read/write file |
| 5 | `RC_RUNTIME_ERROR` | Loader / cache / runtime construction fail |
| 6 | `RC_BENCHMARK_ERROR` | No valid measurements collected |
| 7 | `RC_VERIFICATION_ERROR` | CRC mismatch during verification |

---

## asema-pack

Generate a synthetic MoE container (parameters-driven).

```
asema-pack --output <dir> [--tier 1|2|3] [--layers N] [--experts N]
                        [--hidden D] [--ffn D] [--top-k K] [--dtype fp16|fp32]
                        [--seed N] [--force]
```

Examples:

```
asema-pack --tier 1 --output examples/synthetic_moe/model_synthetic
asema-pack --layers 8 --experts 32 --hidden 256 --ffn 1024 --output model_big --force
```

Behavior:

- Streams deterministic expert weights (normal(0, 0.02²)).
- Aligns every expert to 64 bytes.
- Computes CRC32 for each expert.
- Checks free disk space before writing (aborts if insufficient).
- Writes `model.asema` (binary container) + `manifest.json`.
- Reports bytes written, elapsed time, throughput.

What it does **not** support:

- HuggingFace / PyTorch / Safetensors / GGUF conversion. Only the internal synthetic generator. The CLI prints "Unsupported input format" if asked to convert anything else (future work).

---

## asema-run

Run the synthetic runtime pipeline against a model container.

```
asema-run --model <model.asema|model_dir> [--tokens N] [--cache BYTES]
          [--lookahead K] [--workers N] [--seed N]
          [--telemetry <path>] [--report <path>] [--report-json <path>]
          [--quiet]
```

Example:

```
asema-run --model examples/synthetic_moe/model_synthetic --tokens 64 --cache 8MB
```

Behavior:

- Loads manifest + opens storage via `AsyncExpertLoader`.
- Routes deterministically with the synthetic router.
- Reports `tokens`, `wall_ms`, `ms/token`, `first_token_ms`, cache + prefetch stats.
- Does **not** perform real LLM text generation — v0.1 supports only the synthetic execution path. The CLI does not pretend otherwise.

---

## asema-bench

M7 experiment runner. Iterates over an experiment matrix and saves raw + summarized data.

```
asema-bench --model <model_dir> [--tokens N] [--iterations N] [--warmup N]
            [--cache BYTES | --cache-sweep]
            [--lookahead K | --lookahead-sweep]
            [--workers N | --workers-sweep]
            [--seed N | --seed-sweep]
            [--pattern <label>] [--locality-low | --locality-high] [--warm]
            [--output <dir>] [--format json|csv|text]
            [--quiet]
```

Output (under `<output>/<timestamp>/`):

```
environment.json
configuration.json
raw/all_measurements.csv
raw/<cell>_iter<n>.json
summaries/<cell>.json
summaries/all_summaries.txt
```

Exit codes:

- `0` — at least one cell produced at least one valid measurement
- `6` (`RC_BENCHMARK_ERROR`) — no valid measurements collected

Example:

```
asema-bench --model model_synthetic --cache-sweep --lookahead-sweep --iterations 3
```

---

## Common Argument Syntax

```
--name value         long option with value (space or = separator)
--flag               long boolean flag
-x value             short option with value
-x                   short boolean flag
-xyz                 cluster of short boolean flags
--name=value         long option with = separator
```

All CLIs accept `--help` and `--version` and exit with documented codes.
