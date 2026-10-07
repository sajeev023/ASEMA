# Performance

All numbers on this page were measured on one machine and one set of prompts. They are a
description of that machine, not a general claim. Reproduce them with `asema bench` and
`asema profile` (see the README).

**Reference machine:** AMD Ryzen 7 5700X, 32 GB RAM, AMD Radeon RX 580 8 GB, Windows 11.
Checkpoint storage: one Samsung PM981 NVMe (36% of the data, 171 GB) and one SATA SSD (64%, 304 GB).
Roughly 6-10 GB of RAM was used by other desktop applications during the runs.

## Summary

| Measurement | Result |
|---|---|
| Decode speed over a 40-token generation (warm, tokens 2..40) | **9.3 s/token mean, 0.11 tokens/s** (p50 8.3 s, p95 14.3 s) |
| Cold first token (includes prefill of the chat template) | 179.6 s |
| Storage read per generated token | about 3.7 GB (147 GB total over the run) |
| Expert host-cache hit rate (about 253 experts, 4.7 GB) | 34% |
| Expert VRAM-slot hit rate | 0% (0 hits in 12,480 lookups) |
| CPU use | 0.42 cores on average (the run waits on storage) |
| Process working set | 12.6 GB current, 14.2 GB peak |
| VRAM allocated | 1.7 GB |

The target of about 1 token/s was **not reached and is not reachable on this storage layout**
(see "Why", below).

## Why decode is slow: storage bandwidth

A synthetic probe of random 18.8 MB reads (the size of one expert) against the real shards:

| Drive | Type | Throughput |
|---|---|---|
| D: (64% of the checkpoint) | SATA SSD | about 440 MB/s, **flat from 1 to 8 concurrent reads** |
| E: (36% of the checkpoint) | NVMe | about 2,800 MB/s at 2 or more concurrent reads |

Each token routes 6 of 384 experts in each of 40 layers (240 expert reads, 18.8 MB each, 4.5 GB if
nothing is cached). With a 34% hit rate about 2.4-3.0 GB per token must come from disk, and about
two thirds of that lives on the SATA drive: roughly 5-6 s of pure read time per token, before any
compute. Adding threads does not help because the SATA drive is already saturated.

## What was done, and what each change measured

| Change | Result | Kept |
|---|---|---|
| Use dense FP8 weights directly from memory-mapped shards instead of dequantizing to FP32 every layer of every token | Short warm profile: 12.2 s -> 5.8 s per token (dense load 5.15 s -> 0.07 s). The FP8 kernel is bit-exact against the lookup-table decode and agrees with the old path to about 2e-8 relative error; generated tokens were identical to the old path on the prompts tested | yes |
| Removes an 11 GB private dense cache; pages are now reclaimable file cache | Peak working set 13.3 -> 12.7 GB in the profile run | yes |
| Parallel expert reads (bounded), positioned reads, true LRU cache | Correctness fix (the old shared file pointer was racy) and fewer stalls; overall decode still I/O-bound | yes |
| Memory-pressure governor (SAFE / WARNING / CRITICAL) | Cache shrank and recovered automatically when available RAM dipped to about 6-7 GB | yes |
| Corrected VRAM telemetry | Reported 18 MB while 1.7 GB was resident; now matches the Windows GPU counter | yes |

A short `asema profile` run (4 prefill + 2 warm-up + 1 measured decode step) gave a better number
(5.8 s) than a real 40-token generation (9.3 s) because its cache was unusually favourable. Use
`asema bench` for realistic numbers.

## Cache-policy analysis (offline replay of a recorded trace)

63 tokens x 240 expert lookups were recorded (`ASEMA_TRACE_EXPERTS`) and replayed:

| Cache size | LRU (used) | LFU | Frequency + recency | Optimal (Belady) |
|---|---|---|---|---|
| 253 experts (4.8 GB) | 35.1% | 29.0% | 31.5% | 52.9% |
| 400 experts (7.5 GB) | 39.0% | 36.9% | 40.5% | 59.8% |
| 600 experts (11.3 GB) | 47.0% | 44.9% | 48.5% | 65.3% |
| 1000 experts (18.8 GB) | 56.0% | 54.8% | 58.8% | 70.3% |

A smarter replacement policy gains only about 1-2 points; hit rate rises slowly with memory. The
96-slot VRAM cache gets 0% under LRU because the access pattern is a cyclic scan of 240 experts;
LFU would reach roughly 19% at 128 entries, worth well under 0.2 s/token, so it was not pursued.

## What would help (not measured)

- Putting the whole checkpoint on NVMe storage (or spreading it over several fast drives) attacks
  the dominant cost directly. The NVMe drive here was measured at about 6x the SATA drive.
- More RAM for the expert cache raises the hit rate slowly (table above).
- Prefetching the next layer's experts while the current layer computes could hide at most the
  compute time (about 2 s/token here) and depends on prediction accuracy that has not been measured.

## Per-stage profile of a 50-token generation (asema bench --tokens 50)

Mean over the 49 warm tokens (milliseconds per token; the sum of the rows below is the token time):

| Stage | ms | Share |
|---|---|---|
| Expert cache lookup + storage reads | 6,513 | 75% |
| Expert GPU stage (CPU wall time) | 1,621 | 19% |
| Attention projections (FP8, CPU) | 276 | 3% |
| Dense weight binding | 82 | 1% |
| Shared expert (CPU) | 78 | 1% |
| LM head | 52 | 1% |
| Router, KV store, attention over the KV window, final norm, token selection, detokenize | ~24 | <1% |

- Total: 8,716 ms/token mean (p50 8,014 ms, p95 14,544 ms), 0.11 tokens/s. Cold first token 122 s.
- Storage: 4.5 GB and 480 reads per token at an effective 693 MB/s (a mix of the SATA and NVMe drives).
- GPU: executes only ~930 ms of expert uploads and ~320 ms of compute per token (timestamp queries), i.e.
  busy about 14% of the token. The CPU waits ~1,260 ms for it. The GPU is idle because it is fed from
  storage; its own work is small.
- KV cache: the ring-buffer store costs 0.03 ms/token and attention over the 128-position window 9 ms/token.
  The KV cache persists across tokens and is not rebuilt.
- Persistent state is not being recreated per token: Direct3D resources, the 96 VRAM slots, the expert
  cache, KV cache and tokenizer are created once. Dense weight binding is 82 ms/token (pointer setup over
  memory-mapped pages), not a reload.
- In that run the expert hit rate was only 6%: the memory governor had shrunk the cache to 64-205 experts
  because Windows available memory dipped, and plain LRU gets **zero** hits whenever the cache holds fewer
  experts than one token touches (a cyclic scan of ~240).

## Cache policy change (kept)

Replaced LRU with a frequency-and-recency blend (`score = accesses / (1 + 0.25 * age_in_steps)`, lowest
evicted). A/B on live inference, same prompt, 30 tokens, cache fixed at 3,000 MB (167 experts), no governor events:

| | LRU | Blend |
|---|---|---|
| Expert hit rate | 0.0% | 21.8% (offline replay predicted ~22%) |
| Warm mean | 9,103 ms/token | 8,007 ms/token (-12%) |
| Warm p95 | 15,173 ms | 10,541 ms (-31%) |
| Storage read per token | 4.51 GB | 3.46 GB |

Generated text was identical under both policies. Switch back with `ASEMA_CACHE_POLICY=lru`. The blend
is slightly worse than LRU only near 250-entry caches (about -3.5 points in the replay) and better elsewhere.

## Memory governor hardening

A benchmark run was stopped by an external 3.4 GB safety limit when Windows available memory fell, even though
the governor had shrunk the cache. Two changes: the governor now also checks every layer (growth still only once
per token), and at CRITICAL it hands the memory-mapped checkpoint pages back to Windows (they move to the standby
cache, still counted as available). In a forced test each trim reduced the process working set by 0.3-1.2 GB and
output stayed correct. Limits: trimming forces soft page faults on the next access, so it is a pressure valve and
not something to run every token.

## Overlapping reads with GPU uploads, and moving shards to the NVMe (kept)

Two changes after the profile above:

1. **Pipelined expert loading.** Reads for a layer's experts start right after routing, the CPU shared-expert
   math runs while they load, and each expert is uploaded to the GPU as soon as it arrives (in the original
   order, so the accumulation order and every number are unchanged). Output was identical.
2. **Eight expert shards (layers 0, 1, 3-7, 9; 59 GB) moved from the SATA drive to the NVMe**, using
   `scripts/plan_shard_move.py` (dry run by default; copies are written as .part, size- and SHA-256-verified, then
   renamed; the originals were kept). The engine finds shards by name on either drive, so no code change was
   needed; per-drive read counters during a run showed 46% of expert-file reads coming from the NVMe (it was ~25%).

Clean A/B of the shard move alone (25 tokens, cache fixed at 167 experts, identical hits/misses/evictions and
identical generated text; only the serving drive differs):

| | SATA originals | NVMe copies | Change |
|---|---|---|---|
| Warm mean | 8,371 ms/token | 6,958 ms/token | -17% |
| Warm p50 / p95 | 7,547 / 13,264 ms | 6,452 / 9,398 ms | -15% / -29% |
| Cold first token | 149 s | 113 s | -24% |
| Storage wait per token | 6,289 ms | 4,921 ms | -1,368 ms (the trace replay projected 1,240 ms) |

Full 50-token benchmark with everything enabled and automatic cache sizing (the governor shrank and regrew the
cache during the run): warm mean **7,446 ms/token (0.13 tokens/s)**, p50 7,114 ms, p95 10,072 ms, cold first token
94 s; 32% expert hit rate; 3.0 GB read per token. Compared with the earlier 50-token run of the same prompt
(8,716 ms/token) this includes the cache-policy, shard-move and pipelining changes together; the cache size
differed between the two runs, so the table above is the clean attribution for the shard move.

Remaining time per token is still dominated by storage (~4.9 s of ~7 s) and then the GPU stage (~1.4 s, limited by
upload bandwidth: about 4.5 GB of expert bytes cross to the GPU per token at ~4 GB/s). More layers can be moved to
NVMe only if more NVMe space is available (the NVMe is now at 95% capacity).

## Why ~1 token/s is not reachable on this machine

Simulation over the real shard-to-drive map and the measured drive speeds (layers 0-29 are wholly on the SATA
drive, layers 30-39 on the NVMe): storage time alone is about 4.0-5.0 s/token for the current layout at
253-600 cached experts, and still about 0.8-1.0 s/token if the entire model were on NVMe, before about 1.5 s of
GPU upload/compute. Software changes cannot beat the drive's bandwidth.

## Method notes and caveats

- `asema bench` reports the first token separately (it includes prefill); statistics use tokens
  2..N and nothing is smoothed. GPU utilization is not measured.
- Available RAM varied with other applications; results will differ on other machines.
- A 40-token run on one prompt is a small sample. Treat differences below about 10% as noise.
- Later A/B work (VRAM-tier cache, predictive prefetch) is recorded in
  [OPTIMIZATION_LEDGER.md](OPTIMIZATION_LEDGER.md), including a withdrawn speedup claim. Identical
  configurations were observed to vary by 5.8-7.1 s/token on one day, so figures in this file from
  single unpaired runs (including the 7,446 ms baseline above) carry that uncertainty.
- All speed figures describe an engine that does not execute the model's global compressed
  attention, indexer or Engram modules (see the README's "Known correctness gaps"); a faithful
  implementation would be slower.
