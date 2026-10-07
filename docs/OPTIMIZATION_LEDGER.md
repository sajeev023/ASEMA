# Optimization ledger

Every row is a measurement from the reference machine (Ryzen 7 5700X, 32 GB RAM, Radeon RX 580,
one NVMe + one SATA SSD), greedy decoding, `asema bench` (50 tokens, same prompt) unless noted.
"Warm" means tokens 2..N. **Run-to-run variation on identical configurations is about +/-10%**
(three identical K=0 runs on one day gave 5.8, 6.4 and 7.1 s/token), so any change smaller than that
needs repeated, alternating runs before it can be called an improvement.

## Result summary

| Change | Verdict | Evidence |
|---|---|---|
| Console UTF-8 + complete-UTF-8 streaming | KEEP (fixes garbled console output) | token trace and byte-exact round trip tests |
| Direct FP8 GEMV for dense projections (no FP32 copy) | KEEP | RAM working set ~25 GB -> 12-14 GB (earlier session); output identical to the FP32 path |
| Blend (frequency+recency) expert cache instead of LRU | KEEP | trace replay: LRU gets 0% hits below ~240 slots; blend 20% at 128 |
| Move 8 shards SATA -> NVMe (SHA-256 verified) | KEEP | measured 8,371 -> 6,958 ms in one A/B (single pair; see noise note) |
| Pipelined read -> GPU upload | KEEP (output identical) | isolated gain not measured |
| **VRAM as primary expert cache, resident experts skip storage, no RAM duplicate** | **KEEP for memory; speed gain not demonstrated** | paired 3 vs 3: 6,492 ms (sd 291) -> 6,157 ms (sd 345), -5%, about 1 sd, not significant; RAM working set 11-12 GB -> ~8 GB (deterministic) |
| Cost-aware eviction (SATA experts kept longer) | KEEP (no regression; benefit not isolated) | trace replay predicted -9% storage time; not isolated in a paired run |
| **Predictive next-layer prefetch (K=2, K=4)** | **REJECT, off by default** | see below |

An earlier claim in this repository of "7.4 -> 5.8 s/token (-22%)" for the VRAM-tier cache was based
on two unpaired runs and is **withdrawn**: the paired comparison above does not support it.

## Predictive prefetch experiment

Predictor accuracy (offline, 63-token trace; fraction of the next layer's 6 true experts that appear
in the top-K predictions, using only information available before that layer's router runs):

| Strategy | K=6 | K=12 | K=24 |
|---|---|---|---|
| previous token's experts at the layer | 36.5% | 47.1% | 58.1% |
| frequency | 27.6% | 39.1% | 52.5% |
| co-occurrence with the previous layer's choice | 37.5% | 51.2% | 63.4% |
| co-occurrence + previous token | 37.5% | 51.4% | 63.4% |
| random (K=24) | 6.2% | | |

End-to-end (same binary and prompt, runs executed in this order K=0, 2, 4, 0):

| K | warm mean ms | p50 / p95 ms | useful prefetches | storage MB/token |
|---|---|---|---|---|
| 0 | 6,399 | 5,695 / 9,909 | - | 3,465 |
| 2 | 7,249 | 6,500 / 11,637 | 32% of 4,647 issued | 4,452 |
| 4 | 7,365 | 7,172 / 9,756 | 27% of 9,270 issued | 5,604 |
| 0 (repeat) | 7,085 | 6,607 / 11,161 | - | 3,465 |

The timing differences sit inside the noise band (the two K=0 runs differ by 11%). The byte counts
are exact: prefetching reads 28% (K=2) and 62% (K=4) more data per token, and only 27-32% of the
speculative reads were used. On this machine storage is the bottleneck (SATA drive saturated), so a
wrong guess competes directly with a needed read. Generated text was identical in all four runs.
Prefetch stays in the code behind `ASEMA_PREFETCH_K` (default 0) and may help on a machine whose
storage is not saturated; it was not shown to help here.

## Where the time goes (K=0, VRAM tier, one 50-token run)

| Stage | ms/token | Share |
|---|---|---|
| Waiting for expert reads (storage) | ~4,000-4,700 | ~65-72% |
| GPU expert stage (uploads, compute, sync) | ~1,150 | ~18% |
| Attention projections | ~240 | ~4% |
| Everything else | ~500 | ~8% |

About 3.3-3.5 GB of experts are read per token; 22 of the 40 layers keep their experts on the SATA
drive (~440 MB/s). GPU busy time is about 20% of a token. Storage is the remaining bottleneck, and
no software-only change measured so far removes it.

## Correctness findings made while profiling

- Replies over roughly 60-100 tokens degenerate (garbled words, repeated `<!DOCTYPE html>`). The
  **legacy and new data paths produce the identical degenerate text**, so the cache work did not cause it.
- The engine does not execute the model's compressed-sparse global attention (CSA2), its hierarchical
  indexer, or the Engram modules (layers 1 and 14), only 128-token sliding-window attention with
  attention sinks. Speed numbers therefore describe an incomplete model; a faithful implementation
  will be slower. See "Known correctness gaps" in the README.
