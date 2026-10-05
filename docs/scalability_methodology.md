# ASEMA v0.1 — Scalability Methodology

## Goal

Determine how ASEMA behaves as the working set grows relative to the cache. The fundamental question M7 must answer:

> Under what (cache_size, working_set, locality, lookahead) does ASEMA provide measurable benefit, and where does it stop helping?

## Tier matrix

| Tier | Layers × Experts | Total bytes (fp16) | Working set estimate (top-k × tokens) | Cache sweep range |
|------|------------------|-------------------:|---------------------------------------:|--------------------|
| 1    | 4 × 16           | 24 MB              | ~32 KB / token × top-2                | 1 / 2 / 4 / 8 / 12 MB |
| 2    | 8 × 32           | 96 MB              | ~32 KB / token × top-2                | 1 / 2 / 4 / 8 / 12 MB |
| 3    | 12 × 64          | 288 MB             | ~32 KB / token × top-2                | (skipped this session) |

Per-token working set:

```
active = num_layers × top_k × expert_bytes
       = L × 2 × 393216  bytes
```

For Tier 1: `4 × 2 × 393216 ≈ 3 MB / token`. With 32 tokens and 70 % locality, the resident working set at any moment is roughly `0.7 × 32 × top_k × L × expert_bytes / 2 ≈ 9 MB`.

For Tier 2: `8 × 2 × 393216 ≈ 6 MB / token`. Same locality pattern → ~18 MB resident.

This is why:
- Tier 1 cache threshold is around 2 MB (one cache slot per layer worth of experts).
- Tier 2 cache threshold is around 2-4 MB (residents per layer).

## Working-set ratio (key metric)

```
ratio = cache_size / active_resident_set
```

A ratio ≥ 1.0 should yield > 70 % hit rate under locality. Below 0.5 → thrashing.

## Scaling observations (Tier 1 vs Tier 2)

Both tiers show the same qualitative behavior:

1. Below the working set (Tier 1: 1 MB; Tier 2: 1 MB) → 0 % hit rate, ~28 ms/token.
2. At the working set (Tier 1: 2 MB; Tier 2: 2 MB) → ~73 % hit rate, ~15.4 ms/token.
3. Above the working set (Tier 1: 4+ MB; Tier 2: 4+ MB) → diminishing returns.

The **cache ratio threshold** is what matters, not the absolute size.

## What was NOT tested (and why)

- **Tier 3 (12L × 64E, 288 MB)** — disk write would consume ~300 MB; the host's free space was not sufficient when combined with other artifacts. The synthetic generator enforces a 1.5× safety factor and would have refused.
- **Cross-tier working-set sweep** — would require a Tier-2-sized workload paired with a Tier-1-sized cache (e.g., 4 MB cache vs 96 MB model). This is the most informative scaling experiment but takes ~30 minutes per run on this hardware. Recommended for a future, dedicated session.
- **Multi-process / multi-node** — out of scope for v0.1.

## Bottleneck analysis (from telemetry)

Time per token ≈ Σ (load latency + queue wait + execution + cache ops).

For Tier 1 at 8 MB cache, observed:
- Load latency: ~1.5 ms per expert load (cold), 0 ms (cache hit).
- Queue wait: < 0.1 ms (single producer).
- Execution: ~1 µs (synthetic 4-KB-stride buffer walk).
- Cache ops: < 0.1 ms.

→ **SSD load latency dominates the cold-miss path** (1.5 ms vs 0.1 ms elsewhere). When the cache absorbs ~77 % of requests, ms/token drops from 28 to 15 — exactly what is observed.

## What ASEMA does not yet measure

- End-to-end GPU execution latency (CPU-only synthetic path).
- Network/cluster latency for a distributed loader.
- Power consumption.
- Cold-OS-cache behavior (Windows portable flush not available).

These are documented as future work, not pretended to be measured.
