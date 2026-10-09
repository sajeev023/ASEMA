# ASEMA NEXT-GEN: VALIDATION AND ARCHITECTURE (FINAL)

Evidence classes: **[M]** measured on this machine, **[T]** theoretical or simulated from measured inputs (real traces and
real bandwidths through a model), **[E]** estimated, **[S]** speculative. Nothing in the production source, the checkpoint
or the tokenizer was modified. All tools live in `experiments/nextgen/`; raw logs are in `reports/` (git-ignored).

## 1. Executive summary

1. Today ASEMA runs at **0.16 tok/s (6.4 s/token)** [M]. Its bottleneck is storage (4.5 s of the 6.4 s), and the reason is
   not the drives: it is **where the bytes live and how they are read**.
2. **A large part of the NVMe is wasted [M]:** 109.5 GB of the 238 GiB NVMe holds Engram and MTP weights that the engine never
   reads, while 22 of 40 layers of live experts sit on a SATA drive that delivers 404 MB/s.
3. **Layers run one after another, so only one drive works at a time [M]** (measured storage rate 765 MB/s while the three
   drives can deliver 4.2 GB/s together). Storage time is the *sum* over drives, not the max.
4. **Rebalancing the layout** (32 layers on NVMe, 8 on the two SATA drives, ~110 GB moved) cuts the modeled storage time per
   token from 4.2 s to 1.9 s at the same cache size [T]. This needs no model change.
5. **The router can be run ahead [M].** Applying layer L+1's real router to layer L's router input predicts L+1's experts with
   67% recall at 6 reads, 82% at 12 (history baselines reach 37%). A scheduler that uses *calibrated probabilities x drive
   refetch cost* reduces the modeled token time by 31% on the rebalanced layout. On the *current* layout it makes things worse.
6. **CPU experts beat GPU experts on this machine [M]:** a validated AVX2 FP4 kernel runs all 240 experts of a token in 219 ms
   (the GPU expert stage takes 1,210 ms today, 925 ms of it being PCIe upload at 41% of link speed).
7. **Projected speed [T]:** 0.58 tok/s without prediction, **0.84 tok/s** with the value scheduler (13 GB cache, 13 ms/layer
   compute), 1.25 tok/s with a *perfect* predictor. **Reaching 1 tok/s is UNCERTAIN, not demonstrated.**
8. **The runtime is not the full model.** It does not run CSA2 global attention, the indexer, Engram or MTP. Replies degrade
   after ~60-100 tokens. Every speed figure here, and every projection, is for an incomplete model.

## 2. Current runtime truth
See `CURRENT_RUNTIME_TRUTH.md` (25 items with file, function, thread, device, sync and measured cost). Key facts:
one `active_layer_` rebinds dense weights every layer; prefill is token by token; expert reads are buffered `ReadFile`s from
`std::async` threads (one thread created per missing expert); experts go to the GPU through `UpdateSubresource`
(4.3 GB/s); MLA attention runs on the CPU; sampling is greedy only.

## 3. Hardware truth [M]
Windows 11 Pro; Ryzen 7 5700X (8C/16T, AVX2/FMA, 32 MB L3); 2 x 16 GB DDR4-3200; RX 580 2048SP (7.95 GB, D3D11 FL 11.0;
ROCm/HIP/Vulkan not evaluated); storage: **C: SATA SSD** (OS, 49 GB free), **D: SATA SSD** (model, 226 GB free),
**E: Samsung PM981 NVMe** (238 GiB, 12 GB free). **There is one NVMe, not two.**

## 4. Model truth [M from config, index and shard headers]
40 layers, hidden 5120, 384 routed experts (top-6), 1 shared expert, MoE intermediate 2304, FP8 dense, FP4 experts with UE8M0
scales (32-element blocks). Checkpoint 510 GB: routed experts 296.0, **Engram 203.1 (two FP8 tables of 98.3 GB + 3.07 GB
scales; 256-byte rows)**, attention 5.15, embed+LM head 2.65, shared experts 1.42, router/other 0.46, MTP 0.71, vision 0.82.
The tech report states layers 0-1 use sliding-window attention and the other 38 use **CSA2 (compressed sparse global
attention + hierarchical indexer) plus sliding-window attention**, plus Engram at layers 1 and 14. The official reference code
(`inference/`) is **not** in the repository, so none of this could be validated numerically.

## 5. Current baseline [M]
Warm decode, 50 tokens (6 recent runs): **5.8-7.1 s/token (0.14-0.17 tok/s)**; run-to-run noise about +/-10%. Stage
breakdown (one run): storage wait 4,528 ms; GPU expert stage 1,210 (uploads 925 GPU-busy, wait for GPU 962 CPU-blocked, compute
366); attention projections 222; shared expert 79; HC mixing/norms 224; binding 77; router 13; LM head 37. Cold first token 9-13 s.
Per-run p50/p95 are in `reports/ab_*.txt`. A 100-token run was not done because replies degrade beyond ~60-100 tokens (section 25).

## 6. Bottleneck analysis
Storage 70% (4.5 s) because (a) 22 layers read from a 404 MB/s drive, (b) one drive at a time, (c) buffered reads cost the NVMe
up to 25% and 9-17 points of CPU, (d) 22% hit rate. Then the GPU expert stage 19% (upload path at 4.3 of 10.5 GB/s, per-layer
blocking readback, 30 dispatches per layer). Then CPU dense work 11%. GPU busy time is 20% of a token.

## 7. Physical lower bounds per token [T from M inputs]
Bytes per token: experts touched 240 x 18.8 MB = **4.51 GB** (misses 3.46 GB at today's 22% hit rate); dense **6.9 GB**; LM head
1.32 GB; KV 10 MB; Engram ~12.7 KB [S: ~48 rows x 264 B]; PCIe 3.46 GB with GPU experts, 0 with CPU experts.

| Resource | Floor |
|---|---|
| Storage, current layout A, sequential | 4.2-5.25 s |
| B: 8 layers on SATA, rest NVMe, sequential / overlapped | 2.56 s / 1.73 s (13 GB cache: 1.44 / 0.73 s) |
| B2: SATA layers split over C: and D: | 2.39 s / 0.87 s (13 GB cache: 1.36 / 0.70 s) |
| all-NVMe (impossible: 296 GB of experts > 256 GB) | 1.03 s |
| dual NVMe (hypothetical) | 0.51-1.03 s |
| RAM bandwidth, CPU dense 6.9 GB at 20 GB/s measured | 0.34 s (0.14 s at the 51 GB/s theoretical peak) |
| CPU experts, all 240 | 0.22 s [M] |
| PCIe uploads at 10.5 GB/s | 0.33 s (zero with CPU experts) |
| D3D11 round trip, one per layer | 4 ms [M] |
| VRAM bandwidth, dense 5 GB resident | 0.03 s [T] |

**Absolute physical floor** (everything perfectly overlapped, B2 layout, 13 GB cache): max(storage 0.70, compute ~0.5) ~ **0.7 s/token,
~1.4 tok/s** [T]. This is a bound, not a prediction.

## 8-13. Experiment results (E1-E10)
Full tables are in `RESULTS_NOTES.md`.

| Exp | Question | Result |
|---|---|---|
| E1 lookahead router | Can the router run ahead? | Validated replica 99.96% of 20,927 calls. k=1 recall 67/82/89/93% at 6/12/24/48 reads; k=2 60/74/84/90; k=3 54/68/79/87; history baseline 37% |
| E2 long trace | Cache behaviour | 524 tokens, 8 prompts: 71% of experts touched, 8.7% compulsory floor, median reuse 720, top 10% of experts = 53% of accesses. LRU/CLOCK/ARC/LRFU within 1 point; production "blend" worse from 400 experts up; LFU worst; Belady 53% at 240 and 79% at 1,500 |
| E3 CPU FP4 | CPU vs GPU experts | kernel validated (rel-L2 5.6e-7 vs production); 240 experts in 219 ms at 16 threads, ~20 GB/s aggregate; 1 thread 3.8 ms/expert |
| E4 D3D11 | Round trip / upload | 20 KB round trip 101 us p50 (450 us p99). Upload 4.28 GB/s current vs 10.5 GB/s DMA ceiling |
| E5 VRAM | Dense residency | stable to 6 GB (141-154 GB/s); 6.5 GB = 86 GB/s; 7 GB = 56 GB/s; budget 7.2 GB |
| E6 storage | True bandwidth | C 511, D 404, E 3,370 MB/s at 18.9 MB reads, QD 2-4 (more QD hurts); D+E 3.73, C+D 0.92, C+D+E 4.24 GB/s |
| E7 eviction | Policy | recency policies tie; see E2 |
| E8 prefill histogram | Useful? | **NOT RUN** |
| E10 zero-copy | NO_BUFFERING | NVMe +34% (QD2) / +14% (QD4) and 15 points less CPU; no change on SATA |

Pipeline simulation (524 real tokens; drives as measured, demand priority over speculative reads, 64-expert staging buffer;
`pipeline_sim.py`, `run_pipe.py`):

| Layout | RAM cache | compute/layer | no prefetch | naive lookahead K=6 | value scheduler | oracle k=1 |
|---|---|---|---|---|---|---|
| A (current) | 13 GB | 13 ms | 3.76 s | - | **4.20 s (worse)** | 2.68 s |
| B2 | 4.5 GB | 13 ms | 2.26 s (0.44 tok/s) | - | 1.54 s (0.65) | 1.02 s (0.98) |
| B2 | 13 GB | 13 ms | 1.72 s (0.58) | 1.41 s (0.71) | **1.19 s (0.84)** | 0.80 s (1.25) |
| B2 | 13 GB | 21 ms | 2.04 s (0.49) | - | 1.42 s (0.70) | 0.89 s (1.13) |
| B2 | 13 GB | 46 ms | 3.04 s (0.33) | - | 2.39 s (0.42) | 1.87 s (0.54) |

The value scheduler prefetches a lookahead candidate only if its calibrated hit probability (0.95 at rank 1, 0.62 at rank 4,
0.36 at rank 6) exceeds a per-drive threshold (0.6 NVMe, 0.1 SATA). It recovers 58% of the gap between "no prefetch" and the
oracle. A wider budget loses (K=8: 1.55 s, K=12: 1.58 s); lookahead beyond one layer adds nothing even for the oracle;
NVMe-only prefetch is worse (1.57 s) than prefetching for all drives.

## 14. Architecture candidates and comparison

| Architecture | Storage (seq / overlap) | Compute floor | Sync | RAM | VRAM | Expected tok/s | Confidence |
|---|---|---|---|---|---|---|---|
| CURRENT | 4.5 s [M] | 1.9 s [M] | high | 8 GB | 3.4 GB | **0.16 [M]** | - |
| A: current engine + layout B2 + NO_BUFFERING | ~1.9 s | 1.9 s | high | 8 GB | 3.4 GB | ~0.26 [E] | high |
| B: A + staging-DMA upload + pipelined GPU stage | ~1.9 s | ~1.2 s [E] | medium | 8 GB | 3.4 GB | ~0.35-0.45 [E] | medium |
| C: dense on GPU + CPU experts + B2, 13 GB cache | 1.36 / 0.70 s [T] | ~0.5 s | low | 14 GB | 5 GB | **0.58 [T]** | medium |
| D: C + value lookahead scheduler | 0.7-1.4 s | ~0.5 s | low | 14 GB | 5 GB | **0.84 [T]** | medium-low |
| E: D with a perfect predictor (upper bound) | - | - | - | - | - | 1.25 [T] | bound |
| FINAL (D + adaptive governor, better predictor, larger cache) | | | | 14-16 GB | 5 GB | **0.7-1.0 [S]** | low |

Scores 0-10 (performance / confidence / correctness / simplicity / hardware fit / memory / storage / maintainability /
scalability / novelty): A 3/9/9/9/8/7/6/9/4/2 = 6.6 | B 4/7/8/7/8/7/6/7/5/3 = 6.2 | C 6/6/7/5/9/6/8/6/7/6 = 6.6 |
**D 8/5/7/4/9/6/9/5/8/9 = 7.0**. D is selected because the evidence supports it, not because it is novel;
**A is the mandatory first step of D**.

## 15. Failed or rejected ideas
- History/co-occurrence prefetch (earlier session): +28-62% bytes, 27-32% useful, no speedup [M].
- VRAM-primary expert cache: ~5% faster, within the +/-10% noise; its real effect is -3.7 GB RAM [M, paired 3 v 3].
- "Blend" cache policy: loses to LRU/CLOCK/ARC from 400 experts up [M/T].
- Wide speculative budgets (K>=8) and lookahead deeper than 1 layer [T]. NVMe-only prefetch [T].
- Queue depth above 4 for 18.9 MB reads (slower on both SSD types) [M].
- Not adopted because unmeasured: lossless compression of FP4 weights (entropy near 4 bits), Vulkan/ROCm, DSpark/speculative decoding.

## 16. Successful ideas
Layer rebalancing with Engram/MTP moved off the NVMe [M+T]; dual-SATA aggregation (C+D sum to 918 MB/s [M]); lookahead router replay
(validated replica) [M]; per-drive value-based speculation [T]; CPU FP4 AVX2 experts [M]; NO_BUFFERING for NVMe [M]; staging-DMA upload
ceiling of 10.5 GB/s [M].

## 17. Novel ideas discovered
(1) the 110 GB of dead weight on the only fast drive; (2) cross-drive overlap as what lookahead really buys (turns sum into max);
(3) probability-calibrated, drive-cost-weighted speculation (SATA mistakes cost 8x NVMe mistakes); (4) layer placement co-designed with the
cache to a fixed point (the optimiser puts the most reusable layers on the slow drives).

## 18. Final architecture (D, "storage-planned CPU/GPU hybrid")
```
 tokens -> embed -> [ 40 x ( GPU dense: attention (MLA + CSA2 later), shared expert, router, norms )
                       |  router input -> lookahead router(L+1) ----> Prediction Engine
                       v                                             |  (P(rank) x refetch cost)
              exact experts for layer L                              v
   RAM expert cache <-- Storage Scheduler <-- demand queue (priority) + speculative queue
        (13 GB, CLOCK/ARC-class)      \--> NVMe E: 32 layers | SATA D: 4 layers | SATA C: 4 layers (NO_BUFFERING, 18.9 MB, QD2-3)
              |  CPU experts (AVX2 FP4, 16 threads) -> weighted sum -> GPU (one 20 KB round trip per layer) ]
 final norm -> LM head (GPU or CPU) -> greedy token
```
Memory hierarchy: NVMe/SATA (complete checkpoint) -> RAM warm cache + staging (experts) -> VRAM (dense weights, <= 5 GB) -> CPU
(experts) / GPU (dense). Storage hierarchy: E: 32 layers; D:+C: the 8 layers whose experts are most reusable; Engram table and MTP weights
on D:/C: (random 264-byte rows). Residency: CLOCK/ARC-class recency with drive-cost weighting for SATA-resident layers; speculative
entries in a separate staging area, promoted on demand.

## 19. Scheduler, DAG, responsibilities, sync model
- Per layer: GPU attention -> router input ready: issue lookahead for L+1 -> demand reads for L's experts (priority) -> CPU experts as
  bytes arrive -> reduction -> GPU. Independent work: CPU experts of layer L overlap the storage scheduler serving L+1 speculation and
  the GPU shared expert.
- Sync model: one blocking D3D11 readback per layer (4 ms/token), lock-free queues between predictor/scheduler/workers, a thread pool
  (no `std::async` per read), one 20 KB hidden state across PCIe each way per layer.

## 20-23. Correctness model, performance model, 1 tok/s feasibility, risks
**Correctness** is currently *not established* (section 25). Any new architecture is gated on (i) the AVX2 expert matching the
production kernel (done: rel-L2 5.6e-7), (ii) implementing CSA2/indexer/Engram against the official `inference/` code, and (iii)
token-for-token agreement with that reference for 1/5/20/50/100 tokens. **Performance model:** token time = f(layout, cache size,
compute/layer, predictor), checked only against today's engine (model storage 4.2-5.25 s vs measured 4.0-4.7 s). **Risks:** the
missing model parts add unknown compute and storage (CSA2 KV, indexer scoring, Engram lookups); the simulator assumes single-queue
drives and ignores OS cache and thermal/SATA-state effects (measured +/-10% run-to-run); predictions come from 8 prompts and
524 tokens; 13 ms/layer compute is an estimate [E].

## 24. Implementation phases, pass/fail, rollback
| Phase | Change | Expected | Pass | Fail | Rollback |
|---|---|---|---|---|---|
| P0 | Reproduce baseline (done) | 6.4 s | within 10% | - | - |
| P1 | Move Engram/MTP off E:, 14 layers D:->E:, 4 layers D:->C: (SHA-256-verified copies; `scripts/plan_shard_move.py` exists) | storage 4.5 -> ~2 s | >=30% token-time cut in a paired A/B (3 v 3) | <10% | originals stay until verified; copy back |
| P2 | NO_BUFFERING aligned reads into a preallocated arena; read thread pool | +14-34% NVMe, less CPU | bytes/s up, CPU down | no gain | env flag |
| P3 | CPU expert engine (AVX2 FP4) + staging upload for GPU-resident experts | GPU stage 1.2 -> ~0.3 s | output matches production within 1e-6 rel-L2 | any divergence | keep GPU path |
| P4 | Dense weights resident in VRAM (<=5 GB), pipelined GPU stage | non-I/O ~0.5 s | stable at 5 GB, no TDR | paging | stream from RAM |
| P5 | Storage scheduler (demand priority, staging) | enables P6 | no starvation of demand reads | demand latency up | disable speculation |
| P6 | Lookahead + value scheduler | -20-30% token time | paired A/B beats P5 by more than noise | worse | env flag (default off) |
| P7 | Adaptive governor (thresholds, cache, QD from telemetry) | stability | >= 8 GB free for Windows | pressure | static config |
| P8 | **CSA2 + indexer + Engram per the official reference** | fidelity | token-exact vs reference | divergence | block release |
| P9 | Integrate; re-benchmark; then optimise the faithful model | final | table in section 14 | | |
P8 should come *before* any claim of speed: P1-P7 optimise an incomplete model.

## 25. Attention / CSA / Engram audit
The runtime executes sliding-window MLA with attention sinks (128 positions) in all 40 layers. Not executed: the compressor (ratio 2/1
layers), the indexer (top-512), KV sharing across layers 2/8/14/20, Engram (layers 1 and 14), MTP/DSpark. Observed: coherent for ~50
tokens, degenerate after ~60-100 tokens, identical on the legacy and new data paths. The missing paths are the lead suspects; not proven.

## 26-27. Benchmark protocol and targets
Same model, binary, prompt, greedy decoding; alternating A/B with 3+ repeats because noise is +/-10%; report p50/p95. Targets: primary
1.0 tok/s, secondary 1.25, stretch 1.5. **Projected (not achieved):** current 0.16 [M]; A ~0.26 [E]; C 0.58 and D 0.84 [T]; oracle bound 1.25 [T].

## 31. Verdict (answers to section 45)
1. **Current measured:** 0.16 tok/s (6.4 s/token; range 5.8-7.1 s).
2. **Physical bottleneck:** storage: one drive at a time, 22 layers on a 404 MB/s SATA drive, 765 MB/s effective.
3. **Absolute physical floor:** ~0.7 s/token (~1.4 tok/s) with perfect overlap on this hardware, B2 layout and a 13 GB cache [T].
4. **Strongest architecture:** D (rebalanced storage + CPU experts + dense on GPU + calibrated lookahead scheduler).
5. **Why:** it removes the dead weight from the fast drive, uses all three drives, deletes the PCIe expert upload, and reads ahead only when
   the expected saving beats the wasted bytes.
6. **Proven (measured):** drive bandwidths and their additivity; the dead weight on E:; CPU kernel speed and correctness; round-trip and
   upload costs; VRAM residency limits; NO_BUFFERING gains; lookahead recall with a validated router replica; cache policy ordering.
7. **Uncertain:** simulated token times (a model, not the engine); compute per layer (13-21 ms assumed); cost of the missing model parts;
   the Engram access pattern; predictor generalisation beyond 8 prompts.
8. **Minimum implementation to test it:** P1 (move shards) and P2 (direct reads): both testable against the existing engine.
9. **Projected:** D 0.84 tok/s [T], range 0.65-1.0 depending on cache size and predictor; on the faithful model likely lower [S].
10. **A measurement that would prove it wrong:** a paired engine benchmark after P1 failing to cut storage wait by >=30%, or the lookahead
    scheduler failing to beat no-prefetch by more than the noise after P5.
11. **Can this machine reach 1 tok/s?** **UNCERTAIN.** For: the oracle simulation gives 1.25 tok/s and the physical bound is ~1.4. Against: the
    realistic predictor gives 0.84, the model is incomplete (CSA2/indexer/Engram will add cost), and nothing here has been built.
12. **Fastest plausible without quantization/pruning/distillation/substitution:** about 0.8-1.0 tok/s on the incomplete model; a faithful model
    is expected to be slower [S].

---

# ASEMA NEXT-GEN: IMPLEMENTATION SPECIFICATION

Each subsystem: inputs / outputs / memory / threads / sync / algorithm / failure modes / performance target / correctness requirement.

**1. Static Model Planner.** In: shard index, drive probe (bus type, measured bandwidth), config. Out: a *model plan* (tensor -> drive/offset,
layer -> drive, GPU memory map, arena sizes). Memory <10 MB; startup thread only. Algorithm: classify drives, assign layers with the co-designed
optimiser (`cache_sim2.py: optimise_layout`), keep Engram/MTP on the slowest drives. Failure: missing shard -> fail loudly. Target: plan in <1 s.
Correctness: every tensor resolves to exactly one existing byte range.

**2. Expert Page Manager.** In: (layer, expert) requests with priority. Out: pinned 4 KiB-aligned 18.8 MB pages in a preallocated arena (240-1,000
pages from the governor). Thread: lock-free state machine FREE -> LOADING -> READY -> IN_USE with pin counts. Algorithm: CLOCK/ARC-class recency with
extra weight for refetch cost of SATA-resident layers; speculative pages in a separate 64-page staging area, promoted on first demand. Failure:
arena exhaustion -> evict speculative first. Target: >=51% hit at 700 pages (E2). Correctness: bytes identical to the shard (CRC spot checks).

**3. Storage Scheduler.** In: demand and speculative read requests. Out: completions. Threads: one reader per drive plus a pool; `NO_BUFFERING`,
sector-aligned 18.9 MB reads, QD 2-3 per drive. Algorithm: per-drive priority queues (demand > speculative); a speculative read starts only when the
drive's demand queue is empty; cancel queued (never in-flight) speculation for a mispredicted layer. Failure: demand read error -> retry, then fail
loudly; speculative errors ignored. Target: >=95% of measured per-drive bandwidth. Correctness: length and alignment checks.

**4. RAM Residency Manager / 5. VRAM Residency Manager.** RAM: expert pages + staging + KV; never below 8 GB available to Windows. VRAM: dense
weights only (<=5 GB plus <=64 MB scratch); no expert cache in VRAM in design D (a hot-expert-on-GPU variant is unmeasured). Failure: DXGI budget
drop -> streaming mode. Target: no paging (sweep bandwidth >=140 GB/s).

**6. Prediction Engine.** In: per-layer router input (post-FFN-norm, FP32). Out: ranked candidates for layer L+1 with calibrated P(rank).
Algorithm: apply the real gate of L+1 (384x5120 GEMV, ~2 MFLOP); a table P[k][rank] learned online; prefetch a candidate when P x drive refetch
cost clears a per-drive threshold (start 0.6 NVMe / 0.1 SATA; the governor adapts). Failure: low accuracy -> thresholds rise, speculation off.
Target: 67% recall at 6 reads. Correctness: never changes which experts are computed; the real router decides.

**7. CPU Expert Engine.** AVX2 FP4 GEMV (pshufb LUT decode, FP32 activations), 16 workers, rows split across threads, 3 stages with barriers; weights
read from the page arena; scratch ~100 KB per thread. Target: <=6 ms per layer (6 experts) [M 5.8 ms]. Correctness: rel-L2 <= 1e-6 vs the
production scalar kernel on real weights (tested: 5.6e-7).

**8. GPU Dense Engine.** D3D11 compute with persistent buffers; dense FP8 weights resident (<=5 GB; the remainder streams); attention projections,
shared expert, router, norms, LM head. One blocking 20 KB readback per layer. Target: <=2 ms per layer. Correctness: per-layer cosine >= 0.9999 vs
the CPU path.

**9-11. Attention, Engram and KV Engines.** *Not implemented today.* Required for fidelity: CSA2 compressed KV + hierarchical indexer; Engram lookups
(addresses are a function of the last n tokens, so they are known before the layer runs and can be prefetched; rows 256 B + 8 B scale); a KV store
with a per-layer compression ratio. Correctness: token-exact vs the official reference. This is the largest unknown cost.

**12. Execution DAG.** Nodes: READ, CPU-EXPERT, GPU-DENSE, REDUCE, SYNC. Edges per layer as in section 19. Nodes issue as dependencies resolve;
speculation never blocks demand.

**13. Adaptive Governor.** Inputs: tokens/s, stage times, per-drive utilisation and queue depth, hit/useful/late/wasted speculation, Windows available
memory. Controls: arena size, thresholds, speculative budget, read QD (2-3), worker count. Rule: pressure -> reduce speculation, then cache, then
concurrency; never starve the desktop. Failure: oscillation -> hysteresis.

**14. Benchmark/Telemetry Engine.** Per-token JSON: stage times, per-drive bytes and busy time, speculation outcomes, GPU timestamp queries, CPU
utilisation. Alternating A/B harness with >=3 repeats and noise reporting.
