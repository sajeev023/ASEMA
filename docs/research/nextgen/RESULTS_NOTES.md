# NextGen research notes (working file; folded into the final report)

Labels: [M] measured, [T] theoretical, [E] estimated, [S] speculative.

## E6 - Storage throughput  [M]

QUESTION: what bandwidth do the three drives really sustain for expert-sized reads, alone and together?
METHOD: `experiments/nextgen/io_bench.cpp`, FILE_FLAG_NO_BUFFERING, random 4 KiB-aligned reads of real shard
files (C: a 3 GiB random-data file), one thread per outstanding request, 2.5-4 s per point.
Raw output: `reports/io_D.txt`, `io_E.txt`, `io_DE.txt` (git-ignored).

| Drive | bus | 18.9 MB reads, best | at QD | notes |
|---|---|---|---|---|
| C: | SATA SSD (OS drive) | 511 MB/s | 2 | |
| D: | SATA SSD | 404 MB/s | 2-4 | 4 KB reads collapse to ~3 MB/s with 62 ms stalls |
| E: | NVMe PM981 | 3,370 MB/s | 2-4 | QD1 2.68 GB/s; QD>=8 gets slower (2.79 GB/s) |

Concurrency [M]: D+E = 3.73 GB/s (sum of the two), C+D = 918 MB/s (sum), C+D+E = 4.24 GB/s.
The drives are independent; there is no shared bottleneck up to 4.2 GB/s.
More queue depth than 2-4 never helps for expert-sized reads and hurts latency.
CPU cost of the I/O is negligible (<2% of 16 threads).

## Topology  [M]
Disks: 2x "CONSISTENT SSD S7 512GB" (SATA; C: and D:) and 1x Samsung PM981 256 GB NVMe (E:). NOT two NVMe.
Free space: C 49 GB, D 226 GB, E 12 GB. RAM 2x16 GB DDR4-3200. GPU RX 580 2048SP (8 GB via DXGI).

## Checkpoint layout finding  [M]
Shard contents (from the index) and drive:
- layers 0,1,3-7,9 and 30-39 experts (7.4 GB per layer): E:  (18 layers)
- layers 2, 8, 10-29 experts: D: (22 layers)
- shard 47 (101.5 GB, Engram table) on D:, shard 48 (101.5 GB, Engram table) on E:
- shards 44-46 (MTP weights, 8.0 GB total) on E:
The current engine never reads the Engram tables or the MTP weights, so about **109.5 GB of the 238 GiB NVMe
holds data the engine does not touch**, while 22 layers of live experts are on SATA.

## Physical storage floor per token (expert bytes only)  [T from M bandwidths + M hit rate]
Expert = 18,800,640 B. 240 expert accesses/token. Measured hit rate with 192 VRAM slots: 22.3% -> 184 reads
= 3.46 GB/token (the benchmark measures 3.46 GB, so the byte model matches).

IMPORTANT CORRECTION [M]: the engine's measured "storage rate while loading" is only 765 MB/s, far below the 4.2 GB/s the
three drives deliver together. Layers run one after another and every layer's 6 experts live in ONE shard on ONE drive,
so only one drive works at a time: storage time is the SUM over drives (sequential), not the MAX. Two columns:
"sequential" = what the current engine can do; "overlapped" = needs reads of layer L+k issued while layer L waits.

| Layout | bytes per drive (3.46 GB/token total) | sequential (sum) | overlapped (max) |
|---|---|---|---|
| A current: 22 layers D, 18 E | D 1.93 GB, E 1.58 GB | 4.78 + 0.47 = **5.25 s** (measured wait 4.0-4.7 s) | 4.78 s |
| B: Engram+MTP off E, 14 layers D->E (8 D, 32 E) | D 0.70, E 2.80 | 1.73 + 0.83 = **2.56 s** | 1.73 s |
| B2: as B, SATA layers split 4 on D + 4 on C | D 0.35, C 0.35, E 2.80 | 0.87 + 0.69 + 0.83 = **2.39 s** | 0.87 s |
| all on NVMe (impossible: experts alone ~296 GB > 256 GB) | E 3.46 | 1.03 s | 1.03 s |
| dual NVMe (hypothetical) | 1.73 each | 1.03 s | 0.51 s |

Layout B/B2 needs no change to the model bytes: it moves ~110 GB of data the engine never reads (Engram + MTP) off the
NVMe and 14 layers of experts onto it.

## E3 - CPU FP4 expert throughput  [M]
QUESTION: how fast can the 5700X run the real FP4 experts with AVX2, vs the current GPU expert stage?
METHOD: `experiments/nextgen/cpu_expert_bench.cpp` (AVX2 pshufb-LUT decode + FMA, FP32 activations, same
E2M1/UE8M0 math as the production scalar kernel). Real expert bytes (layer 30, experts 0-5) copied into up to
240 DISTINCT 18.8 MB buffers so DRAM, not the 32 MB L3, is measured. Raw: `reports/cpu_expert_bench.txt`.
CORRECTNESS: vs production scalar kernel on real weights: max|diff| 4.8e-6, rel-L2 5.6e-7, cosine 1.000000000.
RESULT:
- 1 thread: 3.8 ms/expert (scalar production kernel: 13.5 ms) -> 4.9 GB/s
- Expert-parallel aggregate saturates at ~20 GB/s with 16 threads (DDR4-3200 dual channel peaks ~51 GB/s [T])
- N=240 experts, 16 threads: 219 ms total (0.91 ms/expert); 8 threads: 265 ms; 4 threads: 359 ms
- One decode layer (6 experts, each split over T threads): T=16 5.8 ms (232 ms/token), T=8 8.0 ms (321 ms/token)
- SMT helps modestly: 8 -> 16 threads gives 265 -> 219 ms
CONSEQUENCE: executing experts on the CPU costs ~0.22-0.32 s/token IF their bytes are already in RAM. The current GPU expert
stage costs ~1.15 s/token, dominated by the 3.5 GB/token upload (see E4). CPU experts remove the upload entirely.
Caveat: this takes all 16 threads while it runs; attention/dense work and I/O threads compete for them.

## E4 - D3D11 round trip and upload  [M]  (`d3d_bench.cpp`; raw `reports/d3d_rt.txt`, `d3d_upload.txt`)
- Round trip (upload, trivial compute, CopyResource to staging, blocking Map): 1 KB 90 us p50; **20 KB 101 us p50 / 123 avg /
  450 p99**; 256 KB 187 us; 1 MB 536 us. Dynamic-buffer Map(WRITE_DISCARD) is equivalent (96 us at 20 KB, p99 2.4 ms).
  => one small round trip per layer costs ~4 ms/token. VIABLE.
- Upload of one 18.8 MB expert: current `UpdateSubresource` path **4.28 GB/s (4.4 ms/expert)**; VirtualLock'ed source is slower
  (3.63); staging Map+memcpy+CopyResource ring 5.88 GB/s; CPU memcpy into mapped staging alone 23 GB/s; staging->DEFAULT
  CopyResource (pure DMA) **10.5 GB/s (1.78 ms/expert)**. Practical PCIe ceiling ~10.5 GB/s. The current uploader uses 41% of it.

## E5 - Dense VRAM residency  [M]  (`d3d_bench.cpp vram`; raw `reports/d3d_vram.txt`)
128 MiB buffers swept by a compute kernel: 1 GB 115 GB/s, 2 GB 128, 4 GB 145, 5 GB 154, **6 GB 141**, 6.5 GB **86**, 7 GB **56** GB/s.
DXGI local budget 7.2 GB (7.0 once 6 GB are in use). Allocation never failed up to 7 GB, but sustained bandwidth collapses past
~6 GB (paging pressure). Practical resident ceiling ~6 GB on an idle desktop; <=5 GB leaves room for the desktop. No TDR observed.

## Byte accounting per token  [M from shard headers]
routed experts 296.0 GB total | engram 203.1 GB (2 tables of 98.3 GB FP8 + 3.07 GB scales; rows are 256 B + 8 B) |
attention 5.15 | embed+LM head 2.65 (LM head 1.32 GB BF16) | shared experts 1.42 | router 0.16 | other 0.30 | mtp 0.71 | vision 0.82.
Dense bytes touched per token ~6.9 GB (+1.3 GB LM head). Experts touched per token: 240 x 18.8 MB = 4.51 GB (3.46 GB are SSD misses).

## E10 - Zero-copy / NO_BUFFERING storage path  [M]  (`io_bench.cpp`, IO_BUFFERED=1; fresh seeds, cold cache)
| drive | buffered (engine today) QD2 / QD4 | NO_BUFFERING QD2 / QD4 | CPU of 16 threads, buffered vs direct |
|---|---|---|---|
| D: SATA | 402 / 405 MB/s | 404 / 404 MB/s | ~1.5% vs ~0.2% |
| E: NVMe | 2,508 / 2,955 MB/s | 3,355 / 3,369 MB/s | 8.6% / 16.9% vs 1.1% / 1.3% |
Direct reads: +34% / +14% NVMe throughput, ~15 points of 16-thread CPU saved (~2.5 threads), and the Windows file cache is
not polluted (it competes with the RAM expert cache). No benefit on the saturated SATA drives.
(A first buffered QD4 run reusing the QD2 seeds hit the file cache and showed impossible 830 / 4,154 MB/s; discarded.)

## E2 - Long trace and cache behaviour  [M]  (524 tokens, 8 prompts, 125,562 accesses; `cache_sim2.py`; raw `reports/cache_full.txt`)
Prompts: story, Python prime checker, photosynthesis, French translation, French Revolution, cookie recipe, TCP vs UDP, capital of
Australia (the last two shorter). Tokens include the prompt prefill steps. Caches carry over between prompts (a chat session).
- 10,930 of 15,360 experts (71%) touched; compulsory misses 8.7% of accesses; median reuse distance 720 accesses.
- Skew is weaker than the 63-token trace suggested: top 5% of experts = 36.6% of accesses, top 10% = 53.0%, top 25% = 79.2%.
| policy | 240 | 400 | 700 | 1000 | 1300 | 1500 |   (hit rate %, cache size in experts)
|---|---|---|---|---|---|---|
| LRU | 26.2 | 40.8 | 51.4 | 57.5 | 62.4 | 65.0 |
| CLOCK | 28.2 | 40.9 | 52.1 | 58.4 | 62.8 | 65.3 |
| LFU | 16.5 | 23.5 | 33.4 | 40.6 | 46.5 | 50.0 |
| LRFU | 24.6 | 41.3 | 51.5 | 57.5 | 62.5 | 65.0 |
| ARC | 26.1 | 41.4 | 51.9 | 58.1 | 62.6 | 64.9 |
| blend (production) | 28.9 | 38.0 | 48.7 | 55.6 | 60.4 | 63.0 |
| blend x drive cost | 23.3 | 30.6 | 40.9 | 48.5 | 54.5 | 57.6 |
| Belady oracle | 53.3 | 61.3 | 69.3 | 73.9 | 76.9 | 78.5 |
Recency-based policies (LRU/CLOCK/ARC/LRFU) are within ~1 point of each other and beat the production blend from 400 experts up.
CLOCK-Pro was not simulated (plain CLOCK stands in). The Belady gap (25+ points) is the headroom for prediction.
Storage time per token with the drive-cost-aware policy (sequential / perfectly overlapped):
| cache | GB/token read | A current | B (8 SATA layers on D) | B2 (4 D + 4 C) |
|---|---|---|---|---|
| 240 (4.5 GB) | 3.45 | 4.22 / 3.63 s | 2.02 / 1.13 s | 1.90 / 0.89 s |
| 700 (13 GB) | 2.66 | 2.94 / 2.44 | 1.44 / 0.73 | 1.36 / 0.70 |
| 1000 (19 GB) | 2.32 | 2.51 / 2.07 | 1.23 / 0.62 | 1.16 / 0.62 |

## E1 - Lookahead router  [M]  (FULL: 8 prompts, 20,927 layer calls; raw `reports/lookahead_full.txt`)
QUESTION: does router(L+k) applied to the router input of layer L predict layer L+k's true experts well enough to read early?
METHOD: instrumented COPY of the engine (`experiments/nextgen/instrumented`, production untouched) dumps the router input
(post-FFN-norm hidden state, FP32) at every layer call; `lookahead.py` replays it through the real gate weights.
VALIDATION: my replica reproduces the engine's recorded top-6 on 20,918 / 20,927 layer calls (99.96%; near-ties at layers 36, 39).
RESULT (recall of the TRUE top-6 of layer L+k within the top-K predicted):
| k | R@6 | R@12 | R@24 | R@48 | history baseline (previous token's 6 experts) |
|---|---|---|---|---|---|
| 1 | 67.1 | 81.6 | 89.0 | 93.4 | 36.7 |
| 2 | 59.6 | 74.3 | 83.7 | 90.2 | 36.8 |
| 3 | 54.1 | 68.4 | 79.1 | 87.0 | 37.1 |
| 4 | 50.4 | 64.3 | 75.2 | 84.1 | 37.4 |
CONSEQUENCE: lookahead recall is ~1.8x the best history predictor. End-to-end value is decided by `pipeline_sim.py` (below).
