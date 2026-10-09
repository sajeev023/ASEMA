# Phase 1 — Storage rebalance: final report

All numbers are **[M] measured** on this machine unless tagged **[T]** (modeled) or **[I]** (inference, not directly measured).
Raw data: `reports/storage/*.json|txt|csv` (git-ignored), summary `phase1_bench_summary.json`, plan `PHASE1_STORAGE_MIGRATION_PLAN.md`,
manifest `ASEMA_STORAGE_LAYOUT_PHASE1.json`.

## Decision gate

| Question | Answer |
|---|---|
| Did storage rebalancing work? | **PARTIAL** — correct, safe, faster in every paired run, but about +15 %, not the ~2x the model predicted |
| Baseline | **0.140 tok/s** (7,131 ms/token, same exe, original layout via manifest A); 0.144 tok/s with the pre-change exe |
| Rebalanced | **0.162 tok/s** (6,185 ms/token) |
| Speedup | **+15.3 % tok/s (−13.3 % time/token)** vs the same-exe control; +12.5 % vs the original exe |
| Did the measured result approach the research model? | **NO** on magnitude [T: 3.76 s → 1.78 s per token, 2.1x], YES on direction |
| Dominant bottleneck now | expert load path: **4.17 s of 6.19 s per token (67 %)** is still "waiting on storage", with the NVMe running at roughly a quarter of its measured capability |
| Proceed to Phase 2? | **YES** — Phase 2 = the loader read path (section 16) |

## 1–2. Before / after layout

Expert layers (layer L = shard L+3) per drive:

| | before | after |
|---|---|---|
| E: NVMe | 18 (0,1,3–7,9,30–39) | **32** |
| D: SATA | 22 | **5** (10,14,16,18,25) |
| C: SATA | 0 | **3** (17,26,28) |

Deviation from research: 3 layers on C: instead of 4 (C: is the system drive; 20 GiB free floor). Simulator difference between the two: 1.776 vs 1.763 s/token [T].
The layout is "minimum-move": every layer already on E: stayed; 14 layers came from D: (the lowest-miss 8 of D:'s 22 stayed on SATA). It is not the simulator's B2 layout, which also shuffled layers off E:.

## 3–5. Files moved, bytes, hashes — all PASS

* Step 1 (relocate, E:→D:): shards 44, 45, 46 (MTP) and 48 (Engram layer 14), 101.95 GiB. **Their E: copies were removed** after the D: copy was re-read unbuffered and its SHA-256 matched. These are the only files removed this phase, and the engine never reads them. Rollback: copy back.
* Step 2 (copy, D:→E: `shards_p1`): 14 layer shards (layers 2,8,11,12,13,15,19–24,27,29), 96.5 GiB.
* Step 3 (copy, D:→C: `shards_p1`): 3 layer shards (layers 17,26,28), 20.7 GiB.
* 21 operations, **21 ok, 0 failed**; every record has `sha256_src == sha256_dst` and equal size (`reports/storage/migration.jsonl`). 27 further unmoved files were hashed for the manifest (`hashes.jsonl`). Total moved: 219.0 GiB.
* Tool: `shardcopy.cpp` (unbuffered read/write, SHA-256 via CNG, `.part` + rename, refuses to overwrite, resumable). Its 20 MB self-test caught a truncation bug before any shard was touched.
* Static check `verify_layout.py` on both manifests: 48 shards, 96,085 tensors all present in the headers of the shards the index names, data region ends exactly at file size, each layer on exactly one drive → **INTEGRITY PASS**.
* Every original layer shard still exists on D:/E:. **Nothing from the old placement was cleaned up.**

## 6–7. Free space (GiB)

| Drive | before | after step 1 | now |
|---|---|---|---|
| C: | 44.5 (after deleting my own 3 GiB test file; 41.5 earlier) | 44.7 | **21.5** (floor 20, thin; Windows also writes here) |
| D: | 226.5 | 124.5 | **124.5** |
| E: | 11.9 | 113.9 | **17.5** |

## 8–9. Runtime change (storage mapping only)

`ASEMA_STORAGE_MANIFEST` / config key `storage_manifest` → `M8MultiVolumeManager::load_storage_manifest` (`src/m8/m8_multi_volume.cpp`). With a manifest set, **only the listed files are indexed and resolved; directories are not scanned**. An invalid or incomplete manifest is a hard error (no silent fallback). With no manifest the old behaviour is unchanged. No scheduler, kernel, cache or model semantics changed.
Why a manifest: `register_volume` indexes every `.safetensors` in a directory, so duplicate copies would have made the active copy ambiguous.

## 10. Benchmark (20 tokens, greedy, same prompt, 3 runs each; `bench_harness.py`)

| | ms/token (runs) | mean | median | min | max | tok/s |
|---|---|---|---|---|---|---|
| Baseline, original exe | 7142 / 7594 / 6139 | 6958 | 7142 | 6139 | 7594 | 0.144 |
| **Control**, new exe, manifest A (alternated) | 7310 / 7036 / 7047 | **7131** | 7047 | 7036 | 7310 | **0.140** |
| **Rebalanced**, manifest P1 (alternated) | 6363 / 6329 / 5864 | **6185** | 6329 | 5864 | 6363 | **0.162** |

Ranges do not overlap (control 7036–7310 vs rebalanced 5864–6363). Earlier I measured ±10 % run-to-run noise, so +15 % is real but modest.

| Per warm token (means) | control | rebalanced |
|---|---|---|
| expert cache + storage wait | 5,168 ms | **4,171 ms** (−19 %) |
| expert GPU stage (wall) | 1,234 ms | 1,277 ms |
| storage rate while loading | 704 MB/s | **885 MB/s** |
| engine-reported storage bytes | 3,633 MB | 3,672 MB |
| CPU cores busy | 0.64 | 0.85 |
| GPU engine utilisation (sampled, sum of engines) | 11 % | 14 % |
| cold first token (prefill) | 132–146 s | 94–99 s |

Per-drive **physical** reads per warm token (OS counters, warm window): control D 1.07 GB / C ~0.02 / E 1.03 GB → rebalanced **D 0.22 / C 0.14–0.32 / E 1.7–1.9 GB**. The placement did what it was meant to. (The engine's "3.6 GB/token" counts buffered reads; about half is served by the Windows file cache.)
The control's cold first token (132–146 s) was slower than the original exe's (94–101 s) for a reason I did not find. The warm metric is unaffected.

## 11. Correctness

The 20-token greedy output is **byte-identical in all 9 runs** across both layouts and both executables (`output:` line hash equal). STORAGE INTEGRITY = PASS; runtime load = PASS; generation = PASS.

## 12. Did all three drives work concurrently? — **No** [I]

* Layers execute sequentially and each layer's experts live on one drive, so at any instant the loader waits on **one** drive. Per-second OS counters show 2–3 drives "active" in most seconds, but that is alternation between consecutive layers, not overlap. One second is the finest typeperf resolution, so sub-second concurrency cannot be proven; this is inference from the structure.
* The NVMe is far from busy: peak 1.0 GB/s (1-s samples) against 3.37 GB/s measured with 18.9 MB unbuffered reads at QD 2–4; roughly 1.8 GB over ~3 s of E-layer wait ≈ 0.6 GB/s [I]. SATA layers (8 of 40) read at about 0.4–0.5 GB/s as expected.
* So the placement cut the bytes served by slow drives by ~80 %, but the loader does not drive the fast drive at its capability. Candidate causes, **not yet isolated**: buffered reads (the earlier `NO_BUFFERING` test gave +14–34 % on the NVMe), low queue depth (per-layer demand reads, tensors read separately), thread hand-off/locking, and non-overlapped layers.
* C: wrote at only 21–26 MB/s and verified at 220–280 MB/s during the copy (511 MB/s in the earlier read test), so the 3 layers on C: may be slower than modeled. Not measured separately per drive.

## 13. Why the model (2.1x) and the machine (1.15x) disagree

The simulator assumed each layer's reads run at the drive's best sustained rate and that compute is 13 ms/layer. In reality the storage rate is 0.7–0.9 GB/s whatever the drive mix, and about 1.3 s of GPU upload/sync plus 0.6 s of other CPU work sit on the critical path. The rebalance removes the SATA part of the wait, not the loader inefficiency.

## 14. Rollback status

Intact. Unsetting `ASEMA_STORAGE_MANIFEST` (or using `ASEMA_STORAGE_LAYOUT_BASELINE_A.json`) restores the old placement exactly: originals were not touched, and the new copies live in separate `shards_p1` directories that no directory scan sees. The relocated MTP/Engram shards are on D: (scanned, harmless). **Cleanup of the old copies has NOT been done and should wait for Phase 2.**
Known follow-up before any cleanup: `asema models` / `asema doctor` scan only shard directories (they pass today only because the originals remain); they must learn the manifest. Not changed in this phase.

## 15. Caveats

* 20-token runs, one prompt; prefill (cold token ~95 s) dominates wall time but not the warm metric. n=3 per arm.
* Baseline used the pre-change exe; the control used the new exe with the equivalent manifest. They agree within noise (6958 vs 7131 ms).
* GPU utilisation is a best-effort 1-s sample started 25 s into the run.
* The model is still the incomplete one (no CSA2 / indexer / Engram / MTP); these speeds are for that model.

## 16. Phase 2 recommendation (do not start automatically)

**Phase 2 = fix the expert read path, one variable at a time, on the new layout:** (a) `FILE_FLAG_NO_BUFFERING` + aligned reads for expert tensors; (b) issue all of a layer's tensor reads at once with 2–4 outstanding reads per drive and no global lock; (c) add per-drive wait counters to the loader's telemetry and re-measure. Expected from measured I/O capability [T]: E-layer wait 3 s → ≤1 s, i.e. about 0.3–0.35 tok/s *before* any lookahead. Lookahead prefetch and CPU experts stay later phases, as in the architecture report. Gate: paired 3v3 on this manifest with identical output.
