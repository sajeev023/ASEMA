# Phase 1 storage migration plan

Generated 2026-10-08 20:08 from the LIVE filesystem (not from CURRENT_RUNTIME_TRUTH.md).

## Free space (GiB)

| Drive | Total | Free now | Free after step 1 | Free after migration | Required margin | OK |
|---|---|---|---|---|---|---|
| C: | 475.90 | 44.46 | 44.46 | 23.81 | 20 | yes |
| D: | 476.92 | 226.47 | 124.52 | 124.52 | 50 | yes |
| E: | 237.84 | 11.90 | 113.85 | 17.45 | 15 | yes |

**Plan internally consistent: YES**

## Steps

Step 1 *relocates* (copy, verify by unbuffered re-read, only then remove the E: copy) the Engram/MTP shards the engine never reads, which frees the NVMe. It is the one step that removes a file: the verified identical copy on D: becomes the only copy. Rollback = copy it back.

Steps 2-3 are plain verified copies into new `shards_p1` directories; **every original stays where it is** (rollback = unset the manifest). Nothing is cleaned up in this phase.

| Step | Kind | Layer | Shard | Size (bytes) | GiB | Source | Destination | Source mtime | Checksum | Rollback |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | relocate | (mtp/engram) | model-00044-of-00048.safetensors | 2652728736 | 2.47 | E:/asema_models/DeepSeek-V4.1-Flash/shards/model-00044-of-00048.safetensors | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00044-of-00048.safetensors | 2026-10-02 17:10 | SHA-256 streamed while copying, re-verified by re-reading the destination | copy back D->E |
| 1 | relocate | (mtp/engram) | model-00045-of-00048.safetensors | 2573998176 | 2.40 | E:/asema_models/DeepSeek-V4.1-Flash/shards/model-00045-of-00048.safetensors | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00045-of-00048.safetensors | 2026-10-02 17:23 | SHA-256 streamed while copying, re-verified by re-reading the destination | copy back D->E |
| 1 | relocate | (mtp/engram) | model-00046-of-00048.safetensors | 2706402896 | 2.52 | E:/asema_models/DeepSeek-V4.1-Flash/shards/model-00046-of-00048.safetensors | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00046-of-00048.safetensors | 2026-10-02 17:38 | SHA-256 streamed while copying, re-verified by re-reading the destination | copy back D->E |
| 1 | relocate | (mtp/engram) | model-00048-of-00048.safetensors | 101537926640 | 94.56 | E:/asema_models/DeepSeek-V4.1-Flash/shards/model-00048-of-00048.safetensors | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00048-of-00048.safetensors | 2026-10-03 08:39 | SHA-256 streamed while copying, re-verified by re-reading the destination | copy back D->E |
| 2 | copy | 2 | model-00005-of-00048.safetensors | 7405953784 | 6.90 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00005-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00005-of-00048.safetensors | 2026-10-01 20:09 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 8 | model-00011-of-00048.safetensors | 7405953784 | 6.90 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00011-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00011-of-00048.safetensors | 2026-10-02 12:02 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 11 | model-00014-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00014-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00014-of-00048.safetensors | 2026-10-02 14:17 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 12 | model-00015-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00015-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00015-of-00048.safetensors | 2026-10-02 14:56 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 13 | model-00016-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00016-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00016-of-00048.safetensors | 2026-10-02 15:33 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 15 | model-00018-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00018-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00018-of-00048.safetensors | 2026-10-02 16:48 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 19 | model-00022-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00022-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00022-of-00048.safetensors | 2026-10-03 07:34 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 20 | model-00023-of-00048.safetensors | 7400713088 | 6.89 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00023-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00023-of-00048.safetensors | 2026-10-03 09:07 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 21 | model-00024-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00024-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00024-of-00048.safetensors | 2026-10-03 10:21 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 22 | model-00025-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00025-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00025-of-00048.safetensors | 2026-10-03 11:38 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 23 | model-00026-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00026-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00026-of-00048.safetensors | 2026-10-03 13:00 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 24 | model-00027-of-00048.safetensors | 7395337384 | 6.89 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00027-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00027-of-00048.safetensors | 2026-10-03 14:28 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 27 | model-00030-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00030-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00030-of-00048.safetensors | 2026-10-03 17:44 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 2 | copy | 29 | model-00032-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00032-of-00048.safetensors | E:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00032-of-00048.safetensors | 2026-10-03 14:50 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 3 | copy | 17 | model-00020-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00020-of-00048.safetensors | C:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00020-of-00048.safetensors | 2026-10-02 18:20 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 3 | copy | 26 | model-00029-of-00048.safetensors | 7389761368 | 6.88 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00029-of-00048.safetensors | C:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00029-of-00048.safetensors | 2026-10-03 17:39 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |
| 3 | copy | 28 | model-00031-of-00048.safetensors | 7395337384 | 6.89 | D:/asema_models/DeepSeek-V4.1-Flash/shards/model-00031-of-00048.safetensors | C:/asema_models/DeepSeek-V4.1-Flash/shards_p1/model-00031-of-00048.safetensors | 2026-10-03 16:39 | SHA-256 streamed while copying, re-verified by re-reading the destination | original remains at source; unset manifest |

## Resulting expert placement (layer -> drive)

- E: (NVMe) existing 18 layers + new [2, 8, 11, 12, 13, 15, 19, 20, 21, 22, 23, 24, 27, 29] = 32 layers
- C: (SATA) [17, 26, 28]
- D: (SATA) [10, 14, 16, 18, 25]

Chosen by lowest cache-miss rate among D:'s 22 layers (experiments/nextgen/storage/pick_layout.py); simulated 1.78 s/token vs 3.76 s today (no prefetch). Deviation from the research layout (4 layers on C:): 3 on C: because C: is the system drive and a 20 GiB free floor is kept.
