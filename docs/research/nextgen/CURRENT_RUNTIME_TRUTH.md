# CURRENT RUNTIME TRUTH (verified against source, 2026-10)

Everything below was read from the code that actually executes in `asema chat | generate | bench`, not from the
older documentation. Costs are [M] measured on this machine (warm mean of one 50-token `asema bench` run, K=0,
VRAM tier; `reports/` logs) unless marked [E] estimated or [T] theoretical.

Call path of one decode step:
`asema.exe main` -> `cmd_chat` / `cmd_generate_advanced` / `cmd_bench` -> `AsemaEngine::stream_text` ->
`M8ModelRunner::generate` -> for each token `M8ModelRunner::step` -> for l in 0..39 { `load_layer(l)`, `forward_hc(...)` } ->
final norm -> `compute_lm_head_logits` -> greedy token.

| # | Item | File : function | Thread | Memory / device | Sync | Measured cost |
|---|---|---|---|---|---|---|
| 1 | Executable entry | `tools/asema_cli.cpp : main` (UTF-8 console, `ui::init`, dispatch) | main | - | - | - |
| 2 | Generation entry | `src/m8/m8_generation_api.cpp : AsemaEngine::stream_text` -> `src/m8/m8_model_runner.cpp : generate` (544), `step` (446) | main | - | per-token callback | 6.4 s/token total |
| 3 | Tokenization | `M8ModelRunner::generate` -> `Tokenizer::encode` (`m8_model_adapter.cpp`, BPE, 129,280 vocab from `tokenizer.json`); official chat ids hard-coded: BOS 0, User 128803, Assistant 128804, `</think>` 128822 (`generate` 566-573) | main | CPU | - | ~0 |
| 4 | Embedding | `lookup_token_embedding` (310): BF16 row from memory-mapped shard 2 at `352 + id*10240`; expanded to 4 hyper-connection copies | main | RAM (mmap) | - | 0.5 ms |
| 5 | Layer loop | `step` (474-511): ONE `active_layer_`; `load_layer(l)` rebinds dense tensors (mmap pointers) every layer of every token | main | RAM | governor call per layer | binding 77 ms/token |
| 6 | Attention | `m8_mla_attention.cpp : M8MLAAttention::forward`: Q/KV/O projections as direct FP8 GEMV on CPU worker pool, **128-position sliding-window KV ring**, attn sinks. `set_gpu_mla_acceleration(false)` so MLA is CPU | main + `M8WorkerPool` (min(6, cores-2)) | RAM; dense weights mmap'd | pool barriers | projections 222 ms, attend 8.5 ms, KV store 0.03 ms |
| 7 | Router | `m8_router.cpp : M8Router::route`: FP32 384x5120 AVX2 dot, sqrt(softplus), +bias for selection, top-6, weights from unbiased scores normalised x1.5 | main | RAM (FP32 copy of gate) | - | 13 ms/token |
| 8 | Expert loading | `m8_byte_loader.cpp : begin_layer_experts` -> `std::async` per missing expert -> `internal_read_expert`: 2 positioned `ReadFile` (scales 1.1 MB, weights 17.7 MB), **buffered** (no `NO_BUFFERING`), into `std::vector` | up to 6 async threads (io slot limiter) | RAM (heap) | `shared_future::get` in the GPU provider | wait 4,528 ms/token |
| 9 | Expert cache | GPU path: **no host cache** (`publish_to_cache=false`). VRAM: 192 slots, score = freq/(1+0.25 age/240) x drive cost (`m8_gpu_expert_kernel.cpp : pick_victim_slot`) | main | VRAM 3.4 GB | pinning per layer | hit rate 22.3% |
| 10 | GPU path | `forward_top6_layer_streamed`: per expert `UpdateSubresource` x2 (scales, weights), 5 dispatches (w1, w3, swiglu, w2, accumulate), 30 dispatches/layer, one `CopyResource`+`Map(READ)` readback (20 KB) per layer | main (D3D11 immediate context) | VRAM; staging 20 KB | `Map(READ)` blocks every layer | stage wall 1,210 ms: upload calls 240, **wait for GPU 962**, GPU busy upload 925 / compute 366 |
| 11 | CPU work | attention, router, shared expert (FP8 GEMV, 78.7 ms), HC mixing/norms (224 ms), LM head, token selection | main + pool | RAM | - | see above |
| 12 | Storage | dense: `MapViewOfFile` of whole shards (`m8_safetensors_index.cpp`); experts: positioned reads; two roots (D:, E:) via `m8_paths.hpp` | async threads | SSD | - | 3,465 MB/token at 765 MB/s |
| 13 | Sync points per layer | `pending.get(e)` x(<=6), `Map(READ)`, `cache_mutex_`/`io_mutex_`+cv per read, one `std::async` thread **created per missing expert** (~4.6/layer), governor `GlobalMemoryStatusEx` per layer | - | - | ~15-20 per layer | not separated |
| 14 | Memory copies | SSD -> Windows file cache -> `std::vector` (buffered) -> D3D11 driver staging -> VRAM (`UpdateSubresource`) | - | - | - | upload path 4.3 GB/s [M, E4b] |
| 15 | Cache hierarchy | VRAM 192 slots only; OS page cache implicit; dense mmap | - | - | - | - |
| 16 | Prefetch | experimental co-occurrence prefetch exists, **off by default** (`ASEMA_PREFETCH_K=0`); rejected (no gain, +28-62% bytes) | - | - | - | - |
| 17 | Scheduler | none: sequential layer loop; reads of ONE layer are issued together after its router | - | - | - | - |
| 18 | KV system | per-layer ring, 128 x 512 floats (256 KB/layer), CPU; persistent across tokens, reset per `generate()` | main | RAM | - | negligible |
| 19 | Engram | **not executed**: config parsed; tables (203 GB, shards 47/48) never read | - | - | - | - |
| 20 | CSA2 / indexer | **not executed** (`compress_ratios`, `index_*`, `kv_source_layer_ids` parsed only) | - | - | - | - |
| 21 | DSpark / MTP | **not executed** (shards 44-46 never read) | - | - | - | - |
| 22 | Final norm | `step` 532: `rms_norm` with `final_norm_` | main | RAM | - | 0.01 ms |
| 23 | LM head | `compute_lm_head_logits` (322): BF16 129,280x5120 from mapped shard 43, 8 `std::async` threads | 8 threads | RAM mmap | futures | 36.6 ms |
| 24 | Sampling | greedy argmax + repetition penalty (`generate`); temperature/top-k not implemented | main | - | - | 0.07 ms |
| 25 | Known correctness gaps | no compressed/sparse global attention, no indexer, no Engram, no MTP/DSpark; 128-token window only; replies degrade after ~60-100 tokens; never compared with the official reference implementation | | | | |

Other facts that matter for the redesign:
- **Prefill is token by token**: each prompt token runs a full decode step (all experts), so TTFT = prompt length x ~6 s.
- **Per-token byte traffic** [M]: experts 3.46 GB read from SSD (+ the same data uploaded over PCIe); dense ~6.9 GB read
  from RAM by the CPU (at ~20 GB/s, which is why attention + shared expert + binding cost ~0.4 s); LM head 1.3 GB.
- **Storage is touched by one drive at a time**: layers are sequential and each layer's experts are in one shard.
- GPU busy time is 20% of a token [M, timestamp queries]; CPU averages 0.69 cores busy.
