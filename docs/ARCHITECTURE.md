# Architecture

ASEMA executes the full DeepSeek-V4.1-Flash checkpoint with the weights **resident on disk**. Only
the working set needed for the current token is held in host RAM and VRAM. Everything below
describes what the code in this repository does; limitations are listed at the end.

```
text -> tokenizer (byte-level BPE, 129,280 vocab) -> token ids
     -> embedding lookup (memory-mapped, bf16)
     -> 40 x transformer layer  [hyper-connections -> MLA attention -> real router -> shared expert
                                 + top-6 routed experts]
     -> final RMSNorm -> LM head (memory-mapped, bf16) -> logits -> argmax/sampling -> token id
     -> tokenizer decode -> complete-UTF-8 streaming -> console
```

## 1. Storage layer and tensor indexing

- `M8SafetensorsIndex` parses each shard's safetensors JSON header and records, for all 96,085
  tensors, the shard, absolute file offset, byte length, dtype and shape.
- `M8MultiVolumeManager` registers one or two shard directories (configured via `asema.config` /
  environment, see `m8_paths.hpp`) and resolves any tensor to its physical file and byte range.
- Dense tensors are exposed **zero-copy**: `map_tensor()` memory-maps the shard read-only and
  returns a pointer. The pages live in the OS file cache, so Windows can reclaim them under
  memory pressure; ASEMA owns no private copy.

## 2. Dense path: MLA attention, router, shared expert

- Attention (MLA) projections and the shared expert are FP8 (E4M3) with 32x32 block scales (E8M0).
  `fp8_gemv` (`m8_fp8_gemv.hpp`) multiplies directly from the mapped FP8 bytes with an exact
  bit-manipulation decode and AVX2/FMA, split over a bounded worker pool (default at most 6 threads).
- The MLA block uses a low-rank query path (q_lora 1280), 64 heads of dimension 512 (64 rotary
  dimensions), an attention-sink term, and a grouped output projection (8 groups).
- Each layer mixes four residual-stream copies with learned hyper-connection weights
  (Sinkhorn-normalized) before and after attention and the MoE block.
- The **real router** scores the 384 routed experts and selects the top 6 with their weights;
  nothing about expert selection is modified.

## 3. KV cache

Each layer owns a ring buffer of 128 positions x 512 floats in host memory, updated by position
(`pos % 128`) and kept across decode steps; it is reset only at the start of a new prompt.

## 4. Routed experts: async I/O, RAM cache, GPU execution

- Each expert is 18.8 MB: FP4-packed weights (w1, w2, w3) plus per-32-element scales. They are
  contiguous in the shard, so one expert costs two positioned reads (scales, weights).
- `M8ByteRangeLoader::load_layer_experts_parallel` resolves cache hits, then reads all misses
  concurrently in waves of at most N reads (N set by the governor, default 6). Reads use
  positioned (OVERLAPPED-offset) `ReadFile`, which is safe on a shared handle.
- The host cache is a bounded true-LRU of expert payloads. Its capacity is set by the governor.
- Experts execute on the GPU via Direct3D 11 compute shaders (`m8_gpu_expert_kernel`): a GEMV for
  w1 and w3, SwiGLU, a GEMV for w2, and weighted accumulation, one readback per layer. A pool of
  96 VRAM slots (about 1.7 GB) holds uploaded experts; under the observed access pattern it has a
  0% hit rate, so every expert is uploaded each time it is used (see PERFORMANCE.md).
- A CPU fallback path exists for machines without a usable GPU.

## 5. Resource governor and shutdown

- Once per decode step the runner reads Windows available memory and sets SAFE / WARNING /
  CRITICAL: SAFE grows the cache slowly (only while available memory stays above 10 GB by
  default), WARNING shrinks it and lowers I/O concurrency, CRITICAL halves it, minimizes I/O and
  pauses briefly. All thresholds are environment-configurable.
- Not monitored: paging rate, commit charge, NVMe queue depth and GPU load (these are not
  implemented). Available memory is the only live signal.
- Ctrl+C sets a shutdown flag checked between layers and between tokens; generation stops at the
  next check and the process exits normally, restoring the console code page.

## 6. Autoregressive decoding and output

Prefill runs the prompt through the layers token by token (no logits until the last prompt token),
then decode runs one token per step. Token selection is greedy argmax (the CLI applies a repetition penalty of 1.15 unless --greedy is given). Temperature sampling is not implemented: the --temperature flag is parsed but has no effect.
Decoded bytes are streamed only as complete UTF-8 sequences.

## Limitations (honest list)

- Windows-only (Win32 mapping, Direct3D 11) and specific to DeepSeek-V4.1-Flash: the embedding and
  LM-head byte offsets, tensor names, the chat-template token ids and several dimensions are
  constants tied to this checkpoint.
- Decode is I/O-bound; throughput on the reference machine is about 0.1 token/s (PERFORMANCE.md).
- The attention implementation keeps a 128-position window per layer. Behaviour on contexts longer
  than that has not been validated against a reference implementation.
- No batching; prefill is sequential; no speculative decoding.
- The VRAM expert cache is ineffective for this access pattern.
- GPU utilization, paging and commit pressure are not measured.
