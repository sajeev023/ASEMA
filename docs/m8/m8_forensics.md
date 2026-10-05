# ASEMA v0.2 — M8 Real-Model Integration: Architecture Forensics

**Date**: 2026-09-26
**Author**: Agent #5 (M8)
**Scope**: DeepSeek-V4.1-Flash architectural forensics, hardware feasibility, and scope of the achievable M8 subset on this host.

---

## 1. Source of truth

| Field | Value |
|-------|-------|
| Hugging Face repo | `deepseek-ai/DeepSeek-V4.1-Flash` |
| Commit SHA | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Released | 2026-09-10 |
| Architecture class | `DeepseekV41ForCausalLM` |
| License | MIT |
| Total checkpoint size on HF | **510 GB** |
| Number of safetensors shards | **48** |

## 2. Model dimensions (from `config.json`)

```
text_config:
    vocab_size:               129 280
    hidden_size:                5 120
    moe_intermediate_size:     2 304
    num_hidden_layers:             40
    num_attention_heads:           64
    num_key_value_heads:            1  (MLA)
    head_dim:                     512
    qk_rope_head_dim:              64
    q_lora_rank:                 1 280
    o_lora_rank:                 1 024
    o_groups:                        8
    max_position_embeddings:  1 048 576  (1 M tokens)
    rope_scaling:               YaRN (factor 16, original 65 536)
    sliding_window:                128
    n_routed_experts:              384
    n_shared_experts:                1
    num_experts_per_tok:             6
    scoring_func:              sqrtsoftplus
    topk_method:               noaux_tc
    norm_topk_prob:               true
    routed_scaling_factor:         1.5

sparse_attention / CSA2:
    compress_ratios:       per-layer (40 entries, values 0/1/2)
    kv_source_layer_ids:   [2, 8, 14, 20]
    index_source_layer_ids:[2, 8, 14, 20, 24, 28, 32, 36]
    index_n_heads:                32
    index_head_dim:              128
    index_topk:                  512
    candidate_source_layer_id:    20
    candidate_topk_blocks:      2 048
    candidate_block_size:           8

hyper_connections (mHC):
    hc_mult:                         4
    hc_sinkhorn_iters:              20
    hc_eps:                    1e-06

engram (n-gram memory):
    engram_layer_ids:        [1, 14]
    engram_num_embeddings:    [384 006 168, 384 016 682]
    engram_max_ngram_size:           4
    engram_vocab_size:        16 000 000
    engram_n_heads:                    8
    engram_head_dim:                256
    engram_pad_token_id:              2
    engram_compressed_vocab_size:99092

dspark (speculative decode):
    num_nextn_predict_layers:          3
    dspark_block_size:                5
    dspark_noise_token_id:        128799
    dspark_target_layer_ids:  [37, 38, 39]
    dspark_markov_rank:              256
    dspark_n_routed_experts:          128
    dspark_num_experts_per_tok:         3

vision_config:
    num_hidden_layers:               32
    hidden_size:                   1024
    num_attention_heads:             16
    intermediate_size:             2816
    patch_size:                       14
    max_image_tokens:               1024
    min_pixels:                   295936
```

## 3. Activation footprint

| Phase   | Active params per token |
|---------|-------------------------:|
| Prefill | 8 B                      |
| Decode  | 16 B                     |

(Matches README and tech report summary.)

## 4. Tensor naming convention (per layer `layers.{N}`)

For each of the 40 text layers:

```
layers.{N}.hc_attn_base            # hyper-connection residual
layers.{N}.hc_ffn_base             # hyper-connection residual
layers.{N}.hc_attn_fn              # hyper-connection transform
layers.{N}.hc_attn_scale           # hyper-connection scale
layers.{N}.hc_ffn_fn               # hyper-connection transform
layers.{N}.hc_ffn_scale            # hyper-connection scale
layers.{N}.attn.attn_sink          # attention sink
layers.{N}.attn.wq_a.{weight,scale}     # MLA Q compressed
layers.{N}.attn.wq_b.{weight,scale}     # MLA Q uncompressed
layers.{N}.attn.q_norm.weight
layers.{N}.attn.wkv.{weight,scale}      # MLA KV
layers.{N}.attn.kv_norm.weight
layers.{N}.attn.wo_a.{weight,scale}     # MLA O compressed
layers.{N}.attn.wo_b.{weight,scale}     # MLA O uncompressed
layers.{N}.attn_norm.weight
layers.{N}.ffn_norm.weight

# 1 shared expert
layers.{N}.ffn.shared_experts.w1.{weight,scale}  # SwiGLU gate
layers.{N}.ffn.shared_experts.w3.{weight,scale}  # SwiGLU up
layers.{N}.ffn.shared_experts.w2.{weight,scale}  # SwiGLU down

# 384 routed experts
for e in 0..383:
    layers.{N}.ffn.experts.{e}.w1.{weight,scale}
    layers.{N}.ffn.experts.{e}.w3.{weight,scale}
    layers.{N}.ffn.experts.{e}.w2.{weight,scale}
```

**Total tensors per text layer**: 8 HC + 12 attn + 2 norms + 6 shared + 6 × 384 routed = 2 332 tensors.

**Total tensors across 40 layers**: 93 280 expert tensors alone.

**Quantization**: FP8 with block size 32 × 32, scale format `ue8m0`, expert dtype `fp4`. Each routed expert ≈ 35 MB FP8 (≈ 18 MB FP4 after re-quantization).

## 5. Per-expert size (FP8, fp4 expert)

```
gate (w1): 5120 × 2304 = 11 796 480 elements ≈ 11.25 MB FP8 ≈ 5.6 MB FP4
up   (w3): 5120 × 2304 = 11 796 480 elements ≈ 11.25 MB FP8 ≈ 5.6 MB FP4
down (w2): 2304 × 5120 = 11 796 480 elements ≈ 11.25 MB FP8 ≈ 5.6 MB FP4
per-expert total ≈ 33.75 MB FP8 / 16.8 MB FP4
```

For 384 experts × 40 layers: **518 GB FP8 routed experts** (matches HF storage).

## 6. Engram memory tables

```
engram_layer_ids = [1, 14]
engram_num_embeddings = [384_006_168, 384_016_682]   # two tables
engram_head_dim = 256
engram_compressed_vocab_size = 99 092
```

Estimated size: 2 × 384M × 256 × dtype bytes ≈ **196 GB** (matches README).

These are sparse n-gram lookup tables, NOT standard MoE experts. They live in dedicated Engram modules at layers 1 and 14 only.

## 7. Hardware feasibility on this host

| Constraint | Value | Notes |
|-----------|------:|-------|
| Disk free (C:) | 71.7 GB | far below 510 GB checkpoint |
| RAM total | 34 GB | far below ~50 GB minimum for full FP8 inference |
| GPU | none | CPU-only execution path |
| Compiler | Clang 19 / LLVM-MinGW | OK |
| Python | 3.11 | transformers/vLLM not installed |

### 7.1 What this means

- **The full `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint (510 GB) cannot be downloaded to this host.** Disk capacity is 71.7 GB; the model is 510 GB; we would need ~7× more space.
- **CPU-only execution of a 552B-parameter MoE is not feasible.** Even a single forward pass requires loading 384 expert weights × 40 layers ≈ 538 GB of expert data.
- **No GPU is available** — the official vLLM/SGLang/transformers inference paths require GPU.

Therefore the **full-model M8 path is BLOCKED** on this host for these reasons.

### 7.2 What CAN be done without the full checkpoint

1. Download the model definition files (~6 MB total):
   - `config.json`
   - `tokenizer.json`, `tokenizer_config.json`
   - `model.safetensors.index.json` — the full tensor index
   - `README.md`, `LICENSE`, technical report PDF
   - The official `inference/` directory (model.py, generate.py, kernel.py, engram.py, vision.py, etc.)
   - The official `encoding/` and `evaluation/` directories
2. Build the architecture forensics, tensor inventory, and tokenizer verification.
3. Build the M8 adapter scaffold and reference-vs-ASEMA comparison framework.
4. If allowed by disk budget, download ONE shard (≈ 10 GB) to demonstrate end-to-end expert extraction on a real expert tensor. This would still leave 60 GB free for ASEMA experiments.

## 8. Achievable M8 scope on this host

| Phase | Achievable? | Notes |
|-------|:-----------:|-------|
| M8.0 Acquisition + environment lock | PARTIAL | Small files downloaded; full checkpoint blocked by disk |
| M8.1 Architecture forensics | ✅ | Done from config + code |
| M8.2 Tensor inventory | ✅ | Done from index.json |
| M8.3 Reference runner | ⚠ STUB | Cannot execute full forward; can stub tensor shapes |
| M8.4 Tokenizer equivalence | ✅ | Tokenizer alone is ~6 MB, fits; no model weights needed |
| M8.5 Expert extraction | ⚠ | Can extract ONE expert (single shard download) |
| M8.6 Expert weight equivalence | ⚠ | Only for the ONE extracted expert |
| M8.7 Router equivalence | ⚠ | Router is in early shard; one-shard sample possible |
| M8.8 Expert output equivalence | ❌ | Requires runtime model weights, blocked |
| M8.9 Intermediate activation equivalence | ❌ | Requires full forward pass, blocked |
| M8.10 Final logit equivalence | ❌ | Requires full forward pass, blocked |
| M8.11 DeepSeek → ASEMA storage adapter | ⚠ | Skeleton only; full extraction blocked |
| M8.12 Real routing trace | ❌ | Requires full model, blocked |
| M8.13 Real ASEMA cache behavior | ❌ | Requires full model, blocked |
| M8.14 Real prefetch behavior | ❌ | Requires full model, blocked |
| M8.15 CPU constrained inference | ❌ | 552B params cannot run on CPU host |
| M8.16 Performance characterization | ❌ | Blocked by M8.8-15 |

## 9. What is recorded as honest output

- Real architectural facts (from official config + code + index).
- Real download status (24 of 24 small files; 0 of 48 shards).
- Honest BLOCKED status for everything that requires the full checkpoint or GPU.
- No fabricated inference results. No "we ran DeepSeek-V4.1-Flash" claim that would be a lie.

## 10. Recommended path forward

To complete the full M8 path, the host must provide:
- At least **600 GB** of free disk (510 GB checkpoint + working space + ASEMA artifacts).
- At least **64 GB** RAM (for FP8 inference; more for safety).
- A GPU with ≥ 24 GB VRAM (the official README explicitly requires vLLM/SGLang/Docker with `--gpus all`).

OR a smaller real sparse-MoE model (e.g., `deepseek-ai/deepseek-moe-16b-base`) could be substituted under a separate, smaller M8-bis milestone — but the directive is to use DeepSeek-V4.1-Flash, so that substitution is OUT OF SCOPE for this session.
