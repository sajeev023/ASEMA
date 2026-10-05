# ASEMA: Adaptive Sparse-Expert Memory Architecture

**Run massive Mixture-of-Experts models locally using storage-aware sparse expert execution.**

Current support: **DeepSeek-V4.1-Flash** (763B-parameter MoE, full checkpoint, no extra
quantization, pruning or layer reduction), on Windows with a Direct3D 11 GPU. The model stays on
disk; only the working set is held in RAM and VRAM. Other models are not supported.

*Made by Ajay.*

> **Status: research code.** Output correctness has been verified on sample prompts (see
> [docs/VERIFICATION.md](docs/VERIFICATION.md)). Speed is **limited by storage bandwidth** and is
> far from interactive on the reference machine (see [Measured performance](#measured-performance)).
> Nothing here is a production service.

The model weights are **not** part of this repository and are **not** redistributed. You must
obtain the DeepSeek-V4.1-Flash checkpoint from its official source under its own license.

---

## How it works (short version)

- The checkpoint (48 `.safetensors` shards, 96,085 tensors, 40 layers, 384 routed experts per layer,
  top-6 routing) stays on disk, optionally spread over several drives.
- **Dense weights** (attention/MLA, router, shared expert) are memory-mapped and multiplied directly
  from their FP8 bytes. No FP32 copy is made, and Windows can reclaim the pages under pressure.
- **Routed experts** (FP4 weights + scales) are read on demand for the 6 experts the *real router*
  selects, cached in a bounded LRU host cache, and executed on the GPU with Direct3D 11 compute shaders.
- A **resource governor** watches Windows available memory and grows or shrinks the expert cache and
  I/O concurrency so the desktop stays responsive.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Requirements

| | |
|---|---|
| OS | Windows 10/11 x64 |
| CPU | x86-64 with AVX2 + FMA |
| RAM | 32 GB tested. The process used 12-14 GB working set; 16 GB machines are untested |
| GPU | Direct3D 11 feature level 11.0 compute (tested: AMD Radeon RX 580 8 GB). A CPU fallback exists but is much slower |
| Storage | ~480 GB for the checkpoint. **NVMe strongly recommended**: a SATA SSD (~450 MB/s) is the main bottleneck |
| Toolchain | LLVM/clang++ (tested: LLVM-MinGW UCRT), CMake >= 3.20, Ninja |

## Build

```bat
scripts\build.bat
```

or manually:

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build -LE requires-data     :: tests that need no model files
```

Seventeen tests are labelled `requires-data` (they need the checkpoint and/or the locally generated
synthetic model) and fail or time out without it. Once `asema.config` points at your checkpoint,
run them with `ctest --test-dir build -L requires-data` (several run real inference and take minutes).

## Configure model locations

Copy `asema.config.example` to `asema.config` and set the paths (the file is git-ignored), or use
environment variables, which take precedence:

| Setting | Environment variable | Meaning |
|---|---|---|
| `shards_primary` | `ASEMA_SHARDS_PRIMARY` | directory with `model-*.safetensors` |
| `shards_secondary` | `ASEMA_SHARDS_SECONDARY` | optional second directory (another drive) |
| `model_root` | `ASEMA_MODEL_ROOT` | directory with `model.safetensors.index.json` |
| `hf_dir` | `ASEMA_HF_DIR` | Hugging Face metadata directory (`tokenizer.json`) |

See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

## Run

```bat
build\asema.exe                            :: same as "asema chat"
build\asema.exe chat [--max-tokens N]      :: interactive chat ("exit"/"quit" to leave; Ctrl+C stops a reply)
build\asema.exe models                     :: READY / MODEL NOT FOUND / CHECKPOINT INCOMPLETE
build\asema.exe info                       :: model, GPU, RAM, configured paths
build\asema.exe doctor                     :: CPU/RAM/GPU/model/shards/storage/permissions -> READY or PROBLEM (+ fixes)
build\asema.exe verify-model               :: check shards and index
build\asema.exe generate "hi" --max-tokens 32 --greedy --telemetry
build\asema.exe profile                    :: per-layer timing of a warm decode step
build\asema.exe benchmark --tokens 50      :: reproducible decode benchmark (alias: bench)
build\asema.exe --help | --version
```

Only greedy decoding is implemented: there is no temperature / top-k sampling, and no RAM/VRAM
limit flags (limits are the governor settings below). `asema chat` first checks the checkpoint and
refuses to start if it is missing or incomplete; it never falls back to another model.

The CLI switches the console to UTF-8 while it runs and restores your previous code page on exit.

### Tuning and safety knobs (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `ASEMA_EXPERT_CACHE_MB` | 6144 | ceiling for the expert host cache (a ceiling, not a target) |
| `ASEMA_EXPERT_CACHE_START_MB` | 2048 | initial cache size before the governor grows it |
| `ASEMA_GROW_AVAIL_MB` | 10000 | cache only grows while Windows available RAM stays above this |
| `ASEMA_WARNING_AVAIL_MB` | 7500 | below this the governor shrinks the cache and lowers I/O concurrency |
| `ASEMA_CRITICAL_AVAIL_MB` | 4500 | below this it halves the cache, minimizes I/O and pauses briefly |
| `ASEMA_THREADS` | min(6, cores-2) | worker threads for dense math and expert reads |
| `ASEMA_DENSE_FP8_DIRECT` | 1 | `0` selects the older, slower FP32-dequantize path |
| `ASEMA_TRACE_EXPERTS` | unset | write the expert access trace to a file (cache-policy analysis) |
| `ASEMA_CACHE_POLICY` | blend | `blend` (frequency + recency) or `lru`; blend gets hits at small cache sizes where LRU gets none |
| `ASEMA_CACHE_ALPHA` | 0.25 | recency weight of the blend policy |
| `ASEMA_CHAT_MAX_TOKENS` | 512 | maximum tokens per `chat` reply (every reply prints why it ended: EOS, MAX_TOKENS, USER_STOP, ERROR) |

## Measured performance

Reference machine: Ryzen 7 5700X, 32 GB RAM, Radeon RX 580 8 GB, one NVMe (~2.8 GB/s) and one SATA
SSD (~0.44 GB/s) holding 36% / 64% of the checkpoint respectively. `asema bench --tokens 40`,
greedy decoding:

| Metric | Result |
|---|---|
| Cold first token (includes prompt prefill) | ~180 s |
| Warm decode, mean | **5.8 s/token (0.17 tokens/s)**, 50-token run; 7.4 s before the VRAM-tier cache, ~9.3 s before any tuning |
| Warm decode, p50 / p95 | 5.7 s / 7.3 s |
| Data read from storage | ~3.5 GB per token (~4.0 s of each token is spent waiting on it) |
| Expert cache hit rate | 22% (192 VRAM slots; slow-drive experts are kept longer) |
| Process working set | ~8 GB (includes reclaimable memory-mapped weights) |
| VRAM allocated | 3.4 GB (override slot count with `ASEMA_VRAM_SLOTS`) |

This is **far from 1 token/s** and this project does not claim otherwise. Numbers are from one
machine and one benchmark run; GPU utilization is estimated from timestamp queries, not measured
externally.

Decode is **I/O-bound** (the CPU is ~0.4 cores busy): most of each token is spent reading experts
from the slower drive. See [docs/PERFORMANCE.md](docs/PERFORMANCE.md) for the analysis, what was
tried, and what would help (more NVMe capacity). No claim is made that interactive speeds are
reachable on this storage layout.

## Documentation

[ARCHITECTURE](docs/ARCHITECTURE.md) · [BUILD](BUILD.md) · [MODEL_SETUP](docs/MODEL_SETUP.md) ·
[HARDWARE](HARDWARE.md) · [PERFORMANCE](docs/PERFORMANCE.md) · [VERIFICATION](docs/VERIFICATION.md) ·
[DEBUGGING](DEBUGGING.md) · [CONTRIBUTING](CONTRIBUTING.md) · [SECURITY](SECURITY.md) ·
[OPEN_SOURCE_READINESS](docs/OPEN_SOURCE_READINESS.md)

## License

MIT, copyright 2026 Ajay. See [LICENSE](LICENSE). The DeepSeek-V4.1-Flash model weights are governed
by their own license and are not included in or licensed by this repository. The vendored
`include/nlohmann/json.hpp` is nlohmann/json (MIT) and keeps its own license header.
