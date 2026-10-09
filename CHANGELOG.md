# Changelog

All notable changes are listed here. The project is pre-release; versions follow 0.x semantics.

## [Unreleased] - 0.1.0

### Added
- Optional storage manifest (`storage_manifest` / `ASEMA_STORAGE_MANIFEST`): an explicit JSON list of shard files
  (`{"shards":[{"name","path","size"}]}`) that replaces directory scanning, so shards can live on any number of
  drives. An invalid manifest is a hard error. Behaviour without it is unchanged.
- `docs/research/nextgen/`: architecture validation report, measurements, simulators and the Phase 1 storage
  rebalance report (measured +15 % tok/s on one machine; the full-size modeled gain was not reached).
- `asema` with no arguments starts an interactive chat; new commands `chat`/`run`, `models`,
  `info`, `version`, `--help`, `--version`. `models` reports `READY`, `MODEL NOT FOUND` or
  `CHECKPOINT INCOMPLETE` and never falls back to another model.
- Startup banner that shows the detected GPU and system RAM (nothing hard-coded) and
  "Made by Ajay".
- Chat options `--max-tokens`, `--expert-cache-mb`, `--model`; unknown options are rejected
  before any model is loaded.
- `asema bench`: 50-token benchmark with cold/warm latency, p50/p95, per-stage breakdown,
  storage bytes per token, GPU busy time and the reason generation ended.
- Generation end reasons (EOS, MAX_TOKENS, USER_STOP, ERROR, RESOURCE_GOVERNOR, TIMEOUT) reported
  by `chat`, `generate` and `bench`.
- Resource governor (SAFE / WARNING / CRITICAL) with hysteresis, expert-cache growth and
  working-set trimming; routine state flips are only printed with `ASEMA_VERBOSE=1`.
- True-LRU and frequency+recency ("blend") expert cache; pipelined expert reads overlapped with
  GPU upload; direct FP8 GEMV for dense projections; positioned reads with an I/O slot limiter.
- `scripts/plan_shard_move.py` to relocate shards to the faster drive with SHA-256 verification.
- Documentation set (README, BUILD, HARDWARE, docs/*), CONTRIBUTING, SECURITY,
  CODE_OF_CONDUCT.

- VRAM is now the primary expert cache (192 slots, frequency+recency eviction, weighted by the
  re-read cost of the drive each expert lives on). Experts already resident in VRAM are not read from
  storage and no longer duplicated in RAM. RAM working set ~11.7 -> ~8 GB. Speed: a paired 3-vs-3
  comparison gave 6.49 s -> 6.16 s/token warm (-5%), which is within run-to-run noise (about 10%),
  so no speedup is claimed. (An earlier unpaired "7.4 -> 5.8 s" figure was withdrawn.)
- Experimental predictive next-layer expert prefetch (`ASEMA_PREFETCH_K`, default 0 = off). Measured
  to add 28-62% more storage reads with only 27-32% of them used and no demonstrated speedup on this
  machine; kept disabled. See docs/OPTIMIZATION_LEDGER.md.

- New terminal experience: logo, a short startup sequence in which every line reflects a real check
  (GPU, memory, storage bus type per shard drive, model index, layer audit, GPU backend), distinct
  `You ❯` / `DeepSeek ❯` styling, fenced-code colouring, and a status box after each reply with
  measured RAM/VRAM, tok/s, cache hit rate and storage MB/s. Colours and animation switch off when
  output is redirected, `NO_COLOR` is set, or `ASEMA_NO_ANIMATION` is set.
- `asema doctor` rewritten: CPU brand, RAM, GPU/VRAM, config/tokenizer/index, shard header
  sanity (not a hash), per-drive bus type and free space, read/write permissions, shader compiler,
  runtime DLLs; ends with READY or PROBLEM and a fix hint per failure.
- `asema benchmark` is now the measured benchmark (same as `bench`) and prints hardware context.
- Ctrl+C stops the current reply and keeps the session; Ctrl+C at the prompt ends it.

### Fixed
- Garbled chat output on Windows (console code page 437 and multi-byte characters split across
  tokens): console switched to UTF-8 and only complete UTF-8 sequences are streamed.
- Chat replies truncated at a hard-coded 64 tokens (now 512 by default).

### Known limitations
- Greedy decoding only; temperature / top-k sampling is not implemented.
- Windows + Direct3D 11 only. Decode speed is storage-bound; see docs/PERFORMANCE.md for
  measured numbers.
- Licensed under MIT (see LICENSE). Model weights are not covered by it.
