# Changelog

All notable changes are listed here. The project is pre-release; versions follow 0.x semantics.

## [Unreleased] - 0.1.0

### Added
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
  storage and no longer duplicated in RAM. Warm decode 7.4 -> 5.8 s/token, working set ~11.7 -> ~8 GB.

### Fixed
- Garbled chat output on Windows (console code page 437 and multi-byte characters split across
  tokens): console switched to UTF-8 and only complete UTF-8 sequences are streamed.
- Chat replies truncated at a hard-coded 64 tokens (now 512 by default).

### Known limitations
- Greedy decoding only; temperature / top-k sampling is not implemented.
- Windows + Direct3D 11 only. Decode speed is storage-bound; see docs/PERFORMANCE.md for
  measured numbers.
- The license file is a placeholder until the maintainer chooses a license.
