# Open-source readiness

**Published** at https://github.com/sajeev023/ASEMA from a fresh single-commit history. No model
files were uploaded. This document records what was checked and changed before publication, and the
decisions that were made.

## Decisions

1. **License: MIT**, copyright 2026 Ajay (chosen for simplicity and wide reuse; Apache-2.0 would
   add a patent grant). Model weights are not covered; they keep their publisher's license.
2. **Git history.** The original local repository (two commits) contained build output,
   model-weight slices, a tokenizer, reports with local paths and large synthetic files. It was not
   published; the public repository started from a clean tree in a new `git init`.
3. **Security contact:** GitHub private vulnerability reporting (no e-mail address is published).
4. **Model files.** `tokenizer.json`, `model.safetensors.index.json` and the other files under
   `examples/real_model/` come from the model publisher. They are now git-ignored and must not be
   redistributed unless that publisher's license allows it. Users obtain them themselves.
5. **Author identity.** Commits use the author `Byte <byte@asema.internal>`; no real name or e-mail
   appears in history.

## Audit performed

| Check | Method | Result |
|---|---|---|
| Secrets (API keys, tokens, private keys, passwords) | regex scan of all tracked non-JSON files | none found |
| Private paths and user name in `src/`, `include/`, `tools/`, `tests/` | repository-wide search | 88 occurrences in 43 files -> all replaced by `asema::m8::paths::*` (`m8_paths.hpp`); 0 remain |
| Private paths in `scripts/`, `docs/` | search | replaced; `build.bat` no longer contains toolchain paths |
| Generated reports with local paths (`reports/`, 690 files) | inventory | untracked and git-ignored |
| Build output tracked in git (`build/`, `build_clean/`, `dist/`; objects, executables, ninja state) | inventory | untracked and git-ignored |
| Model data tracked in git (weight slices `*.bin`, `*.npy`, `*.npz`, ~450 MB of synthetic `*.asema`, tokenizer/index) | inventory | untracked and git-ignored; files remain on disk |
| `.part` downloads, crash dumps, IDE/agent state | search | none present in the tree; patterns added to `.gitignore` |
| Missing sources: several headers needed to build were not tracked | `git status` | to be added (see below) |
| False or stale documentation (memory bounds, "dual NVMe", throughput claims, fake clone URL) | manual review | README, HARDWARE, PERFORMANCE, STORAGE, GPU, BUILD, CONTRIBUTING, SECURITY rewritten from measurements |

## Local configuration

`asema.config` (git-ignored) holds this machine's drive paths. `asema.config.example` is the
committed template. The program resolves paths from environment variables first, then
`asema.config`, then neutral relative defaults (`models/DeepSeek-V4.1-Flash/...`).

## Third-party code

| Component | Where | License |
|---|---|---|
| nlohmann/json 3.11.3 | `include/nlohmann/json.hpp` (vendored, license header intact) | MIT |

No other third-party sources are vendored. Windows SDK / Direct3D headers are used from the system.
Model weights, tokenizer and index files are **not** part of this project.

## Pre-publication checklist (run from a clean checkout of the intended tree)

```
git status --short                       # nothing unexpected
git ls-files | findstr /i /r "\.bin$ \.npy$ \.npz$ \.safetensors$ \.exe$ \.obj$ \.asema$"   # must be empty
git grep -nIE "[A-Za-z]:[/\\](Users|asema_models)|ADMIN"                                     # must be empty
git grep -nIE "(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|BEGIN .*PRIVATE KEY)"   # must be empty
scripts/build.bat && ctest --test-dir build -LE requires-data
```

A fresh export of the staged tree (230 files, no `build/`, no `asema.config`, no models) was configured and
built with all targets (66 executables), and `ctest -LE requires-data` passed 26 of 26. The other 17 tests
are labelled `requires-data`; they need the checkpoint and fail or time out without it.
