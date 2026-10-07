# Verification

What has been checked, how to re-run it, and what has NOT been checked.

## Checkpoint integrity (reference machine)

| Check | Result |
|---|---|
| Shards required by `model.safetensors.index.json` | 48 |
| Shards present on disk (across both volumes) | 48 of 48, none missing, none extra, none empty |
| Leftover `.part` download files | 0 |
| Tensors in the weight map | 96,085 |
| Layers / hidden size / routed experts | 40 / 5120 / 384 (`asema verify-model`) |
| Total shard size | 510.3 GB |

Not verified: per-tensor checksums against the publisher's hashes (the publisher's hashes were not
used). Shard files are checked for presence, non-zero size and a parseable header/index only.

## Tokenizer and output decoding

`build/test_m8_tokenizer_roundtrip.exe` (needs only `tokenizer.json`, no weights):

- vocabulary size is exactly 129,280 and every encoded id is in range;
- encode -> decode round-trips for `hi`, `hello`, `2 + 2`, `NVMe`, a punctuation sentence, and a
  non-ASCII string (`cafe` with an accent, an em dash, Chinese text, an emoji);
- streaming simulation (emit only complete UTF-8 after each token) reassembles every string exactly
  and never emits an invalid sequence, including an emoji whose 4 bytes are split across 2 tokens.

History: a report of garbled ("mojibake") output was traced to (1) the Windows console using the
OEM code page 437 while the program wrote UTF-8, and (2) the streaming code emitting half of a
multi-byte character when one character spans tokens. It was not model or kernel corruption. Both
are fixed (see `tools/asema_cli.cpp` `enable_utf8_console` and `include/asema/m8/m8_utf8.hpp`).

## Numerical parity of the FP8 dense path

`build/test_m8_fp8_gemv.exe`:

- all 254 non-NaN E4M3 codes decode bit-identically to the lookup table (the two NaN codes are
  verified to be absent from the dense tensors that were scanned);
- random matrices and real checkpoint tensors (layers 0 and 19: attention projections, shared
  expert) match a double-precision reference to about 1.5-2.3e-8 relative error (FP32 rounding noise).

Layer 39 tensors are on the second volume and were not covered by this unit test; the end-to-end
runs below cover them.

## End-to-end generation (greedy, all 40 layers)

| Prompt | Output |
|---|---|
| `yooo hi` | `Hey! What's up?` (token ids 28826, 3, 1999, 734, 890, 33) |
| `hi` | Chinese greeting with an emoji (valid UTF-8) |
| `Explain what an NVMe SSD does in one sentence.` | `An NVMe SSD is a high-speed storage drive that keeps data accessible even when your computer is powered off.` |
| `What is 2 + 2?` | `2 + 2 = 4` |
| `Write a short story about a lighthouse keeper.` | coherent English narrative (40 tokens) |
| `Hello` | Chinese greeting with an emoji and a bulleted offer to help (valid, coherent) |
| `Explain what an NVMe SSD does.` | **Poor:** "NVMe isn't a standard term - you likely mean NVMe as a typo..." (identical with 192 and with 8 VRAM slots, so not a cache bug) |
| `Explain ASEMA.` | starts coherently (treats the unknown name as a possible typo of "ASME"), then **degenerates into garbled words** by about token 100 |
| `Write a small HTML file.` | valid `<!DOCTYPE html>`, `<head>`, `<style>` start, then **repeats `<!DOCTYPE html>`** at roughly token 70-90 |

The last two rows are failures. The cause is not established (see "Known correctness gaps" in the
README: the compressed-attention, indexer and engram paths are not executed, but the HTML
degeneration began inside the 128-token window).

For `hi` and `2 + 2` the output was compared token-for-token with the older FP32-dequantize path
(`ASEMA_DENSE_FP8_DIRECT=0`) and was identical. Other prompts were checked for plausibility only.
Greedy decoding is deterministic for a given build and configuration, but cache sizes and
thread counts change floating-point summation order only in the dense kernels; tiny differences
could in principle flip a near-tie token on other inputs.

Not verified: comparison against a reference implementation's logits for long generations;
long-context behaviour. Sampling is not implemented (only greedy argmax, with an optional repetition penalty).

## Clean shutdown

`asema generate` was started in its own console process group and sent a console break event
(the same handler path as Ctrl+C) in the middle of a cold prompt prefill. The program logged the
shutdown request, stopped, and the process exited within 2.8 s with no leftover `asema.exe`.
Generation checks the shutdown flag between layers and between tokens, so the worst-case delay is
about one layer's time. Not tested: interrupting during the GPU upload of an expert (the check
happens after it), or Ctrl+C typed into an interactive `chat` session by a person.

## Fresh-clone build

The staged tree was exported to an empty directory (230 files, no `build/`, no `asema.exe`, no
`asema.config`, no models) and built from scratch with Clang and Ninja: all targets compile (66
executables). Without any model files `ctest -LE requires-data` passes 26 of 26. The 17
tests labelled `requires-data` need the checkpoint and fail, abort or time out without it. With the
checkpoint configured, the lighter ones were run on the reference machine: 11 of 12 passed, and
`test_m8_run` (real inference) exceeded the 150 s cap I gave it. The heaviest tests (full-model
generation suites) were not re-run after the path/loader changes; the end-to-end `generate` runs
above cover the same pipeline.

## Not verified

- Responsiveness of the Windows desktop (browser, video, two monitors) while generating: the
  governor was observed to react to memory pressure, but desktop responsiveness was not measured.
- Console rendering of UTF-8 in an interactive terminal (output was verified as valid UTF-8 when
  redirected to a file; the console code page switch is a standard Win32 call but was not observed
  on screen).
- GPU utilization, paging and commit pressure.

## How to re-run

```
build/test_m8_tokenizer_roundtrip.exe
build/test_m8_fp8_gemv.exe
build/asema.exe verify-model
build/asema.exe generate "yooo hi" --max-tokens 12 --greedy --diagnostic
ctest --test-dir build -LE requires-data
```
