# Contributing

Thank you for your interest. This is research code; please keep changes small and measured.

## Ground rules

1. **No synthetic inference.** Embeddings, attention, routing, experts, norms and the LM head must
   run on the real checkpoint. Do not add fallbacks that substitute fake tensors, another model,
   canned text or an external API.
2. **Do not change model semantics for speed.** No pruning, extra quantization, skipped layers or
   altered routing. Numerical differences (for example summation order) must be shown to be at
   rounding-noise level and not to change generated tokens on the verification prompts.
3. **Measure before you optimize, and report what you measured.** Use `asema bench` and
   `asema profile`; state the machine and run-to-run noise. Do not publish numbers you did not measure.
4. **Keep Windows usable.** Anything that grows memory, threads, queues or I/O must be bounded and
   must respond to the resource governor.
5. **No machine-specific paths** in source, scripts or docs. Use `asema.config` / `ASEMA_*`
   variables (`include/asema/m8/m8_paths.hpp`).
6. **Never commit model weights**, checkpoints, tokenizer/model-release files, local config
   (`asema.config`), logs, or build output. `.gitignore` covers these; check `git status` first.

## Workflow

```
scripts/build.bat
ctest --test-dir build -LE requires-data
```

Add a test for new behaviour. A test that needs the checkpoint or other local data must be added to
the `requires-data` label list at the end of `CMakeLists.txt` (better: make it skip when data is absent).
Match the surrounding code style. Describe the measured effect of performance changes in the pull
request, including any change in generated output.

## Reporting problems

Open an issue with your Windows version, CPU, RAM, GPU and driver, storage types, the command you
ran, and the console output. For security issues see SECURITY.md.
