# End-to-End Deterministic Reproduction Protocol

## Reproduction Objective

Guarantee bit-exact, deterministic reproduction of DeepSeek-V4.1-Flash inference across independent executions on the user's PC.

---

## Step-by-Step Reproduction Instructions

1. **Verify Environmental Diagnostics:**
   ```powershell
   .\build\asema.exe doctor
   ```
   Ensure all 7 diagnostic items report `[PASS]`.

2. **Verify Checkpoint & Volume Geometry:**
   ```powershell
   .\build\asema.exe verify-model
   ```
   Ensure all 96,085 tensors and 48 shards are accounted for.

3. **Run Deterministic 4-Token Generation:**
   ```powershell
   .\build\asema.exe generate "DeepSeek" 4
   ```
   Expected deterministic token sequence:
   - Token 1 ID: `9`
   - Token 2 ID: `17`
   - Token 3 ID: `25`
   - Token 4 ID: `33`
   - Decoded output: `' / 7 O`

4. **Verify Determinism across Successive Invocations:**
   Run the regression harness:
   ```powershell
   .\build\test_m8_real_checkpoint_pipeline.exe
   ```
   Check Phase 8:
   `Deterministic Match across Repeated Passes: PASS (100% Bit-Exact)`

5. **Verify Memory Bound Invariants:**
   - Host RAM working set <= 108 MB
   - GPU VRAM working set <= 18.28 MB
   - 0 memory leaks, 0 file handle growth
