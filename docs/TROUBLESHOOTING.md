# ASEMA Diagnostic & Troubleshooting Handbook

## Diagnostic Command

Always begin by running the system diagnostic doctor:
```powershell
.\build\asema.exe doctor
```

---

## Common Issues & Resolutions

### 1. Direct3D 11 Initialization Failure
- **Symptom:** `asema doctor` reports `[FAIL] GPU : DirectCompute Device`.
- **Cause:** GPU driver crash, outdated AMD Adrenalin drivers, or remote desktop session without hardware acceleration.
- **Resolution:**
  - Verify AMD drivers are up to date (Adrenalin 23.11.1+).
  - Verify that Direct3D 11 runtime is installed via DirectX End-User Runtimes.
  - Check device manager for AMD Radeon RX 580.
  - If GPU is temporarily unavailable, ASEMA will automatically degrade to the verified CPU AVX2 fallback path.

### 2. Checkpoint Verification Failure (`config.json` or Index missing)
- **Symptom:** `asema verify-model` returns `VERIFICATION FAILED`.
- **Cause:** Missing files in `examples/real_model/DeepSeek-V4.1-Flash/hf` or corrupted `model.safetensors.index.json`.
- **Resolution:**
  - Verify `config.json`, `tokenizer.json`, and `model.safetensors.index.json` exist in the HF directory.
  - Check file permissions.

### 3. Insufficient Disk Space on Volume D: or E:
- **Symptom:** `asema install` reports `INSUFFICIENT SPACE`.
- **Cause:** The drive(s) holding the shards have less free space than the checkpoint needs (about 510 GB in total).
- **Resolution:**
  - Clear temporary files or download caches on the respective drive.
  - Run `asema install` to verify updated drive space.

### 4. Generation Token Output Is Garbled or Empty
- **Symptom:** Terminal output displays `<tok_N>` or unexpected characters.
- **Cause:** BPE special tokens or vocabulary mapping issue.
- **Resolution:**
  - Check `tokenizer.json` for integrity.
  - Ensure greedy argmax sampling is used for deterministic reproduction.
