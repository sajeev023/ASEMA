# GPU execution (Direct3D 11 compute)

- API: Direct3D 11 compute shaders (shader model 5.0), compiled at start-up with D3DCompile.
  Tested on an AMD Radeon RX 580 8 GB.
- Kernels for each routed expert: GEMV for w1 and w3 (FP4 weights, per-32 scales), SwiGLU, GEMV
  for w2, and weighted accumulation into a single output vector in VRAM; one readback per layer.
- VRAM use: about 1.7 GB, almost all of it 96 expert slots (18.8 MB each) allocated at start-up.
  The slot cache hit rate measured 0% in a 40-token generation, so each expert is uploaded again
  every time it is used. Upload and readback wait account for roughly 30-40 ms per layer.
- GPU MLA attention (`m8_gpu_mla_kernel`) exists but is disabled by default; attention runs on the CPU
  directly from FP8 weights.
- Reported VRAM was wrong (18 MB) before telemetry was fixed; it now matches the Windows
  "dedicated GPU memory" counter (1,721 MB reported vs 1,730 MB observed).
- GPU utilization is not measured.
