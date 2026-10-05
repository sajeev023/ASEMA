# ASEMA: Forensic Debugging & Tracing Guide

## Diagnostic Tracing Flags

ASEMA provides granular observability into each transformation along the autoregressive inference path:

### 1. Tokenizer Diagnostics (`--trace-tokenizer`)
Traces byte-level conversions, BPE merge ranks, and round-trip decoding:
```powershell
build\asema.exe generate "Explain neural network" 4 --trace-tokenizer
```
Output includes:
- Prompt UTF-8 bytes and token length.
- Token IDs produced by the tokenizer.
- Token text pieces with special-token delimiters.
- Round-trip UTF-8 decoded text.

### 2. Physical Embedding Diagnostics (`--trace-embedding <id>`)
Verifies direct resolution of embedding tensors from physical safetensors shards:
```powershell
build\asema.exe generate "Hello" 1 --trace-embedding 65106
```
Output includes:
- Token ID and shard location (`model-00002-of-00048.safetensors`).
- Exact byte offset, tensor shape `[129280, 5120]`, and dtype (`BF16`).
- First 16 values converted to FP32.
- Vector L2 norm / checksum.

### 3. MoE Router Diagnostics (`--trace-router`)
Traces expert gating scores and top-6 selection:
```powershell
build\asema.exe generate "Hello" 1 --trace-router
```
Output includes:
- Layer ID and token position.
- Top-6 selected expert indices.
- Raw gate projections and bias-adjusted values.
- Normalized softmax routing weights summing to 1.0.

### 4. LM Head & Logit Diagnostics (`--trace-logits`)
Verifies vocabulary projection directly from physical shard 43:
```powershell
build\asema.exe generate "Hello" 1 --trace-logits
```
Output includes:
- Final normalized hidden state L2 norm.
- Physical LM head tensor mapping (`model-00043-of-00048.safetensors` @ offset 184).
- Top-20 token IDs, raw logits, and decoded string representations.
- Selected token ID and greedy argmax confirmation.
