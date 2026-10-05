#!/usr/bin/env python3
"""
Synthetic MoE Model Generator & ASEMA-SSF Container Packer
Generates a parametric sparse-MoE checkpoint with:
- N layers
- M experts per layer
- top-k routing configuration
- deterministic & locality-heavy routing profiles
- ASEMA-SSF binary packaging with 64-byte alignment and CRC32 checksums
"""

import os
import sys
import json
import zlib
import hashlib
import struct
import numpy as np

def generate_synthetic_moe(
    output_dir: str,
    num_layers: int = 4,
    experts_per_layer: int = 16,
    hidden_dim: int = 128,
    ffn_dim: int = 512,
    active_top_k: int = 2,
    dtype: str = "float32"
):
    os.makedirs(output_dir, exist_ok=True)
    layers_dir = os.path.join(output_dir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    
    np_dtype = np.float32 if dtype == "float32" else np.float16
    bytes_per_elem = 4 if dtype == "float32" else 2

    # Weight matrices per expert:
    # w_gate: [hidden_dim, ffn_dim]
    # w_up:   [hidden_dim, ffn_dim]
    # w_down: [ffn_dim, hidden_dim]
    expert_elements = (hidden_dim * ffn_dim * 2) + (ffn_dim * hidden_dim)
    expert_bytes = expert_elements * bytes_per_elem

    manifest = {
        "format": "ASEMA-SSF",
        "version": 1,
        "model_name": f"Synthetic-MoE-{num_layers}L-{experts_per_layer}E",
        "base_architecture": "synthetic-sparse-moe",
        "num_layers": num_layers,
        "experts_per_layer": experts_per_layer,
        "active_experts_per_token": active_top_k,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "num_heads": 4,
        "total_parameters": int(num_layers * experts_per_layer * expert_elements),
        "total_storage_bytes": int(num_layers * experts_per_layer * expert_bytes),
        "layers": []
    }

    # Binary container file for unified SSD storage
    bin_path = os.path.join(output_dir, "model.asema")
    with open(bin_path, "wb") as f_out:
        current_offset = 0

        for layer_id in range(num_layers):
            layer_info = {
                "layer_id": layer_id,
                "num_experts": experts_per_layer,
                "shared_weights_offset": 0,
                "shared_weights_size": 0,
                "shared_weights_file": "",
                "experts": []
            }

            for exp_id in range(experts_per_layer):
                # Deterministic reproducible synthetic weights
                seed = layer_id * 1000 + exp_id
                rng = np.random.default_rng(seed)
                
                w_gate = rng.standard_normal((hidden_dim, ffn_dim), dtype=np_dtype) * 0.02
                w_up   = rng.standard_normal((hidden_dim, ffn_dim), dtype=np_dtype) * 0.02
                w_down = rng.standard_normal((ffn_dim, hidden_dim), dtype=np_dtype) * 0.02
                
                raw_bytes = w_gate.tobytes() + w_up.tobytes() + w_down.tobytes()
                crc = zlib.crc32(raw_bytes)
                sha = hashlib.sha256(raw_bytes).hexdigest()

                # Align offset to 64 bytes
                pad_len = (64 - (current_offset % 64)) % 64
                if pad_len > 0:
                    f_out.write(b'\x00' * pad_len)
                    current_offset += pad_len

                start_offset = current_offset
                f_out.write(raw_bytes)
                current_offset += len(raw_bytes)

                expert_meta = {
                    "layer": layer_id,
                    "expert": exp_id,
                    "storage_offset": start_offset,
                    "storage_length": len(raw_bytes),
                    "alignment": 64,
                    "dtype": dtype,
                    "quant_type": "none",
                    "checksum_sha256": sha,
                    "checksum_crc32": crc,
                    "tensor_shape": [3, hidden_dim, ffn_dim]
                }
                layer_info["experts"].append(expert_meta)

            manifest["layers"].append(layer_info)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"[ASEMA] Generated Synthetic MoE at: {output_dir}")
    print(f"[ASEMA] Layers: {num_layers}, Experts/Layer: {experts_per_layer}, Active Top-K: {active_top_k}")
    print(f"[ASEMA] Container size: {current_offset / (1024*1024):.2f} MB")
    print(f"[ASEMA] Manifest written to: {manifest_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate a synthetic MoE ASEMA-SSF container.")
    p.add_argument("--output", default=os.path.join("examples", "synthetic_moe", "model_synthetic"),
                   help="Output directory.")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--ffn", type=int, default=512)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--dtype", default="fp32", choices=["fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not os.path.isabs(args.output):
        args.output = os.path.abspath(args.output)
    generate_synthetic_moe(
        output_dir=args.output,
        num_layers=args.layers,
        experts_per_layer=args.experts,
        hidden_dim=args.hidden,
        ffn_dim=args.ffn,
        active_top_k=args.top_k,
        dtype=args.dtype,
    )
