#!/usr/bin/env python3
"""ASEMA v0.1 — Tier 2 anomaly investigation tool.

Parses the JSONL telemetry stream emitted by ASEMA runtime and computes
access-pattern statistics that explain the difference between Tier 1 and
Tier 2 cache hit behavior.

Outputs (per layer):
    * total unique experts accessed
    * unique_experts / total_experts ratio
    * top-1, top-2, top-3, top-5 expert access frequencies
    * working-set size estimate (number of distinct experts within a sliding
      window of N tokens)

Usage:
    python analyze_access_pattern.py <run_dir> [output_json]
"""
import json
import os
import sys
from collections import Counter, defaultdict


def parse_jsonl(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "anomaly_report.json"

    # Accept either a directory (search recursively) or a single JSONL file.
    jsonl_files = []
    if os.path.isfile(run_path):
        jsonl_files.append(run_path)
    else:
        for root, _dirs, files in os.walk(run_path):
            for f in files:
                if f.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, f))
                elif f.endswith(".json") and "iter" in f:
                    jsonl_files.append(os.path.join(root, f))

    if not jsonl_files:
        print(f"No JSONL/iter JSON files found under {run_path}")
        sys.exit(1)

    events = []
    for f in jsonl_files:
        events.extend(parse_jsonl(f))

    if not events:
        print("No events parsed")
        sys.exit(1)

    # EXPERT_REQUEST events carry the access pattern.
    access_by_layer = defaultdict(Counter)
    access_order = []  # (layer, expert) tuples
    for e in events:
        if e.get("event") in ("EXPERT_REQUEST",):
            layer = e.get("layer")
            expert = e.get("expert")
            if layer is None or expert is None:
                continue
            access_by_layer[layer][(layer, expert)] += 1
            access_order.append((layer, expert))

    layers = sorted(access_by_layer.keys())
    total_experts_per_layer = max((max(eid for (_l, eid) in c) for c in access_by_layer.values()), default=0) + 1

    report = {
        "run_path": run_path,
        "files": jsonl_files,
        "events_parsed": len(events),
        "num_layers": len(layers),
        "total_experts_per_layer_observed": total_experts_per_layer,
        "per_layer": {},
    }

    for layer in layers:
        c = access_by_layer[layer]
        total_accesses = sum(c.values())
        unique_experts = len(c)
        if total_accesses == 0:
            continue
        # Top-N frequencies
        top = c.most_common(min(10, len(c)))
        top_pct = [(f"e{eid}", cnt, 100.0 * cnt / total_accesses)
                   for (_l, eid), cnt in top]
        report["per_layer"][f"L{layer}"] = {
            "total_accesses": total_accesses,
            "unique_experts": unique_experts,
            "unique_ratio": unique_experts / max(1, total_experts_per_layer),
            "top10_freq_pct": top_pct,
        }

    # Working-set size: number of distinct experts per sliding window of 32 tokens.
    # We use a token-window; we approximate "tokens" by counting access_order
    # entries. With top_k=2 × num_layers, each token contributes ~16 expert
    # accesses for Tier 2 (8L*2).
    window_size = 32 * max(1, len(layers)) * 2  # approx top_k × num_layers × tokens
    if len(access_order) >= window_size:
        seen = set()
        working_set = len(seen)
        max_ws = 0
        for i, key in enumerate(access_order):
            seen.add(key)
            if i >= window_size:
                seen.discard(access_order[i - window_size])
            if len(seen) > max_ws:
                max_ws = len(seen)
        report["working_set_max_in_32token_window"] = max_ws
        # Bytes assuming 393216 B per expert
        report["working_set_max_bytes"] = max_ws * 393216

    # Top-K concentration: how concentrated are accesses in top-1/top-2?
    for layer in layers:
        c = access_by_layer[layer]
        total = sum(c.values())
        sorted_counts = sorted(c.values(), reverse=True)
        for k in (1, 2, 3, 4, 5):
            if len(sorted_counts) >= k:
                pct = sum(sorted_counts[:k]) / total * 100.0
                report["per_layer"][f"L{layer}"][f"top{k}_pct"] = pct

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")

    # Print a one-line summary per layer.
    print("\nLayer summary (accesses / unique / top1% / top2%):")
    for layer in layers:
        d = report["per_layer"][f"L{layer}"]
        print(f"  L{layer}: accesses={d['total_accesses']:5d}  "
              f"unique={d['unique_experts']:3d}  "
              f"top1={d.get('top1_pct', 0):.1f}%  "
              f"top2={d.get('top2_pct', 0):.1f}%")


if __name__ == "__main__":
    main()
