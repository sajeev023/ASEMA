"""Builds the two storage manifests the runtime consumes via ASEMA_STORAGE_MANIFEST:
  ASEMA_STORAGE_LAYOUT_PHASE1.json     - the new placement (authoritative)
  ASEMA_STORAGE_LAYOUT_BASELINE_A.json - the old placement expressed as a manifest (control for the paired benchmark)
Every shard of the checkpoint is listed exactly once with drive, path, layer, size and SHA-256.
SHA-256 comes from reports/storage/migration.jsonl (copies) and reports/storage/hashes.jsonl (unmoved files).
usage (project root): python -I experiments/nextgen/storage/build_manifest.py"""
import json, os

LOG_COPY, LOG_HASH = "reports/storage/migration.jsonl", "reports/storage/hashes.jsonl"
OUT = "experiments/nextgen/storage"
SHARD = lambda n: f"model-{n:05d}-of-00048.safetensors"
D_DIR = "D:/asema_models/DeepSeek-V4.1-Flash/shards"
E_DIR = "E:/asema_models/DeepSeek-V4.1-Flash/shards"

def role(n):
    if 3 <= n <= 42: return n - 3, "layer-experts+dense"
    return None, {1: "vision/aligner", 2: "embed", 43: "head+norm", 44: "mtp", 45: "mtp", 46: "mtp", 47: "engram(layer1)", 48: "engram(layer14)"}[n]

def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []

def main():
    steps = json.load(open(f"{OUT}/migration_steps.json"))["steps"]
    moved = {s["shard"]: s for s in steps}
    sha, size = {}, {}
    for r in read_jsonl(LOG_COPY):
        if r["op"] == "copy" and r["ok"]:
            for k in ("dst", "src"):
                sha[r[k].replace("\\", "/")] = r["sha256_" + k]; size[r[k].replace("\\", "/")] = r["size"]
    for r in read_jsonl(LOG_HASH):
        sha[r["path"].replace("\\", "/")] = r["sha256"]; size[r["path"].replace("\\", "/")] = r["size"]

    def entry(n, path):
        layer, rl = role(n)
        h = sha.get(path)
        if h is None: raise SystemExit(f"STOP: no SHA-256 recorded for {path}")
        if not os.path.exists(path) and not (path.startswith("D:") and moved.get(SHARD(n), {}).get("kind") == "copy"):
            raise SystemExit(f"STOP: {path} missing")
        return {"name": SHARD(n), "path": path, "drive": path[0], "tier": "nvme" if path[0] == "E" else "sata", "layer": layer, "role": rl,
                "size": size[path], "sha256": h}

    base, new = [], []
    for n in range(1, 49):
        nm = SHARD(n); s = moved.get(nm)
        if s is not None and s["kind"] == "relocate":
            old = new_p = s["dst"]                # dead shard, never read: listed at its new home in both manifests
        elif s is not None:
            old, new_p = s["src"], s["dst"]       # layer shard copied: original (baseline) vs. new copy (phase 1)
        else:
            old = new_p = (f"{E_DIR}/{nm}" if os.path.exists(f"{E_DIR}/{nm}") else f"{D_DIR}/{nm}")
        base.append(entry(n, old)); new.append(entry(n, new_p))

    for name, lst, desc in (("ASEMA_STORAGE_LAYOUT_BASELINE_A.json", base, "pre-migration placement (control)"),
                            ("ASEMA_STORAGE_LAYOUT_PHASE1.json", new, "Phase 1 rebalanced placement")):
        assert len({e["name"] for e in lst}) == 48 and len({e["path"] for e in lst}) == 48
        per = {d: sum(1 for e in lst if e["drive"] == d and e["layer"] is not None) for d in "CDE"}
        json.dump({"version": 1, "model": "DeepSeek-V4.1-Flash", "description": desc, "expert_layers_per_drive": per, "shards": lst},
                  open(f"{OUT}/{name}", "w"), indent=1)
        print(name, "layers per drive:", per)

if __name__ == "__main__":
    main()
