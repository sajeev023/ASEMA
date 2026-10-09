"""Static post-migration verification of a storage manifest against model.safetensors.index.json (read-only, cheap).
Checks: every shard in the index is listed exactly once; the file exists with the manifest size; the safetensors header
parses and its data region ends exactly at the file size; every index tensor is present in the header of the shard the
index names; and every expert layer 0..39 resolves to exactly one drive.
usage (project root): python -I experiments/nextgen/storage/verify_layout.py <manifest.json>"""
import collections, json, os, re, struct, sys

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    end = 8 + n + max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
    return h, end

def main():
    man = json.load(open(sys.argv[1]))
    idx = json.load(open("D:/asema_models/DeepSeek-V4.1-Flash/model.safetensors.index.json"))["weight_map"]
    by_name = {}
    for s in man["shards"]:
        if s["name"] in by_name: raise SystemExit(f"FAIL duplicate shard {s['name']}")
        by_name[s["name"]] = s
    need = set(idx.values())
    errs = []
    if need != set(by_name): errs.append(f"index shards {len(need)} vs manifest {len(by_name)}; missing {sorted(need - set(by_name))[:3]}")
    heads = {}
    for nm, s in by_name.items():
        p = s["path"]
        if not os.path.exists(p): errs.append(f"missing file {p}"); continue
        if os.path.getsize(p) != s["size"]: errs.append(f"size mismatch {p}")
        try:
            h, end = header(p)
        except Exception as e:
            errs.append(f"bad header {p}: {e}"); continue
        if end != s["size"]: errs.append(f"{nm}: header data end {end} != file size {s['size']}")
        heads[nm] = h
    for t, nm in idx.items():
        if nm in heads and t not in heads[nm]: errs.append(f"tensor {t} not in header of {nm}")
    layer_drive = collections.defaultdict(set)
    for t, nm in idx.items():
        m = re.match(r"(?:model\.)?layers\.(\d+)\..*experts", t)
        if m and nm in by_name: layer_drive[int(m.group(1))].add(by_name[nm]["drive"])
    for L in range(40):
        if len(layer_drive[L]) != 1: errs.append(f"layer {L} spans drives {sorted(layer_drive[L])}")
    per = collections.Counter(next(iter(v)) for v in layer_drive.values() if len(v) == 1)
    print(f"{os.path.basename(sys.argv[1])}: shards {len(by_name)}, tensors checked {len(idx)}, expert layers per drive {dict(per)}")
    print("INTEGRITY PASS" if not errs else "INTEGRITY FAIL:\n  " + "\n  ".join(errs[:20]))
    sys.exit(1 if errs else 0)

if __name__ == "__main__":
    main()
