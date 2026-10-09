"""E1: lookahead-router replay. Read-only on shards; consumes trace_N.txt + hid_N.bin from the instrumented run.
usage: python lookahead.py reports/traces 8
Step 1 validates my numpy replica of the router against the engine's recorded expert choices.
Step 2 applies router(L+k) to the router INPUT of layer L (what is known k layers early) and scores recall of the
        true top-6 of layer L+k at several budgets, next to a history-only baseline on the same samples.
"""
import json, os, struct, sys
import numpy as np

SHARD_DIRS = ["D:/asema_models/DeepSeek-V4.1-Flash/shards", "E:/asema_models/DeepSeek-V4.1-Flash/shards"]
INDEX = "examples/real_model/DeepSeek-V4.1-Flash/hf/model.safetensors.index.json"
D, E, TOPK = 5120, 384, 6

def read_tensor(shard, name):
    for d in SHARD_DIRS:
        p = os.path.join(d, shard)
        if os.path.exists(p): break
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); t = hdr[name]
        a, b = t["data_offsets"]; f.seek(8 + n + a); raw = f.read(b - a)
    return raw, t["dtype"], t["shape"]

def load_routers():
    wm = json.load(open(INDEX))["weight_map"]; W = []; B = []
    for l in range(40):
        raw, dt, sh = read_tensor(wm[f"layers.{l}.ffn.gate.weight"], f"layers.{l}.ffn.gate.weight")
        assert dt == "BF16" and sh == [E, D], (dt, sh)
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        W.append(u.view(np.float32).reshape(E, D))
        raw, dt, sh = read_tensor(wm[f"layers.{l}.ffn.gate.bias"], f"layers.{l}.ffn.gate.bias")
        assert dt == "F32", dt
        B.append(np.frombuffer(raw, dtype=np.float32).copy())
    return W, B

def sel_scores(x, W, B):
    logit = W @ x
    sp = np.where(logit > 20, logit, np.log1p(np.exp(np.minimum(logit, 20))))
    return np.sqrt(sp) + B

def parse_trace(path):
    toks = []; cur = None
    for line in open(path):
        s = line.strip()
        if s == "T": cur = {}; toks.append(cur)
        elif s:
            l, e = s.split(); cur.setdefault(int(l), []).append(int(e))
    return toks

def parse_hidden(path):
    raw = np.fromfile(path, dtype=np.uint8); rec = 4 + 4 * D; n = len(raw) // rec
    out = []
    for i in range(n):
        r = raw[i * rec:(i + 1) * rec]
        out.append((int(np.frombuffer(r[:4], dtype=np.int32)[0]), np.frombuffer(r[4:], dtype=np.float32)))
    return out

def main():
    d, nfiles = sys.argv[1], int(sys.argv[2])
    W, B = load_routers()
    print("routers loaded for 40 layers")
    match = tot = 0; per_layer_bad = {}
    samples = []   # (file, token, layer, hidden, true_set, history_set)
    for i in range(1, nfiles + 1):
        tp, hp = f"{d}/trace_{i}.txt", f"{d}/hid_{i}.bin"
        if not (os.path.exists(tp) and os.path.exists(hp)): continue
        toks = parse_trace(tp); hid = parse_hidden(hp)
        calls = [(t, l) for t, tk in enumerate(toks) for l in sorted(tk)]
        n = min(len(calls), len(hid))
        if len(calls) != len(hid): print(f"  file {i}: trace has {len(calls)} layer calls, dump has {len(hid)}; using the common prefix")
        prev_sets = {}
        for (t, l), (lid, x) in zip(calls[:n], hid[:n]):
            assert lid == l, (lid, l)
            true = set(toks[t][l]); sc = sel_scores(x, W[l], B[l]); top = set(np.argpartition(-sc, TOPK)[:TOPK].tolist())
            tot += 1; ok = (top == true); match += ok
            if not ok: per_layer_bad[l] = per_layer_bad.get(l, 0) + 1
            samples.append((i, t, l, x, true, prev_sets.get(l, set())))
            prev_sets[l] = true
    print(f"\n[validation] my router replica reproduces the engine's top-6 exactly on {match}/{tot} layer calls ({100 * match / max(tot, 1):.2f}%)")
    if per_layer_bad: print("  mismatches by layer:", dict(sorted(per_layer_bad.items())))
    if tot == 0: return
    print("\n[lookahead] recall of the TRUE top-6 of layer L+k, using router(L+k) applied to the router input of layer L")
    by = {(s[0], s[1], s[2]): s for s in samples}
    Ks = (6, 12, 24, 48)
    print(f"{'k':>2} {'samples':>8}   lookahead R@6 R@12 R@24 R@48   |  history baseline (previous token, same layer) R@6 R@12 R@24 R@48")
    for k in (0, 1, 2, 3, 4):
        hit = {K: 0 for K in Ks}; hh = {K: 0 for K in Ks}; cnt = 0
        for (fi, t, l), (_, _, _, x, true, hist) in by.items():
            tgt = by.get((fi, t, l + k))
            if tgt is None: continue
            true_t = tgt[4]
            order = np.argsort(-sel_scores(x, W[l + k], B[l + k]))
            for K in Ks: hit[K] += len(true_t & set(order[:K].tolist()))
            ph = tgt[5]
            for K in Ks: hh[K] += len(true_t & ph)      # history set has at most 6 experts, so it is the same for every K
            cnt += 1
        if cnt:
            print(f"{k:>2} {cnt:8d}                {100 * hit[6] / (6 * cnt):5.1f} {100 * hit[12] / (6 * cnt):5.1f} {100 * hit[24] / (6 * cnt):5.1f} {100 * hit[48] / (6 * cnt):5.1f}"
                  f"   |                                       {100 * hh[6] / (6 * cnt):5.1f} (previous token's 6 experts)")
    print("\n(k=0 is the router applied to its own input: it must be ~100%; that is the validation above in recall form.)")

if __name__ == "__main__":
    main()
