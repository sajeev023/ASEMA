"""Offline cache-policy + placement study on real expert traces (read-only).
usage: python cache_sim2.py trace1.txt [trace2.txt ...]
Trace format: line "T" starts a token; other lines are "layer expert".
Policies: LRU, CLOCK(second chance), LFU(perfect), LRFU, ARC, blend (production), drive-aware blend, Belady (oracle).
CLOCK-Pro is NOT simulated (plain CLOCK stands in; stated in the report).
"""
import sys, heapq, math
from collections import OrderedDict, defaultdict
import numpy as np

EXPERT_B = 18_800_640
BW = {"C": 511e6, "D": 404e6, "E": 3370e6}      # measured sustained B/s for 18.9 MB reads

def load(paths):
    acc = []
    t = -1
    for p in paths:
        for line in open(p):
            s = line.strip()
            if s == "T": t += 1
            elif s:
                l, e = s.split(); acc.append(int(l) * 384 + int(e))
    return np.array(acc, dtype=np.int64), t + 1

def layer_of(k): return k // 384

# ---------------- policies: each returns a boolean hit array --------------------------------------
def sim_lru(a, cap):
    c = OrderedDict(); hit = np.zeros(len(a), bool)
    for i, k in enumerate(a):
        if k in c: c.move_to_end(k); hit[i] = True
        else:
            c[k] = 1
            if len(c) > cap: c.popitem(last=False)
    return hit

def sim_clock(a, cap):
    keys = [-1] * cap; ref = [0] * cap; pos = {}; hand = 0; hit = np.zeros(len(a), bool)
    for i, k in enumerate(a):
        if k in pos: ref[pos[k]] = 1; hit[i] = True; continue
        while True:
            if keys[hand] == -1 or ref[hand] == 0: break
            ref[hand] = 0; hand = (hand + 1) % cap
        if keys[hand] != -1: del pos[keys[hand]]
        keys[hand] = k; ref[hand] = 0; pos[k] = hand; hand = (hand + 1) % cap
    return hit

def sim_scored(a, cap, score_fn, update_fn=None, cost=None):
    """Generic: keep arrays for cached items; evict argmin(score)."""
    n_keys = 40 * 384
    freq = np.zeros(n_keys); last = np.zeros(n_keys); crf = np.zeros(n_keys)
    in_c = np.zeros(n_keys, bool); members = []
    hit = np.zeros(len(a), bool)
    for i, k in enumerate(a):
        freq[k] += 1
        if update_fn: update_fn(crf, last, k, i)
        if in_c[k]:
            hit[i] = True
        else:
            if len(members) >= cap:
                m = np.array(members)
                sc = score_fn(m, freq, last, crf, i)
                if cost is not None: sc = sc * cost[m]
                v = m[int(np.argmin(sc))]
                in_c[v] = False; members.remove(v)
            in_c[k] = True; members.append(k)
        last[k] = i
    return hit

LAM = 0.01
def lfu_score(m, freq, last, crf, i): return freq[m] + 1e-9 * last[m]
def blend_score(m, freq, last, crf, i): return freq[m] / (1.0 + 0.25 * (i - last[m]) / 240.0)
def lrfu_score(m, freq, last, crf, i): return crf[m] * np.power(0.5, LAM * (i - last[m]))
def lrfu_update(crf, last, k, i):
    crf[k] = 1.0 + crf[k] * math.pow(0.5, LAM * (i - last[k]))

def sim_arc(a, cap):
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(); p = 0; hit = np.zeros(len(a), bool)
    def replace(k):
        nonlocal p
        if t1 and ((k in b2 and len(t1) == p) or len(t1) > p):
            old, _ = t1.popitem(last=False); b1[old] = 1
        else:
            old, _ = t2.popitem(last=False); b2[old] = 1
    for i, k in enumerate(a):
        if k in t1: del t1[k]; t2[k] = 1; hit[i] = True
        elif k in t2: t2.move_to_end(k); hit[i] = True
        elif k in b1:
            p = min(cap, p + max(1, len(b2) // max(1, len(b1)))); replace(k); del b1[k]; t2[k] = 1
        elif k in b2:
            p = max(0, p - max(1, len(b1) // max(1, len(b2)))); replace(k); del b2[k]; t2[k] = 1
        else:
            if len(t1) + len(b1) == cap:
                if len(t1) < cap: b1.popitem(last=False); replace(k)
                else: t1.popitem(last=False)
            elif len(t1) + len(t2) + len(b1) + len(b2) >= cap:
                if len(t1) + len(t2) + len(b1) + len(b2) >= 2 * cap and b2: b2.popitem(last=False)
                if len(t1) + len(t2) >= cap: replace(k)
            t1[k] = 1
    return hit

def sim_belady(a, cap):
    n = len(a); nxt = np.full(n, n + 1, dtype=np.int64); seen = {}
    for i in range(n - 1, -1, -1):
        nxt[i] = seen.get(a[i], n + 1); seen[a[i]] = i
    cache = {}; heap = []; hit = np.zeros(n, bool)
    for i, k in enumerate(a):
        if k in cache: hit[i] = True
        else:
            if len(cache) >= cap:
                while True:
                    nu, v = heapq.heappop(heap); nu = -nu
                    if v in cache and cache[v] == nu: del cache[v]; break
        cache[k] = nxt[i]; heapq.heappush(heap, (-nxt[i], k))
    return hit

# ---------------- placement ------------------------------------------------------------------------
def storage_time(miss_per_layer, layout, sequential=False):
    """layout: dict layer->drive; returns (seconds/token, per-drive seconds).
    sequential=False: drives overlap perfectly (needs cross-layer prefetch) -> max over drives.
    sequential=True : layers run one after another and each layer's reads hit ONE drive -> sum over drives
                      (this is how the current engine behaves; it matches the measured storage wait)."""
    per = defaultdict(float)
    for l, m in miss_per_layer.items(): per[layout[l]] += m * EXPERT_B / BW[layout[l]]
    return (sum(per.values()) if sequential else max(per.values())), dict(per)

def optimise_layout(miss_per_layer, caps, sequential=False):
    """minimise storage time (sequential: sum over drives, overlapped: max over drives); caps = layers per drive.
    LPT greedy start + pairwise swaps."""
    layers = sorted(miss_per_layer, key=lambda l: -miss_per_layer[l]); load = {d: 0.0 for d in caps}; cnt = {d: 0 for d in caps}; lay = {}
    for l in layers:
        cands = [d for d in caps if cnt[d] < caps[d]]
        best = min(cands, key=lambda d: load[d] + miss_per_layer[l] * EXPERT_B / BW[d])
        lay[l] = best; cnt[best] += 1; load[best] += miss_per_layer[l] * EXPERT_B / BW[best]
    for l in range(40):
        if l not in lay:   # layers that never missed still need a home: fill remaining capacity
            d = next(d for d in caps if cnt[d] < caps[d]); lay[l] = d; cnt[d] += 1
    improved = True
    while improved:
        improved = False
        for l1 in layers:
            for l2 in layers:
                if lay[l1] == lay[l2]: continue
                cur = storage_time(miss_per_layer, lay, sequential)[0]
                lay[l1], lay[l2] = lay[l2], lay[l1]
                if storage_time(miss_per_layer, lay, sequential)[0] < cur - 1e-12: improved = True
                else: lay[l1], lay[l2] = lay[l2], lay[l1]
    return lay

LAYOUT_A = {l: ("D" if (l in (2, 8) or 10 <= l <= 29) else "E") for l in range(40)}

def main():
    a, ntok = load(sys.argv[1:])
    n = len(a)
    print(f"{ntok} tokens, {n} accesses ({n / ntok:.0f}/token), {len(set(a.tolist()))} distinct (layer,expert) of 15360 "
          f"({100 * len(set(a.tolist())) / 15360:.1f}%)")
    last = {}; dist = []
    for i, k in enumerate(a):
        if k in last: dist.append(i - last[k])
        last[k] = i
    dist = np.array(dist)
    if len(dist):
        print(f"reuse distance (accesses between uses): median {np.median(dist):.0f}, p90 {np.percentile(dist, 90):.0f}; "
              f"compulsory misses {len(set(a.tolist()))} = {100 * len(set(a.tolist())) / n:.1f}% of accesses (floor on miss rate)")
    freqs = np.bincount(a, minlength=15360); top = np.sort(freqs)[::-1]
    for frac in (0.05, 0.1, 0.25):
        k = int(15360 * frac); print(f"  top {frac*100:.0f}% of experts ({k}) receive {100 * top[:k].sum() / n:.1f}% of accesses")
    sizes = [240, 400, 700, 1000, 1300, 1500]
    cost = np.array([1.0 / BW[LAYOUT_A[layer_of(k)]] for k in range(15360)]) * 1e9
    pols = [("LRU", lambda c: sim_lru(a, c)), ("CLOCK", lambda c: sim_clock(a, c)),
            ("LFU", lambda c: sim_scored(a, c, lfu_score)), ("LRFU", lambda c: sim_scored(a, c, lrfu_score, lrfu_update)),
            ("ARC", lambda c: sim_arc(a, c)), ("blend", lambda c: sim_scored(a, c, blend_score)),
            ("blend x drive-cost", lambda c: sim_scored(a, c, blend_score, cost=cost)),
            ("Belady (oracle)", lambda c: sim_belady(a, c))]
    print("\nhit rate % by policy and cache size (experts); 1 expert = 18.8 MB")
    print(f"{'policy':20s}" + "".join(f"{s:>8d}" for s in sizes))
    results = {}
    for name, fn in pols:
        row = []
        for s in sizes:
            h = fn(s); results[(name, s)] = h; row.append(100 * h.mean())
        print(f"{name:20s}" + "".join(f"{v:8.1f}" for v in row), flush=True)
    print(f"{'size (GB)':20s}" + "".join(f"{s * EXPERT_B / 1e9:8.1f}" for s in sizes))
    print("\nstorage time per token, policy = blend x drive-cost; layout A = current, B = optimised (32 layers NVMe + 8 on D), "
          "B2 = optimised (32 NVMe + 4 D + 4 C):")
    for s in sizes:
        h = results[("blend x drive-cost", s)]
        miss_pl = defaultdict(float)
        for k, hh in zip(a, h):
            if not hh: miss_pl[layer_of(int(k))] += 1.0 / ntok
        mb = sum(miss_pl.values()) * EXPERT_B / 1e9
        out = f"  cache {s:5d} ({s * EXPERT_B / 1e9:5.1f} GB): {mb:5.2f} GB/token |"
        layB = optimise_layout(miss_pl, {"E": 32, "D": 8}, True); layB2 = optimise_layout(miss_pl, {"E": 32, "D": 4, "C": 4}, True)
        layB2o = optimise_layout(miss_pl, {"E": 32, "D": 4, "C": 4}, False)
        for name, lay in (("A", LAYOUT_A), ("B", layB), ("B2", layB2), ("B2*", layB2o)):
            seq = storage_time(miss_pl, lay, True)[0]; par = storage_time(miss_pl, lay, False)[0]
            out += f" {name}: seq {seq:4.2f}s / overlapped {par:4.2f}s |"
        print(out)

def codesign(a, ntok, cap, caps, sequential, iters=3):
    """Iterate cache policy and placement to a fixed point: cost-aware cache -> miss profile -> best layout -> new costs."""
    cost = np.ones(15360); lay = None
    for _ in range(iters):
        h = sim_scored(a, cap, blend_score, cost=cost)
        miss_pl = defaultdict(float)
        for k, hh in zip(a, h):
            if not hh: miss_pl[layer_of(int(k))] += 1.0 / ntok
        lay = optimise_layout(miss_pl, caps, sequential)
        cost = np.array([1.0 / BW[lay[layer_of(k)]] for k in range(15360)]) * 1e9
    t_seq = storage_time(miss_pl, lay, True)[0]; t_par = storage_time(miss_pl, lay, False)[0]
    return h.mean(), sum(miss_pl.values()) * EXPERT_B / 1e9, t_seq, t_par, lay

def main_codesign():
    a, ntok = load(sys.argv[2:])
    print(f"co-designed cache + placement (blend x drive cost, layout re-optimised for that cache), {ntok} tokens")
    print(f"{'cache':>6} {'GB':>5} | {'layout':8s} {'hit%':>6} {'GB/tok':>7} {'seq s':>6} {'overlap s':>9}  SATA layers")
    for cap in (240, 400, 700, 1000, 1300, 1500):
        for name, caps in (("B", {"E": 32, "D": 8}), ("B2", {"E": 32, "D": 4, "C": 4})):
            hit, gb, ts, tp, lay = codesign(a, ntok, cap, caps, True)
            sata = sorted(l for l, d in lay.items() if d != "E")
            print(f"{cap:6d} {cap * EXPERT_B / 1e9:5.1f} | {name:8s} {100 * hit:6.1f} {gb:7.2f} {ts:6.2f} {tp:9.2f}  {sata}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--codesign": main_codesign()
    else: main()
