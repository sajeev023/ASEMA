"""Event-driven token-time simulator driven by REAL traces and REAL lookahead predictions (read-only).
usage: python pipeline_sim.py <traces_dir> <nfiles>
Model of the machine (all parameters measured earlier in this study):
  * drives C/D/E are independent single servers with sustained bandwidth BW (18.9 MB reads), each expert read = 18.8 MB / BW
  * demand reads have priority over queued speculative reads; an in-service read cannot be preempted
  * RAM expert cache (blend x drive-cost score) with capacity C experts; an entry in flight is usable when its read completes
  * each layer: attention (a ms) -> router input known -> issue speculative reads for layers l+1..l+k -> demand reads for l's
    6 experts -> wait for all 6 -> expert execution + rest (c - a ms)
Token time = end of layer 39 minus start of layer 0. Cache and drive queues persist across tokens and prompts.
"""
import os, pickle, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookahead import load_routers, sel_scores, parse_trace, parse_hidden, D
from cache_sim2 import optimise_layout, storage_time, EXPERT_B, BW, LAYOUT_A

KMAX_LOOK = 3; TOPN = 24

def build(d, nfiles):
    cache_path = f"{d}/preds.pkl"
    if os.path.exists(cache_path):
        data = pickle.load(open(cache_path, "rb"))
        if data["nfiles"] == nfiles: return data
    W, B = load_routers()
    seq = []   # list of (file, token, layer, true list, preds{k: top-24 list})
    for i in range(1, nfiles + 1):
        tp, hp = f"{d}/trace_{i}.txt", f"{d}/hid_{i}.bin"
        if not (os.path.exists(tp) and os.path.exists(hp)): continue
        toks = parse_trace(tp); hid = parse_hidden(hp)
        calls = [(t, l) for t, tk in enumerate(toks) for l in sorted(tk)]
        n = min(len(calls), len(hid))
        for (t, l), (lid, x) in zip(calls[:n], hid[:n]):
            preds = {}
            for k in range(1, KMAX_LOOK + 1):
                if l + k < 40: preds[k] = np.argsort(-sel_scores(x, W[l + k], B[l + k]))[:TOPN].tolist()
            seq.append((i, t, l, toks[t][l], preds))
    data = {"nfiles": nfiles, "seq": seq}
    pickle.dump(data, open(cache_path, "wb"))
    return data

class Drive:
    def __init__(self, bw): self.bw = bw; self.busy_until = 0.0; self.queue = []   # queued speculative reads: (key, svc)
    def svc(self): return EXPERT_B / self.bw * 1000.0   # ms

INF = float("inf")

def simulate(seq, layout, cap, c_ms, a_frac, strategy, kk=None, costs=True, staging_cap=64, spec_drives=None, truth=None, prank=None, pmin=None):
    """strategy: 'none' | 'look' (kk = {k: K}); returns per-token times (ms), stats, total demand wait (ms).
    Speculative reads land in a small separate STAGING buffer (staging_cap experts, FIFO) so they cannot pollute the cache;
    a demand hit on staging promotes the expert into the main cache."""
    from collections import OrderedDict
    drives = {d: Drive(BW[d]) for d in set(layout.values())}
    ready = {}          # key -> time its bytes are in RAM (INF while a speculative read is queued but not started)
    freq = defaultdict(float); last = {}
    members = []; mset = set()
    cost = {k: 1.0 / BW[layout[k // 384]] * 1e9 for k in range(15360)} if costs else None
    stats = defaultdict(float)
    token_times = []; now = 0.0; tok_start = 0.0; cur = None; wait_total = 0.0
    spec_pending = set()          # queued speculative reads not yet started
    staging = OrderedDict()       # speculative entries (queued, running or done) not yet consumed
    t_idx = 0

    def advance(dr, t):
        while dr.queue and dr.busy_until <= t:
            key, sv = dr.queue.pop(0)
            if key not in spec_pending: continue
            spec_pending.discard(key)
            start = dr.busy_until; dr.busy_until = start + sv; ready[key] = dr.busy_until

    def insert(key):
        if key in mset: return
        if len(members) >= cap:
            best = None; bs = 0.0
            for m in members:
                sc = freq[m] / (1.0 + 0.25 * (t_idx - last.get(m, 0)) / 240.0) * (cost[m] if cost else 1.0)
                if best is None or sc < bs: best, bs = m, sc
            members.remove(best); mset.discard(best); ready.pop(best, None)
        members.append(key); mset.add(key)

    for (fi, tk, l, true, preds) in seq:
        if (fi, tk) != cur:
            if cur is not None: token_times.append(now - tok_start)
            cur = (fi, tk); tok_start = now
        t_idx += 1
        now += c_ms * a_frac                      # attention done: router input of layer l is known
        drv = drives[layout[l]]
        if strategy in ("look", "oracle", "value"):
            for k, K in kk.items():
                if l + k >= 40: continue
                if strategy in ("look", "value") and k not in preds: continue
                if spec_drives is not None and layout[l + k] not in spec_drives: continue
                d2 = drives[layout[l + k]]; issued = 0
                cand = preds[k] if strategy in ("look", "value") else truth.get((fi, tk, l + k), [])
                for rank, e in enumerate(cand):
                    if issued >= K: break
                    if strategy == "value" and prank[k][rank] < pmin[layout[l + k]]: break   # ranks are ordered by probability
                    key = (l + k) * 384 + e
                    if key in mset or key in staging: continue
                    if len(staging) >= staging_cap:
                        old, _ = staging.popitem(last=False); spec_pending.discard(old); ready.pop(old, None); stats["spec_wasted"] += 1
                    staging[key] = 1; ready[key] = INF; spec_pending.add(key)
                    d2.queue.append((key, d2.svc())); stats["spec_issued"] += 1; issued += 1
                advance(d2, now)
        arrivals = [now]
        for e in true:
            key = l * 384 + e; freq[key] += 1
            if key in mset:
                stats["hit"] += 1; arrivals.append(ready.get(key, 0.0))
            elif key in staging:
                del staging[key]; stats["spec_used"] += 1
                if key in spec_pending:               # still queued: it becomes a demand read now
                    spec_pending.discard(key); stats["promoted"] += 1
                    advance(drv, now); start = max(now, drv.busy_until); drv.busy_until = start + drv.svc(); ready[key] = drv.busy_until
                arrivals.append(ready[key]); insert(key)
                if ready[key] > now: stats["spec_late"] += 1
            else:
                insert(key); advance(drv, now)
                start = max(now, drv.busy_until); drv.busy_until = start + drv.svc(); ready[key] = drv.busy_until
                arrivals.append(drv.busy_until); stats["miss"] += 1
            last[key] = t_idx
        fin = max(arrivals); wait_total += fin - now; now = fin
        now += c_ms * (1 - a_frac)
    token_times.append(now - tok_start)
    return np.array(token_times), stats, wait_total

def summarize(name, tt, st, skip=15):
    x = tt[skip:] if len(tt) > skip + 5 else tt
    used = st["spec_used"]; iss = max(st["spec_issued"], 1)
    print(f"    {name:30s} mean {x.mean() / 1000:6.3f}s p50 {np.percentile(x, 50) / 1000:6.3f} p95 {np.percentile(x, 95) / 1000:6.3f}"
          f" -> {1000 / x.mean():5.2f} tok/s | demand-miss {st['miss']:6.0f} hit {st['hit']:6.0f} | spec issued {st['spec_issued']:6.0f} used {100 * used / iss:4.0f}% (late {st['spec_late']:.0f})", flush=True)

def main():
    d, nfiles = sys.argv[1], int(sys.argv[2])
    data = build(d, nfiles); seq = data["seq"]
    ntok = len({(s[0], s[1]) for s in seq})
    truth = {(s[0], s[1], s[2]): s[3] for s in seq}
    print(f"{len(seq)} layer calls, {ntok} tokens over {nfiles} prompts (token times skip the first 15 tokens as warm-up)")
    from cache_sim2 import sim_scored, blend_score
    acc = np.array([l * 384 + e for (_, _, l, true, _) in seq for e in true], dtype=np.int64)
    layouts = {"A  current layout": LAYOUT_A}
    for cap in (240, 700):
        h = sim_scored(acc, cap, blend_score); miss = defaultdict(float)
        for k, hh in zip(acc, h):
            if not hh: miss[int(k) // 384] += 1.0 / ntok
        layouts[f"B2 (32 E,4 D,4 C) tuned for cache {cap}"] = optimise_layout(miss, {"E": 32, "D": 4, "C": 4}, True)
    scen = [(13.0, "compute 13 ms/layer (GPU dense + CPU experts)"), (21.0, "compute 21 ms/layer (all-CPU)"), (46.0, "compute 46 ms/layer (today's non-I/O time)")]
    if len(sys.argv) > 3: scen = [s for s in scen if str(int(s[0])) in sys.argv[3].split(",")]
    for c_ms, label in scen:
        print(f"\n=== {label} ===")
        for lname, lay in layouts.items():
            for cap in (240, 700):
                print(f"  layout {lname}, RAM cache {cap} experts ({cap * EXPERT_B / 1e9:.1f} GB):")
                tt, st, _ = simulate(seq, lay, cap, c_ms, 0.4, "none"); summarize("no prefetch", tt, st)
                for kk in ({1: 6}, {1: 8}, {1: 12}, {1: 8, 2: 6}, {1: 8, 2: 6, 3: 4}):
                    tt, st, _ = simulate(seq, lay, cap, c_ms, 0.4, "look", kk); summarize("lookahead " + str(kk), tt, st)
                for kk in ({1: 6}, {1: 8, 2: 6}):
                    tt, st, _ = simulate(seq, lay, cap, c_ms, 0.4, "look", kk, spec_drives={"E"}); summarize("NVMe-only lookahead " + str(kk), tt, st)
                for kk in ({1: 6}, {1: 6, 2: 6, 3: 6}, {k: 6 for k in range(1, 40)}):
                    tt, st, _ = simulate(seq, lay, cap, c_ms, 0.4, "oracle", kk, truth=truth); summarize(f"ORACLE k<={max(kk)}", tt, st)

if __name__ == "__main__":
    main()
