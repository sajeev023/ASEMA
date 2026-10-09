"""Batch runner for pipeline_sim.py: key configurations, results flushed line by line.
usage: python run_pipe.py <config_index ...>   (configs listed in CONFIGS)"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_sim as P
from cache_sim2 import sim_scored, blend_score, optimise_layout

CONFIGS = [(13.0, "B2_700", 700), (13.0, "A", 700), (21.0, "B2_700", 700), (46.0, "B2_700", 700), (46.0, "A", 240), (13.0, "B2_240", 240)]

def main():
    d = P.build("reports/traces", 8); seq = d["seq"]; ntok = len({(s[0], s[1]) for s in seq})
    truth = {(s[0], s[1], s[2]): s[3] for s in seq}
    acc = np.array([l * 384 + e for (_, _, l, true, _) in seq for e in true], dtype=np.int64)
    import pickle
    lp = "reports/traces/layouts.pkl"
    if os.path.exists(lp):
        lay = pickle.load(open(lp, "rb"))
    else:
        lay = {"A": P.LAYOUT_A}
        for cap in (240, 700):
            h = sim_scored(acc, cap, blend_score); miss = defaultdict(float)
            for k, hh in zip(acc, h):
                if not hh: miss[int(k) // 384] += 1.0 / ntok
            lay[f"B2_{cap}"] = optimise_layout(miss, {"E": 32, "D": 4, "C": 4}, True)
        pickle.dump(lay, open(lp, "wb"))
    print("SATA layers in B2_700:", sorted(l for l, dr in lay["B2_700"].items() if dr != "E"), flush=True)
    # calibrated P(true | lookahead rank) for k = 1..3 from the data itself (rank 0 = most likely)
    prank = {}
    for k in (1, 2, 3):
        hits = np.zeros(24); tot = 0
        for (fi, tk, l, true, preds) in seq:
            if k in preds and (fi, tk, l + k) in truth:
                t2 = set(truth[(fi, tk, l + k)]); tot += 1
                for r_, e in enumerate(preds[k]): hits[r_] += (e in t2)
        prank[k] = hits / max(tot, 1)
    print("P(true | rank), k=1:", " ".join(f"{p:.2f}" for p in prank[1][:12]), flush=True)
    # args: "<config index>:<group>" with group a = no prefetch + history-free lookahead, b = NVMe-only lookahead + oracle
    for arg in sys.argv[1:]:
        idx, grp = (arg.split(":") + ["ab"])[:2]
        c, lname, cap = CONFIGS[int(idx)]; L = lay[lname]
        print(f"\n=== compute {c:.0f} ms/layer | layout {lname} | RAM cache {cap} experts ({cap * 18.8e6 / 1e9:.1f} GB) [{grp}] ===", flush=True)
        if "a" in grp:
            tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "none"); P.summarize("no prefetch", tt, st)
            for kk in ({1: 6}, {1: 8}, {1: 12}, {1: 8, 2: 6}):
                tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "look", kk); P.summarize("lookahead " + str(kk), tt, st)
        if "x" in grp:   # compact: no prefetch, best value policy, oracle k=1
            tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "none"); P.summarize("no prefetch", tt, st)
            tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "value", {1: 24}, prank=prank, pmin={"E": 0.6, "D": 0.1, "C": 0.1}); P.summarize("value pmin E=0.6 SATA=0.1", tt, st)
            tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "oracle", {1: 6}, truth=truth); P.summarize("ORACLE k=1", tt, st)
        if "v" in grp:
            for pm in ({"E": 0.5, "D": 0.5, "C": 0.5}, {"E": 0.3, "D": 0.3, "C": 0.3}, {"E": 0.5, "D": 0.2, "C": 0.2},
                       {"E": 0.4, "D": 0.15, "C": 0.15}, {"E": 0.6, "D": 0.1, "C": 0.1}, {"E": 0.9, "D": 0.1, "C": 0.1}):
                tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "value", {1: 24}, prank=prank, pmin=pm); P.summarize(f"value pmin E={pm['E']} SATA={pm['D']}", tt, st)
        if "b" in grp:
            for kk in ({1: 6}, {1: 8, 2: 6}):
                tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "look", kk, spec_drives={"E"}); P.summarize("NVMe-only lookahead " + str(kk), tt, st)
            for kk in ({1: 6}, {1: 6, 2: 6, 3: 6}):
                tt, st, _ = P.simulate(seq, L, cap, c, 0.4, "oracle", kk, truth=truth); P.summarize("ORACLE " + str(kk), tt, st)

if __name__ == "__main__":
    main()
