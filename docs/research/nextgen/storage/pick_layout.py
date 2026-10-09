"""Pick a minimal-move Phase-1 layout: every layer currently on E: stays on E:; of the 22 layers on D: the
lowest-miss ones stay on SATA (split D/C), the rest move to E:.  Evaluates candidates in the pipeline simulator.
Run from the project root:  python -I experiments/nextgen/storage/pick_layout.py"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.join("experiments", "nextgen"))
import pipeline_sim as P
from cache_sim2 import sim_scored, blend_score, LAYOUT_A

d = P.build("reports/traces", 8); seq = d["seq"]; ntok = len({(s[0], s[1]) for s in seq})
acc = np.array([l * 384 + e for (_, _, l, true, _) in seq for e in true], dtype=np.int64)
h = sim_scored(acc, 700, blend_score); miss = defaultdict(float)
for k, hh in zip(acc, h):
    if not hh: miss[int(k) // 384] += 1.0 / ntok
d_layers = sorted([l for l, dr in LAYOUT_A.items() if dr == "D"], key=lambda l: miss[l])
print("D-resident layers by miss/token (low first):", [(l, round(miss[l], 2)) for l in d_layers])

def evaluate(name, lay, c=13.0):
    tt, st, _ = P.simulate(seq, lay, 700, c, 0.4, "none")
    x = tt[15:]
    print(f"  {name:34s} {x.mean() / 1000:6.3f} s/token -> {1000 / x.mean():5.2f} tok/s (no prefetch, 700-expert cache, 13 ms/layer)")

evaluate("current (A)", LAYOUT_A)
for n_sata, n_c in ((8, 4), (8, 3), (9, 3), (8, 0)):
    sata = d_layers[:n_sata]
    byhigh = sorted(sata, key=lambda l: -miss[l])      # C (511 MB/s) gets the higher-miss SATA layers
    lay = dict(LAYOUT_A)
    for l in sata: lay[l] = "D"
    for l in byhigh[:n_c]: lay[l] = "C"
    for l in d_layers:
        if l not in sata: lay[l] = "E"
    cnt = {k: sum(1 for v in lay.values() if v == k) for k in "CDE"}
    evaluate(f"{n_sata} SATA ({n_c} on C) {cnt}", lay)
    print("     C:", sorted(l for l, v in lay.items() if v == "C"), " D:", sorted(l for l, v in lay.items() if v == "D"))
