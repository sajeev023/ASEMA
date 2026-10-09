"""Phase-1 migration planner (read-only).  Builds the move list from the LIVE filesystem, checks the free-space
arithmetic and the safety margins, and writes experiments/nextgen/storage/migration_steps.json plus
PHASE1_STORAGE_MIGRATION_PLAN.md.  Moves nothing.
Run from the project root:  python -I experiments/nextgen/storage/make_plan.py"""
import json, os, shutil, sys, time

GIB = 1024 ** 3
ROOT = {"D": "D:/asema_models/DeepSeek-V4.1-Flash", "E": "E:/asema_models/DeepSeek-V4.1-Flash", "C": "C:/asema_models/DeepSeek-V4.1-Flash"}
# layer L's routed experts + layer tensors live in shard L+3 (verified against model.safetensors.index.json)
SHARD = lambda n: f"model-{n:05d}-of-00048.safetensors"
LAYER_TO_E = [2, 8, 11, 12, 13, 15, 19, 20, 21, 22, 23, 24, 27, 29]     # 14 layers D -> E
LAYER_TO_C = [17, 26, 28]                                                # 3 layers D -> C
LAYER_STAY_D = [10, 14, 16, 18, 25]                                      # stay on D
DEAD_E_TO_D = [44, 45, 46, 48]                                           # MTP x3 + Engram(layer 14) off the NVMe
MARGIN_GIB = {"E": 15.0, "C": 20.0, "D": 50.0}                           # minimum free space after the whole migration

def src_of(shard_no):
    for d in ("D", "E"):
        p = f"{ROOT[d]}/shards/{SHARD(shard_no)}"
        if os.path.exists(p): return d, p
    raise SystemExit(f"STOP: {SHARD(shard_no)} not found on D or E")

def main():
    steps = []
    for n in DEAD_E_TO_D:
        d, p = src_of(n)
        if d != "E": raise SystemExit(f"STOP: shard {n} expected on E, found on {d}")
        steps.append({"step": 1, "kind": "relocate", "shard": SHARD(n), "layer": None, "src": p, "dst": f"{ROOT['D']}/shards/{SHARD(n)}", "dst_drive": "D", "src_drive": "E"})
    for L in LAYER_TO_E:
        d, p = src_of(L + 3)
        if d != "D": raise SystemExit(f"STOP: layer {L} shard expected on D, found on {d}")
        steps.append({"step": 2, "kind": "copy", "shard": SHARD(L + 3), "layer": L, "src": p, "dst": f"{ROOT['E']}/shards_p1/{SHARD(L + 3)}", "dst_drive": "E", "src_drive": "D"})
    for L in LAYER_TO_C:
        d, p = src_of(L + 3)
        if d != "D": raise SystemExit(f"STOP: layer {L} shard expected on D, found on {d}")
        steps.append({"step": 3, "kind": "copy", "shard": SHARD(L + 3), "layer": L, "src": p, "dst": f"{ROOT['C']}/shards_p1/{SHARD(L + 3)}", "dst_drive": "C", "src_drive": "D"})
    for s in steps:
        st = os.stat(s["src"]); s["size"] = st.st_size; s["src_mtime"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))

    free = {d: shutil.disk_usage(f"{d}:/").free for d in "CDE"}
    total = {d: shutil.disk_usage(f"{d}:/").total for d in "CDE"}
    delta = {d: 0 for d in "CDE"}
    for s in steps:
        delta[s["dst_drive"]] -= s["size"]
        if s["kind"] == "relocate": delta[s["src_drive"]] += s["size"]      # the E copy is removed after the D copy verifies
    final = {d: free[d] + delta[d] for d in "CDE"}
    step1 = sum(s["size"] for s in steps if s["step"] == 1)
    e_in = sum(s["size"] for s in steps if s["dst_drive"] == "E")
    e_after_step1 = free["E"] + step1
    d_after_step1 = free["D"] - step1
    ok = (all(final[d] / GIB >= MARGIN_GIB[d] for d in "CDE") and d_after_step1 > 0 and e_after_step1 - e_in > 0)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "steps": steps, "free_before": free, "total": total, "free_after": final,
           "free_after_step1": {"E": e_after_step1, "D": d_after_step1}, "margin_gib": MARGIN_GIB, "plan_consistent": ok,
           "layers": {"E_new": LAYER_TO_E, "C_new": LAYER_TO_C, "D_stay": LAYER_STAY_D}}
    json.dump(out, open("experiments/nextgen/storage/migration_steps.json", "w"), indent=1)
    gib = lambda b: f"{b / GIB:.2f}"
    L = ["# Phase 1 storage migration plan", "", f"Generated {out['generated']} from the LIVE filesystem (not from CURRENT_RUNTIME_TRUTH.md).", "",
         "## Free space (GiB)", "", "| Drive | Total | Free now | Free after step 1 | Free after migration | Required margin | OK |", "|---|---|---|---|---|---|---|"]
    for d in "CDE":
        a1 = gib(e_after_step1) if d == "E" else (gib(d_after_step1) if d == "D" else gib(free[d]))
        L.append(f"| {d}: | {gib(total[d])} | {gib(free[d])} | {a1} | {gib(final[d])} | {MARGIN_GIB[d]:.0f} | {'yes' if final[d] / GIB >= MARGIN_GIB[d] else 'NO'} |")
    L += ["", f"**Plan internally consistent: {'YES' if ok else 'NO - STOP'}**", "",
          "## Steps", "",
          "Step 1 *relocates* (copy, verify by unbuffered re-read, only then remove the E: copy) the Engram/MTP shards the engine never reads, which frees the NVMe. "
          "It is the one step that removes a file: the verified identical copy on D: becomes the only copy. Rollback = copy it back.", "",
          "Steps 2-3 are plain verified copies into new `shards_p1` directories; **every original stays where it is** (rollback = unset the manifest). "
          "Nothing is cleaned up in this phase.", "",
          "| Step | Kind | Layer | Shard | Size (bytes) | GiB | Source | Destination | Source mtime | Checksum | Rollback |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in steps:
        rb = "copy back D->E" if s["kind"] == "relocate" else "original remains at source; unset manifest"
        L.append(f"| {s['step']} | {s['kind']} | {s['layer'] if s['layer'] is not None else '(mtp/engram)'} | {s['shard']} | {s['size']} | {gib(s['size'])} | {s['src']} | {s['dst']} | {s['src_mtime']} | SHA-256 streamed while copying, re-verified by re-reading the destination | {rb} |")
    L += ["", "## Resulting expert placement (layer -> drive)", "", f"- E: (NVMe) existing 18 layers + new {LAYER_TO_E} = 32 layers",
          f"- C: (SATA) {LAYER_TO_C}", f"- D: (SATA) {LAYER_STAY_D}", "",
          "Chosen by lowest cache-miss rate among D:'s 22 layers (experiments/nextgen/storage/pick_layout.py); simulated 1.78 s/token vs 3.76 s today (no prefetch). "
          "Deviation from the research layout (4 layers on C:): 3 on C: because C: is the system drive and a 20 GiB free floor is kept.", ""]
    open("experiments/nextgen/storage/PHASE1_STORAGE_MIGRATION_PLAN.md", "w", encoding="utf-8").write("\n".join(L))
    print(json.dumps({"consistent": ok, "free_GiB": {d: round(free[d] / GIB, 1) for d in "CDE"}, "final_GiB": {d: round(final[d] / GIB, 1) for d in "CDE"},
                      "moved_GiB": round(sum(s["size"] for s in steps) / GIB, 1), "steps": len(steps)}))
    if not ok: sys.exit(1)

if __name__ == "__main__":
    main()
