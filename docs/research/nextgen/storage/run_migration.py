"""Runs migration_steps.json with shardcopy.exe, one step group at a time, logging to reports/storage/migration.jsonl.
usage (project root):  python -I experiments/nextgen/storage/run_migration.py <step 1|2|3>
Safety: stops at the first failed/refused copy.  Only `relocate` entries (step 1) ever remove a file, and only after the
log holds an ok record whose dst size equals the src size and whose source/destination SHA-256 are identical."""
import json, os, subprocess, sys

EXE = os.path.abspath("experiments/nextgen/storage/shardcopy.exe")
LOG = "reports/storage/migration.jsonl"

def verified(src, dst, size):
    if not (os.path.exists(LOG) and os.path.exists(dst) and os.path.getsize(dst) == size): return False
    for line in open(LOG, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") == "copy" and r["src"] == src and r["dst"] == dst and r["size"] == size and r["ok"] and r["sha256_src"] == r["sha256_dst"]:
            return True
    return False

def main():
    group = int(sys.argv[1])
    plan = json.load(open("experiments/nextgen/storage/migration_steps.json"))
    os.makedirs("reports/storage", exist_ok=True)
    for s in plan["steps"]:
        if s["step"] != group: continue
        os.makedirs(os.path.dirname(s["dst"]), exist_ok=True)
        if not os.path.exists(s["src"]) and verified(s["src"], s["dst"], s["size"]):
            print(f"-- {s['shard']} already relocated and verified; skipping", flush=True); continue
        print(f"-- {s['shard']} ({s['size'] / 2**30:.2f} GiB) {s['src']} -> {s['dst']}", flush=True)
        r = subprocess.run([EXE, "copy", s["src"], s["dst"], LOG])
        if r.returncode != 0:
            print(f"STOP: shardcopy exit {r.returncode} for {s['shard']}", flush=True); sys.exit(r.returncode)
        if s["kind"] == "relocate":
            if not verified(s["src"], s["dst"], s["size"]):
                print("STOP: refusing to remove source - no verified record", flush=True); sys.exit(5)
            if os.path.getsize(s["src"]) != s["size"]:
                print("STOP: source changed size", flush=True); sys.exit(6)
            os.remove(s["src"])
            print(f"   removed verified-duplicate source {s['src']}", flush=True)
    print(f"STEP {group} COMPLETE", flush=True)

if __name__ == "__main__":
    main()
