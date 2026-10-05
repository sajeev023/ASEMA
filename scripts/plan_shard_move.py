#!/usr/bin/env python3
"""Plan (and optionally perform) moving checkpoint shards from the slow drive to the fast drive.

Why: decode speed is dominated by reading routed experts from storage. If the primary shard
directory is on a slower drive than the secondary one, moving shards onto the faster drive
reduces the time spent waiting. Every expert lives wholly inside one shard, and the loader finds
a shard by name in either directory, so no code or index change is needed after a move.

SAFE BY DEFAULT: with no flags this script only READS (index json, file sizes, free space) and
prints a plan plus a projected gain. Nothing is copied or deleted.

  --execute        copy the planned shards to the secondary directory (as <name>.part, then
                   rename), verifying size and SHA-256 against the source. Sources are kept.
  --delete-source  only together with --execute: delete each source shard AFTER its copy has
                   been verified. Without this flag the originals stay (duplicate shards are
                   harmless to the loader but use space).

Locations come from asema.config / ASEMA_SHARDS_PRIMARY / ASEMA_SHARDS_SECONDARY (same as the
engine). Optional --trace <file> (recorded with ASEMA_TRACE_EXPERTS) projects the I/O time.
"""
import argparse, glob, hashlib, json, os, re, shutil, sys
from collections import defaultdict

EXPERT_MB = 18.8006
# The runner memory-maps the embedding and LM head from these two shards; leave them where they are.
KEEP_SHARDS = {"model-00002-of-00048.safetensors", "model-00043-of-00048.safetensors"}


def read_config():
    cfg = {}
    path = os.environ.get("ASEMA_CONFIG", "asema.config")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"')
    return cfg


def resolve(env, key, default=""):
    return os.environ.get(env) or read_config().get(key) or default


def sha256(path, label):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
            done += len(b)
            print(f"\r    hashing {label}: {done / size * 100:5.1f}%", end="", flush=True)
    print()
    return h.hexdigest()


def load_trace(path):
    tokens, cur = [], None
    for line in open(path):
        s = line.strip()
        if s == "T":
            cur = []
            tokens.append(cur)
        elif s:
            l, e = map(int, s.split())
            cur.append((l, e))
    return tokens


def simulate(tokens, cap, drive_of, slow_mbps, fast_mbps):
    """Mean storage ms/token for a layout: blend cache (alpha 0.25), drives read concurrently per layer."""
    freq, cache, clock, per_token = defaultdict(int), {}, 0, []
    for tk in tokens:
        by_layer = defaultdict(list)
        for (l, e) in tk:
            by_layer[l].append(e)
        t = 0.0
        for l, es in by_layer.items():
            slow_mb = fast_mb = 0.0
            for e in es:
                k = (l, e)
                clock += 1
                freq[k] += 1
                if k in cache:
                    cache[k] = clock
                    continue
                if drive_of[k] == "fast":
                    fast_mb += EXPERT_MB
                else:
                    slow_mb += EXPERT_MB
                if len(cache) >= cap:
                    victim = min(cache, key=lambda x: freq[x] / (1.0 + 0.25 * (clock - cache[x]) / 240.0))
                    del cache[victim]
                cache[k] = clock
            t += max(slow_mb / slow_mbps, fast_mb / fast_mbps) * 1000.0
        per_token.append(t)
    warm = per_token[5:] if len(per_token) > 10 else per_token
    return sum(warm) / len(warm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reserve-gb", type=float, default=12.0, help="free space to keep on the fast drive (default 12)")
    ap.add_argument("--slow-mbps", type=float, default=440.0, help="measured slow-drive random-read MB/s")
    ap.add_argument("--fast-mbps", type=float, default=2800.0, help="measured fast-drive random-read MB/s")
    ap.add_argument("--cache-experts", type=int, default=167, help="expert cache size used for the projection")
    ap.add_argument("--trace", help="expert access trace written by ASEMA_TRACE_EXPERTS")
    ap.add_argument("--execute", action="store_true", help="copy the planned shards (sources are kept)")
    ap.add_argument("--delete-source", action="store_true", help="with --execute: delete verified sources")
    ap.add_argument("--delete-duplicates", action="store_true",
                    help="delete the PRIMARY-drive copy of every shard that exists on both drives, but only after "
                         "re-hashing both copies and finding them identical (use after a verified --execute)")
    args = ap.parse_args()

    primary = resolve("ASEMA_SHARDS_PRIMARY", "shards_primary")
    secondary = resolve("ASEMA_SHARDS_SECONDARY", "shards_secondary")
    model_root = resolve("ASEMA_MODEL_ROOT", "model_root")
    hf_dir = resolve("ASEMA_HF_DIR", "hf_dir")
    if not (primary and secondary):
        sys.exit("Need both shards_primary and shards_secondary configured (asema.config or ASEMA_* env vars).")
    index_path = next((p for p in (os.path.join(hf_dir, "model.safetensors.index.json"),
                                   os.path.join(model_root, "model.safetensors.index.json")) if os.path.exists(p)), None)
    if not index_path:
        sys.exit("model.safetensors.index.json not found under hf_dir / model_root.")
    weight_map = json.load(open(index_path))["weight_map"]

    on_primary = {os.path.basename(f): os.path.getsize(f) for f in glob.glob(os.path.join(primary, "model-*.safetensors"))}
    on_secondary = {os.path.basename(f): os.path.getsize(f) for f in glob.glob(os.path.join(secondary, "model-*.safetensors"))}
    free_gb = shutil.disk_usage(secondary).free / 1e9
    budget_gb = max(0.0, free_gb - args.reserve_gb)

    if args.delete_duplicates:
        dups = sorted(s for s in on_primary if s in on_secondary)
        print(f"Shards present on BOTH drives: {len(dups)}")
        freed = 0
        for s in dups:
            src, dst = os.path.join(primary, s), os.path.join(secondary, s)
            if on_primary[s] != on_secondary[s]:
                print(f"  {s}: sizes differ ({on_primary[s]} vs {on_secondary[s]}); leaving both untouched")
                continue
            if sha256(src, f"{s} (primary)") != sha256(dst, f"{s} (secondary)"):
                sys.exit(f"  checksum mismatch for {s}; NOT deleting anything further.")
            os.remove(src)
            freed += on_primary[s]
            print(f"  {s}: identical copy verified on secondary; primary copy deleted")
        print(f"Done. Freed {freed / 1e9:.1f} GB on the primary drive.")
        return

    pat = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.weight$")
    experts_in = defaultdict(list)   # shard -> [(layer, expert)]
    for name, shard in weight_map.items():
        m = pat.match(name)
        if m:
            experts_in[shard].append((int(m.group(1)), int(m.group(2))))
    total_experts = sum(len(v) for v in experts_in.values())

    print(f"Primary   : {primary}  ({len(on_primary)} shards, {sum(on_primary.values()) / 1e9:.0f} GB)")
    print(f"Secondary : {secondary}  ({len(on_secondary)} shards, {sum(on_secondary.values()) / 1e9:.0f} GB, "
          f"{free_gb:.0f} GB free, keeping {args.reserve_gb:.0f} GB -> budget {budget_gb:.0f} GB)")

    # Candidates: on the slow drive, hold experts, not the embedding/LM-head shards. Rank by experts per GB.
    cands = [(s, on_primary[s], len(experts_in[s])) for s in on_primary if s not in KEEP_SHARDS and experts_in.get(s)]
    cands.sort(key=lambda c: c[2] / (c[1] / 1e9), reverse=True)
    plan, used = [], 0.0
    for s, size, n_exp in cands:
        if used + size / 1e9 <= budget_gb:
            plan.append((s, size, n_exp))
            used += size / 1e9

    moved = {s for s, _, _ in plan}
    n_moved_exp = sum(n for _, _, n in plan)
    print(f"\nPlan: move {len(plan)} shard(s), {used:.1f} GB, {n_moved_exp} experts "
          f"({100.0 * n_moved_exp / max(total_experts, 1):.1f}% of all experts)")
    for s, size, n_exp in sorted(plan):
        ls = sorted({l for (l, _) in experts_in[s]})
        print(f"  {s}  {size / 1e9:6.2f} GB  {n_exp:4d} experts  layers {ls[0]}-{ls[-1]}")
    if not plan:
        print("  (nothing fits in the budget)")

    if args.trace and os.path.exists(args.trace):
        tokens = load_trace(args.trace)
        now_map, new_map = {}, {}
        for shard, exps in experts_in.items():
            now_fast = shard in on_secondary
            for k in exps:
                now_map[k] = "fast" if now_fast else "slow"
                new_map[k] = "fast" if (now_fast or shard in moved) else "slow"
        before = simulate(tokens, args.cache_experts, now_map, args.slow_mbps, args.fast_mbps)
        after = simulate(tokens, args.cache_experts, new_map, args.slow_mbps, args.fast_mbps)
        print(f"\nProjected storage time per token (trace replay of {len(tokens)} tokens, blend cache of "
              f"{args.cache_experts} experts, {args.slow_mbps:.0f}/{args.fast_mbps:.0f} MB/s):")
        print(f"  now   : {before:7.0f} ms")
        print(f"  after : {after:7.0f} ms   (saves {before - after:.0f} ms/token, {100 * (before - after) / before:.0f}% of storage time)")
        print("  This is a projection, not a measurement; re-run `asema bench` after any move.")
    else:
        print("\n(pass --trace <file> to project the gain)")

    if not args.execute:
        print("\nDRY RUN: nothing was copied or deleted. Re-run with --execute to copy (sources are kept).")
        return

    print("\nEXECUTING: copying with verification. Close other heavy programs; this reads/writes several GB.")
    for s, size, _ in sorted(plan):
        src, dst = os.path.join(primary, s), os.path.join(secondary, s)
        part = dst + ".part"
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            print(f"  {s}: already present on secondary, skipping copy")
            if args.delete_source and sha256(src, "source") != sha256(dst, "existing copy"):
                sys.exit(f"  checksum mismatch for {s}; NOT deleting the source.")
        else:
            print(f"  {s}: copying {size / 1e9:.2f} GB ...")
            shutil.copyfile(src, part)
            if os.path.getsize(part) != size:
                sys.exit(f"  size mismatch for {s}; aborting (source untouched).")
            if sha256(src, "source") != sha256(part, "copy"):
                os.remove(part)
                sys.exit(f"  checksum mismatch for {s}; removed the bad copy, source untouched.")
            os.replace(part, dst)
            print(f"  {s}: verified and in place")
        if args.delete_source:
            os.remove(src)
            print(f"  {s}: source deleted")
    print("\nDone." + ("" if args.delete_source else " Originals were kept; delete them (or re-run with --delete-source) to free space."))


if __name__ == "__main__":
    main()
