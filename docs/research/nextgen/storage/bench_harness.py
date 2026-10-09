"""Phase-1 benchmark harness.  One run = `asema.exe benchmark --tokens N` with a fixed prompt, while Windows
performance counters (typeperf, 1 s) record per-physical-disk read bytes, total CPU and GPU-engine utilisation.
No engine changes: per-drive bytes come from the OS, not from ASEMA telemetry.

usage (project root):  python -I experiments/nextgen/storage/bench_harness.py <label> <tokens> [KEY=VAL env ...]
Writes reports/storage/<label>.txt (engine output) and reports/storage/<label>.json (parsed result)."""
import csv, json, os, re, subprocess, sys, threading, time

PROMPT = "Write a short story about a lighthouse keeper."
EXE = os.path.join("build", "asema.exe")
OUT = os.path.join("reports", "storage")

def typeperf(counters, path):
    return subprocess.Popen(["typeperf", *counters, "-si", "1", "-f", "CSV", "-o", path, "-y"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def integrate(path, want):
    """Sum 1-second samples (bytes/sec * 1 s) per column whose header matches a key in `want`."""
    tot = {k: 0.0 for k in want}; peak = {k: 0.0 for k in want}; n = 0
    try:
        rows = list(csv.reader(open(path, newline="", encoding="utf-8", errors="replace")))
    except OSError:
        return tot, peak, 0
    if len(rows) < 2: return tot, peak, 0
    hdr = rows[0]
    for r in rows[1:]:
        n += 1
        for k, pat in want.items():
            s = 0.0
            for i, h in enumerate(hdr):
                if i and re.search(pat, h):
                    try: s += float(r[i])
                    except (ValueError, IndexError): pass
            tot[k] += s; peak[k] = max(peak[k], s)
    return tot, peak, n

def main():
    label, tokens = sys.argv[1], int(sys.argv[2])
    env = dict(os.environ)
    for kv in sys.argv[3:]:
        k, v = kv.split("=", 1); env[k] = v
    os.makedirs(OUT, exist_ok=True)
    disk_csv = os.path.join(OUT, label + ".disk.csv"); gpu_csv = os.path.join(OUT, label + ".gpu.csv")
    for p in (disk_csv, gpu_csv):
        if os.path.exists(p): os.remove(p)
    disk = typeperf([r"\PhysicalDisk(0 D:)\Disk Read Bytes/sec", r"\PhysicalDisk(1 C:)\Disk Read Bytes/sec",
                     r"\PhysicalDisk(2 E:)\Disk Read Bytes/sec", r"\Processor(_Total)\% Processor Time"], disk_csv)
    t0 = time.time()
    p = subprocess.Popen([EXE, "benchmark", "--tokens", str(tokens), PROMPT], env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    gpu = None
    chunks = []
    # drain the engine's stdout continuously: a full pipe would block the engine and silently stall the benchmark
    drain = threading.Thread(target=lambda: [chunks.append(l) for l in p.stdout], daemon=True)
    drain.start()
    # GPU-engine instances are per pid; start the GPU sampler once the process has had time to create its device
    while p.poll() is None:
        time.sleep(1.0)
        if gpu is None and time.time() - t0 > 25:
            gpu = typeperf([rf"\GPU Engine(pid_{p.pid}*)\Utilization Percentage"], gpu_csv)
    drain.join(); out = "".join(chunks); wall = time.time() - t0
    for pr in (disk, gpu):
        if pr: pr.terminate()
    time.sleep(1.0)
    open(os.path.join(OUT, label + ".txt"), "w", encoding="utf-8").write(out)
    tot, peak, ns = integrate(disk_csv, {"D": r"0 D:", "C": r"1 C:", "E": r"2 E:"})
    gtot, gpeak, gn = integrate(gpu_csv, {"gpu_sum": r"Utilization"})
    cpu_tot, _, _ = integrate(disk_csv, {"cpu": r"Processor"})

    def f(pat, cast=float):
        m = re.search(pat, out); return cast(m.group(1)) if m else None
    res = {
        "label": label, "tokens_requested": tokens, "wall_s": wall,
        "tokens_generated": f(r"tokens generated:\s+(\d+)", int),
        "warm_ms_per_token": f(r"warm ms/token \(mean\):\s+([\d.]+)"),
        "warm_tok_s": f(r"warm tokens/sec:\s+([\d.]+)"),
        "p50_ms": f(r"warm p50 / p95 ms:\s+([\d.]+)"), "p95_ms": f(r"warm p50 / p95 ms:\s+[\d.]+ / ([\d.]+)"),
        "cold_first_ms": f(r"cold first token \(incl\. prefill\):\s+([\d.]+)"),
        "total_wall_ms_engine": f(r"total wall time:\s+([\d.]+)"),
        "engine_storage_MB": f(r"storage bytes read:\s+(\d+) MB", int),
        "cpu_cores_busy": f(r"avg CPU cores busy:\s+([\d.]+)"),
        "os_read_MB": {k: tot[k] / 1048576.0 for k in tot}, "os_read_peak_MBps": {k: peak[k] / 1048576.0 for k in peak},
        "os_cpu_mean_pct": (cpu_tot["cpu"] / ns) if ns else None,
        "gpu_util_sum_mean_pct": (gtot["gpu_sum"] / gn) if gn else None,
        "gpu_util_sum_peak_pct": gpeak["gpu_sum"] if gn else None,
        "samples": ns,
    }
    json.dump(res, open(os.path.join(OUT, label + ".json"), "w"), indent=1)
    print(json.dumps(res))

if __name__ == "__main__":
    main()
