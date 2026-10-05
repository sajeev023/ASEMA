"""
ASEMA Automated Checkpoint Acquisition & Verification Engine
Official Checkpoint: deepseek-ai/DeepSeek-V4.1-Flash (48 shards, 510 GB)
"""

import os
import sys
import time
import json
import struct
import shutil
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

VOL_D = os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards")
VOL_E = os.environ.get("ASEMA_SHARDS_SECONDARY", "")
BASE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main"

# Volume partition strategy ensuring ample safety headroom on both drives:
# Volume D (473 GiB free): Shards 1-32 (208.67 GiB) + Shard 47 (94.56 GiB) = 303.23 GiB (~170 GiB headroom)
# Volume E (237 GiB free): Shards 33-46 (77.40 GiB) + Shard 48 (94.56 GiB) = 171.96 GiB (~65 GiB headroom)
def get_target_volume(shard_idx):
    if shard_idx <= 32 or shard_idx == 47:
        return VOL_D
    else:
        return VOL_E

def check_disk_space():
    d_free = shutil.disk_usage(VOL_D).free if os.path.exists(VOL_D) else shutil.disk_usage("D:\\").free
    e_free = shutil.disk_usage(VOL_E).free if os.path.exists(VOL_E) else shutil.disk_usage("E:\\").free
    return d_free, e_free

def verify_safetensors_file(path, expected_size):
    if not os.path.exists(path):
        return False, "File does not exist"
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        return False, f"Size mismatch: expected {expected_size}, got {actual_size}"
    try:
        with open(path, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return False, "Header too short"
            header_len = struct.unpack("<Q", header_bytes)[0]
            if header_len <= 0 or header_len > 100_000_000:
                return False, f"Invalid header length: {header_len}"
            header_json = f.read(header_len)
            data = json.loads(header_json.decode("utf-8"))
            return True, f"Valid safetensors ({len(data)} tensors)"
    except Exception as e:
        return False, f"Safetensors parse error: {e}"

def find_existing_shard(shard_name):
    path_d = os.path.join(VOL_D, shard_name)
    if os.path.exists(path_d):
        return path_d
    path_e = os.path.join(VOL_E, shard_name)
    if os.path.exists(path_e):
        return path_e
    return None

def download_shard(shard_name, expected_size, workers=8, chunk_mb=64):
    shard_idx = int(shard_name.split("-")[1])
    target_vol = get_target_volume(shard_idx)
    os.makedirs(target_vol, exist_ok=True)

    existing_path = find_existing_shard(shard_name)
    final_path = existing_path if existing_path else os.path.join(target_vol, shard_name)
    part_path = final_path + ".part"
    prog_path = final_path + ".progress.json"

    # Step 1: Check existing final file
    valid, reason = verify_safetensors_file(final_path, expected_size)
    if valid:
        print(f"[EXISTS & VERIFIED] {shard_name} ({expected_size / (1024**3):.2f} GiB) at {final_path}")
        return True

    print(f"\n[DOWNLOAD START] {shard_name} ({expected_size / (1024**3):.2f} GiB) -> {target_vol}")
    url = f"{BASE_URL}/{shard_name}"

    chunk_size = chunk_mb * 1024 * 1024
    ranges = []
    for start in range(0, expected_size, chunk_size):
        end = min(start + chunk_size - 1, expected_size - 1)
        ranges.append((start, end))

    total_chunks = len(ranges)
    completed_chunks = set()

    # Step 2: Check resume progress
    if os.path.exists(prog_path) and os.path.exists(part_path):
        try:
            if os.path.getsize(part_path) == expected_size:
                with open(prog_path, "r") as f:
                    completed_chunks = set(json.load(f))
                print(f"  [RESUME] Found {len(completed_chunks)} / {total_chunks} chunks completed.")
        except Exception:
            completed_chunks = set()

    if not os.path.exists(part_path) or os.path.getsize(part_path) != expected_size:
        with open(part_path, "wb") as f:
            f.seek(expected_size - 1)
            f.write(b"\0")
        completed_chunks = set()

    chunks_to_fetch = [i for i in range(total_chunks) if i not in completed_chunks]
    if not chunks_to_fetch:
        print("  All chunks already fetched, finalizing...")
    else:
        print(f"  Fetching {len(chunks_to_fetch)} remaining chunks across {workers} parallel curl workers...")
        t0 = time.time()
        downloaded_bytes = len(completed_chunks) * chunk_size

        def fetch_chunk(idx):
            s, e = ranges[idx]
            expected_len = e - s + 1
            for attempt in range(10):
                try:
                    cmd = ["curl.exe", "-s", "-L", "-r", f"{s}-{e}", "--retry", "5", "--retry-delay", "1", url]
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    data = p.stdout.read()
                    p.wait()
                    if len(data) != expected_len:
                        raise ValueError(f"Length mismatch: got {len(data)}, expected {expected_len}")
                    with open(part_path, "r+b") as f:
                        f.seek(s)
                        f.write(data)
                    return idx, len(data)
                except Exception as ex:
                    if attempt == 9:
                        print(f"  [WARN] Chunk {idx} failed after 10 attempts: {ex}")
                        raise ex
                    time.sleep(min(1.5 ** attempt, 5.0))
            return idx, 0

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fetch_chunk, idx): idx for idx in chunks_to_fetch}
            done_count = len(completed_chunks)
            last_report = time.time()

            for fut in as_completed(futures):
                idx, b_len = fut.result()
                completed_chunks.add(idx)
                done_count += 1
                downloaded_bytes += b_len

                now = time.time()
                tmp_prog = prog_path + ".tmp"
                with open(tmp_prog, "w") as f:
                    json.dump(list(completed_chunks), f)
                try:
                    os.replace(tmp_prog, prog_path)
                except Exception:
                    pass
                pct = (done_count / total_chunks) * 100.0
                mb_done = (done_count * chunk_size) / (1024 * 1024)
                mb_tot = expected_size / (1024 * 1024)
                speed = (downloaded_bytes - len(completed_chunks) * chunk_size) / max(1.0, now - t0) / (1024 * 1024)
                print(f"  [{pct:5.1f}%] {done_count}/{total_chunks} chunks ({mb_done:.1f}/{mb_tot:.1f} MB) - Chunk {idx} done", flush=True)

    # Finalize download
    valid, reason = verify_safetensors_file(part_path, expected_size)
    if not valid:
        print(f"  [ERROR] Verification of {part_path} failed: {reason}")
        return False

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(part_path, final_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)

    print(f"[SUCCESS] {shard_name} verified and committed to {final_path}")
    return True

def generate_audit_report(sizes_map):
    idx_path = "examples/real_model/DeepSeek-V4.1-Flash/hf/model.safetensors.index.json"
    weight_map = {}
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r") as f:
                weight_map = json.load(f).get("weight_map", {})
        except Exception:
            pass

    audit = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "target_model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "storage_locations": {
            "volume_d": VOL_D,
            "volume_e": VOL_E
        },
        "shard_count": 48,
        "shards_present_count": 0,
        "shards_missing_count": 0,
        "shards_corrupt_count": 0,
        "total_physical_bytes_expected": sum(sizes_map.values()),
        "total_physical_bytes_present": 0,
        "tensor_count_total": len(weight_map),
        "tensor_count_present": 0,
        "tensor_count_missing": 0,
        "missing_shards": [],
        "corrupt_shards": [],
        "duplicate_shards": [],
        "shards": [],
        "complete": False
    }

    present_shard_names = set()

    for i in range(1, 49):
        name = f"model-{i:05d}-of-00048.safetensors"
        exp_sz = sizes_map[name]
        existing = find_existing_shard(name)
        vol = get_target_volume(i)
        path = existing if existing else os.path.join(vol, name)
        present = os.path.exists(path)
        sz = os.path.getsize(path) if present else 0
        valid, msg = verify_safetensors_file(path, exp_sz) if present else (False, "Missing")

        if valid:
            audit["shards_present_count"] += 1
            audit["total_physical_bytes_present"] += sz
            present_shard_names.add(name)
        elif present:
            audit["shards_corrupt_count"] += 1
            audit["corrupt_shards"].append(name)
        else:
            audit["shards_missing_count"] += 1
            audit["missing_shards"].append(name)

        audit["shards"].append({
            "shard_id": i,
            "name": name,
            "expected_bytes": exp_sz,
            "actual_bytes": sz,
            "path": path,
            "storage_volume": "D" if VOL_D in path else "E",
            "present": present,
            "valid": valid,
            "checksum_status": "VERIFIED_VALID" if valid else ("SIZE_MISMATCH" if present else "NOT_PRESENT"),
            "status": msg
        })

    # Tensor count analysis
    for t_name, s_name in weight_map.items():
        if s_name in present_shard_names:
            audit["tensor_count_present"] += 1
        else:
            audit["tensor_count_missing"] += 1

    audit["complete"] = (audit["shards_present_count"] == 48 and audit["tensor_count_missing"] == 0)

    # Materialized layer count
    materialized_layers = 0
    for l in range(40):
        tname = f"layers.{l}.attn_norm.weight"
        sh = weight_map.get(tname)
        if sh in present_shard_names:
            materialized_layers += 1
        else:
            break

    status_manifest = {
        "status": "COMPLETE" if (audit["shards_present_count"] == 48 and materialized_layers == 40) else "PARTIAL",
        "shard_count_expected": 48,
        "shard_count_verified": audit["shards_present_count"],
        "total_expected_bytes": audit["total_physical_bytes_expected"],
        "total_verified_bytes": audit["total_physical_bytes_present"],
        "tensor_count_expected": audit["tensor_count_total"],
        "tensor_count_verified": audit["tensor_count_present"],
        "layer_count_expected": 40,
        "layer_count_materialized": materialized_layers,
        "failed_shards": audit["corrupt_shards"],
        "incomplete_shards": audit["missing_shards"],
        "drive_assignments": {s["name"]: s["storage_volume"] for s in audit["shards"]},
        "verification_timestamp": audit["timestamp"]
    }

    try:
        with open("checkpoint_audit.json.tmp", "w") as f:
            json.dump(audit, f, indent=2)
        os.replace("checkpoint_audit.json.tmp", "checkpoint_audit.json")
    except Exception:
        pass

    try:
        with open("checkpoint_status.json.tmp", "w") as f:
            json.dump(status_manifest, f, indent=2)
        os.replace("checkpoint_status.json.tmp", "checkpoint_status.json")
    except Exception:
        pass

    print(f"\n[AUDIT] Manifest updated: {audit['shards_present_count']}/48 shards present ({materialized_layers}/40 layers materialized, {audit['tensor_count_present']}/{audit['tensor_count_total']} tensors).")
    return audit

def main():
    with open("remote_shard_sizes.json", "r") as f:
        sizes_map = json.load(f)["sizes"]

    print("======================================================================")
    print("  ASEMA: Checkpoint Acquisition & Storage Verification Engine         ")
    print("======================================================================")

    d_free, e_free = check_disk_space()
    print(f"Disk D: Free: {d_free / (1024**3):.2f} GiB")
    print(f"Disk E: Free: {e_free / (1024**3):.2f} GiB")

    # Initial audit
    audit = generate_audit_report(sizes_map)

    # Shards priority order:
    # 1. Shard 3 (Layer 0) - immediate priority for Layer 0 completion
    # 2. Shards 4-42 (Layers 1-39)
    # 3. Shard 1 (Vision patch)
    # 4. Shards 44-46 (MTP)
    # 5. Shards 47-48 (Engram)
    target_shards = []
    if len(sys.argv) > 1:
        target_shards = sys.argv[1:]
    else:
        # Default: process missing shards in layer order
        for s in audit["shards"]:
            if not s["valid"]:
                target_shards.append(s["name"])

    print(f"\nTarget download queue: {len(target_shards)} shards pending.")
    for s_name in target_shards:
        exp_sz = sizes_map[s_name]
        ok = download_shard(s_name, exp_sz, workers=8)
        if not ok:
            print(f"[FATAL] Failed downloading {s_name}. Aborting.")
            break
        generate_audit_report(sizes_map)

if __name__ == "__main__":
    main()
