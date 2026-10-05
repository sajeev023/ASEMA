import os
import sys
import time
import struct
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TARGET_DIR = os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards")
BASE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main"

def download_file(filename, workers=16, chunk_mb=8):
    os.makedirs(TARGET_DIR, exist_ok=True)
    out_path = os.path.join(TARGET_DIR, filename)
    part_path = out_path + ".part"
    prog_path = out_path + ".progress.json"
    url = f"{BASE_URL}/{filename}"

    # Get total size
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        total_size = int(resp.headers.get("Content-Length"))

    # Check if already fully downloaded and valid
    if os.path.exists(out_path) and os.path.getsize(out_path) == total_size:
        try:
            with open(out_path, "rb") as f:
                header_bytes = f.read(8)
                if len(header_bytes) == 8:
                    hlen = struct.unpack("<Q", header_bytes)[0]
                    if 0 < hlen < 50_000_000:
                        hdata = f.read(hlen)
                        json.loads(hdata.decode("utf-8"))
                        print(f"[EXISTS & VALID] {filename} is complete ({total_size / (1024*1024):.2f} MB)")
                        return out_path
        except Exception:
            pass

    print(f"[START] Downloading {filename}: {total_size / (1024*1024):.2f} MB across {workers} workers...")
    t0 = time.time()

    chunk_size = chunk_mb * 1024 * 1024
    ranges = []
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size - 1, total_size - 1)
        ranges.append((start, end))

    total_chunks = len(ranges)
    completed_chunks = set()

    if os.path.exists(prog_path) and os.path.exists(part_path) and os.path.getsize(part_path) == total_size:
        try:
            with open(prog_path, "r") as f:
                completed_chunks = set(json.load(f))
            print(f"  [RESUME] Found {len(completed_chunks)} / {total_chunks} chunks already downloaded.")
        except Exception:
            completed_chunks = set()

    if not os.path.exists(part_path) or os.path.getsize(part_path) != total_size:
        with open(part_path, "wb") as f:
            f.seek(total_size - 1)
            f.write(b"\0")
        completed_chunks = set()

    chunks_to_fetch = [i for i in range(total_chunks) if i not in completed_chunks]
    downloaded_bytes = len(completed_chunks) * chunk_size

    def fetch_chunk(idx):
        s, e = ranges[idx]
        r_req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={s}-{e}", "User-Agent": "Mozilla/5.0"}
        )
        for attempt in range(20):
            try:
                with urllib.request.urlopen(r_req, timeout=60) as resp:
                    chunk = resp.read()
                    with open(part_path, "r+b") as f:
                        f.seek(s)
                        f.write(chunk)
                    return idx, len(chunk)
            except Exception as ex:
                if attempt == 19:
                    print(f"  Fatal error on chunk {idx} ({s}-{e}): {ex}")
                    raise ex
                time.sleep(min(1.5 ** attempt, 8.0))
        return idx, 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch_chunk, idx): idx for idx in chunks_to_fetch}
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            idx, b_len = fut.result()
            completed_chunks.add(idx)
            downloaded_bytes += b_len
            with open(prog_path, "w") as f:
                json.dump(list(completed_chunks), f)
            mb_done = (len(completed_chunks) * chunk_size) / (1024 * 1024)
            pct = (len(completed_chunks) / total_chunks) * 100.0
            print(f"  [{pct:5.1f}%] Finished chunk {len(completed_chunks)}/{total_chunks} ({mb_done:.1f} MB)", flush=True)

    # Validate header before renaming
    with open(part_path, "rb") as f:
        header_bytes = f.read(8)
        hlen = struct.unpack("<Q", header_bytes)[0]
        hdata = f.read(hlen)
        h = json.loads(hdata.decode("utf-8"))
        print(f"  Verified Safetensors Header: {list(h.keys())}")

    # Rename to final
    if os.path.exists(out_path):
        os.remove(out_path)
    os.rename(part_path, out_path)
    if os.path.exists(prog_path):
        os.remove(prog_path)

    print(f"[COMPLETE] {filename} downloaded and verified in {time.time() - t0:.1f}s")
    return out_path

if __name__ == "__main__":
    shards = ["model-00043-of-00048.safetensors", "model-00002-of-00048.safetensors"]
    if len(sys.argv) > 1:
        shards = sys.argv[1:]
    for s in shards:
        download_file(s)
