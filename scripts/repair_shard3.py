"""
Script to repair model-00003-of-00048.safetensors by fetching all zero/unpopulated chunks.
"""
import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

SHARD_PATH = os.path.join(os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards"), "model-00003-of-00048.safetensors")
BASE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main"
SHARD_NAME = "model-00003-of-00048.safetensors"
URL = f"{BASE_URL}/{SHARD_NAME}"

CHUNK_SIZE = 64 * 1024 * 1024
EXPECTED_SIZE = 7389759032

def main():
    if not os.path.exists(SHARD_PATH):
        print(f"Error: {SHARD_PATH} does not exist!")
        return 1

    actual_size = os.path.getsize(SHARD_PATH)
    if actual_size != EXPECTED_SIZE:
        print(f"Size mismatch: {actual_size} vs {EXPECTED_SIZE}")
        return 1

    ranges = []
    for start in range(0, EXPECTED_SIZE, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, EXPECTED_SIZE - 1)
        ranges.append((start, end))

    total_chunks = len(ranges)
    print(f"Scanning {total_chunks} chunks of {SHARD_PATH} for zero regions...")

    zero_chunks = []
    with open(SHARD_PATH, "rb") as f:
        for idx, (s, e) in enumerate(ranges):
            f.seek(s)
            d = f.read(e - s + 1)
            # Check if chunk is all zero
            if not any(d):
                zero_chunks.append(idx)

    print(f"Found {len(zero_chunks)} zero chunks out of {total_chunks} to fetch.")
    if not zero_chunks:
        print("Shard 3 is already 100% populated with physical tensors!")
        return 0

    print(f"Fetching {len(zero_chunks)} missing chunks using 4 parallel curl workers...")
    t0 = time.time()

    def fetch_chunk(idx):
        s, e = ranges[idx]
        expected_len = e - s + 1
        for attempt in range(10):
            try:
                cmd = ["curl.exe", "-s", "-L", "-r", f"{s}-{e}", "--retry", "5", "--retry-delay", "1", URL]
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                data = p.stdout.read()
                p.wait()
                if len(data) != expected_len:
                    raise ValueError(f"Length mismatch: got {len(data)}, expected {expected_len}")
                with open(SHARD_PATH, "r+b") as f:
                    f.seek(s)
                    f.write(data)
                return idx, len(data)
            except Exception as ex:
                if attempt == 9:
                    print(f"Chunk {idx} failed after 10 attempts: {ex}")
                    raise ex
                time.sleep(min(1.5 ** attempt, 5.0))
        return idx, 0

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_chunk, idx): idx for idx in zero_chunks}
        done = 0
        total_to_fetch = len(zero_chunks)
        for fut in as_completed(futures):
            idx, blen = fut.result()
            done += 1
            elapsed = time.time() - t0
            pct = (done / total_to_fetch) * 100.0
            print(f"[{pct:5.1f}%] {done}/{total_to_fetch} chunks repaired - Chunk {idx} ({blen} bytes) committed", flush=True)

    print(f"Shard 3 successfully repaired in {time.time() - t0:.1f}s!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
