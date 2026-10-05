import os
import sys
import time
from huggingface_hub import hf_hub_download

REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a"
TARGET_DIR = os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards")

def download_shard(shard_filename):
    os.makedirs(TARGET_DIR, exist_ok=True)
    target_path = os.path.join(TARGET_DIR, shard_filename)
    if os.path.exists(target_path):
        size_gb = os.path.getsize(target_path) / (1024**3)
        print(f"[EXISTS] {shard_filename} already present: {size_gb:.2f} GiB")
        return target_path

    print(f"[DOWNLOADING] {shard_filename} to {TARGET_DIR} ...")
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_filename,
            revision=REVISION,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False
        )
        elapsed = time.time() - t0
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"[SUCCESS] {shard_filename} downloaded in {elapsed:.1f}s ({size_mb / elapsed:.2f} MB/s)")
        return path
    except Exception as e:
        print(f"[ERROR] Failed to download {shard_filename}: {e}")
        return None

if __name__ == "__main__":
    shard = sys.argv[1] if len(sys.argv) > 1 else "model-00003-of-00048.safetensors"
    download_shard(shard)
