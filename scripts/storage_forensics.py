import os
import sys
import json
import ctypes
import glob

def get_drive_space(drive):
    free_bytes = ctypes.c_ulonglong(0)
    total_bytes = ctypes.c_ulonglong(0)
    total_free_bytes = ctypes.c_ulonglong(0)
    ret = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(drive + ":\\"),
        ctypes.byref(free_bytes),
        ctypes.byref(total_bytes),
        ctypes.byref(total_free_bytes)
    )
    return total_bytes.value, free_bytes.value

d_total, d_free = get_drive_space('D')
e_total, e_free = get_drive_space('E')

print("=================================================================")
print("             DUAL NVME PHYSICAL STORAGE FORENSICS               ")
print("=================================================================")
print(f"Drive D: Total Capacity = {d_total:,} bytes ({d_total / (1024**3):.4f} GiB | {d_total / 1e9:.4f} GB)")
print(f"Drive D: Free Space      = {d_free:,} bytes ({d_free / (1024**3):.4f} GiB | {d_free / 1e9:.4f} GB)")
print(f"Drive E: Total Capacity = {e_total:,} bytes ({e_total / (1024**3):.4f} GiB | {e_total / 1e9:.4f} GB)")
print(f"Drive E: Free Space      = {e_free:,} bytes ({e_free / (1024**3):.4f} GiB | {e_free / 1e9:.4f} GB)")

total_free = d_free + e_free
target_checkpoint_bytes = 510296708312

print("-----------------------------------------------------------------")
print(f"Combined Free Space      = {total_free:,} bytes ({total_free / (1024**3):.4f} GiB | {total_free / 1e9:.4f} GB)")
print(f"Target Checkpoint Bytes  = {target_checkpoint_bytes:,} bytes ({target_checkpoint_bytes / (1024**3):.4f} GiB | {target_checkpoint_bytes / 1e9:.4f} GB)")
print("-----------------------------------------------------------------")

def catalog_dir(base_path):
    catalog = {
        'safetensors': [],
        'part': [],
        'progress': [],
        'cache': [],
        'bin_weights': [],
        'other': []
    }
    if not os.path.exists(base_path):
        return catalog
    for root, dirs, files in os.walk(base_path):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            entry = {'path': fp, 'size': sz, 'name': f, 'rel': os.path.relpath(fp, base_path)}
            if f.endswith('.safetensors'):
                catalog['safetensors'].append(entry)
            elif f.endswith('.part'):
                catalog['part'].append(entry)
            elif f.endswith('.progress.json'):
                catalog['progress'].append(entry)
            elif '.cache' in root or f.endswith('.incomplete') or f.endswith('.lock'):
                catalog['cache'].append(entry)
            elif f.endswith('.bin'):
                catalog['bin_weights'].append(entry)
            else:
                catalog['other'].append(entry)
    return catalog

cat_d = catalog_dir(os.environ.get("ASEMA_SHARDS_PRIMARY", "models/DeepSeek-V4.1-Flash/shards"))
cat_e = catalog_dir(os.environ.get("ASEMA_SHARDS_SECONDARY", "")) if os.environ.get("ASEMA_SHARDS_SECONDARY", "") else {}

# Load remote shard sizes
with open('remote_shard_sizes.json', 'r') as f:
    remote_data = json.load(f)
    remote_sizes = remote_data.get('sizes', remote_data)

# Also check for any other directories on D: and E: matching deepseek, checkpoint, etc.
def find_checkpoint_dirs(drive_letter):
    matches = []
    root = drive_letter + ':\\'
    try:
        for entry in os.listdir(root):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and any(k in entry.lower() for k in ['deepseek', 'checkpoint', 'safetensor', 'asema', 'huggingface']):
                matches.append(full)
    except Exception as e:
        pass
    return matches

dirs_d = find_checkpoint_dirs('D')
dirs_e = find_checkpoint_dirs('E')

print(f"Discovered potential checkpoint directories on D: {dirs_d}")
print(f"Discovered potential checkpoint directories on E: {dirs_e}")

# Check verified safetensors on D and E
resident_verified = []
resident_unverified = []

all_safetensors = [(entry, 'D') for entry in cat_d['safetensors']] + [(entry, 'E') for entry in cat_e['safetensors']]

for entry, drive in all_safetensors:
    fname = entry['name']
    actual_sz = entry['size']
    expected_sz = remote_sizes.get(fname, None)
    if expected_sz is not None and actual_sz == expected_sz:
        resident_verified.append((fname, drive, actual_sz, entry['path']))
    else:
        resident_unverified.append((fname, drive, actual_sz, expected_sz, entry['path']))

print("\n--- 1. VERIFIED CHECKPOINT RESIDENT SHARDS ---")
verified_bytes = sum(x[2] for x in resident_verified)
print(f"Total Verified Shards Present: {len(resident_verified)} / 48")
print(f"Total Verified Bytes Present : {verified_bytes:,} bytes ({verified_bytes / (1024**3):.4f} GiB)")
for name, drive, sz, path in sorted(resident_verified):
    print(f"  [VERIFIED] {drive}: {name:36s} | {sz:14,d} bytes")

print("\n--- 2. INCOMPLETE / IN-PROGRESS (.part) FILES ---")
all_parts = [(entry, 'D') for entry in cat_d['part']] + [(entry, 'E') for entry in cat_e['part']]
part_bytes = sum(x[0]['size'] for x in all_parts)
print(f"Total In-Progress .part Files: {len(all_parts)}, consuming {part_bytes:,} bytes ({part_bytes / (1024**3):.4f} GiB)")
for entry, drive in all_parts:
    print(f"  [IN-PROGRESS] {drive}: {entry['name']:40s} | {entry['size']:14,d} bytes | {entry['path']}")

print("\n--- 3. CACHE & INCOMPLETE ARTIFACTS ---")
all_cache = [(entry, 'D') for entry in cat_d['cache']] + [(entry, 'E') for entry in cat_e['cache']]
cache_bytes = sum(x[0]['size'] for x in all_cache)
print(f"Total Cache Files: {len(all_cache)}, consuming {cache_bytes:,} bytes")
for entry, drive in all_cache:
    print(f"  [CACHE] {drive}: {entry['name']} | {entry['size']:,} bytes | {entry['path']}")

print("\n--- 4. EXTRACTED BINARY ASSETS (Legacy/Pre-extracted Layer 0 / Experts) ---")
all_bins = [(entry, 'D') for entry in cat_d['bin_weights']] + [(entry, 'E') for entry in cat_e['bin_weights']]
bin_bytes = sum(x[0]['size'] for x in all_bins)
print(f"Total Extracted .bin Files: {len(all_bins)}, consuming {bin_bytes:,} bytes ({bin_bytes / (1024**3):.4f} GiB)")

print("\n--- 5. EXACT STORAGE ACCOUNTING ---")
# If the entire checkpoint is 510,296,708,312 bytes:
# We already have `verified_bytes` resident.
remaining_checkpoint_bytes = target_checkpoint_bytes - verified_bytes
print(f"Full Checkpoint Size Required: {target_checkpoint_bytes:,} bytes ({target_checkpoint_bytes / (1024**3):.4f} GiB)")
print(f"Already Verified On Disk     : {verified_bytes:,} bytes ({verified_bytes / (1024**3):.4f} GiB)")
print(f"Remaining Bytes To Acquire   : {remaining_checkpoint_bytes:,} bytes ({remaining_checkpoint_bytes / (1024**3):.4f} GiB)")

# Note: The .part files currently occupy disk space. When they complete, their .part file is renamed directly to .safetensors.
# Downloader allocates 0 additional temporary working space because it preallocates the file as .part and writes chunks in-place, then renames to .safetensors!
# That means the part bytes ALREADY count against current used disk space.
# Therefore, Net new space required from current free space:
net_new_space_needed = remaining_checkpoint_bytes - part_bytes
print(f"Space Currently In .part Files: {part_bytes:,} bytes ({part_bytes / (1024**3):.4f} GiB)")
print(f"Net Additional Free Space Needed To Reach 510 GB Checkpoint: {net_new_space_needed:,} bytes ({net_new_space_needed / (1024**3):.4f} GiB)")
print(f"Current Combined Free Space  : {total_free:,} bytes ({total_free / (1024**3):.4f} GiB)")

surplus = total_free - net_new_space_needed
print(f"Storage Margin (Surplus)     : {surplus:,} bytes ({surplus / (1024**3):.4f} GiB)")
if surplus >= 0:
    print(f"[VERDICT] SUFFICIENT STORAGE PROVEN. Surplus of {surplus / (1024**3):.2f} GiB ({surplus / 1e9:.2f} GB) beyond full 510 GB checkpoint.")
else:
    print(f"[VERDICT] INSUFFICIENT STORAGE. Deficit of {-surplus:,} bytes ({-surplus / (1024**3):.2f} GiB / {-surplus / 1e9:.2f} GB).")
