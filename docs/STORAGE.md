# Storage layout and expert paging

The checkpoint is about 510 GB (48 shards). It is never loaded into memory as a whole; each token
reads only the bytes it needs.

## What is read per token

- **Dense tensors** (attention projections, router, shared expert, norms): memory-mapped from the
  shards and read in place. After the first pass these live in the OS file cache and are served
  from RAM if Windows keeps them resident.
- **Routed experts**: 6 of 384 per layer, 40 layers -> 240 expert reads of 18.8 MB each
  (1.1 MB of scales + 17.7 MB of FP4 weights, contiguous in the shard), about 4.5 GB per token if
  nothing is cached. Reads go through a bounded LRU host cache; typical hit rate measured here: 34%.

## Using two volumes

`shards_primary` and `shards_secondary` (see MODEL_SETUP.md) may point to different drives. The
loader resolves each tensor to the directory that contains its shard and issues reads for different
drives concurrently, so a layer's misses on two drives overlap.

On the reference machine 34 shards (304 GB) are on a SATA SSD and 14 shards (171 GB) on an NVMe
drive. Measured random 18.8 MB read throughput: SATA about 440 MB/s (flat from 1 to 8 concurrent
reads), NVMe about 2,800 MB/s. Because most experts live on the SATA drive, it sets the pace
(PERFORMANCE.md). There is no combined-bandwidth guarantee: throughput is the sum over drives only
if a layer's misses are spread over both.

## Free space

Plan for the checkpoint size plus headroom on whichever drives hold it. ASEMA only reads the shards
and never writes to them.

## Measuring your own drives

Any tool that issues random 16-32 MB reads at several queue depths will do. The `asema bench`
command reports bytes read, read operations and cache hit rate for a real generation.

## Rebalancing shards between drives

`scripts/plan_shard_move.py` plans (and, with `--execute`, performs) moving whole expert shards from the slower
drive to the faster one when the fast drive has free space. It is read-only by default, prints the shards chosen
and a trace-based projection of the saving, copies with size and SHA-256 verification, and keeps the originals
unless `--delete-source` is given. On the reference machine, moving 8 layer shards (59 GB) to the NVMe measured a
17% faster warm decode (see PERFORMANCE.md). If the same shard name exists in both directories, the secondary
directory wins.

After a verified move, `--delete-duplicates` removes the primary-drive copy of every shard that exists on both drives, but only after re-hashing both copies and finding them identical (it refuses to delete on any mismatch). Without it, duplicates are harmless but use space.
