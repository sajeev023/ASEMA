# Hardware

## Reference machine (the only one measured)

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 7 5700X (8 cores / 16 threads), AVX2 + FMA |
| RAM | 32 GB DDR4 |
| GPU | AMD Radeon RX 580 8 GB (Direct3D 11 compute, "DirectCompute") |
| Storage A | SATA SSD, about 440 MB/s random-read, holds 34 of 48 shards (304 GB) |
| Storage B | NVMe SSD (PCIe 3.0 x4), about 2,800 MB/s random-read, holds 14 of 48 shards (171 GB) |
| OS | Windows 11 x64 |

## What the engine needs

- **RAM:** the process working set measured 12-14 GB on the reference machine. Most of that is
  memory-mapped checkpoint pages (reclaimable by Windows) plus a bounded expert cache (grows from
  2 GB up to 6 GB by default, only while Windows available memory stays healthy). A 16 GB machine
  has not been tested.
- **VRAM:** about 1.7 GB allocated (96 expert slots plus scratch buffers). The configured ceiling
  is 5 GB.
- **Storage:** about 510 GB for the checkpoint. Storage throughput dominates decode speed;
  NVMe is strongly recommended (PERFORMANCE.md).
- **GPU:** needs Direct3D 11 compute shader 5.0 support. Without it a slower CPU path is used.

## Desktop responsiveness

A governor reduces the expert cache and I/O concurrency when Windows available memory falls
(thresholds are configurable, see the README). On the reference machine, with a browser and other
applications open, available memory stayed above about 6-7 GB during runs and the governor reacted
as designed. Behaviour with video playback or two monitors under load was not instrumented here.
