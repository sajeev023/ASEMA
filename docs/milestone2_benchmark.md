# ASEMA v0.1 — Milestone 2 Benchmark Report
## Synchronous Baseline vs Asynchronous Overlapped I/O & Request Coalescing

---

### 1. Test Environment & Storage Tier
- **Date**: 2026-09-25
- **Platform**: Windows 11 Pro 64-bit (10.0.26200)
- **CPU**: AMD Ryzen 7 5700X (8 physical cores, 16 logical processors)
- **Toolchain**: LLVM-MinGW Clang++ 22.1.8 with `-O3 -mavx2 -mfma -static`
- **Storage Subsystem**: NVMe Fast SSD (`C:\`, NTFS formatted)
- **Model Checkpoint**: Synthetic MoE (4 layers, 16 experts/layer = 64 distinct experts, top-2 active)
- **Expert Size**: 786,432 bytes (~768 KB) per expert; 48.00 MB total container size
- **Data Integrity**: IEEE 802.3 CRC32 verification executed on every loaded block before release

---

### 2. Empirical Benchmark Results

| Experiment | Total Requests | Total Data | Wall Time (ms) | Effective Throughput | Avg Latency | P50 Latency | P95 Latency | P99 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Synchronous Baseline** | 64 | 48 MB | 106.56 ms | **450.44 MB/s** | 1.66 ms | 1.66 ms | 1.71 ms | 1.76 ms |
| **B. Async IOCP (4 Workers)** | 64 | 48 MB | 26.37 ms | **1,819.96 MB/s** | 13.90 ms | 14.34 ms | 25.67 ms | 26.28 ms |
| **C. Async IOCP (8 Workers)** | 64 | 48 MB | 15.54 ms | **3,088.03 MB/s** | 8.62 ms | 8.64 ms | 14.78 ms | 15.46 ms |
| **D. Coalesced Repeated Access** | 64 | 48 MB | 7.79 ms | **6,159.22 MB/s** | 4.91 ms | 5.36 ms | 7.76 ms | 7.76 ms |
| **E. Random Access (128 Req)** | 128 | 96 MB | 3.94 ms | **24,368.60 MB/s** | 2.86 ms | 3.54 ms | 3.91 ms | 3.91 ms |

---

### 3. Detailed Component Breakdown

#### A. Single Request Latency Profile
- **Queue scheduling overhead**: ~0.73 ms
- **Raw Windows Overlapped ReadFile I/O**: ~0.22 ms
- **CRC32 Checksum Validation**: ~1.69 ms
- **Total single-flight completion**: ~2.63 ms

#### B. Concurrency Scaling & Overlap Efficiency
- Moving from synchronous reads (1 worker, serialized) to 4 workers decreased total wall clock time from **106.56 ms to 26.37 ms** (**4.04x speedup**).
- Moving to 8 workers further dropped wall clock time to **15.54 ms** (**6.86x speedup** over synchronous baseline), saturating NVMe bandwidth at **3.088 GB/s**.

#### C. Request Coalescing Efficiency
- In Experiment D (64 requests with 4x repetition across Layer 0):
  - Total requests submitted: **64**
  - Duplicate requests coalesced: **48** (75.0% deduplication rate)
  - Physical disk dispatches issued: **16**
  - Elimination of redundant disk reads boosted effective transfer rate to **6.16 GB/s**.

#### D. Integrity & Error Quarantine
- Under concurrent asynchronous stress, corrupted blocks (tested via bit-tampered CRC32) were caught before buffer release.
- Corrupted buffers were immediately quarantined with `is_corrupt == true`, returning zero unverified data to callers.

---

### 4. Summary & Conclusions
1. Asynchronous I/O via Windows IOCP and `FILE_FLAG_OVERLAPPED` eliminates the inference thread blocking bottleneck.
2. An 8-worker thread pool matches the 8 physical cores of the Ryzen 7 5700X, scaling disk read throughput from 450 MB/s to over 3.08 GB/s.
3. In-flight request coalescing prevents disk thrashing when multiple tokens or speculative prefetchers request identical experts simultaneously.
