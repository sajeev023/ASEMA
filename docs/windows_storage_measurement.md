# Windows Storage Measurement Methodology

**Date**: 2026-09-26
**Scope**: What we can and cannot measure about storage behavior on Windows without unsafe or fragile APIs.

---

## 1. Layered measurement model

There are four layers between the ASEMA runtime and the SSD hardware. Each layer can serve a read, and each layer is observable at a different level:

```
   ┌───────────────────────────────────────────┐
   │  (1) Runtime application requests        │   ← logical_bytes_requested
   └────────────────────┬──────────────────────┘
                        ▼
   ┌───────────────────────────────────────────┐
   │  (2) AsyncExpertLoader (post-coalescing) │   ← ssd_bytes_physical,
   │     ReadFile / IOCP                       │     ssd_reads,
   │     CRC32 verification                    │     coalesced_requests
   └────────────────────┬──────────────────────┘
                        ▼
   ┌───────────────────────────────────────────┐
   │  (3) Windows OS file-system cache         │   ← OS_FILE_CACHE_BYTES (unknown)
   │     (Standby list, modified list, etc.)   │   ← not directly measurable
   │     ReadFile API returns from cache if   │     without ETW tracing
   │     data is present                        │
   └────────────────────┬──────────────────────┘
                        ▼
   ┌───────────────────────────────────────────┐
   │  (4) Storage device / SSD firmware       │   ← HARDWARE_BYTES (unknown)
   │     Actual NAND reads / wear leveling    │   ← only via vendor SMART
   └───────────────────────────────────────────┘
```

ASEMA v0.1 measures (1) and (2) directly. (3) and (4) are **NOT** measured and would require either ETW tracing (Windows kernel-level) or vendor-specific SMART counters (storage hardware-level). Neither is currently integrated because:

- ETW tracing requires elevated privileges and risks destabilizing the runtime.
- SMART counters are not portable across NVMe / SATA / cloud storage.

---

## 2. What ASEMA v0.1 measures

### `runtime_logical_requests`
- **Source**: `loader->total_requests_submitted()`
- **What it means**: The number of expert loads the ASEMA runtime (or its prefetcher) asked the M2 loader to perform. Each `loader->submit_future()` increments by 1; each coalesced duplicate does NOT increment.

### `runtime_logical_bytes`
- **Source**: `runtime_logical_requests × expert_bytes_per_request`
- **What it means**: Bytes the ASEMA runtime requested at the application layer. This is the **upper bound on storage I/O** if the loader had no coalescing and the OS had no cache.

### `ssd_bytes_physical`
- **Source**: `loader->total_bytes_loaded()`
- **What it means**: Bytes actually transferred through `ReadFile` (or its IOCP equivalent) by the M2 loader, after internal coalescing. This is the application-OS boundary.

### `ssd_reads`
- **Source**: `loader->total_io_dispatches()`
- **What it means**: Number of physical I/O operations issued by the M2 loader. Each dispatch corresponds to one `ReadFile` call.

### `coalesced_requests`
- **Source**: `loader->total_coalesced_requests()`
- **What it means**: Number of logical requests that the M2 loader folded into an in-flight load, avoiding a duplicate physical I/O.

### `cache_bytes_served`
- **Source**: `cache_hits × expert_bytes_per_request`
- **What it means**: Bytes served from the RAM cache without touching storage.

---

## 3. What ASEMA v0.1 does NOT measure (and why)

### OS file-cache bytes
- `ReadFile` on Windows transparently returns from the OS file cache if the data is present. The M2 loader has no API to know whether the bytes came from disk or from the OS cache.
- Measuring this requires ETW (`Microsoft-Windows-Kernel-File` provider) with `FILE_IO` events. ETW setup is invasive (admin privileges, kernel tracing sessions, JSON export tooling). Not done in v0.1.

### Hardware-level physical bytes
- The actual NAND read count depends on the SSD's internal wear leveling, deduplication, and compression. Vendor-specific (Samsung, Intel, etc.). Not done in v0.1.

### True cold-cache benchmarking
- Windows does not provide a portable API to flush the OS file cache. The closest is `SetSystemFileCacheSize` (requires admin), which can destabilize the system.
- ASEMA v0.1 runs are therefore on a **partially warm** cache state. Each `--cold` flag invocation just records the intent; actual cold behavior cannot be enforced.

---

## 4. Implications for interpreting ASEMA v0.1 results

| Claim | Supported by v0.1 data? |
|-------|------------------------|
| "ASEMA reduced application-level storage demand by X%" | ✅ Yes — `runtime_logical_bytes vs cache_bytes_served` |
| "The M2 loader performed Y physical I/O operations" | ✅ Yes — `ssd_reads` |
| "The M2 loader coalesced Z duplicate loads" | ✅ Yes — `coalesced_requests` |
| "Physical SSD bandwidth was X MB/s" | ❌ No — OS file cache may reduce hardware-level throughput |
| "ASEMA achieved a 10× SSD bandwidth multiplier" | ❌ No — coalescing reduces logical demand, not hardware bandwidth |
| "ASEMA works in true cold-cache conditions" | ❌ No — Windows OS cache is not flushed |
| "The SSD physically read X MB" | ❌ No — not measurable without vendor SMART |

Any future claim of hardware-level storage performance requires either ETW integration or vendor SMART counters. These are deferred to M9/M10.

---

## 5. The "logical vs physical" reduction is meaningful even with OS caching

Coalescing operates **above** the OS file cache: it folds multiple application requests for the same expert into a single `ReadFile` call. Whether the OS serves that single `ReadFile` from cache or from disk is irrelevant to the coalescing benefit.

Therefore `ssd_reads < runtime_logical_requests` is a valid application-level win, even when the OS cache is warm.

The same is NOT true for `ssd_bytes_physical` — those bytes were transferred through the OS API and may or may not have hit the disk.

---

## 6. Reproducing v0.1 numbers

Every measurement in `reports/runs/<run>/<timestamp>/raw/` can be reproduced by:

```
asema-bench --model <dir> --tokens N --iterations N --warmup N ...
```

The OS cache state at the start of each run is uncontrolled. Numbers should be interpreted as **typical application behavior under unknown cache state**, not as raw SSD hardware performance.

---

## 7. Future work (M9+)

If real cold-cache measurement is needed:

1. ETW `Microsoft-Windows-Kernel-File` provider with `FILE_IO_READ` events filtered to the model.asema file.
2. Parse ETW events into per-read log.
3. Compute true cold-cache hit rate and bandwidth.

This requires elevated privileges and is deferred.
