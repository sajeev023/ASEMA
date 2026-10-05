# ASEMA v0.1 — Milestone 2 Technical Design Document
## Asynchronous Expert Loader & Overlapped I/O Worker Pool

---

### 1. Current Storage Flow (Milestone 1)
In Milestone 1, expert weight loading is synchronous and blocking:
```
Runtime Thread
     │
     ▼
StorageBackend::read_expert_sync()
     │
     ├── Set OVERLAPPED offset
     ├── Call ReadFile()
     ├── Block on GetOverlappedResult()
     ├── Execute CRC32 checksum verification on returned buffer
     └── Return ExpertBuffer
```
**Limitation**: The calling thread cannot execute attention, compute active routing tokens, or schedule speculative expert prefetches while blocked waiting for disk I/O.

---

### 2. Proposed Asynchronous Architecture (Milestone 2)
Milestone 2 decouples submission, I/O dispatch, and completion handling:
```
Inference / Prefetch Thread
     │
     ▼
AsyncExpertLoader::submit(coord, priority, callback)
     │
     ├── Check in-flight registry (Request Coalescing)
     │     ├─ If duplicate in-flight: attach listener/promise, avoid duplicate disk I/O
     │     └─ If new: allocate OperationContext, register in in-flight map
     │
     ▼
Priority Request Queue (REQUIRED_NOW > PREFETCH > BACKGROUND)
     │
     ▼
I/O Dispatcher Thread
     │
     ├── Dequeue highest priority request
     ├── Allocate aligned ExpertBuffer (64-byte alignment)
     ├── Set OVERLAPPED structure
     └── Issue ReadFile() with FILE_FLAG_OVERLAPPED to IOCP
             │
             ▼
Windows I/O Completion Port (IOCP)
             │
             ▼
IOCP Worker Pool (Configurable threads, e.g. 4-8 on Ryzen 7 5700X)
     │
     ├── GetQueuedCompletionStatus() wakes worker thread on hardware I/O finish
     ├── Record I/O completion timestamp & delta
     ├── Execute CRC32 validation on loaded ExpertBuffer
     │     ├─ If CRC matches: Mark SUCCESS, attach valid buffer
     │     └─ If CRC mismatch: Mark CORRUPTED, discard buffer
     ├── Remove from in-flight registry
     └── Fan-out dispatch to all coalesced callbacks/promises
```

---

### 3. Threading Model
- **Caller Thread(s)**: Inference runtime or prefetcher submitting asynchronous requests. Non-blocking.
- **Dispatch Thread**: Single scheduler ordering requests strictly by priority (`REQUIRED_NOW` first, then `PREFETCH`, then `BACKGROUND`) and submitting them to the Windows I/O subsystem.
- **IOCP Worker Pool**: Dedicated worker threads running `GetQueuedCompletionStatus()`. They perform parallel hardware completion handling and checksum validation.
- **No thread explosion**: Worker count is fixed and configurable (defaults to 4 workers).

---

### 4. Request Lifecycle & Explicit States
Each asynchronous request moves through strict state transitions:
1. `QUEUED`: Enqueued in priority queue awaiting I/O dispatch.
2. `IN_FLIGHT`: Issued to Windows IOCP via `ReadFile()`.
3. `VALIDATING`: Hardware transfer finished; worker executing CRC32 check.
4. `COMPLETED`: Data valid; dispatched to callback/promise.
5. `FAILED` / `CORRUPT`: I/O error or checksum failure; error payload dispatched.
6. `CANCELLED`: Cancelled prior to execution; buffer released immediately.

---

### 5. Memory Safety & Ownership Model
- **`OperationContext`**: Managed via `std::shared_ptr<AsyncOpContext>`.
- The `OVERLAPPED` structure is embedded directly inside `AsyncOpContext`. It is guaranteed to remain valid and pinned in memory until `GetQueuedCompletionStatus` returns for that operation.
- **Buffer Lifetime**: `ExpertBuffer` is allocated with 64-byte alignment. If an operation fails, is corrupted, or cancelled, the buffer is freed by RAII destructor.
- **No dangling callbacks**: Listener callbacks hold weak/shared references and are invoked on worker threads or detached callers.

---

### 6. Request Coalescing (Duplicate Suppression)
- If token 1 demands `L2/E5` (`REQUIRED_NOW`) while speculative prefetcher already requested `L2/E5` (`PREFETCH`):
  - No second `ReadFile` is issued.
  - The second request attaches its future/callback to the pending operation.
  - The operation's priority is boosted to `REQUIRED_NOW` if still in queue.
  - When the single I/O finishes, both callers receive the identical buffer pointer.

---

### 7. Deterministic Shutdown Sequence
1. Set `shutdown_` atomic flag to refuse new requests.
2. Cancel remaining queued requests (notify listeners with `is_cancelled = true`).
3. Broadcast `PostQueuedCompletionStatus` with exit sentinel to all IOCP worker threads.
4. Join dispatch thread and IOCP worker threads.
5. Close IOCP handle (`CloseHandle`).
6. Close container file handle.
