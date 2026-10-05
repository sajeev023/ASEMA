# ASEMA v0.1 — Integration Contract (Agent #2 ↔ Agent #1)

This document defines the precise contract between Agent #2's subsystems
(RAM cache, request queue, prefetch engine, telemetry, runtime) and
Agent #1's `AsyncExpertLoader`.

---

## 1. The `IExpertLoader` Interface

Agent #2 will **not** call `AsyncExpertLoader` directly. Instead, it depends
on the abstract interface declared in `include/asema/loader_adapter.hpp`:

```cpp
class IExpertLoader {
public:
    virtual ~IExpertLoader() = default;

    // Submit a load request, returning a load handle.
    // `priority` mirrors asema::LoadPriority.
    virtual LoadHandle submit(
        ExpertCoord coord,
        LoadPriority priority,
        LoadCallback callback = nullptr) = 0;

    // Submit and obtain a future.
    virtual std::future<ExpertLoadResult> submit_future(
        ExpertCoord coord,
        LoadPriority priority) = 0;

    // Cancel a previously-submitted handle. Returns true if known.
    virtual bool cancel(LoadHandle handle) = 0;

    // Non-blocking poll.
    virtual std::optional<ExpertLoadResult> try_get_result(LoadHandle handle) = 0;

    // Blocking wait.
    virtual ExpertLoadResult wait(LoadHandle handle) = 0;

    // Telemetry hooks (read-only).
    virtual uint64_t total_requests_submitted() const noexcept = 0;
    virtual uint64_t total_coalesced_requests() const noexcept = 0;
    virtual uint64_t total_io_dispatches() const noexcept = 0;
    virtual uint64_t total_bytes_loaded() const noexcept = 0;
    virtual uint64_t total_checksum_failures() const noexcept = 0;
    virtual uint64_t active_in_flight() const noexcept = 0;

    // Lifecycle.
    virtual void shutdown() = 0;
};
```

This interface is a **structurally identical mirror** of Agent #1's
`AsyncExpertLoader` public API. The mapping is one-to-one so the adapter
has zero behavioral reinterpretation.

---

## 2. The Adapter

`src/runtime/loader_adapter.cpp` provides:

```cpp
class AsyncLoaderAdapter final : public IExpertLoader {
public:
    explicit AsyncLoaderAdapter(
        const std::string& container_path,
        const ModelManifest& manifest,
        size_t num_workers = 4);

    // All IExpertLoader methods forward to the wrapped AsyncExpertLoader.
    ...
private:
    std::unique_ptr<AsyncExpertLoader> impl_;
};
```

A factory function is provided:

```cpp
std::unique_ptr<IExpertLoader> create_async_loader_adapter(
    const std::string& container_path,
    const ModelManifest& manifest,
    size_t num_workers = 4);
```

---

## 3. Coordination Rules

1. **Single owner of state transitions.** Only `ASEMARuntime` calls
   `cache->insert(...)` and `cache->set_state(RAM_RESIDENT)`. The loader
   does not know about the cache's state machine.

2. **Cache-first lookup.** Before any `loader->submit(coord, ...)`, the
   runtime first asks the cache. If `cache->get(coord)` returns a hit, the
   loader is never called.

3. **Loader second.** If the cache misses, the runtime calls
   `loader->submit_future(coord, REQUIRED_NOW)`. The runtime then awaits
   the future, inserts the resulting buffer into the cache (calling
   `pin`/`unpin` around the execute step), and proceeds.

4. **Prefetch third.** The prefetch engine can speculatively call
   `loader->submit(coord, PREFETCH, callback)`. The callback is responsible
   for inserting into the cache **only if** the entry is still beneficial
   (cancellation check) and capacity allows. If not, the buffer is
   discarded.

5. **Request coalescing.** Two layers of dedup exist:
   - **Logical layer (queue):** the request queue suppresses duplicate
     `REQUIRED_NOW` requests against in-flight entries.
   - **Physical layer (loader):** the loader coalesces duplicates across
     its own internal queue.

   Both layers are idempotent. The logical layer exists so the runtime can
   reason about back-pressure before paying the `submit()` cost; the
   physical layer is the ultimate safety net.

6. **Cancellation.** Prefetch requests may be cancelled by the runtime when
   the prefetch lookahead horizon moves past the expert. The runtime calls
   `loader->cancel(handle)`. **REQUIRED_NOW requests are never cancelled**
   by Agent #2 — the only cancellation path for them is a runtime-level
   `shutdown()`.

---

## 4. Lifecycle Order

```
ASEMARuntime::init()
   ├── cache   = std::make_unique<ExpertRAMCache>(ram_budget);
   ├── queue   = std::make_unique<ExpertRequestQueue>(queue_capacity);
   ├── prefetch= std::make_unique<PrefetchEngine>(*queue, *cache, *loader, lookahead);
   ├── telemetry = std::make_unique<TelemetryEngine>(jsonl_path);
   └── loader  = create_async_loader_adapter(container, manifest, workers);

ASEMARuntime::run_tokens(tokens)
   ├── for each token:
   │     experts = router(token)
   │     telemetry->token_begin(...)
   │     for each expert:
   │           if (cache->get(coord)) { hit } else { queue->submit(coord, REQUIRED_NOW) }
   │     await futures → cache->insert → execute
   │     prefetch->on_layer_complete(current_layer + 1)
   │     telemetry->token_end(...)
   └── emit aggregate report

ASEMARuntime::shutdown()
   ├── prefetch->shutdown()
   ├── queue->shutdown()
   ├── loader->shutdown()
   ├── cache->clear()
   └── telemetry->flush()
```

---

## 5. Build-System Integration

Agent #2 will only add lines to `CMakeLists.txt`. Existing lines are
preserved verbatim. The added section is fenced inside a comment marker:

```cmake
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# AGENT #2 ADDITIONS — Milestones 3, 4, 5
# (Do not modify this block from another agent)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

list(APPEND ASEMA_SOURCES
    src/cache/cache.cpp
    src/queue/request_queue.cpp
    src/prefetch/prefetch.cpp
    src/telemetry/telemetry.cpp
    src/runtime/runtime.cpp
    src/runtime/loader_adapter.cpp)

add_executable(test_cache tests/test_cache.cpp)
target_link_libraries(test_cache PRIVATE asema_core)
add_test(NAME test_cache COMMAND test_cache)

add_executable(test_queue tests/test_queue.cpp)
target_link_libraries(test_queue PRIVATE asema_core)
add_test(NAME test_queue COMMAND test_queue)

add_executable(test_prefetch tests/test_prefetch.cpp)
target_link_libraries(test_prefetch PRIVATE asema_core)
add_test(NAME test_prefetch COMMAND test_prefetch)

add_executable(test_telemetry tests/test_telemetry.cpp)
target_link_libraries(test_telemetry PRIVATE asema_core)
add_test(NAME test_telemetry COMMAND test_telemetry)

add_executable(test_runtime tests/test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE asema_core)
add_test(NAME test_runtime COMMAND test_runtime)

add_executable(asema-full-bench tools/asema_full_bench.cpp)
target_link_libraries(asema-full-bench PRIVATE asema_core)
```

No existing CMake variable, target, or test is renamed, removed, or
re-ordered.
