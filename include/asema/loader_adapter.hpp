#pragma once

// ASEMA v0.1 — AsyncLoaderAdapter
// -----------------------------------------------------------------------------
// Adapter that wraps Agent #1's AsyncExpertLoader behind a thin interface
// so the runtime, cache, queue, and prefetch engine do not depend on the
// concrete loader type.
// -----------------------------------------------------------------------------

#include "async_loader.hpp"

#include <future>
#include <memory>
#include <optional>

namespace asema {

class IExpertLoader {
public:
    virtual ~IExpertLoader() = default;
    virtual LoadHandle submit(
        ExpertCoord coord,
        LoadPriority priority,
        LoadCallback callback = nullptr) = 0;
    virtual std::future<ExpertLoadResult> submit_future(
        ExpertCoord coord,
        LoadPriority priority) = 0;
    virtual bool cancel(LoadHandle handle) = 0;
    virtual std::optional<ExpertLoadResult> try_get_result(LoadHandle handle) = 0;
    virtual ExpertLoadResult wait(LoadHandle handle) = 0;
    virtual void shutdown() = 0;
    virtual uint64_t total_requests_submitted() const noexcept = 0;
    virtual uint64_t total_coalesced_requests() const noexcept = 0;
    virtual uint64_t total_io_dispatches() const noexcept = 0;
    virtual uint64_t total_bytes_loaded() const noexcept = 0;
    virtual uint64_t total_checksum_failures() const noexcept = 0;
    virtual uint64_t active_in_flight() const noexcept = 0;
};

class AsyncLoaderAdapter final : public IExpertLoader {
public:
    explicit AsyncLoaderAdapter(
        const std::string& container_path,
        const ModelManifest& manifest,
        size_t num_workers = 4)
        : impl_(std::make_unique<AsyncExpertLoader>(container_path, manifest, num_workers)) {}

    LoadHandle submit(ExpertCoord coord, LoadPriority priority,
                      LoadCallback callback = nullptr) override {
        return impl_->submit(coord, priority, std::move(callback));
    }

    std::future<ExpertLoadResult> submit_future(ExpertCoord coord, LoadPriority priority) override {
        return impl_->submit_future(coord, priority);
    }

    bool cancel(LoadHandle handle) override { return impl_->cancel(handle); }
    std::optional<ExpertLoadResult> try_get_result(LoadHandle handle) override {
        return impl_->try_get_result(handle);
    }
    ExpertLoadResult wait(LoadHandle handle) override { return impl_->wait(handle); }
    void shutdown() override { impl_->shutdown(); }

    uint64_t total_requests_submitted() const noexcept override {
        return impl_->total_requests_submitted();
    }
    uint64_t total_coalesced_requests() const noexcept override {
        return impl_->total_coalesced_requests();
    }
    uint64_t total_io_dispatches() const noexcept override {
        return impl_->total_io_dispatches();
    }
    uint64_t total_bytes_loaded() const noexcept override {
        return impl_->total_bytes_loaded();
    }
    uint64_t total_checksum_failures() const noexcept override {
        return impl_->total_checksum_failures();
    }
    uint64_t active_in_flight() const noexcept override {
        return impl_->active_in_flight();
    }

private:
    std::unique_ptr<AsyncExpertLoader> impl_;
};

std::unique_ptr<IExpertLoader> create_async_loader_adapter(
    const std::string& container_path,
    const ModelManifest& manifest,
    size_t num_workers = 4);

} // namespace asema
