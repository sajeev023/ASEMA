// ASEMA v0.1 — AsyncLoaderAdapter implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/loader_adapter.hpp"

namespace asema {

std::unique_ptr<IExpertLoader> create_async_loader_adapter(
    const std::string& container_path,
    const ModelManifest& manifest,
    size_t num_workers)
{
    return std::make_unique<AsyncLoaderAdapter>(container_path, manifest, num_workers);
}

} // namespace asema
