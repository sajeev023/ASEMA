#include "asema/m8/m8_gpu_memory_arena.hpp"

#include <iostream>
#include <algorithm>

namespace asema {
namespace m8 {

M8GPUMemoryArena::M8GPUMemoryArena(ID3D11Device* device, size_t slab_size_bytes)
    : device_(device), slab_size_bytes_(slab_size_bytes) {
    telemetry_.total_slab_bytes = slab_size_bytes;
    telemetry_.free_bytes = slab_size_bytes;
}

M8GPUMemoryArena::~M8GPUMemoryArena() {
    reset();
}

bool M8GPUMemoryArena::initialize() {
    if (!device_) return false;

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(slab_size_bytes_);
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;

    HRESULT hr = device_->CreateBuffer(&desc, nullptr, slab_buffer_.GetAddressOf());
    if (FAILED(hr)) {
        return false;
    }

    ArenaBlock initial_block;
    initial_block.offset = 0;
    initial_block.size = slab_size_bytes_;
    initial_block.in_use = false;
    initial_block.allocation_id = 0;
    blocks_.push_back(initial_block);

    return true;
}

ArenaSuballocation M8GPUMemoryArena::suballocate(size_t size_bytes, const std::string& tag) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_.allocation_requests++;

    // Strict 256-byte alignment
    size_t aligned_size = (size_bytes + 255) & ~size_t(255);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (!blocks_[i].in_use && blocks_[i].size >= aligned_size) {
            uint64_t id = next_alloc_id_++;
            size_t remaining = blocks_[i].size - aligned_size;

            blocks_[i].size = aligned_size;
            blocks_[i].in_use = true;
            blocks_[i].allocation_id = id;
            blocks_[i].tag = tag;

            if (remaining > 0) {
                ArenaBlock rem_block;
                rem_block.offset = blocks_[i].offset + aligned_size;
                rem_block.size = remaining;
                rem_block.in_use = false;
                rem_block.allocation_id = 0;
                blocks_.insert(blocks_.begin() + i + 1, rem_block);
            }

            telemetry_.allocated_bytes += aligned_size;
            telemetry_.free_bytes -= aligned_size;
            if (telemetry_.allocated_bytes > telemetry_.peak_allocated_bytes) {
                telemetry_.peak_allocated_bytes = telemetry_.allocated_bytes;
            }
            telemetry_.active_allocations++;
            telemetry_.suballoc_hits++;

            auto t1 = std::chrono::high_resolution_clock::now();
            telemetry_.suballoc_latency_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

            ArenaSuballocation sub;
            sub.allocation_id = id;
            sub.buffer = slab_buffer_.Get();
            sub.byte_offset = blocks_[i].offset;
            sub.byte_size = aligned_size;
            return sub;
        }
    }

    return ArenaSuballocation{}; // Out of memory in slab
}

void M8GPUMemoryArena::free_suballocation(uint64_t allocation_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& b : blocks_) {
        if (b.allocation_id == allocation_id && b.in_use) {
            b.in_use = false;
            b.allocation_id = 0;
            telemetry_.allocated_bytes -= b.size;
            telemetry_.free_bytes += b.size;
            if (telemetry_.active_allocations > 0) telemetry_.active_allocations--;
            break;
        }
    }
    coalesce_free_blocks();
}

void M8GPUMemoryArena::coalesce_free_blocks() {
    for (size_t i = 0; i + 1 < blocks_.size(); ) {
        if (!blocks_[i].in_use && !blocks_[i + 1].in_use) {
            blocks_[i].size += blocks_[i + 1].size;
            blocks_.erase(blocks_.begin() + i + 1);
        } else {
            ++i;
        }
    }
}

ComPtr<ID3D11Buffer> M8GPUMemoryArena::get_staging_buffer(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t aligned = (size_bytes + 255) & ~size_t(255);

    auto& pool = staging_pool_[aligned];
    if (!pool.empty()) {
        ComPtr<ID3D11Buffer> buf = pool.back();
        pool.pop_back();
        return buf;
    }

    // Allocate persistent staging buffer
    if (!device_) return nullptr;

    D3D11_BUFFER_DESC desc{};
    desc.ByteWidth = static_cast<UINT>(aligned);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;

    ComPtr<ID3D11Buffer> buf;
    device_->CreateBuffer(&desc, nullptr, buf.GetAddressOf());
    return buf;
}

void M8GPUMemoryArena::release_staging_buffer(ComPtr<ID3D11Buffer> buf, size_t size_bytes) {
    if (!buf) return;
    std::lock_guard<std::mutex> lock(mutex_);
    size_t aligned = (size_bytes + 255) & ~size_t(255);
    staging_pool_[aligned].push_back(buf);
}

ArenaTelemetry M8GPUMemoryArena::get_telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_;
}

void M8GPUMemoryArena::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_.clear();
    slab_buffer_.Reset();
    staging_pool_.clear();
    telemetry_ = ArenaTelemetry{};
    telemetry_.total_slab_bytes = slab_size_bytes_;
    telemetry_.free_bytes = slab_size_bytes_;
}

} // namespace m8
} // namespace asema
