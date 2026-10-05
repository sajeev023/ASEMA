#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <chrono>

namespace asema {

// Explicit lifecycle state representation
enum class ExpertState : uint8_t {
    NOT_RESIDENT = 0,
    LOADING      = 1,
    RAM_RESIDENT = 2,
    GPU_RESIDENT = 3,
    IN_USE       = 4,
    WARM         = 5,
    EVICTING     = 6
};

inline const char* to_string(ExpertState state) {
    switch (state) {
        case ExpertState::NOT_RESIDENT: return "NOT_RESIDENT";
        case ExpertState::LOADING:      return "LOADING";
        case ExpertState::RAM_RESIDENT: return "RAM_RESIDENT";
        case ExpertState::GPU_RESIDENT: return "GPU_RESIDENT";
        case ExpertState::IN_USE:       return "IN_USE";
        case ExpertState::WARM:         return "WARM";
        case ExpertState::EVICTING:     return "EVICTING";
        default:                        return "UNKNOWN";
    }
}

// Coordinate identifying an individual expert
struct ExpertCoord {
    uint32_t layer_id{0};
    uint32_t expert_id{0};

    bool operator==(const ExpertCoord& other) const noexcept {
        return layer_id == other.layer_id && expert_id == other.expert_id;
    }

    bool operator<(const ExpertCoord& other) const noexcept {
        if (layer_id != other.layer_id) return layer_id < other.layer_id;
        return expert_id < other.expert_id;
    }
};

// Metadata for an individual expert stored in ASEMA-SSF
struct ExpertMetadata {
    uint32_t layer_id{0};
    uint32_t expert_id{0};
    uint64_t storage_offset{0};
    uint64_t storage_length{0};
    uint32_t alignment{64};
    std::string dtype{"fp16"};
    std::string quant_type{"none"};
    std::string checksum_sha256;
    uint32_t checksum_crc32{0};
    std::vector<int64_t> tensor_shape;
};

// Contiguous aligned memory buffer holding expert weight tensors
class ExpertBuffer {
public:
    ExpertBuffer(size_t size_bytes, size_t alignment = 64)
        : size_(size_bytes), alignment_(alignment) {
        if (size_bytes > 0) {
#if defined(_WIN32)
            data_ = static_cast<uint8_t*>(_aligned_malloc(size_bytes, alignment));
#else
            data_ = static_cast<uint8_t*>(std::aligned_alloc(alignment, size_bytes));
#endif
        }
    }

    ~ExpertBuffer() {
        if (data_) {
#if defined(_WIN32)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
            data_ = nullptr;
        }
    }

    // Disable copy, enable move
    ExpertBuffer(const ExpertBuffer&) = delete;
    ExpertBuffer& operator=(const ExpertBuffer&) = delete;

    ExpertBuffer(ExpertBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_), alignment_(other.alignment_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    ExpertBuffer& operator=(ExpertBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) {
#if defined(_WIN32)
                _aligned_free(data_);
#else
                std::free(data_);
#endif
            }
            data_ = other.data_;
            size_ = other.size_;
            alignment_ = other.alignment_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    size_t alignment() const noexcept { return alignment_; }

private:
    uint8_t* data_{nullptr};
    size_t size_{0};
    size_t alignment_{64};
};

// Managed Expert object containing buffer, lifecycle state, and telemetry stats
struct ManagedExpert {
    ExpertCoord coord;
    ExpertMetadata metadata;
    std::shared_ptr<ExpertBuffer> buffer;
    std::atomic<ExpertState> state{ExpertState::NOT_RESIDENT};
    std::atomic<uint64_t> last_access_timestamp{0};
    std::atomic<uint32_t> access_count{0};
    std::atomic<uint32_t> pin_count{0}; // For IN_USE protection during active compute

    ManagedExpert(const ExpertCoord& c, const ExpertMetadata& meta)
        : coord(c), metadata(meta) {}
};

} // namespace asema
