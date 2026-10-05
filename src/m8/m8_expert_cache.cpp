#include "asema/m8/m8_expert_cache.hpp"

#include <algorithm>

namespace asema {
namespace m8 {

M8ExpertCacheEngine::M8ExpertCacheEngine(size_t capacity_bytes)
    : capacity_bytes_(capacity_bytes) {}

M8ExpertCacheEngine::~M8ExpertCacheEngine() {
    clear();
}

void M8ExpertCacheEngine::set_capacity_bytes(size_t capacity_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_bytes_ = capacity_bytes;
    if (used_bytes_ > capacity_bytes_) {
        evict_to_fit(used_bytes_ - capacity_bytes_);
    }
}

void M8ExpertCacheEngine::set_capacity_mb(size_t capacity_mb) {
    set_capacity_bytes(capacity_mb * 1024ULL * 1024ULL);
}

size_t M8ExpertCacheEngine::capacity_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_bytes_;
}

size_t M8ExpertCacheEngine::used_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return used_bytes_;
}

void M8ExpertCacheEngine::detach_node(CacheNode* node) {
    if (!node) return;
    if (node->prev) node->prev->next = node->next;
    else head_ = node->next;

    if (node->next) node->next->prev = node->prev;
    else tail_ = node->prev;

    node->prev = nullptr;
    node->next = nullptr;
}

void M8ExpertCacheEngine::attach_mru(CacheNode* node) {
    if (!node) return;
    node->prev = nullptr;
    node->next = head_;
    if (head_) head_->prev = node;
    head_ = node;
    if (!tail_) tail_ = node;
}

void M8ExpertCacheEngine::remove_node(CacheNode* node) {
    if (!node) return;
    detach_node(node);
    CacheKey key{node->layer_id, node->expert_id};
    map_.erase(key);
    used_bytes_ -= node->total_bytes;
    delete node;
    evictions_++;
}

std::shared_ptr<CacheNode> M8ExpertCacheEngine::get(int layer_id, int expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheKey key{layer_id, expert_id};
    auto it = map_.find(key);
    if (it == map_.end()) {
        misses_++;
        return nullptr;
    }

    hits_++;
    CacheNode* node = it->second;
    // Promote to MRU if not pinned
    if (node->pin_count.load() == 0 && node != head_) {
        detach_node(node);
        attach_mru(node);
    }

    node->ref_count.fetch_add(1);
    // Custom deleter to decrement ref_count
    return std::shared_ptr<CacheNode>(node, [](CacheNode* n) {
        if (n) n->ref_count.fetch_sub(1);
    });
}

bool M8ExpertCacheEngine::put(int layer_id,
                             int expert_id,
                             std::shared_ptr<std::vector<uint8_t>> weights,
                             std::shared_ptr<std::vector<uint8_t>> scales,
                             CachePriority priority) {
    if (!weights || !scales) return false;

    size_t node_bytes = weights->size() + scales->size() + sizeof(CacheNode);

    std::lock_guard<std::mutex> lock(mutex_);
    CacheKey key{layer_id, expert_id};
    auto it = map_.find(key);
    if (it != map_.end()) {
        // Already cached, update priority if higher
        if (static_cast<uint8_t>(priority) > static_cast<uint8_t>(it->second->priority)) {
            it->second->priority = priority;
        }
        return true;
    }

    if (node_bytes > capacity_bytes_) {
        rejected_admissions_++;
        return false;
    }

    // Evict unpinned from LRU (tail) until room is available
    while (used_bytes_ + node_bytes > capacity_bytes_ && tail_) {
        CacheNode* curr = tail_;
        CacheNode* to_evict = nullptr;
        while (curr) {
            if (curr->pin_count.load() == 0 && curr->ref_count.load() == 0) {
                to_evict = curr;
                break;
            }
            curr = curr->prev;
        }

        if (!to_evict) {
            // All active nodes are pinned or referenced
            rejected_admissions_++;
            return false;
        }

        remove_node(to_evict);
    }

    auto* node = new CacheNode();
    node->layer_id = layer_id;
    node->expert_id = expert_id;
    node->priority = priority;
    node->weights = std::move(weights);
    node->scales = std::move(scales);
    node->total_bytes = node_bytes;

    attach_mru(node);
    map_[key] = node;
    used_bytes_ += node_bytes;

    return true;
}

bool M8ExpertCacheEngine::pin(int layer_id, int expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheKey key{layer_id, expert_id};
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    it->second->pin_count.fetch_add(1);
    return true;
}

bool M8ExpertCacheEngine::unpin(int layer_id, int expert_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheKey key{layer_id, expert_id};
    auto it = map_.find(key);
    if (it == map_.end()) return false;
    int32_t prev = it->second->pin_count.fetch_sub(1);
    if (prev < 1) {
        it->second->pin_count.store(0);
    }
    return true;
}

size_t M8ExpertCacheEngine::evict_to_fit(size_t required_bytes) {
    size_t freed = 0;
    while (freed < required_bytes && tail_) {
        CacheNode* curr = tail_;
        CacheNode* to_evict = nullptr;
        while (curr) {
            if (curr->pin_count.load() == 0 && curr->ref_count.load() == 0) {
                to_evict = curr;
                break;
            }
            curr = curr->prev;
        }

        if (!to_evict) break;

        size_t b = to_evict->total_bytes;
        remove_node(to_evict);
        freed += b;
    }
    return freed;
}

void M8ExpertCacheEngine::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheNode* curr = head_;
    while (curr) {
        CacheNode* next = curr->next;
        delete curr;
        curr = next;
    }
    head_ = nullptr;
    tail_ = nullptr;
    map_.clear();
    used_bytes_ = 0;
}

CacheStats M8ExpertCacheEngine::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    CacheStats s;
    s.capacity_bytes = capacity_bytes_;
    s.used_bytes = used_bytes_;
    s.count = map_.size();
    s.pinned_count = 0;
    for (const auto& kv : map_) {
        if (kv.second->pin_count.load() > 0) s.pinned_count++;
    }
    s.hits = hits_;
    s.misses = misses_;
    s.evictions = evictions_;
    s.rejected_admissions = rejected_admissions_;
    uint64_t total = hits_ + misses_;
    s.hit_rate = (total > 0) ? (static_cast<double>(hits_) / total) : 0.0;
    return s;
}

void M8ExpertCacheEngine::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    evictions_ = 0;
    rejected_admissions_ = 0;
}

} // namespace m8
} // namespace asema
