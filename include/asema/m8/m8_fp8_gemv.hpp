#pragma once

// FP8 (E4M3) block-scaled matrix-vector product that reads the checkpoint bytes directly.
//
// The previous dense path dequantized every FP8 matrix into FP32 (about 650 MB of writes per
// layer per token) before multiplying. This kernel keeps the weights as 1 byte/element and
// converts them in registers, so memory traffic drops ~8x and no FP32 weight copy exists.
//
// Decode is exact: for an E4M3 byte b,
//   float(b) = reinterpret_f32(((b & 0x7F) << 20) | ((b & 0x80) << 24)) * 2^120
// (the 2^120 rebias also maps E4M3 subnormals correctly). Block scales are E8M0 powers of two,
// so scaling is exact as well. Only the FP32 summation order differs from the LUT path.
// Codes 0x7F / 0xFF (E4M3 NaN) are decoded as +/-480 here but as 0 by the LUT path; the dense
// tensors never contain them (checked by tests/test_m8_fp8_gemv.cpp).

#include "asema/m8/m8_fp8_lut.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

namespace asema {
namespace m8 {

// Fixed-size worker pool. Hard upper bound on threads (never one thread per task) so that
// dense compute cannot starve the Windows desktop. Override with ASEMA_THREADS.
class M8WorkerPool {
public:
    static M8WorkerPool& instance() {
        static M8WorkerPool pool;
        return pool;
    }

    int worker_count() const { return static_cast<int>(workers_.size()) + 1; }

    // Runs fn(task) for task in [0, num_tasks) across the pool plus the calling thread.
    void run(int num_tasks, const std::function<void(int)>& fn) {
        if (num_tasks <= 0) return;
        if (workers_.empty() || num_tasks == 1) {
            for (int t = 0; t < num_tasks; ++t) fn(t);
            return;
        }
        std::lock_guard<std::mutex> serialize(run_mutex_);
        {
            std::lock_guard<std::mutex> lk(mutex_);
            fn_ = &fn;
            num_tasks_ = num_tasks;
            next_task_.store(0);
            pending_workers_ = static_cast<int>(workers_.size());
            ++generation_;
        }
        cv_start_.notify_all();
        drain(fn, num_tasks);
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [&] { return pending_workers_ == 0; });
        fn_ = nullptr;
    }

private:
    M8WorkerPool() {
        int n = 6;
        if (const char* env = std::getenv("ASEMA_THREADS")) {
            n = std::max(1, std::atoi(env));
        } else {
            int hw = static_cast<int>(std::thread::hardware_concurrency());
            if (hw > 0) n = std::min(6, std::max(1, hw - 2));
        }
        for (int i = 0; i < n - 1; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~M8WorkerPool() {
        {
            std::lock_guard<std::mutex> lk(mutex_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& w : workers_) w.join();
    }

    void drain(const std::function<void(int)>& fn, int num_tasks) {
        for (;;) {
            int t = next_task_.fetch_add(1);
            if (t >= num_tasks) break;
            fn(t);
        }
    }

    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            const std::function<void(int)>* fn = nullptr;
            int num_tasks = 0;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_start_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                fn = fn_;
                num_tasks = num_tasks_;
            }
            drain(*fn, num_tasks);
            {
                std::lock_guard<std::mutex> lk(mutex_);
                --pending_workers_;
            }
            cv_done_.notify_one();
        }
    }

    std::vector<std::thread> workers_;
    std::mutex run_mutex_;
    std::mutex mutex_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    const std::function<void(int)>* fn_{nullptr};
    int num_tasks_{0};
    std::atomic<int> next_task_{0};
    int pending_workers_{0};
    uint64_t generation_{0};
    bool stop_{false};
};

// Kill switch for the zero-copy FP8 dense path. Set ASEMA_DENSE_FP8_DIRECT=0 to fall back to the
// legacy dequantize-to-FP32 path (identical weights, slower, uses ~800 MB more RAM).
inline bool dense_fp8_direct_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("ASEMA_DENSE_FP8_DIRECT");
        return !(env && env[0] == '0');
    }();
    return enabled;
}

// Non-owning view of an FP8 matrix [rows, cols] with E8M0 scales [rows/32, cols/32].
struct Fp8MatView {
    const uint8_t* weights{nullptr};
    const uint8_t* scales{nullptr};
    int rows{0};
    int cols{0};
    bool valid() const { return weights && scales && rows > 0 && cols > 0 && (cols % 32) == 0 && (rows % 32) == 0; }

    // Sub-matrix of `row_count` rows starting at `row_begin` (both multiples of 32).
    Fp8MatView slice_rows(int row_begin, int row_count) const {
        Fp8MatView v;
        v.weights = weights + static_cast<size_t>(row_begin) * cols;
        v.scales = scales + static_cast<size_t>(row_begin / 32) * (cols / 32);
        v.rows = row_count;
        v.cols = cols;
        return v;
    }
};

namespace detail {

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// Eight E4M3 bytes -> eight exact floats (before block scaling).
inline __m256 decode8_e4m3(const uint8_t* p, __m256i mant_mask, __m256i sign_mask, __m256 rebias) {
    __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
    __m256i em = _mm256_slli_epi32(_mm256_and_si256(v, mant_mask), 20);
    __m256i sg = _mm256_slli_epi32(_mm256_and_si256(v, sign_mask), 24);
    return _mm256_mul_ps(_mm256_castsi256_ps(_mm256_or_si256(em, sg)), rebias);
}

inline float fp8_row_dot(const uint8_t* w_row, const uint8_t* s_row, int cols, const float* x) {
    const __m256i mant_mask = _mm256_set1_epi32(0x7F);
    const __m256i sign_mask = _mm256_set1_epi32(0x80);
    const __m256 rebias = _mm256_castsi256_ps(_mm256_set1_epi32(247 << 23)); // 2^120
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < cols; b += 32) {
        __m256 blk0 = _mm256_mul_ps(decode8_e4m3(w_row + b, mant_mask, sign_mask, rebias), _mm256_loadu_ps(x + b));
        __m256 blk1 = _mm256_mul_ps(decode8_e4m3(w_row + b + 8, mant_mask, sign_mask, rebias), _mm256_loadu_ps(x + b + 8));
        blk0 = _mm256_fmadd_ps(decode8_e4m3(w_row + b + 16, mant_mask, sign_mask, rebias), _mm256_loadu_ps(x + b + 16), blk0);
        blk1 = _mm256_fmadd_ps(decode8_e4m3(w_row + b + 24, mant_mask, sign_mask, rebias), _mm256_loadu_ps(x + b + 24), blk1);
        __m256 scale = _mm256_set1_ps(k_e8m0_lut[s_row[b >> 5]]);
        acc = _mm256_fmadd_ps(_mm256_add_ps(blk0, blk1), scale, acc);
    }
    return hsum256(acc);
}

} // namespace detail

// out[r] = sum_c dequant(W[r, c]) * x[c], rows split across the bounded worker pool.
inline void fp8_gemv(const Fp8MatView& m, const float* x, float* out) {
    const int rows = m.rows;
    const int s_cols = m.cols / 32;
    const int rows_per_task = 64; // multiple of 32 so each task owns whole scale rows
    const int num_tasks = (rows + rows_per_task - 1) / rows_per_task;
    M8WorkerPool::instance().run(num_tasks, [&](int t) {
        const int r0 = t * rows_per_task;
        const int r1 = std::min(rows, r0 + rows_per_task);
        for (int r = r0; r < r1; ++r) {
            out[r] = detail::fp8_row_dot(m.weights + static_cast<size_t>(r) * m.cols,
                                         m.scales + static_cast<size_t>(r / 32) * s_cols, m.cols, x);
        }
    });
}

} // namespace m8
} // namespace asema
