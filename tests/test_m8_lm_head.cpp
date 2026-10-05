#include "asema/m8/m8_paths.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <future>
#include <immintrin.h>
#include <windows.h>

static float dot_product_bf16_fp32(const uint16_t* b16_row, const float* x, int size) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= size - 32; i += 32) {
        __m128i raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i));
        __m128i raw1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 8));
        __m128i raw2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 16));
        __m128i raw3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b16_row + i + 24));

        __m256i f32_0_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw0), 16);
        __m256i f32_1_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw1), 16);
        __m256i f32_2_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw2), 16);
        __m256i f32_3_int = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw3), 16);

        __m256 f32_0 = _mm256_castsi256_ps(f32_0_int);
        __m256 f32_1 = _mm256_castsi256_ps(f32_1_int);
        __m256 f32_2 = _mm256_castsi256_ps(f32_2_int);
        __m256 f32_3 = _mm256_castsi256_ps(f32_3_int);

        __m256 x0 = _mm256_loadu_ps(x + i);
        __m256 x1 = _mm256_loadu_ps(x + i + 8);
        __m256 x2 = _mm256_loadu_ps(x + i + 16);
        __m256 x3 = _mm256_loadu_ps(x + i + 24);

        sum0 = _mm256_fmadd_ps(f32_0, x0, sum0);
        sum1 = _mm256_fmadd_ps(f32_1, x1, sum1);
        sum2 = _mm256_fmadd_ps(f32_2, x2, sum2);
        sum3 = _mm256_fmadd_ps(f32_3, x3, sum3);
    }

    __m256 sum = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    __m128 low = _mm256_castps256_ps128(sum);
    __m128 high = _mm256_extractf128_ps(sum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float total = _mm_cvtss_f32(sum128);

    for (; i < size; ++i) {
        uint32_t u = static_cast<uint32_t>(b16_row[i]) << 16;
        float val = *reinterpret_cast<float*>(&u);
        total += val * x[i];
    }
    return total;
}

int main() {
    std::cout << "Testing LM Head and Embedding mmap...\n";
    const std::string p_emb_s = asema::m8::paths::primary_shards() + "/model-00002-of-00048.safetensors";
    const char* p_emb = p_emb_s.c_str();
    const std::string p_head_s = asema::m8::paths::primary_shards() + "/model-00043-of-00048.safetensors";
    const char* p_head = p_head_s.c_str();

    HANDLE h_head = CreateFileA(p_head, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h_head == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open head file: " << GetLastError() << "\n";
        return 1;
    }
    HANDLE m_head = CreateFileMappingA(h_head, NULL, PAGE_READONLY, 0, 0, NULL);
    const uint8_t* ptr_head = (const uint8_t*)MapViewOfFile(m_head, FILE_MAP_READ, 0, 0, 0);
    if (!ptr_head) {
        std::cerr << "Failed to map head file: " << GetLastError() << "\n";
        return 1;
    }

    std::cout << "Successfully mapped head file (1.23 GB)\n";

    // Read norm weight
    const uint16_t* norm_raw = reinterpret_cast<const uint16_t*>(ptr_head + 1323827384);
    std::vector<float> norm(5120);
    for (int i = 0; i < 5120; ++i) {
        uint32_t u = static_cast<uint32_t>(norm_raw[i]) << 16;
        norm[i] = *reinterpret_cast<float*>(&u);
    }
    std::cout << "Norm[0]: " << norm[0] << ", Norm[1]: " << norm[1] << "\n";

    // Head weight rows
    const uint16_t* head_rows = reinterpret_cast<const uint16_t*>(ptr_head + 184);

    // Dummy test vector
    std::vector<float> test_h(5120, 0.01f);
    std::vector<float> logits(129280);

    const int num_threads = 8;
    int chunk_size = 129280 / num_threads;

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::future<void>> futures;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? 129280 : (t + 1) * chunk_size;
        futures.push_back(std::async(std::launch::async, [&, start, end]() {
            for (int v = start; v < end; ++v) {
                const uint16_t* row = head_rows + static_cast<size_t>(v) * 5120;
                logits[v] = dot_product_bf16_fp32(row, test_h.data(), 5120);
            }
        }));
    }
    for (auto& f : futures) f.get();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Computed all 129,280 logits in " << ms << " ms!\n";

    float max_l = -1e9f;
    int best_v = -1;
    for (int v = 0; v < 129280; ++v) {
        if (logits[v] > max_l) {
            max_l = logits[v];
            best_v = v;
        }
    }
    std::cout << "Best token: " << best_v << " with logit: " << max_l << "\n";

    UnmapViewOfFile(ptr_head);
    CloseHandle(m_head);
    CloseHandle(h_head);
    return 0;
}
