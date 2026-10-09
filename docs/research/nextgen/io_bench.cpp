// Read-only storage benchmark: random aligned reads from real shard files with FILE_FLAG_NO_BUFFERING
// (so the Windows cache cannot inflate results). One thread per outstanding request (QD = threads).
// usage: io_bench <seconds> <label=file[,label=file]> <block_bytes[,..]> <qd[,..]>
//   Several drives given in one spec run CONCURRENTLY (each gets `qd` threads) to measure D+E together.
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Target { std::string label, path; uint64_t size{0}; };

static std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out; std::string cur;
    for (char ch : s) { if (ch == c) { out.push_back(cur); cur.clear(); } else cur += ch; }
    out.push_back(cur); return out;
}

static double cpu_seconds() {
    FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto t = [](FILETIME f) { return ((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) / 1e7; };
    return t(k) + t(u);
}

static void worker(const Target& t, uint64_t block, double seconds, unsigned seed, std::vector<double>& lat, uint64_t& bytes) {
    // IO_BUFFERED=1 reads through the Windows file cache exactly like the engine does today (no NO_BUFFERING).
    const bool buffered = std::getenv("IO_BUFFERED") != nullptr;
    HANDLE h = CreateFileA(t.path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           (buffered ? 0 : FILE_FLAG_NO_BUFFERING) | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "open failed %s\n", t.path.c_str()); return; }
    void* buf = VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::mt19937_64 rng(seed);
    const uint64_t slots = (t.size > block) ? (t.size - block) / 4096 : 1;
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() < end) {
        const uint64_t off = (rng() % slots) * 4096;
        OVERLAPPED ov{}; ov.Offset = (DWORD)(off & 0xFFFFFFFF); ov.OffsetHigh = (DWORD)(off >> 32);
        DWORD got = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (!ReadFile(h, buf, (DWORD)block, &got, &ov)) break;
        lat.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        bytes += got;
    }
    VirtualFree(buf, 0, MEM_RELEASE);
    CloseHandle(h);
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: io_bench seconds label=file,.. blocks qds\n"); return 2; }
    const double seconds = std::atof(argv[1]);
    std::vector<Target> targets;
    for (auto& spec : split(argv[2], ',')) {
        auto p = split(spec, '=');
        Target t; t.label = p[0]; t.path = p[1];
        WIN32_FILE_ATTRIBUTE_DATA fa{}; GetFileAttributesExA(t.path.c_str(), GetFileExInfoStandard, &fa);
        t.size = (uint64_t(fa.nFileSizeHigh) << 32) | fa.nFileSizeLow;
        targets.push_back(t);
    }
    std::printf("%-8s %10s %4s %9s %9s %9s %8s %8s %6s\n", "drives", "block", "QD", "MB/s", "IOPS", "avg ms", "p50 ms", "p99 ms", "CPU%");
    for (auto& bs : split(argv[3], ','))
        for (auto& q : split(argv[4], ',')) {
            const uint64_t block = std::strtoull(bs.c_str(), nullptr, 10);
            const int qd = std::atoi(q.c_str());
            std::vector<std::vector<double>> lats(targets.size() * qd);
            std::vector<uint64_t> bytes(targets.size() * qd, 0);
            std::vector<std::thread> th;
            const double cpu0 = cpu_seconds();
            const auto w0 = std::chrono::steady_clock::now();
            for (size_t d = 0; d < targets.size(); ++d)
                for (int i = 0; i < qd; ++i) {
                    const size_t k = d * qd + i;
                    const unsigned seed_base = std::getenv("IO_SEED_BASE") ? unsigned(std::atoi(std::getenv("IO_SEED_BASE"))) : 1000u;
                    th.emplace_back(worker, std::cref(targets[d]), block, seconds, seed_base + unsigned(k), std::ref(lats[k]), std::ref(bytes[k]));
                }
            for (auto& t : th) t.join();
            const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            const double cpu = cpu_seconds() - cpu0;
            std::string names;
            for (auto& t : targets) names += (names.empty() ? "" : "+") + t.label;
            std::vector<double> all; uint64_t tot = 0;
            for (auto& l : lats) all.insert(all.end(), l.begin(), l.end());
            for (auto b : bytes) tot += b;
            std::sort(all.begin(), all.end());
            double avg = 0; for (double v : all) avg += v; if (!all.empty()) avg /= all.size();
            std::printf("%-8s %10llu %4d %9.1f %9.0f %9.3f %8.3f %8.3f %6.1f\n", names.c_str(), (unsigned long long)block, qd,
                        tot / 1048576.0 / wall, all.size() / wall, avg,
                        all.empty() ? 0 : all[all.size() / 2], all.empty() ? 0 : all[size_t(all.size() * 0.99)],
                        100.0 * cpu / wall / 16.0);
            std::fflush(stdout);
        }
    return 0;
}
