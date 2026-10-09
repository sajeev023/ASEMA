// shardcopy: resumable, verified file copy + SHA-256 for multi-GB checkpoint shards.  Never deletes anything.
//   shardcopy hash <file> <log.jsonl>                    -> SHA-256 of a file (unbuffered read), appended to the log
//   shardcopy copy <src> <dst> <log.jsonl>               -> copy to <dst>.part, SHA-256 of the bytes read from src,
//                                                           rename to <dst>, re-read <dst> unbuffered, compare hash + size
// Resumable at file granularity: a `copy` is skipped when the log already holds an ok record for the same src/dst/size
// and dst exists with that size.  Reads/writes use FILE_FLAG_NO_BUFFERING so the verify pass reads the disk, not the cache.
#include <windows.h>
#include <bcrypt.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const size_t kChunk = 8u << 20;   // 8 MiB, multiple of every sector size

struct Sha256 {
    BCRYPT_ALG_HANDLE alg{}; BCRYPT_HASH_HANDLE h{}; std::vector<UCHAR> obj;
    Sha256() {
        BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0);
        DWORD n = 0, got = 0; BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, (PUCHAR)&n, sizeof(n), &got, 0);
        obj.resize(n); BCryptCreateHash(alg, &h, obj.data(), n, nullptr, 0, 0);
    }
    void update(const void* p, size_t n) { BCryptHashData(h, (PUCHAR)p, (ULONG)n, 0); }
    std::string finish() {
        UCHAR d[32]; BCryptFinishHash(h, d, 32, 0); BCryptDestroyHash(h); BCryptCloseAlgorithmProvider(alg, 0);
        static const char* hx = "0123456789abcdef"; std::string s;
        for (UCHAR b : d) { s += hx[b >> 4]; s += hx[b & 15]; }
        return s;
    }
};

static uint64_t file_size(const std::string& p) {
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (!GetFileAttributesExA(p.c_str(), GetFileExInfoStandard, &a)) return UINT64_MAX;
    return (uint64_t(a.nFileSizeHigh) << 32) | a.nFileSizeLow;
}

static void* aligned_buf() { return VirtualAlloc(nullptr, kChunk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE); }

// Reads a whole file unbuffered; if `out` is valid also writes every chunk there (the final block is zero-padded to a
// 4 KiB multiple; the caller truncates to the true size).  Returns false on any I/O error.
static bool stream(const std::string& path, HANDLE out, std::string& sha, uint64_t& total, double& secs) {
    HANDLE in = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (in == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "open failed: %s (%lu)\n", path.c_str(), GetLastError()); return false; }
    void* buf = aligned_buf(); Sha256 h; total = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        DWORD got = 0;
        if (!ReadFile(in, buf, (DWORD)kChunk, &got, nullptr)) { std::fprintf(stderr, "read error %lu\n", GetLastError()); return false; }
        if (got == 0) break;
        h.update(buf, got); total += got;
        if (out != INVALID_HANDLE_VALUE) {
            DWORD wr = 0; DWORD padded = (got + 4095) & ~4095u;
            if (padded > got) std::memset((char*)buf + got, 0, padded - got);
            if (!WriteFile(out, buf, padded, &wr, nullptr) || wr != padded) { std::fprintf(stderr, "write error %lu\n", GetLastError()); return false; }
        }
        if (got < kChunk) break;
    }
    secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    VirtualFree(buf, 0, MEM_RELEASE); CloseHandle(in);
    sha = h.finish(); return true;
}

static std::string esc(const std::string& s) { std::string o; for (char c : s) { if (c == '\\' || c == '"') o += '\\'; o += c; } return o; }

static bool already_done(const std::string& log, const std::string& src, const std::string& dst, uint64_t size) {
    std::ifstream f(log); std::string line;
    const std::string a = "\"src\":\"" + esc(src) + "\"", b = "\"dst\":\"" + esc(dst) + "\"", c = "\"size\":" + std::to_string(size) + ",";
    while (std::getline(f, line))
        if (line.find(a) != std::string::npos && line.find(b) != std::string::npos && line.find(c) != std::string::npos &&
            line.find("\"ok\":true") != std::string::npos) return file_size(dst) == size;
    return false;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: shardcopy hash <file> <log> | copy <src> <dst> <log>\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "hash") {
        std::string sha; uint64_t n = 0; double s = 0;
        if (!stream(argv[2], INVALID_HANDLE_VALUE, sha, n, s)) return 1;
        std::ofstream(argv[3], std::ios::app) << "{\"op\":\"hash\",\"path\":\"" << esc(argv[2]) << "\",\"size\":" << n
            << ",\"sha256\":\"" << sha << "\",\"read_MBps\":" << (n / 1048576.0 / s) << "}\n";
        std::printf("%s  %llu  %s\n", sha.c_str(), (unsigned long long)n, argv[2]); return 0;
    }
    if (mode != "copy" || argc < 5) return 2;
    const std::string src = argv[2], dst = argv[3], log = argv[4], part = dst + ".part";
    const uint64_t want = file_size(src);
    if (want == UINT64_MAX) { std::fprintf(stderr, "source missing: %s\n", src.c_str()); return 1; }
    if (already_done(log, src, dst, want)) { std::printf("SKIP (already verified) %s\n", dst.c_str()); return 0; }
    if (file_size(dst) != UINT64_MAX) { std::fprintf(stderr, "REFUSE: destination exists without a verified record: %s\n", dst.c_str()); return 3; }
    HANDLE out = CreateFileA(part.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (out == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "cannot create %s (%lu)\n", part.c_str(), GetLastError()); return 1; }
    std::string sha_src; uint64_t n_src = 0; double t_copy = 0;
    if (!stream(src, out, sha_src, n_src, t_copy)) { CloseHandle(out); return 1; }
    FlushFileBuffers(out); CloseHandle(out);
    {   // trim the zero padding of the last block: needs an ordinary (buffered) handle, not the NO_BUFFERING one
        HANDLE t = CreateFileA(part.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        LARGE_INTEGER li; li.QuadPart = (LONGLONG)n_src;
        if (t == INVALID_HANDLE_VALUE || !SetFilePointerEx(t, li, nullptr, FILE_BEGIN) || !SetEndOfFile(t)) {
            std::fprintf(stderr, "truncate failed %lu\n", GetLastError()); if (t != INVALID_HANDLE_VALUE) CloseHandle(t); return 1;
        }
        FlushFileBuffers(t); CloseHandle(t);
    }
    if (!MoveFileExA(part.c_str(), dst.c_str(), 0)) { std::fprintf(stderr, "rename failed %lu\n", GetLastError()); return 1; }
    std::string sha_dst; uint64_t n_dst = 0; double t_ver = 0;
    const bool read_ok = stream(dst, INVALID_HANDLE_VALUE, sha_dst, n_dst, t_ver);
    const bool ok = read_ok && sha_dst == sha_src && n_dst == n_src && n_src == want;
    std::ofstream(log, std::ios::app) << "{\"op\":\"copy\",\"src\":\"" << esc(src) << "\",\"dst\":\"" << esc(dst) << "\",\"size\":" << n_src
        << ",\"sha256_src\":\"" << sha_src << "\",\"sha256_dst\":\"" << sha_dst << "\",\"ok\":" << (ok ? "true" : "false")
        << ",\"copy_MBps\":" << (n_src / 1048576.0 / t_copy) << ",\"verify_MBps\":" << (n_dst / 1048576.0 / t_ver) << "}\n";
    std::printf("%s %s -> %s  %llu B  sha256 %s  copy %.0f MB/s verify %.0f MB/s\n", ok ? "OK  " : "FAIL", src.c_str(), dst.c_str(),
                (unsigned long long)n_src, sha_src.c_str(), n_src / 1048576.0 / t_copy, n_dst / 1048576.0 / t_ver);
    return ok ? 0 : 4;
}
