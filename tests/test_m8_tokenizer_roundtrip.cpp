// Tokenizer / decoder forensics (no model weights needed).
//  - vocabulary size and token-ID range
//  - encode -> decode round trip for ASCII, punctuation and non-ASCII UTF-8
//  - per-token diagnostic: ID, raw token bytes (hex), codepoints
//  - streaming simulation: emitting only the complete-UTF-8 prefix after each token must (a) never
//    emit an invalid sequence and (b) reassemble to exactly the original text
//
// Usage: test_m8_tokenizer_roundtrip [path/to/tokenizer.json]
// The path may also come from ASEMA_TOKENIZER_JSON.
#include "asema/m8/m8_model_adapter.hpp"
#include "asema/m8/m8_utf8.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace asema::m8;

static int g_failures = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            ++g_failures;                                 \
            std::printf("  FAIL: " __VA_ARGS__);          \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

static std::string hex_bytes(const std::string& s, size_t max_bytes = 12) {
    std::string out;
    char buf[4];
    for (size_t i = 0; i < s.size() && i < max_bytes; ++i) {
        std::snprintf(buf, sizeof(buf), "%02X ", static_cast<unsigned char>(s[i]));
        out += buf;
    }
    if (s.size() > max_bytes) out += "...";
    return out;
}

static void run_case(const Tokenizer& tok, const std::string& text) {
    std::printf("\n[case] \"%s\"\n", text.c_str());
    std::vector<int> ids = tok.encode(text);
    CHECK(!ids.empty(), "encode produced no tokens");

    std::printf("  %-7s %-26s %s\n", "ID", "RAW BYTES (hex)", "CODEPOINTS");
    for (int id : ids) {
        CHECK(id >= 0 && id < tok.vocab_size(), "token id %d out of range [0,%d)", id, tok.vocab_size());
        std::string raw = tok.decode({id});
        std::string cps;
        char buf[16];
        for (uint32_t cp : utf8_codepoints(raw)) {
            std::snprintf(buf, sizeof(buf), "U+%04X ", cp);
            cps += buf;
        }
        std::printf("  %-7d %-26s %s\n", id, hex_bytes(raw).c_str(), cps.c_str());
    }

    // Full round trip
    std::string decoded = tok.decode(ids);
    CHECK(decoded == text, "round trip mismatch: got \"%s\"", decoded.c_str());

    // Streaming simulation (what the runner does token by token)
    std::vector<int> so_far;
    std::string streamed;
    size_t emitted = 0;
    for (int id : ids) {
        so_far.push_back(id);
        std::string cur = tok.decode(so_far);
        size_t stable = utf8_complete_prefix_len(cur);
        if (stable > emitted) {
            std::string delta = cur.substr(emitted, stable - emitted);
            CHECK(utf8_is_valid(delta), "streamed delta is not valid UTF-8 (%s)", hex_bytes(delta).c_str());
            streamed += delta;
            emitted = stable;
        }
    }
    CHECK(streamed == text, "streamed text mismatch: got \"%s\"", streamed.c_str());
    std::printf("  tokens=%zu round_trip=%s streaming=%s\n", ids.size(),
                decoded == text ? "OK" : "BAD", streamed == text ? "OK" : "BAD");
}

// A character split across tokens must be held back, then released once complete.
static void test_split_character_streaming() {
    std::printf("\n[case] synthetic split of U+1F60A (4-byte emoji)\n");
    const std::string emoji = "\xF0\x9F\x98\x8A";
    for (size_t cut = 1; cut < emoji.size(); ++cut) {
        std::string partial = emoji.substr(0, cut);
        CHECK(utf8_complete_prefix_len(partial) == 0, "partial sequence (%zu bytes) must be held back", cut);
    }
    CHECK(utf8_complete_prefix_len(emoji) == 4, "complete emoji must be released");
    CHECK(utf8_complete_prefix_len("ok\xE4\xBD") == 2, "ASCII before a truncated CJK char must be released");
    CHECK(utf8_complete_prefix_len("\xE4\xBD\xA0\xE5") == 3, "complete CJK char before truncated one released");
    CHECK(utf8_complete_prefix_len("a\x80z") == 3, "stray continuation byte must not stall the stream");
    std::printf("  hold-back / release logic OK\n");
}

int main(int argc, char** argv) {
    std::string path;
    if (argc > 1) path = argv[1];
    else if (const char* env = std::getenv("ASEMA_TOKENIZER_JSON")) path = env;
    else {
        const char* candidates[] = {
            "examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json",
            "../examples/real_model/DeepSeek-V4.1-Flash/hf/tokenizer.json",
        };
        for (const char* c : candidates) {
            if (std::filesystem::exists(c)) { path = c; break; }
        }
    }

    test_split_character_streaming();

    if (path.empty() || !std::filesystem::exists(path)) {
        std::printf("\ntokenizer.json not found; skipped vocabulary tests (set ASEMA_TOKENIZER_JSON).\n");
        std::printf(g_failures == 0 ? "PASS (partial)\n" : "FAILED (%d)\n", g_failures);
        return g_failures == 0 ? 0 : 1;
    }

    Tokenizer tok;
    if (!tok.load(path)) {
        std::printf("FAIL: could not load %s\n", path.c_str());
        return 1;
    }
    std::printf("tokenizer: %s\nvocab_size=%d\n", path.c_str(), tok.vocab_size());
    CHECK(tok.vocab_size() == 129280, "expected vocab 129280, got %d", tok.vocab_size());

    run_case(tok, "hi");
    run_case(tok, "hello");
    run_case(tok, "2 + 2");
    run_case(tok, "NVMe");
    run_case(tok, "Hey! What's up? It's fine, isn't it (really)?");
    run_case(tok, "caf\xC3\xA9 \xE2\x80\x94 \xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x98\x8A");

    std::printf(g_failures == 0 ? "\nPASS\n" : "\nFAILED (%d)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
