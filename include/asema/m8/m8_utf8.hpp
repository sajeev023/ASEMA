#pragma once

// Streaming-safe UTF-8 helpers.
//
// The tokenizer is byte-level BPE, so one Unicode character (e.g. an emoji or a CJK glyph) can be
// split across several generated tokens. Emitting each token's raw bytes immediately would hand a
// terminal half a character, which renders as mojibake or U+FFFD. Streaming code must therefore
// only emit the longest *complete* UTF-8 prefix and hold back an incomplete trailing sequence
// until the next token completes it.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace asema {
namespace m8 {

// Length (bytes) of the sequence started by lead byte `c`, or 0 if `c` is not a valid lead byte.
inline size_t utf8_sequence_length(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return 2;
    if (c >= 0xE0 && c <= 0xEF) return 3;
    if (c >= 0xF0 && c <= 0xF4) return 4;
    return 0;
}

// Returns the number of leading bytes of `s` that form complete, well-formed UTF-8 sequences.
// A truncated sequence at the very end is excluded (it may be completed by a later token).
// Invalid bytes in the middle are passed through (counted as complete single bytes) so a stray
// byte can never stall the stream forever.
inline size_t utf8_complete_prefix_len(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = utf8_sequence_length(c);
        if (len == 0) { ++i; continue; }          // stray continuation / invalid lead: pass through
        if (i + len > n) {
            // Incomplete only if all bytes present so far are valid continuation bytes.
            bool all_cont = true;
            for (size_t k = i + 1; k < n; ++k) {
                if ((static_cast<unsigned char>(s[k]) & 0xC0) != 0x80) { all_cont = false; break; }
            }
            return all_cont ? i : n;
        }
        bool ok = true;
        for (size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) { ok = false; break; }
        }
        i += ok ? len : 1;
    }
    return n;
}

// True if the whole string is well-formed UTF-8.
inline bool utf8_is_valid(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        size_t len = utf8_sequence_length(static_cast<unsigned char>(s[i]));
        if (len == 0 || i + len > n) return false;
        for (size_t k = 1; k < len; ++k) {
            if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
        }
        i += len;
    }
    return true;
}

// Decodes well-formed UTF-8 into code points (invalid bytes become U+FFFD).
inline std::vector<uint32_t> utf8_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = utf8_sequence_length(c);
        if (len == 0 || i + len > n) { out.push_back(0xFFFD); ++i; continue; }
        uint32_t cp = (len == 1) ? c : (c & (0xFF >> (len + 1)));
        bool ok = true;
        for (size_t k = 1; k < len; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); ++i; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

} // namespace m8
} // namespace asema
