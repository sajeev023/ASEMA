#pragma once
// Terminal presentation for the asema CLI: colours, logo, startup steps, boxes, code-fence styling.
// Everything degrades to plain text when stdout is not a console or NO_COLOR is set, so redirected
// output (logs, tests, CI) stays clean. Nothing in here measures anything; callers pass real values.

#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace asema {
namespace ui {

struct State {
    bool color{false};    // ANSI escapes enabled
    bool animate{false};  // short typed-out startup (only on an interactive console)
};
inline State& state() { static State s; return s; }

inline void init() {
    State& s = state();
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    const bool is_console = out != INVALID_HANDLE_VALUE && GetConsoleMode(out, &mode);
    if (is_console && SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
        s.color = std::getenv("NO_COLOR") == nullptr;
    }
    s.animate = s.color && std::getenv("ASEMA_NO_ANIMATION") == nullptr;
}

inline const char* c(const char* code) { return state().color ? code : ""; }
inline const char* reset() { return c("\x1b[0m"); }
inline const char* bold() { return c("\x1b[1m"); }
inline const char* dim() { return c("\x1b[2m"); }
inline const char* cyan() { return c("\x1b[96m"); }
inline const char* green() { return c("\x1b[92m"); }
inline const char* yellow() { return c("\x1b[93m"); }
inline const char* red() { return c("\x1b[91m"); }

inline void sleep_ms(int ms) {
    if (state().animate && ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Display width of a UTF-8 string, ignoring ANSI escape sequences.
inline size_t display_width(const std::string& s) {
    size_t w = 0;
    for (size_t i = 0; i < s.size();) {
        const unsigned char ch = static_cast<unsigned char>(s[i]);
        if (ch == 0x1b && i + 1 < s.size() && s[i + 1] == '[') {
            i += 2;
            while (i < s.size() && !(s[i] >= '@' && s[i] <= '~')) ++i;
            ++i;
        } else {
            if ((ch & 0xC0) != 0x80) ++w;  // count code points, not continuation bytes
            ++i;
        }
    }
    return w;
}

inline std::string pad_to(const std::string& s, size_t width) {
    const size_t w = display_width(s);
    return w >= width ? s : s + std::string(width - w, ' ');
}

inline std::string repeat(const char* unit, size_t n) {
    std::string out;
    for (size_t i = 0; i < n; ++i) out += unit;
    return out;
}

// Rounded box with left-aligned lines.
inline void box(const std::vector<std::string>& lines, size_t inner_width = 60, const char* tint = nullptr) {
    const char* t = tint ? tint : dim();
    const std::string bar = repeat("─", inner_width + 2);
    std::cout << t << "╭" << bar << "╮" << reset() << "\n";
    for (const auto& l : lines) {
        std::cout << t << "│" << reset() << " " << pad_to(l, inner_width) << " " << t << "│" << reset() << "\n";
    }
    std::cout << t << "╰" << bar << "╯" << reset() << "\n";
}

inline void logo() {
    static const char* rows[6] = {
        " █████╗ ███████╗███████╗███╗   ███╗ █████╗ ",
        "██╔══██╗██╔════╝██╔════╝████╗ ████║██╔══██╗",
        "███████║███████╗█████╗  ██╔████╔██║███████║",
        "██╔══██║╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║",
        "██║  ██║███████║███████╗██║ ╚═╝ ██║██║  ██║",
        "╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝"};
    const size_t inner = 56;
    auto centered = [&](const std::string& text, const char* style) {
        const size_t w = display_width(text);
        const size_t left = w < inner ? (inner - w) / 2 : 0;
        return std::string(left, ' ') + style + text + reset();
    };
    std::vector<std::string> lines;
    lines.push_back("");
    for (const char* r : rows) lines.push_back(centered(r, (std::string(cyan()) + bold()).c_str()));
    lines.push_back("");
    lines.push_back(centered("ADAPTIVE SPARSE-EXPERT MEMORY ARCHITECTURE", bold()));
    lines.push_back("");
    box(lines, inner, cyan());
}

// One startup line: "  LABEL ........ STATUS  detail". The dots are typed out when animating.
inline void step(const std::string& label, bool ok, const std::string& status, const std::string& detail = "") {
    const size_t label_col = 30;
    std::string dots;
    for (size_t i = display_width(label); i < label_col; ++i) dots += '.';
    std::cout << "  " << label << " " << dim();
    if (state().animate) {
        for (char ch : dots) { std::cout << ch << std::flush; sleep_ms(2); }
    } else {
        std::cout << dots;
    }
    std::cout << reset() << " " << (ok ? green() : red()) << bold() << status << reset();
    if (!detail.empty()) std::cout << dim() << "  " << detail << reset();
    std::cout << "\n" << std::flush;
    sleep_ms(18);
}

// Streams model text to the terminal, styling fenced ``` code blocks. State is kept across calls, so
// a fence split over several tokens is still recognised. The text itself is never altered.
class StreamStyler {
public:
    void write(const std::string& piece) {
        for (char ch : piece) {
            if (ch == '`') {
                if (++run_ == 3) { pending_toggle_ = true; run_ = 0; }
            } else {
                run_ = 0;
            }
            std::cout << ch;
            if (ch == '\n' && pending_toggle_) {
                pending_toggle_ = false;
                in_code_ = !in_code_;
                std::cout << (in_code_ ? yellow() : reset());
            }
        }
        std::cout << std::flush;
    }
    void finish() {
        if (in_code_ || pending_toggle_) std::cout << reset();
        in_code_ = false;
        pending_toggle_ = false;
        run_ = 0;
    }
private:
    int run_{0};
    bool pending_toggle_{false};
    bool in_code_{false};
};

} // namespace ui
} // namespace asema
