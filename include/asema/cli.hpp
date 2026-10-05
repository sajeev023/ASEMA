#pragma once

// ASEMA v0.1 — Agent #3: minimal CLI argument parser
// -----------------------------------------------------------------------------
// A small, no-dependency argument parser used by all M6 CLI tools
// (asema-pack, asema-run, asema-bench). Supports:
//
//   * long options:  --name value
//   * long flags:    --name  (boolean)
//   * short flags:   -x      (single character, boolean)
//   * positional:    anything not starting with '-'
//
// Designed for predictable behavior, easy help text, and explicit error
// reporting. This is NOT a replacement for argparse / cxxopts; it is a
// deliberate, auditable 200-line implementation.
// -----------------------------------------------------------------------------

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace asema {
namespace cli {

// -----------------------------------------------------------------------------
// Argument spec
// -----------------------------------------------------------------------------
enum class ArgType {
    Flag,        // --name (no value), --no-name also supported
    Value,       // --name value   (required value)
    Optional,    // --name value   (value optional; default applied if absent)
    Positional,  // anonymous positional (collected into a vector)
};

struct ArgSpec {
    std::string long_name;       // e.g. "model" (-> --model)
    std::string short_name;      // e.g. "m"  (-> -m); empty = none
    ArgType type{ArgType::Value};
    std::string default_value;   // applied when option is absent (Value only)
    std::string help;            // for --help
};

// -----------------------------------------------------------------------------
// Parsed result
// -----------------------------------------------------------------------------
struct ParsedArgs {
    // Map: long_name -> value. For flags, value is "true" or "false".
    std::map<std::string, std::string> values;

    // Positional arguments in order of appearance.
    std::vector<std::string> positional;

    // True if --help was requested.
    bool help_requested{false};

    // True if --version was requested.
    bool version_requested{false};

    // First error encountered.
    std::string error;

    bool ok() const { return error.empty(); }

    bool has(const std::string& name) const {
        auto it = values.find(name);
        return it != values.end() && it->second == "true";
    }

    std::string get(const std::string& name, const std::string& fallback = "") const {
        auto it = values.find(name);
        if (it == values.end()) return fallback;
        return it->second;
    }

    int64_t get_int(const std::string& name, int64_t fallback = 0) const {
        auto v = get(name);
        if (v.empty()) return fallback;
        try { return std::stoll(v); } catch (...) { return fallback; }
    }

    uint64_t get_uint(const std::string& name, uint64_t fallback = 0) const {
        auto v = get(name);
        if (v.empty()) return fallback;
        try { return std::stoull(v); } catch (...) { return fallback; }
    }

    double get_double(const std::string& name, double fallback = 0.0) const {
        auto v = get(name);
        if (v.empty()) return fallback;
        try { return std::stod(v); } catch (...) { return fallback; }
    }
};

// -----------------------------------------------------------------------------
// Program metadata
// -----------------------------------------------------------------------------
struct ProgramInfo {
    std::string name;          // e.g. "asema-bench"
    std::string version;       // e.g. "0.1.0"
    std::string description;   // one-line summary
    std::string usage;         // usage line e.g. "[options] <model>"
};

// -----------------------------------------------------------------------------
// Parser
// -----------------------------------------------------------------------------
class Parser {
public:
    Parser(ProgramInfo info, std::vector<ArgSpec> specs);

    // Parse argv[1..argc-1]. argv[0] is the program name.
    ParsedArgs parse(int argc, char** argv);

    // Render help text.
    std::string render_help() const;

    // Render version.
    std::string render_version() const { return info_.name + " " + info_.version + "\n"; }

private:
    ProgramInfo info_;
    std::vector<ArgSpec> specs_;
    std::map<std::string, size_t> long_index_;     // long_name -> spec index
    std::map<std::string, size_t> short_index_;    // short_name -> spec index
};

// -----------------------------------------------------------------------------
// Exit codes (shared by all M6 CLIs). Renamed to avoid colliding with
// stdlib.h's EXIT_SUCCESS macro.
// -----------------------------------------------------------------------------
constexpr int RC_SUCCESS              = 0;
constexpr int RC_INVALID_ARGUMENT     = 2;
constexpr int RC_MODEL_ERROR          = 3;
constexpr int RC_IO_ERROR             = 4;
constexpr int RC_RUNTIME_ERROR        = 5;
constexpr int RC_BENCHMARK_ERROR      = 6;
constexpr int RC_VERIFICATION_ERROR   = 7;

} // namespace cli
} // namespace asema
