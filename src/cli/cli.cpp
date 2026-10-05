// ASEMA v0.1 — Agent #3: CLI parser implementation
// -----------------------------------------------------------------------------

#include "../../include/asema/cli.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace asema {
namespace cli {

Parser::Parser(ProgramInfo info, std::vector<ArgSpec> specs)
    : info_(std::move(info)), specs_(std::move(specs))
{
    for (size_t i = 0; i < specs_.size(); ++i) {
        if (!specs_[i].long_name.empty()) {
            long_index_[specs_[i].long_name] = i;
        }
        if (!specs_[i].short_name.empty()) {
            short_index_[specs_[i].short_name] = i;
        }
    }
}

ParsedArgs Parser::parse(int argc, char** argv) {
    ParsedArgs out;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        // --help / --version short-circuits
        if (a == "--help" || a == "-h") {
            out.help_requested = true;
            return out;
        }
        if (a == "--version" || a == "-V") {
            out.version_requested = true;
            return out;
        }

        // Long option: --name or --name=value
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            std::string body = a.substr(2);
            std::string name, value;

            auto eq = body.find('=');
            if (eq != std::string::npos) {
                name = body.substr(0, eq);
                value = body.substr(eq + 1);
            } else {
                name = body;
            }

            auto it = long_index_.find(name);
            if (it == long_index_.end()) {
                out.error = "Unknown option: --" + name;
                return out;
            }
            const auto& spec = specs_[it->second];

            if (spec.type == ArgType::Flag) {
                out.values[name] = (value.empty() || value == "true") ? "true" : "false";
            } else {
                if (value.empty()) {
                    // Need next argv element as value.
                    if (i + 1 >= argc) {
                        out.error = "Option --" + name + " requires a value";
                        return out;
                    }
                    value = argv[++i];
                }
                out.values[name] = value;
            }
            continue;
        }

        // Short option cluster: -x, -x value, -xyz (cluster of flags)
        if (a.size() >= 2 && a[0] == '-' && a[1] != '-') {
            std::string body = a.substr(1);
            // Single-char short flag
            if (body.size() == 1) {
                auto it = short_index_.find(body);
                if (it == short_index_.end()) {
                    out.error = "Unknown short option: -" + body;
                    return out;
                }
                const auto& spec = specs_[it->second];
                if (spec.type == ArgType::Flag) {
                    out.values[spec.long_name] = "true";
                } else {
                    if (i + 1 >= argc) {
                        out.error = "Option -" + body + " requires a value";
                        return out;
                    }
                    out.values[spec.long_name] = argv[++i];
                }
            } else {
                // Cluster of boolean flags: -abc means -a -b -c.
                for (char c : body) {
                    std::string s(1, c);
                    auto it = short_index_.find(s);
                    if (it == short_index_.end()) {
                        out.error = "Unknown short option: -" + s;
                        return out;
                    }
                    const auto& spec = specs_[it->second];
                    if (spec.type != ArgType::Flag) {
                        out.error = "Short option -" + s + " is not a flag";
                        return out;
                    }
                    out.values[spec.long_name] = "true";
                }
            }
            continue;
        }

        // Positional
        out.positional.push_back(a);
    }

    // Apply defaults for Value-type options not provided
    for (const auto& spec : specs_) {
        if (spec.type == ArgType::Value && !spec.long_name.empty()) {
            if (out.values.find(spec.long_name) == out.values.end() &&
                !spec.default_value.empty()) {
                out.values[spec.long_name] = spec.default_value;
            }
        }
    }

    return out;
}

std::string Parser::render_help() const {
    std::ostringstream os;
    os << info_.name << " " << info_.version << "\n";
    if (!info_.description.empty()) os << info_.description << "\n";
    os << "\nUsage:\n  " << info_.name << " " << info_.usage << "\n\n";
    os << "Options:\n";

    for (const auto& spec : specs_) {
        std::string lead;
        if (!spec.short_name.empty()) lead = "-" + spec.short_name + ", ";
        else                          lead = "    ";
        os << "  " << lead << "--" << spec.long_name;
        if (spec.type != ArgType::Flag) os << " <value>";
        os << "\n";
        if (!spec.help.empty()) {
            os << "        " << spec.help;
            if (spec.type == ArgType::Value && !spec.default_value.empty()) {
                os << " (default: " << spec.default_value << ")";
            }
            os << "\n";
        }
    }
    os << "  -h, --help\n        Show this help and exit\n";
    os << "  -V, --version\n        Show version and exit\n";
    os << "\nExit codes:\n";
    os << "  0   RC_SUCCESS\n";
    os << "  2   RC_INVALID_ARGUMENT\n";
    os << "  3   RC_MODEL_ERROR\n";
    os << "  4   RC_IO_ERROR\n";
    os << "  5   RC_RUNTIME_ERROR\n";
    os << "  6   RC_BENCHMARK_ERROR\n";
    os << "  7   RC_VERIFICATION_ERROR\n";
    return os.str();
}

} // namespace cli
} // namespace asema
