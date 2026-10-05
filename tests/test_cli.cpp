// ASEMA v0.1 — Agent #3: CLI parser tests
// -----------------------------------------------------------------------------

#include "../include/asema/cli.hpp"

#include <iostream>
#include <cstring>

using namespace asema;

namespace {

int g_failed = 0;
#define ASSERT_TRUE(expr) do {                                                 \
    if (!(expr)) {                                                             \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #expr << "\n";                                      \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)
#define ASSERT_EQ(a, b) do {                                                   \
    auto _av = (a); auto _bv = (b);                                            \
    if (!(_av == _bv)) {                                                       \
        std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                  \
                  << " : " #a " != " #b << "\n";                               \
        g_failed++;                                                            \
        return false;                                                          \
    }                                                                          \
} while (0)

// Helper: build argv from a vector.
static int to_argc(const std::vector<std::string>& v) { return (int)v.size(); }
static std::vector<char*> to_argv(std::vector<std::string>& v) {
    std::vector<char*> out;
    for (auto& s : v) out.push_back(s.data());
    return out;
}

cli::Parser make_parser() {
    return cli::Parser(
        {"test-cli", "0.1.0", "Test CLI.", "[options] <positional>"},
        {
            {"flag",   "f", cli::ArgType::Flag,   "",   "A boolean flag."},
            {"value",  "v", cli::ArgType::Value,  "42", "A value option."},
            {"opt",    "",  cli::ArgType::Optional, "", "An optional value."},
            {"pos",    "",  cli::ArgType::Positional, "", "Positional."},
        }
    );
}

bool test_help_short_circuit() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--help"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.help_requested);
    return true;
}

bool test_version_short_circuit() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "-V"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.version_requested);
    return true;
}

bool test_long_value() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--value", "hello"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ(a.get("value"), std::string("hello"));
    return true;
}

bool test_long_value_equals() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--value=world"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ(a.get("value"), std::string("world"));
    return true;
}

bool test_flag_true() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--flag"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(a.has("flag"));
    return true;
}

bool test_flag_false_by_default() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(!a.has("flag"));
    return true;
}

bool test_short_value() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "-v", "7"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ(a.get("value"), std::string("7"));
    return true;
}

bool test_short_flag_cluster() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "-f"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_TRUE(a.has("flag"));
    return true;
}

bool test_default_applied() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ(a.get("value"), std::string("42"));
    return true;
}

bool test_positional() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "alpha", "beta", "gamma"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ((int)a.positional.size(), 3);
    ASSERT_EQ(a.positional[0], std::string("alpha"));
    ASSERT_EQ(a.positional[2], std::string("gamma"));
    return true;
}

bool test_numeric_helpers() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--value", "1234"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(a.ok());
    ASSERT_EQ(a.get_int("value"), 1234);
    ASSERT_EQ(a.get_uint("value"), (uint64_t)1234);
    return true;
}

bool test_unknown_option() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--nope"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(!a.ok());
    return true;
}

bool test_missing_value() {
    auto p = make_parser();
    std::vector<std::string> argv = {"test-cli", "--value"};
    auto a = p.parse(to_argc(argv), to_argv(argv).data());
    ASSERT_TRUE(!a.ok());
    return true;
}

bool test_render_help_nonempty() {
    auto p = make_parser();
    auto s = p.render_help();
    ASSERT_TRUE(!s.empty());
    ASSERT_TRUE(s.find("--value") != std::string::npos);
    return true;
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "      Running ASEMA CLI parser tests                \n";
    std::cout << "====================================================\n";

    struct T { const char* n; bool (*f)(); };
    T tests[] = {
        {"help_short_circuit",   test_help_short_circuit},
        {"version_short_circuit",test_version_short_circuit},
        {"long_value",           test_long_value},
        {"long_value_equals",    test_long_value_equals},
        {"flag_true",            test_flag_true},
        {"flag_false_default",   test_flag_false_by_default},
        {"short_value",          test_short_value},
        {"short_flag_cluster",   test_short_flag_cluster},
        {"default_applied",      test_default_applied},
        {"positional",           test_positional},
        {"numeric_helpers",      test_numeric_helpers},
        {"unknown_option",       test_unknown_option},
        {"missing_value",        test_missing_value},
        {"render_help",          test_render_help_nonempty},
    };
    int passed = 0, failed = 0;
    for (const auto& t : tests) {
        std::cout << "\n[TEST] " << t.n << "...\n";
        if (t.f()) { passed++; std::cout << "  [PASS] " << t.n << "\n"; }
        else       { failed++; std::cout << "  [FAIL] " << t.n << "\n"; }
    }
    std::cout << "\n  Results: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
