// Version contract: the public macros generated from the top-level VERSION file
// are internally consistent.  No version literal is encoded here; the
// version_matches_project CTest compares the generated values with VERSION and
// the CMake project version.
//
// This is a scenario test (own main, one rank): it makes no MPI call, so it
// only needs the generated header.  The header self-containment target also
// compiles mp1/version.hpp standalone.

#include <cstddef>
#include <cstdio>
#include <mp1/version.hpp>
#include <string_view>

namespace {

  constexpr bool is_digit(char c) {
    return c >= '0' and c <= '9';
  }

  // Accept exactly major.minor.patch: digits only, exactly two dots, at least
  // one digit per component, and no sign, whitespace or suffix.
  constexpr bool parse_version(std::string_view text, int& major, int& minor, int& patch) {
    int* parts[] = {&major, &minor, &patch};
    std::size_t part = 0;
    bool have_digit = false;
    major = 0;
    minor = 0;
    patch = 0;
    for (char c : text) {
      if (c == '.') {
        if (not have_digit or part == 2)
          return false;
        ++part;
        have_digit = false;
      } else if (is_digit(c)) {
        *parts[part] = *parts[part] * 10 + (c - '0');
        have_digit = true;
      } else {
        return false;
      }
    }
    return have_digit and part == 2;
  }

}  // namespace

int main() {
  static_assert(MP1_VERSION_MAJOR >= 0, "MP1_VERSION_MAJOR must be a non-negative integer");
  static_assert(MP1_VERSION_MINOR >= 0, "MP1_VERSION_MINOR must be a non-negative integer");
  static_assert(MP1_VERSION_PATCH >= 0, "MP1_VERSION_PATCH must be a non-negative integer");
  static_assert(sizeof(MP1_VERSION_STRING) > 1, "MP1_VERSION_STRING must not be empty");

  int major = 0;
  int minor = 0;
  int patch = 0;
  if (not parse_version(MP1_VERSION_STRING, major, minor, patch)) {
    std::fprintf(stderr, "version contract: MP1_VERSION_STRING '%s' is not major.minor.patch\n",
                 MP1_VERSION_STRING);
    return 1;
  }
  if (major != MP1_VERSION_MAJOR or minor != MP1_VERSION_MINOR or patch != MP1_VERSION_PATCH) {
    std::fprintf(stderr,
                 "version contract: MP1_VERSION_STRING '%s' does not match "
                 "MP1_VERSION_MAJOR/MINOR/PATCH (%d.%d.%d)\n",
                 MP1_VERSION_STRING, MP1_VERSION_MAJOR, MP1_VERSION_MINOR, MP1_VERSION_PATCH);
    return 2;
  }

  std::printf("version contract: %s (%d.%d.%d)\n", MP1_VERSION_STRING, major, minor, patch);
  return 0;
}
