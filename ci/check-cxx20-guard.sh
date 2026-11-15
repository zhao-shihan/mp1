#!/usr/bin/env bash
# Verify that the installed umbrella fails closed below C++20.
#
# Usage: ci/check-cxx20-guard.sh <install-prefix>
#
# The CMake target raises consumers through cxx_std_20, but a build system can
# bypass the imported target and invoke the compiler directly.  mp1/mp1.hpp
# carries the #error guard for exactly that case; this check compiles the
# installed header as C++17 and requires that diagnostic.
#
# This file is the one place allowed to spell -std=c++17: ci/static-checks.sh
# excludes it from the "build or CI file forces C++17" scan.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s <install-prefix>\n' "${0##*/}" >&2
  exit 2
fi

prefix=$1
mpicxx=${MPICXX:-mpicxx}
log=$(mktemp)
trap 'rm -f "${log}"' EXIT

if [[ ! -f "${prefix}/include/mp1/mp1.hpp" ]]; then
  printf 'check-cxx20-guard: missing %s/include/mp1/mp1.hpp\n' "${prefix}" >&2
  exit 2
fi

if printf '#include <mp1/mp1.hpp>\n' |
  "${mpicxx}" -std=c++17 -I"${prefix}/include" -x c++ -fsyntax-only - \
    2>"${log}"; then
  printf 'check-cxx20-guard: FAIL: a C++17 compilation of mp1/mp1.hpp succeeded\n' >&2
  exit 1
fi

if ! grep -qF 'mp1 requires C++20' "${log}"; then
  printf 'check-cxx20-guard: FAIL: the C++17 compile failed without the guard diagnostic\n' >&2
  cat "${log}" >&2
  exit 1
fi

printf 'check-cxx20-guard: PASS (C++17 compilation rejected by the umbrella guard)\n'
