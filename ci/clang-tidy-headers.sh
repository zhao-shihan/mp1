#!/usr/bin/env bash
# Blocking clang-tidy check of the header-only public API.
#
# Usage: ci/clang-tidy-headers.sh [config] [build]
#   config   clang-tidy configuration file (default: .clang-tidy)
#   build    unused; kept so callers can share the build argument convention
#
# Environment:
#   CLANG_TIDY   clang-tidy executable (default: clang-tidy)
#   MPICXX       MPI C++ compiler wrapper (default: mpicxx)
#   REPORT_ONLY  when set to 1, report diagnostics without failing
#
# The check parses mp1/mp1.hpp once (all public headers) and
# test/test_reflection.cc once (reflection template instantiation).  Header
# filtering keeps MPI and Boost diagnostics out; SystemHeaders is false in the
# configuration.

set -euo pipefail

if ! root=$(git rev-parse --show-toplevel 2>/dev/null); then
  printf 'clang-tidy-headers: not inside a git working tree\n' >&2
  exit 2
fi
cd "${root}"

config=${1:-.clang-tidy}
tidy=${CLANG_TIDY:-clang-tidy}
mpicxx=${MPICXX:-mpicxx}
report_only=${REPORT_ONLY:-0}

if ! command -v "${tidy}" >/dev/null 2>&1; then
  printf 'clang-tidy-headers: %s not found\n' "${tidy}" >&2
  exit 2
fi
if [[ ! -f "${config}" ]]; then
  printf 'clang-tidy-headers: config not found: %s\n' "${config}" >&2
  exit 2
fi

"${tidy}" --verify-config --config-file="${config}" >/dev/null

# Extract only -I flags from the MPI wrapper.  clang-tidy consumes them as
# -isystem so third-party headers stay out of the diagnostics.
mpi_compile=$("${mpicxx}" --showme:compile 2>/dev/null || "${mpicxx}" -show)
mpi_includes=()
for flag in ${mpi_compile}; do
  case "${flag}" in
    -I*) mpi_includes+=("-isystem" "${flag#-I}") ;;
  esac
done

warnings=(-warnings-as-errors='*')
if [[ "${report_only}" == "1" ]]; then
  warnings=()
fi

common=(--config-file="${config}" --header-filter='.*/mp1/.*' "${warnings[@]}")
run_header_check() {
  local source=$1
  printf 'clang-tidy-headers: checking %s\n' "${source}"
  "${tidy}" "${common[@]}" "${source}" -- \
    -std=c++20 -I"${root}" "${mpi_includes[@]}"
}

rc=0
run_header_check mp1/mp1.hpp || rc=1
run_header_check test/test_reflection.cc || rc=1

if ((rc != 0)); then
  printf 'clang-tidy-headers: FAILED\n' >&2
  exit 1
fi
printf 'clang-tidy-headers: PASS\n'
