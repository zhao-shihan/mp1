#!/usr/bin/env bash
# Report-only broad clang-tidy run for nightly CI.
#
# Usage: ci/clang-tidy-nightly.sh [build]
#
# Environment:
#   CLANG_TIDY   clang-tidy executable (default: clang-tidy)
#   JOBS         parallel clang-tidy processes (default: nproc)
#
# Runs one clang-tidy process per unique translation unit in the compilation
# database, excluding test/support/** (the same support TU is compiled into many
# test targets, so its repeated entries are expensive and produce no diagnostics).
# Diagnostics do not fail the script; a missing compilation database or a
# clang-tidy crash does.

set -euo pipefail

if ! root=$(git rev-parse --show-toplevel 2>/dev/null); then
  printf 'clang-tidy-nightly: not inside a git working tree\n' >&2
  exit 2
fi
cd "${root}"

build=${1:-build}
tidy=${CLANG_TIDY:-clang-tidy}
jobs=${JOBS:-$(nproc)}

if [[ ! -f "${build}/compile_commands.json" ]]; then
  printf 'clang-tidy-nightly: missing %s/compile_commands.json\n' "${build}" >&2
  exit 2
fi

mapfile -t files < <(
  python3 - "${build}/compile_commands.json" <<'PY'
import json
import os
import sys

seen = set()
for entry in json.load(open(sys.argv[1], encoding="utf-8")):
    path = os.path.realpath(entry["file"])
    if path in seen or "/test/support/" in path:
        continue
    seen.add(path)
    print(path)
PY
)

printf 'clang-tidy-nightly: %d unique TUs, %d jobs\n' "${#files[@]}" "${jobs}"
printf '%s\0' "${files[@]}" |
  xargs -0 -n1 -P "${jobs}" "${tidy}" -p "${build}" \
    --config-file="${root}/ci/clang-tidy-nightly.yml"

printf 'clang-tidy-nightly: report written (diagnostics are non-blocking)\n'
