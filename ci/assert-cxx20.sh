#!/usr/bin/env bash
# Assert that every C++ translation unit in the CMake compilation database is
# compiled as C++20.
#
# Usage: ci/assert-cxx20.sh [build]
#
# The project baseline is C++20 in three places: the library target's
# INTERFACE feature, the test/example target properties, and the compile-fail
# runner's command line.  This gate reads CMake's own compilation database, so
# a target that silently overrides CXX_STANDARD (or a CI file that adds a
# C++17 standard flag) fails even when configure succeeds.

set -euo pipefail

build=${1:-build}
database="${build}/compile_commands.json"

if [[ ! -f "${database}" ]]; then
  printf 'assert-cxx20: missing %s (configure with CMAKE_EXPORT_COMPILE_COMMANDS=ON)\n' \
    "${database}" >&2
  exit 2
fi

python3 - "${database}" <<'PY'
import json
import sys

database = sys.argv[1]
try:
    with open(database, encoding="utf-8") as handle:
        entries = json.load(handle)
except (OSError, json.JSONDecodeError) as error:
    sys.exit(f"assert-cxx20: cannot read {database}: {error}")

suffixes = (".cc", ".cpp", ".cxx", ".c++", ".C")
cxx = [entry for entry in entries if entry["file"].endswith(suffixes)]
if not cxx:
    sys.exit(f"assert-cxx20: no C++ translation units in {database}")

flags = ("-std=c++20", "-std=gnu++20", "/std:c++20")
bad = []
for entry in cxx:
    command = entry.get("command")
    if command is None:
        command = " ".join(entry.get("arguments", []))
    if not any(flag in command for flag in flags):
        bad.append((entry["file"], command))

if bad:
    print(
        f"assert-cxx20: {len(bad)} of {len(cxx)} C++ translation units are "
        "not compiled as C++20",
        file=sys.stderr,
    )
    for path, command in bad[:10]:
        print(f"  {path}: {command}", file=sys.stderr)
    sys.exit(1)

print(f"assert-cxx20: PASS ({len(cxx)} C++ translation units use C++20)")
PY
