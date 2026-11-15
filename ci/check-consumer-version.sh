#!/usr/bin/env bash
# Verify that an installed consumer reports the version declared in VERSION.
#
# Usage: ci/check-consumer-version.sh <consumer-command> [args...]
#
# The command is run once (normally `mpiexec -n 2 ./consumer`); its first
# output line must begin with `mp1 <VERSION>:`.  This is the package/consumer
# gate for the single version source: the consumer reads MP1_VERSION_STRING
# from the installed generated header, not from a compile definition, so a
# stale header or a duplicated literal fails here.
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'usage: %s <consumer-command> [args...]\n' "${0##*/}" >&2
  exit 2
fi

if ! root=$(git rev-parse --show-toplevel 2>/dev/null); then
  printf 'check-consumer-version: not inside a git working tree\n' >&2
  exit 2
fi

version=$(tr -d '[:space:]' <"${root}/VERSION")
if [[ ! ${version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  printf 'check-consumer-version: invalid VERSION %q\n' "${version}" >&2
  exit 2
fi

output=$("$@")
printf '%s\n' "${output}"
first_line=$(printf '%s\n' "${output}" | head -n 1)
expected_prefix="mp1 ${version}:"
if [[ ${first_line} != "${expected_prefix}"* ]]; then
  printf 'check-consumer-version: FAIL: expected the first output line to begin with %q, got %q\n' \
    "${expected_prefix}" "${first_line}" >&2
  exit 1
fi

printf 'check-consumer-version: PASS (consumer reports %s)\n' "${version}"
