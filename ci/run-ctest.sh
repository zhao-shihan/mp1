#!/usr/bin/env bash
# Run the CTest suite with the configured test-execution parallelism.
#
# Usage:
#   ci/run-ctest.sh [--test-dir <build-dir>] [ctest arguments...]
#
# The job count comes from the first applicable source:
#   1. CTEST_PARALLEL_LEVEL in the environment.  This is CTest's own default
#      for --parallel; the wrapper then leaves the choice entirely to ctest.
#   2. MP1_TEST_JOBS in the environment.
#   3. MP1_TEST_JOBS from <build-dir>/CMakeCache.txt (configured by
#      -DMP1_TEST_JOBS=N or the environment at configure time).
#   4. All online CPUs (getconf, then sysctl, then nproc; 1 if none work).
#
# `ctest -j` limits concurrent CTest entries, not MPI ranks.  Every MPI test
# keeps its fixed 2/4 rank coverage; the PROCESSORS test property reserves
# that many CTest slots per entry, so a rank-4 test normally runs without
# another entry beside it.  A test whose rank count exceeds the job budget is
# still run (not skipped), so it can oversubscribe a smaller host.
#
# Requires bash 3.2 or newer.  This wrapper is for Linux and macOS; Windows
# jobs use the CMake `test` target, which passes the configured -j through
# CMAKE_CTEST_ARGUMENTS, or pass -j N to ctest directly.
set -euo pipefail

usage() {
  cat <<'EOF'
usage: ci/run-ctest.sh [--test-dir <build-dir>] [ctest arguments...]

Runs ctest with the configured MP1_TEST_JOBS (default: all online CPUs).
The build directory defaults to `build`.  Override the job count with
MP1_TEST_JOBS=<n> in the environment or CTEST_PARALLEL_LEVEL=<n>.
EOF
}

detect_jobs() {
  local candidate
  for candidate in \
    "$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)" \
    "$(sysctl -n hw.ncpu 2>/dev/null || true)" \
    "$(nproc 2>/dev/null || true)"; do
    if [[ ${candidate} =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '1\n'
}

test_dir=build
ctest_args=()
while (($# > 0)); do
  case $1 in
    --test-dir)
      if (($# < 2)); then
        printf 'run-ctest: --test-dir requires a directory\n' >&2
        exit 2
      fi
      test_dir=$2
      shift 2
      ;;
    --test-dir=*)
      test_dir=${1#*=}
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      ctest_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n ${CTEST_PARALLEL_LEVEL:-} ]]; then
  exec ctest --test-dir "${test_dir}" ${ctest_args[@]+"${ctest_args[@]}"}
fi

jobs=${MP1_TEST_JOBS:-}
if [[ -z ${jobs} && -f ${test_dir}/CMakeCache.txt ]]; then
  jobs=$(sed -n 's/^MP1_TEST_JOBS:STRING=//p' "${test_dir}/CMakeCache.txt")
fi
if [[ -z ${jobs} ]]; then
  jobs=$(detect_jobs)
fi
if [[ ! ${jobs} =~ ^[1-9][0-9]*$ ]]; then
  printf 'run-ctest: MP1_TEST_JOBS must be a positive integer (got %q)\n' \
    "${jobs}" >&2
  exit 2
fi

exec ctest --test-dir "${test_dir}" -j "${jobs}" \
  ${ctest_args[@]+"${ctest_args[@]}"}
