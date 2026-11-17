#!/usr/bin/env bash
# Assert the MPI implementation on PATH is the one the matrix selected.
#
# Usage: ci/assert-mpi.sh <mpich|openmpi|intel> <provider-version> <tier>
#
# Checks, in this order:
#   1. mpicxx and mpiexec are both on PATH and in the same directory;
#   2. the launcher names the expected family and provider version;
#   3. a probe compiled with mpicxx reports the same MPI_VERSION/
#      MPI_SUBVERSION as its <mpi.h>, and MPI_Get_version agrees;
#   4. the runtime advertises MPI 4.1 or newer;
#   5. a `floor` provider advertises *exactly* MPI 4.1 (MPI 4.1 §9.1.1),
#      while stable/provisional/rc may be newer.
#
# MPI 4.1 §9.1.1: MPI_VERSION/MPI_SUBVERSION are compile-time constants and
# MPI_Get_version returns the runtime pair; §11.4.1/Table 11.1 makes
# MPI_Get_version/MPI_Get_library_version callable at any time, so the probe
# calls them before MPI_Init and does not need a launcher.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s <mpich|openmpi|intel> <provider-version> <tier>\n' "${0##*/}" >&2
  exit 2
fi

family=$1
version=$2
tier=$3

case "${family}" in
  openmpi) expected_family='Open MPI' ;;
  mpich) expected_family='MPICH' ;;
  intel) expected_family='Intel' ;;
  *)
    printf 'assert-mpi: unknown family: %s\n' "${family}" >&2
    exit 2
    ;;
esac
case "${tier}" in
  floor | stable | provisional | rc | dev) ;;
  *)
    printf 'assert-mpi: unknown tier: %s\n' "${tier}" >&2
    exit 2
    ;;
esac

fail() {
  printf 'assert-mpi: FAIL: %s\n' "$1" >&2
  exit 1
}

mpicxx_path=$(command -v mpicxx || true)
mpiexec_path=$(command -v mpiexec || true)
[[ -n "${mpicxx_path}" ]] || fail "mpicxx is not on PATH"
[[ -n "${mpiexec_path}" ]] || fail "mpiexec is not on PATH"

mpicxx_dir=$(cd "$(dirname "${mpicxx_path}")" && pwd -P)
mpiexec_dir=$(cd "$(dirname "${mpiexec_path}")" && pwd -P)
[[ "${mpicxx_dir}" == "${mpiexec_dir}" ]] ||
  fail "mpicxx (${mpicxx_path}) and mpiexec (${mpiexec_path}) are not in the same prefix"

launcher_version=$("${mpiexec_path}" --version 2>&1 || true)
[[ -n "${launcher_version}" ]] || fail "mpiexec --version produced no output"
case "${family}" in
  openmpi)
    grep -qi 'Open MPI' <<<"${launcher_version}" ||
      fail "mpiexec does not identify as Open MPI: ${launcher_version}"
    ;;
  mpich)
    grep -qiE 'MPICH|Hydra' <<<"${launcher_version}" ||
      fail "mpiexec does not identify as MPICH/Hydra: ${launcher_version}"
    ;;
  intel)
    grep -qi 'Intel' <<<"${launcher_version}" ||
      fail "mpiexec does not identify as Intel MPI: ${launcher_version}"
    ;;
esac
grep -qF "${version}" <<<"${launcher_version}" ||
  fail "mpiexec version does not contain the pinned provider version '${version}': ${launcher_version}"

tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT

cat >"${tmp}/version_probe.cc" <<'EOF'
#include <mpi.h>
#include <cstdio>
int main() {
  int major = -1;
  int minor = -1;
  if (MPI_Get_version(&major, &minor) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Get_version failed\n");
    return 1;
  }
  char library[MPI_MAX_LIBRARY_VERSION_STRING + 1] = {};
  int length = 0;
  if (MPI_Get_library_version(library, &length) != MPI_SUCCESS) {
    std::fprintf(stderr, "MPI_Get_library_version failed\n");
    return 1;
  }
  std::printf("header=%d.%d\n", MPI_VERSION, MPI_SUBVERSION);
  std::printf("runtime=%d.%d\n", major, minor);
  std::printf("library=%s\n", library);
  return 0;
}
EOF

"${mpicxx_path}" -std=c++20 -o "${tmp}/version_probe" "${tmp}/version_probe.cc"
probe_output=$("${tmp}/version_probe")
header_version=$(sed -n 's/^header=//p' <<<"${probe_output}")
runtime_version=$(sed -n 's/^runtime=//p' <<<"${probe_output}")
runtime_library=$(sed -n 's/^library=//p' <<<"${probe_output}")

[[ -n "${header_version}" ]] || fail "version probe produced no header version"
[[ -n "${runtime_version}" ]] || fail "version probe produced no runtime version"
[[ "${header_version}" == "${runtime_version}" ]] ||
  fail "mpi.h advertises ${header_version} but MPI_Get_version reports ${runtime_version}"
grep -qF "${expected_family}" <<<"${runtime_library}" ||
  fail "MPI_Get_library_version does not name ${expected_family}: ${runtime_library}"

version_ge() {
  # version_ge A B: true when A >= B.
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

version_ge "${runtime_version}" "4.1" ||
  fail "runtime ${runtime_version} is below the MPI 4.1 floor"
if [[ "${tier}" == "floor" ]]; then
  [[ "${runtime_version}" == "4.1" ]] ||
    fail "floor provider must advertise exactly MPI 4.1, got ${runtime_version}"
fi

if [[ ${family} == intel ]]; then
  # I_MPI_FABRICS=shm:ofi is required by the spawn tests, and Intel MPI's own vars.sh does
  # not put the bundled libfabric on the loader path.  Catch that early instead of letting
  # the first MPI_Init fail inside CTest.
  cat >"${tmp}/init_probe.cc" <<'EOF'
#include <mpi.h>
int main(int argc, char** argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return 1;
  if (MPI_Finalize() != MPI_SUCCESS)
    return 2;
  return 0;
}
EOF
  "${mpicxx_path}" -std=c++20 -o "${tmp}/init_probe" "${tmp}/init_probe.cc"
  if ! I_MPI_FABRICS=shm:ofi "${mpiexec_path}" -n 1 "${tmp}/init_probe" >/dev/null 2>&1; then
    fail "Intel MPI OFI smoke failed: I_MPI_FABRICS=shm:ofi cannot initialize; source the \
oneAPI libfabric env (ci/mpi-env.sh does this) and check FI_PROVIDER_PATH"
  fi
fi

printf 'assert-mpi: PASS %s %s (tier %s, advertises MPI %s)\n' \
  "${expected_family}" "${version}" "${tier}" "${runtime_version}"
