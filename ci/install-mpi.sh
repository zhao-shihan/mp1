#!/usr/bin/env bash
# Build and install the pinned MPI implementation used by the mp1 CI jobs.
#
# Usage: ci/install-mpi.sh <openmpi|mpich> <version> <prefix> <sha256>
#
# The workflow caches <prefix>.  A cache hit is reused only when the recorded
# family/version/checksum match the request and the compiler+launcher exist;
# otherwise the prefix is removed and rebuilt.  The downloaded archive is
# verified before extraction, so an unverified tarball can never become the
# implementation under test.  The expected checksum comes from
# ci/mpi-matrix.json (via ci/matrix-plan.py); this script never reads the
# matrix itself so it can also be run by hand.

set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: %s <openmpi|mpich> <version> <prefix> <sha256>\n' "${0##*/}" >&2
  exit 2
fi

family=$1
version=$2
prefix=$3
expected_sha=$4

if [[ ! "${expected_sha}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'install-mpi: expected a 64-character lowercase hex sha256, got %q\n' \
    "${expected_sha}" >&2
  exit 2
fi

cache_stamp="${prefix}/.mp1-mpi-pin"
cache_ok=0
if [[ -x "${prefix}/bin/mpicxx" ]] &&
  { [[ -x "${prefix}/bin/mpiexec" ]] || [[ -x "${prefix}/bin/mpirun" ]]; } &&
  [[ -f "${cache_stamp}" ]] &&
  [[ "$(cat "${cache_stamp}")" == "${family} ${version} ${expected_sha}" ]]; then
  cache_ok=1
fi

if [[ "${cache_ok}" -eq 1 ]]; then
  printf 'install-mpi: using cached %s %s at %s\n' "${family}" "${version}" "${prefix}"
  exit 0
fi

if [[ -e "${prefix}" ]]; then
  printf 'install-mpi: stale or unverified cache at %s; rebuilding\n' "${prefix}"
  rm -rf "${prefix}"
fi

if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+([a-z0-9.]+)?)?$ ]]; then
  printf 'install-mpi: unsupported version: %s\n' "${version}" >&2
  exit 2
fi

if ! jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null); then
  jobs=2
fi

case "${family}" in
  openmpi)
    major_minor=${version%.*}
    url="https://download.open-mpi.org/release/open-mpi/v${major_minor}/openmpi-${version}.tar.bz2"
    ;;
  mpich)
    url="https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz"
    ;;
  *)
    printf 'install-mpi: unknown MPI family: %s\n' "${family}" >&2
    exit 2
    ;;
esac

work=$(mktemp -d)
trap 'rm -rf "${work}"' EXIT

mkdir -p "${prefix}"
printf 'install-mpi: downloading %s %s\n' "${family}" "${version}"
curl -fsSL --retry 3 --retry-delay 2 "${url}" -o "${work}/mpi.tar"
printf '%s  %s\n' "${expected_sha}" "${work}/mpi.tar" | sha256sum --check --status ||
  {
    printf 'install-mpi: sha256 mismatch for %s %s from %s\n' \
      "${family}" "${version}" "${url}" >&2
    exit 1
  }

mkdir "${work}/src"
case "${family}" in
  openmpi)
    tar -xjf "${work}/mpi.tar" -C "${work}/src" --strip-components=1
    ;;
  mpich)
    tar -xzf "${work}/mpi.tar" -C "${work}/src" --strip-components=1
    ;;
esac

cd "${work}/src"

# mp1 is a C++ wrapper and does not require peer language bindings.  The provider is built
# C-only on purpose; the suite treats an absent Fortran datatype as a valid fail-closed
# configuration (MPICH makes MPI_INTEGER MPI_DATATYPE_NULL, Open MPI gives it zero extent).
case "${family}" in
  openmpi)
    printf 'install-mpi: configuring Open MPI %s\n' "${version}"
    ./configure \
      --prefix="${prefix}" \
      --disable-man-pages \
      --disable-mpi-fortran \
      --with-libevent=internal \
      --with-hwloc=internal
    ;;
  mpich)
    printf 'install-mpi: configuring MPICH %s\n' "${version}"
    ./configure \
      --prefix="${prefix}" \
      --disable-fortran
    ;;
esac

make -j"${jobs}"
make install

printf '%s %s %s\n' "${family}" "${version}" "${expected_sha}" >"${cache_stamp}"

printf 'install-mpi: installed %s %s at %s\n' "${family}" "${version}" "${prefix}"
