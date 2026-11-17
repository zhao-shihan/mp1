#!/usr/bin/env bash
# Select one MPI provider for a build tree.
#
# Usage:
#   eval "$(ci/mpi-env.sh <mpich|openmpi|intel> <version> [--prefix DIR])"
#   ci/mpi-env.sh <family> <version> [--prefix DIR] --github-env
#
# The script resolves the provider's absolute mpicc/mpicxx/mpiexec, prepends its
# bin directory to PATH, and extends LD_LIBRARY_PATH.  It never relies on
# update-alternatives or on a bare `mpicxx` that may belong to another provider.
#
# For openmpi/mpich the default prefix is ~/.cache/mp1-ci-mpi/<family>-<version>
# (the ci/install-mpi.sh cache); if that prefix is absent, the Debian/Ubuntu
# suffixes /usr/bin/mpicxx.<family> and /usr/bin/mpiexec.<family> are used.
# For intel the prefix is /opt/intel/oneapi/mpi/<version>, falling back to the
# `latest` symlink created by the oneAPI installer.
#
# --github-env appends the variables to $GITHUB_ENV and the bin directory to
# $GITHUB_PATH.  Without it the script prints `export` lines for `eval`.
# --print is accepted as an explicit spelling of the default mode.

set -euo pipefail

usage() {
  printf 'usage: %s <mpich|openmpi|intel> <version> [--prefix DIR] [--github-env|--print]\n' \
    "${0##*/}" >&2
  exit 2
}

family=${1-}
version=${2-}
[[ -n ${family} && -n ${version} ]] || usage
shift 2

prefix=
prefix_explicit=0
system_prefix=0
mode=print
while (($# > 0)); do
  case $1 in
    --prefix)
      (($# >= 2)) || usage
      prefix=$2
      prefix_explicit=1
      shift 2
      ;;
    --github-env)
      mode=github
      shift
      ;;
    --print)
      mode=print
      shift
      ;;
    *)
      usage
      ;;
  esac
done

case ${family} in
  openmpi | mpich | intel) ;;
  *)
    printf 'mpi-env: unknown family: %s\n' "${family}" >&2
    exit 2
    ;;
esac

if [[ ${family} == intel ]]; then
  if [[ ${prefix_explicit} -eq 1 && ! -x ${prefix}/bin/mpiexec ]]; then
    printf 'mpi-env: explicit --prefix %s has no bin/mpiexec\n' "${prefix}" >&2
    exit 1
  fi
  prefix=${prefix:-/opt/intel/oneapi/mpi/${version}}
  if [[ ! -x ${prefix}/bin/mpiexec ]]; then
    prefix=/opt/intel/oneapi/mpi/latest
  fi
  if [[ ! -x ${prefix}/bin/mpiexec ]]; then
    printf 'mpi-env: Intel MPI %s is not installed under /opt/intel/oneapi/mpi\n' \
      "${version}" >&2
    exit 1
  fi
  # Intel MPI's vars.sh owns PATH/LD_LIBRARY_PATH/I_MPI_ROOT, but it does not add
  # the bundled libfabric used by I_MPI_FABRICS=shm:ofi.  Source libfabric's
  # vars.sh too so FI_PROVIDER_PATH and the OFI provider are available.
  # shellcheck source=/dev/null
  source "${prefix}/env/vars.sh" >/dev/null
  oneapi_root=$(cd "${prefix}/../.." && pwd -P)
  libfabric_vars="${oneapi_root}/libfabric/latest/env/vars.sh"
  if [[ ! -f ${libfabric_vars} ]]; then
    libfabric_vars=$(find "${oneapi_root}/libfabric" -name vars.sh -path '*/env/*' |
      sort -V | tail -n 1)
  fi
  if [[ -n ${libfabric_vars} && -f ${libfabric_vars} ]]; then
    # shellcheck source=/dev/null
    source "${libfabric_vars}" >/dev/null
  fi
  export I_MPI_ROOT=${I_MPI_ROOT:-${prefix}}
  bin_dir=${prefix}/bin
else
  if [[ -z ${prefix} ]]; then
    prefix=${HOME}/.cache/mp1-ci-mpi/${family}-${version}
  fi
  if [[ ! -x ${prefix}/bin/mpicxx || ! -x ${prefix}/bin/mpiexec ]]; then
    if [[ ${prefix_explicit} -eq 1 ]]; then
      printf 'mpi-env: explicit --prefix %s has no bin/mpicxx or bin/mpiexec\n' \
        "${prefix}" >&2
      exit 1
    fi
    prefix=
    if [[ -x /usr/bin/mpicxx.${family} ]]; then
      prefix=/usr/bin
    fi
    if [[ -z ${prefix} || ! -x ${prefix}/mpiexec.${family} ]]; then
      printf 'mpi-env: %s %s is neither in ~/.cache/mp1-ci-mpi nor in /usr/bin\n' \
        "${family}" "${version}" >&2
      printf 'mpi-env: run ci/install-mpi.sh %s %s <prefix> <sha256>, or install the distro package\n' \
        "${family}" "${version}" >&2
      exit 1
    fi
    system_prefix=1
    bin_dir=${prefix}
    MPI_C_COMPILER=${prefix}/mpicc.${family}
    MPI_CXX_COMPILER=${prefix}/mpicxx.${family}
    MPIEXEC_EXECUTABLE=${prefix}/mpiexec.${family}
  else
    bin_dir=${prefix}/bin
    MPI_C_COMPILER=${prefix}/bin/mpicc
    MPI_CXX_COMPILER=${prefix}/bin/mpicxx
    MPIEXEC_EXECUTABLE=${prefix}/bin/mpiexec
  fi
  if [[ ${system_prefix} -eq 0 ]]; then
    export LD_LIBRARY_PATH="${prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

export PATH="${bin_dir}:${PATH}"

if [[ ${family} == intel ]]; then
  MPI_C_COMPILER=$(command -v mpicc)
  MPI_CXX_COMPILER=$(command -v mpicxx)
  MPIEXEC_EXECUTABLE=$(command -v mpiexec)
fi

export MPI_C_COMPILER MPI_CXX_COMPILER MPIEXEC_EXECUTABLE

if [[ ${mode} == github ]]; then
  [[ -n ${GITHUB_ENV:-} && -n ${GITHUB_PATH:-} ]] || {
    printf 'mpi-env: --github-env requires GITHUB_ENV and GITHUB_PATH\n' >&2
    exit 2
  }
  {
    printf 'MPI_C_COMPILER=%s\n' "${MPI_C_COMPILER}"
    printf 'MPI_CXX_COMPILER=%s\n' "${MPI_CXX_COMPILER}"
    printf 'MPIEXEC_EXECUTABLE=%s\n' "${MPIEXEC_EXECUTABLE}"
    printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH-}"
    if [[ -n ${I_MPI_ROOT:-} ]]; then
      printf 'I_MPI_ROOT=%s\n' "${I_MPI_ROOT}"
    fi
    if [[ -n ${FI_PROVIDER_PATH:-} ]]; then
      printf 'FI_PROVIDER_PATH=%s\n' "${FI_PROVIDER_PATH}"
    fi
  } >>"${GITHUB_ENV}"
  printf '%s\n' "$(dirname "${MPI_CXX_COMPILER}")" >>"${GITHUB_PATH}"
  printf 'mpi-env: %s %s -> %s\n' "${family}" "${version}" "${MPI_CXX_COMPILER}"
else
  printf 'export MPI_C_COMPILER=%q\n' "${MPI_C_COMPILER}"
  printf 'export MPI_CXX_COMPILER=%q\n' "${MPI_CXX_COMPILER}"
  printf 'export MPIEXEC_EXECUTABLE=%q\n' "${MPIEXEC_EXECUTABLE}"
  printf 'export PATH=%q\n' "${PATH}"
  printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH-}"
  if [[ -n ${I_MPI_ROOT:-} ]]; then
    printf 'export I_MPI_ROOT=%q\n' "${I_MPI_ROOT}"
  fi
  if [[ -n ${FI_PROVIDER_PATH:-} ]]; then
    printf 'export FI_PROVIDER_PATH=%q\n' "${FI_PROVIDER_PATH}"
  fi
fi
