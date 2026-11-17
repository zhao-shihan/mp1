#!/usr/bin/env bash
# Install Intel MPI from Intel's official oneAPI apt repository.
#
# Usage: ci/install-oneapi-mpi.sh [package ...]
#
# The default package is intel-oneapi-mpi-devel.  The repository is the supported
# apt channel for Intel MPI; Intel MPI is not in the Ubuntu archive.  The script
# does not pin a package version: the matrix records the expected family version
# and ci/assert-mpi.sh fails closed if the installed launcher does not match it.
#
# The script uses sudo when it is not already root.  GitHub-hosted runners allow
# passwordless sudo; a local developer can run `sudo ci/install-oneapi-mpi.sh`.

set -euo pipefail

packages=("$@")
if [[ ${#packages[@]} -eq 0 ]]; then
  packages=(intel-oneapi-mpi-devel intel-oneapi-libfabric)
fi

if [[ ${EUID} -eq 0 ]]; then
  sudo=()
else
  command -v sudo >/dev/null 2>&1 || {
    printf 'install-oneapi-mpi: root or sudo is required\n' >&2
    exit 2
  }
  sudo=(sudo)
fi

keyring=/usr/share/keyrings/oneapi-archive-keyring.gpg
repo_file=/etc/apt/sources.list.d/oneAPI.list
key_url=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

"${sudo[@]}" install -d -m 0755 "$(dirname "${keyring}")"
curl -fsSL --retry 3 --retry-delay 2 "${key_url}" |
  gpg --dearmor --batch --yes |
  "${sudo[@]}" tee "${keyring}" >/dev/null
printf 'deb [signed-by=%s] https://apt.repos.intel.com/oneapi all main\n' "${keyring}" |
  "${sudo[@]}" tee "${repo_file}" >/dev/null

"${sudo[@]}" env DEBIAN_FRONTEND=noninteractive apt-get update -o Acquire::Retries=3
"${sudo[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"

prefix=/opt/intel/oneapi/mpi/latest
if [[ ! -x "${prefix}/bin/mpiexec" ]]; then
  prefix=$(find /opt/intel/oneapi/mpi -mindepth 1 -maxdepth 1 -type d -name '20*' |
    sort -V | tail -n 1)
fi
if [[ -z "${prefix}" || ! -x "${prefix}/bin/mpiexec" ]]; then
  printf 'install-oneapi-mpi: no usable mpiexec under /opt/intel/oneapi/mpi\n' >&2
  exit 1
fi

printf 'install-oneapi-mpi: installed %s at %s\n' "${packages[*]}" "${prefix}"
"${prefix}/bin/mpiexec" --version 2>&1 | sed -n '1,2p'
