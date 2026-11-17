# Contributing to mp1

mp1 is a header-only C++20 library with an MPI 4.1 floor. The library target
raises consumers to C++20 through `INTERFACE_COMPILE_FEATURES`; C++17 builds
are rejected by `mp1/mp1.hpp` and the CMake compiler checks.

## Toolchain

The formatting and static-analysis versions are pinned so local runs match CI:

- `clang-format` **21.1.8** (installed from PyPI as `clang-format==21.1.8`)
- `clang-tidy` **21.1.8** (official LLVM Linux X64 release, with
  `clang-tidy-diff` and `run-clang-tidy`)

CI downloads the exact LLVM release tarball and checks its SHA256. A distro
clang-tidy is not a supported substitute: its check set and diagnostics differ.

## Build and test

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DMP1_BUILD_EXAMPLES=ON
cmake --build build --parallel
cmake --build build --target test
```

CTest execution defaults to all logical CPUs (`MP1_TEST_JOBS`; set
`-DMP1_TEST_JOBS=N` at configure time or `MP1_TEST_JOBS=N` in the environment
to override). The job count only limits concurrent CTest entries; the fixed
2/4 MPI ranks of each test are coverage and are unaffected. For a filtered run
or a direct `ctest`, use `ci/run-ctest.sh --test-dir build --output-on-failure`,
or set `CTEST_PARALLEL_LEVEL=N` yourself. The `perf` suite stays serial because
it measures timing ratios. See `test/README.md` for details.

Use an MPI 4.1 implementation and a C++20 compiler (GCC 13+, Clang 16+ with
libstdc++ 13+ or a conforming libc++, or MSVC 19.32+). The pull-request matrix
builds MPICH 4.3.2 Debug, Open MPI 6.0.0rc3 Debug, Intel MPI 2021.19 Debug and
a Clang build/test leg; the nightly matrix adds extra ranks, the performance
suite and report-only clang-tidy.

## Version

The top-level `VERSION` file is the only version source. CMake reads it,
derives the project and package version, and generates `mp1/version.hpp` from
`mp1/version.hpp.in`; the generated header is installed and included by
`mp1/mp1.hpp`, and no other file defines the `MP1_VERSION_*` macros.

To prepare a release, edit `VERSION` to `major.minor.patch`, reconfigure (a
build also reconfigures automatically because `VERSION` is a configure
dependency), and run the version gates:

```sh
cmake -S . -B build -DBUILD_TESTING=ON
ci/run-ctest.sh --test-dir build -R 'version'
```

Do not edit `mp1/version.hpp` (it does not exist in the source tree), add a
version literal to CMake or tests, or define the macros in another header. The
first `##` release heading in `CHANGELOG.md` must begin with the `VERSION`
string; `ci/static-checks.sh` enforces both rules.

## Formatting

```bash
python3 -m venv .format-venv
.format-venv/bin/pip install "clang-format==21.1.8"
mapfile -t files < <(git ls-files 'mp1/*.hpp' 'mp1/**/*.hpp')
.format-venv/bin/clang-format --style=file --dry-run --Werror "${files[@]}"
```

The formatting gate currently covers `mp1/**`; format touched tests and
examples locally with the same binary.

## clang-tidy

The repository has two profiles:

- `.clang-tidy` — the blocking pull-request profile. It is C++20-only, has no
  clang-analyzer checks, and checks the main file only. Pull requests run it on
  changed C++ lines and run a separate public-header check so the header-only API
  is covered once.
- `ci/clang-tidy-nightly.yml` — the broad report-only profile. It is not a
  merge gate; nightly CI uploads its log as an artifact.

Local reproduction:

```bash
# Changed lines (use the base commit of your branch):
git diff -U0 "$BASE_SHA" HEAD -- '*.cc' '*.cpp' '*.cxx' |
  clang-tidy-diff.py -p1 -path build -j 4 \
    -config-file .clang-tidy \
    -clang-tidy-binary clang-tidy \
    -warnings-as-errors='*' \
    -regex='^(?!test/(support|compile_fail)/).*\.(cc|cpp|cxx)$'

# Public-header check (blocking):
ci/clang-tidy-headers.sh

# Full report-only run:
ci/clang-tidy-nightly.sh build
```

The public-header check inspects `mp1/mp1.hpp` and `test/test_reflection.cc` with
`--header-filter='.*/mp1/.*'`. `test/support/**` is excluded from full runs: its
per-test-target duplicate compilation-database entries are slow and produce no
diagnostics. The two C examples are covered by the report-only nightly run, not
by the pull-request gate.

### Suppression policy

A suppression must name the check and give a reason:

```cpp
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API requires void*
```

Bare `NOLINT` and `NOLINTNEXTLINE` are rejected by `ci/static-checks.sh`.
Do not disable a check globally to hide a problem; fix it, or document why the
specific line is a false positive. `--fix` is for local branches and nightly
artifacts only; pull requests must not rewrite code during analysis.

### MPI address arithmetic

Never use intrinsic `+`/`-` on `MPI_Aint`. MPI 4.1 §5.1.5 requires
`MPI_AINT_ADD`/`MPI_AINT_DIFF`; the representation is not promised to be a flat
integer address, and §5.1.12 forbids a difference of addresses from different
allocations in a datatype displacement. Use
`detail::checked_address_diff`/`detail::checked_address_add` (which call the
MPI procedures and check the documented round trip) or `MPI_Aint_add` directly
for a non-representability-tolerant run probe. `ci/static-checks.sh` rejects an
intrinsic address subtraction in `mp1/impl` and pins the two procedures in
their owning headers.

A relative iterator map over separate allocations is not a legal datatype; the
p2p non-contiguous sequence send uses the absolute-address `MPI_BOTTOM` form
instead. If a future relative map is reintroduced, it must use
`checked_address_diff` and reject a cross-allocation map before constructing
the datatype.

## CI

- `.github/workflows/clang-tidy.yml` runs `clang-tidy --verify-config`, the
  blocking public-header check, and `clang-tidy-diff` on changed C++ lines.
- `.github/workflows/nightly.yml` has a `clang-tidy report` job that runs the
  broad profile and the public-header check report-only.
- `.github/workflows/ci.yml` runs the build/test matrix, package consumer,
  identity/hygiene checks, formatting and AddressSanitizer.
