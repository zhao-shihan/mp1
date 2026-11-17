# mp1 test suite

The CTest tree is registered by `test/CMakeLists.txt`; `cmake/MP1Test.cmake`
defines the two registration helpers (`mp1_add_test`, `mp1_add_compile_fail`).
`test/compile_fail/README.md` documents the negative-compilation probes.

## Suite inventory

The default (non-nightly) release configuration measures:

| Item | Count |
|---|---:|
| CTest entries (default) | **292** |
| CTest entries with `MP1_TEST_EXTRA_RANKS=ON` | **293** |
| compile-fail CTest entries | **96** (94 contract probes + 2 harness self-checks) |
| contract compile-fail probes (`test/compile_fail/*.cc`) | **94** |
| compile-pass controls (`test/compile_pass/*.cc`) | **64** |
| compile-all API sources | **9** |
| unit-test sources (`test/test_*.cc`) | **90** |
| Boost test cases in those sources | **509** |
| C++20 translation units in the full build | **279** |
| Ninja build edges (tests + examples + perf) | **416** |
| wrapped MPI symbols (`harness_wrap_coverage`) | **177** (34 implementation files) |

`MP1_TEST_EXTRA_RANKS=ON` adds one nightly 5-rank entry (the asymmetric 3+2
intercommunicator run of the persistent-collective unit), for 293 entries; it
is off in the default count. The release build enables `MP1_BUILD_PERF`, so the
deterministic `perf_calls` gate is one of the 292 default entries; the nightly
timing-ratio `perf_ratios` entry needs the separate `MP1_BUILD_PERF_RATIOS`
opt-in and is excluded by `-LE nightly`. Header self-containment and
compile-all are build gates; the 64 compile-pass controls are compiled by the
`harness_compile_pass_controls` CTest entry, and the two compile-fail harness
self-checks are the extra entries included in the 96.

## CTest jobs vs MPI ranks

Two independent numbers control a test run:

- **MPI ranks** — each MPI test is registered at fixed rank counts (2 and 4
  for the unit tests). Those counts are protocol coverage, not a measurement
  of how many CPUs the host has; they do not change with the machine size.
- **CTest jobs** — `ctest -j N` (or `CTEST_PARALLEL_LEVEL`) limits how many
  CTest entries execute concurrently. `mp1_add_test` sets the `PROCESSORS`
  test property to the entry's rank count, so CTest treats a rank-4 entry as
  consuming four of the `N` slots and does not co-schedule another entry with
  it. If the budget is below an entry's rank count (for example the nightly
  rank-5 entry at `-j4`), CTest still runs the entry rather than skipping it,
  so it can oversubscribe a smaller host.

Do not pass the CTest job count to `mpiexec`; the rank count of each entry is
part of the test contract.

## Parallelism default and overrides

Test execution defaults to all detected logical CPUs. CMake's portable
`ProcessorCount` module supplies the number (Linux `nproc`, macOS `sysctl`,
Windows `NUMBER_OF_PROCESSORS`); an undetectable count falls back to 1.

```bash
# Configure-time override (cache variable):
cmake -S . -B build -DBUILD_TESTING=ON -DMP1_TEST_JOBS=4
# ...or via the environment when configuring:
MP1_TEST_JOBS=4 cmake -S . -B build -DBUILD_TESTING=ON

# The generated test target uses the configured value:
cmake --build build --target test          # ctest -j <MP1_TEST_JOBS>

# CI and direct runs:
ci/run-ctest.sh --test-dir build --output-on-failure
ci/run-ctest.sh --test-dir build -R '^test_file' --output-on-failure
MP1_TEST_JOBS=2 ci/run-ctest.sh --test-dir build --output-on-failure
```

`ci/run-ctest.sh` reads `MP1_TEST_JOBS` from `build/CMakeCache.txt`, so the
configured value is used unless the environment overrides it. CTest's own
`CTEST_PARALLEL_LEVEL` takes precedence when set.

A bare `ctest` has no build-system hook for a default `-j`: CTest accepts only
the command-line option or `CTEST_PARALLEL_LEVEL` in the environment. Use
`cmake --build build --target test` or `ci/run-ctest.sh` for the configured
default, or run `ctest` with `-j N`/`CTEST_PARALLEL_LEVEL=N`.

The generated `test` target passes `-j <MP1_TEST_JOBS>` explicitly, so
`CTEST_PARALLEL_LEVEL` in the environment does not override it for that target;
use the wrapper or a bare `ctest` when the environment variable must win. A
`-j`/`--parallel` already present in `CMAKE_CTEST_ARGUMENTS` (for example
`-DCMAKE_CTEST_ARGUMENTS='-j;1'`) is kept and no second job count is appended,
so the caller's explicit value wins.

`ci/run-ctest.sh` is a bash script for Linux and macOS (bash 3.2 or newer).
Windows has no CI job; there use the generated `test` target
(`cmake --build build --target test`), which applies the configured `-j` via
`CMAKE_CTEST_ARGUMENTS`, or pass `-j N` to `ctest` directly.

The `perf` label is intentionally run with `-j1` in nightly CI: timing ratios
require an exclusive machine and are not part of the default test set.
`test_spawn_contract` and `test_file_spawn_contract` are registered with the
`RUN_SERIAL` property: they launch extra processes with `MPI_Comm_spawn`, so
CTest runs those entries alone and they do not compete with other CTest
entries. `RUN_SERIAL` and `PROCESSORS` budget only the parent ranks, not the
spawned children: a launcher with a strict slot limit can still fail a spawn on
a small host without oversubscription (Open MPI needs `--oversubscribe`, which
the CI matrix supplies; MPICH/Hydra allows it by default). Their rank coverage
(1/2/4 and 2/4) is unchanged.
