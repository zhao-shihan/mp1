# AGENTS.md

This file is the working agreement for AI agents and human contributors editing
this repository. It is normative for changes; the public documentation remains
the normative description of the library's behaviour.

mp1 is a header-only C++20 wrapper over MPI 4.1. It is a complete rewrite of
`mpl 0.5.0`, not a compatibility layer. The current release line is versioned in
[`VERSION`](VERSION); `mp1/version.hpp` is generated and must never be edited.

Read, in this order, before changing behaviour:

1. [`README.md`](README.md) — public API and usage.
2. [`API-MATRIX.md`](API-MATRIX.md) — coverage and current surface.
3. [`COMPATIBILITY.md`](COMPATIBILITY.md) — standard classes, provider caveats.
4. [`CONTRIBUTING.md`](CONTRIBUTING.md) — toolchain, formatting, CI.
5. The relevant MPI 4.1 clause. Tests are evidence, never proof.

If a request conflicts with the design philosophy below, stop and ask the
maintainer instead of silently weakening the library.

---

## 1. Design philosophy

These eight points are non-negotiable. They explain most review comments and
should decide ambiguous cases.

### 1.1 MPI semantics first

The wrapper must follow MPI 4.1, not invent a more convenient semantics. Do not
hide collectives, ordering, matching, completion, or error behaviour. Do not
add a hidden collective, barrier, probe, or datatype cache operation to make a
C++ API look simpler.

For every MPI behaviour, distinguish:

- **standard-required** — a conforming implementation must do it;
- **implementation-defined / unspecified** — the standard permits a range of
  behaviour; the wrapper must not assume one provider's choice;
- **provider-dependent** — a real provider may deviate, or expose an extension;
- **test evidence only** — a passing test does not make a behaviour portable.

Provider deviations belong in `COMPATIBILITY.md`, not in an undocumented
workaround. When in doubt, read the clause and write the reason in the code
comment or commit message.

### 1.2 Thin and zero-overhead

The common scalar/dense paths should remain thin:

- one native MPI call for a blocking operation;
- no allocation on the common path;
- no per-call datatype query, attribute query, or string formatting;
- no unnecessary copies or staging when a typed datatype is enough;
- concepts and compile-time checks instead of runtime branching where possible;
- pay for rare cases only when they are actually used.

Header-only does not mean "put everything in every header": respect the include
graph and `mp1/headers.txt`. Benchmark or count calls before claiming a
performance improvement.

### 1.3 Typed and portable wire

Never silently reinterpret a C++ object as a byte string for communication.
Do not use `MPI_BYTE` as a full-object image, do not send padding as a substitute
for fields, and do not use `reinterpret_cast`/`memcpy` to move typed values onto
the wire.

- Use `native_datatype<T>()`, `layout<T>`, and the typed form of each family.
- `std::byte` is the explicit byte type; `region<std::byte>` or
  `std::span<std::byte>` is the supported byte-erasure route.
- Non-contiguous sequences use an absolute-address typed map with
  `MPI_BOTTOM`; never a relative cross-allocation map.
- No assumptions about `MPI_Aint` representation, tag values, thread level,
  status fields, `MPI_UNDEFINED`, or `MPI_ANY_TAG` sign.
- Use `MPI_Aint_add`/`MPI_Aint_diff`; intrinsic arithmetic on `MPI_Aint` is
  forbidden.
- `MPI_TAG_UB` is inclusive. Tag validation uses `0..tag_ub` and equality with
  `any_tag`; it must not rely on `MPI_ANY_TAG < 0`.
- Handle constants are not `constexpr`; only int-valued constants that the
  standard describes as integer constant expressions may be used in
  `constexpr` contexts.

### 1.4 Explicit ownership and lifetime

RAII is the default. Owners own their MPI handles, destructors release them, and
views never outlive their owners.

- `request`, `status`, `layout`, `tag`, `file`, `window`, `communicator`,
  `message`, and buffers must have clear ownership and move/copy rules.
- No dangling native handle, no double free, no use after completion, no
  hidden global mutable state.
- A caller's layout/buffer may be borrowed only for the documented lifetime;
  nonblocking/persistent operations must own or copy what they need.
- `MPI_PROC_NULL` statuses are normalized; do not let a provider-specific
  envelope leak through.
- Errors should be detected locally before the MPI data call whenever the
  contract makes them detectable (null with positive bytes, invalid tag,
  invalid range, unsupported form).
- Source-breaking changes are allowed only when they simplify or correct the
  design; document them in `CHANGELOG.md`/`MIGRATION.md`.

### 1.5 Minimal and symmetric API

Prefer a small vocabulary with consistent shapes:

- blocking / nonblocking / persistent symmetry;
- value / range / region forms where they make sense;
- the same root/role/in-place rules across families;
- one strong `mp1::tag` type, not an `int` with conventions;
- `mp1::empty_data` for an explicit empty message;
- only the necessary mp1 reduction functors (`max`, `min`, `logical_xor`); the
  equivalent `std::` functors are accepted and must not be duplicated;
- no used-side `blocks{}` magic for empty collective operations; empty
  collectives are still collectives and require the normal descriptors/ranks;
- no `layout<T>::empty()` factory; use an empty span or `contiguous(0)`;
- no compatibility aliases unless the maintainer explicitly asks for them.

Keep ADL in mind: most free functions take an mp1 type and can be called
unqualified; do not remove that convenience. In headers and public examples,
prefer explicit `mp1::` qualification.

### 1.6 Tests are evidence, not proof

A test proves only that the tested provider/configuration did not fail that
case. It does not prove MPI portability, absence of UB, or correctness of an
assumption.

- Start from the MPI 4.1 text and the C++ rules, then design the test.
- Run the relevant suite on the pinned providers when the behaviour is
  provider-sensitive.
- Use ASan/UBSan where ownership, lifetime, or callback code changes.
- Do not weaken a test, delete a case, or change its expected result to make a
  change pass.
- Do not encode provider quirks in the library and call them "tests pass".
- Compile-fail probes are part of the API contract; add one when a form must
  be rejected.
- New public forms need runtime tests, compile-pass coverage, and, where
  relevant, compile-fail controls.

### 1.7 Truthful docs and single-source version

Documentation is part of the product. It must match the headers, not an older
design.

- `VERSION` is the only version source. CMake reads it, calls
  `project(mp1 VERSION ...)`, and generates `mp1/version.hpp` from
  `mp1/version.hpp.in`; `mp1/mp1.hpp` includes the generated header. Never add a
  version literal to code, CMake, tests, or another header. Never edit
  `mp1/version.hpp` (it does not exist in the source tree).
- The previous library is **mpl 0.5.0**. Do not write the old fork's name, its
  `MPL`-prefixed macro spellings, the late 0.2x version series, or internal
  development/review/phase/wave references in product docs, source comments,
  or test names.
- `mpl 0.5.0` had no RMA/windows; RMA is new in mp1. `MPL_REFLECTION` is the
  old macro; the later fork's renamed reflection macro is not a 0.5.0 name.
- `MPL_HOMOGENEOUS` was broken in 0.5.0 and has no replacement;
  `MP1_DATATYPE_BASIC` is an RMA annotation, not a byte transport.
- Keep counts (CTest entries, probes, controls, TUs, edges, API census) in one
  place and re-derive them from the actual build; do not copy stale numbers.
- Document provider caveats in `COMPATIBILITY.md`; document standard-required
  behavior in the relevant public section.

### 1.8 Release discipline

A release is a gate, not a commit message.

- Run the full release checks: MPICH, Open MPI and Intel MPI builds, default
  and extra-ranks CTest, compile-fail, compile-pass controls, examples,
  ASan/UBSan targeted, static checks, formatting, install/consumer.
- Do not tag, push, rewrite `main`, or amend a release commit unless the
  maintainer explicitly asks for that exact operation.
- Do not hide a provider workaround in the library to make a gate green; put it
  in the documented environment instructions.
- Leave the working tree clean; keep ad-hoc probes/builds under ignored paths or
  outside the repository.
- Historical releases/tags are retained/removed only by explicit owner
  decision.

---

## 2. Repository shape

```text
mp1/mp1.hpp              public umbrella
mp1/headers.txt          public header manifest (self-containment)
mp1/impl/*.hpp           public family headers
mp1/impl/detail/*.hpp    implementation helpers (not public API)
mp1/version.hpp.in       generated-version template
test/                    Boost.Test suite, compile-fail/pass, perf, harness
examples/                public API examples
ci/                      CI/gate scripts
README.md                public guide
API-MATRIX.md            coverage/API census
COMPATIBILITY.md         standard classes and provider caveats
MIGRATION.md             mpl 0.5.0 -> mp1
CHANGELOG.md             release history (first heading matches VERSION)
CONTRIBUTING.md          toolchain and contribution rules
VERSION                  sole version source
```

`agent-workspace/` contains working material from maintainer/agent sessions; it
is not part of the library and must not be committed as product source.

## 3. Build and test

### 3.1 Normal development build

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DMP1_BUILD_EXAMPLES=ON \
  -DMP1_BUILD_PERF=ON
cmake --build build --parallel
ci/run-ctest.sh --test-dir build --output-on-failure
```

`ci/run-ctest.sh` uses `MP1_TEST_JOBS` (configure-time cache or environment);
the rank coverage inside each test is fixed and independent of `ctest -j`.

Useful focused runs:

```sh
ci/run-ctest.sh --test-dir build -R '^test_file' --output-on-failure
ci/run-ctest.sh --test-dir build -LE nightly --output-on-failure
MP1_TEST_JOBS=2 ci/run-ctest.sh --test-dir build --output-on-failure
```

### 3.2 Extra ranks / release-like suite

```sh
cmake -S . -B build-extra -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DMP1_TEST_EXTRA_RANKS=ON \
  -DMP1_BUILD_PERF=ON
cmake --build build-extra --parallel
ci/run-ctest.sh --test-dir build-extra -LE nightly --output-on-failure
```

`MP1_BUILD_PERF_RATIOS` is a separate opt-in for timing-ratio entries; it is
not part of the default release count.

### 3.3 Provider matrix

The release matrix is MPICH 4.3.2 (tarball), Open MPI 6.0.0rc3 (tarball) and
Intel MPI 2021.19 (Intel oneAPI apt). Use
the provider's `mpicxx`/`mpiexec` through `ci/mpi-env.sh`, which selects
absolute paths; do not rely on a bare `mpicxx` from another provider. For a
tarball provider, ensure `LD_LIBRARY_PATH` points at that prefix's `lib`
directory so the intended `libmpi` is loaded. Do not report a MPICH run that
silently loaded the system MPICH/Open MPI shared library.

Intel MPI is installed from Intel's oneAPI apt repository; it is not in the
Ubuntu archive. Intel's default `shm` fabric cannot start `MPI_Comm_spawn`
children, so the spawn tests export `I_MPI_FABRICS=shm:ofi` (see
`test/CMakeLists.txt`). `ci/mpi-env.sh` also sources the bundled libfabric
`vars.sh`; Intel MPI's own `vars.sh` does not put `FI_PROVIDER_PATH` on the
loader path, and `MPI_Init` under `shm:ofi` fails without it.

Open MPI ASan runs on some hosts need the documented MCA workaround
(`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self,tcp`, or the patcher override); this is
an environment issue, not a reason to change library code.

### 3.4 Gates

```sh
ci/static-checks.sh
ci/assert-cxx20.sh
ci/assert-mpi.sh <family> <provider-version>
```

Formatting is pinned to clang-format 21.1.8 and currently gates `mp1/**`:

```sh
mapfile -t files < <(git ls-files 'mp1/*.hpp' 'mp1/**/*.hpp')
clang-format --style=file --dry-run --Werror "${files[@]}"
```

Format touched tests/examples with the same binary, but do not reformat
unrelated files.

Install/consumer check:

```sh
cmake --install build --prefix /tmp/mp1-install
# configure a small consumer with find_package(mp1 2.0 CONFIG REQUIRED)
```

## 4. C++ and API rules

- C++20 is required. Do not add C++17 fallbacks.
- Public names live in `namespace mp1`; implementation helpers live in
  `mp1::detail`.
- Prefer concepts, `std::span`, `std::ranges`, and designated initializers.
  Keep `std::format`/`std::ranges` out of the foundational header layer unless
  the dependency is genuinely needed.
- No undefined behaviour: no out-of-bounds, no misaligned access, no strict
  aliasing violations, no reading indeterminate values.
- Use the project error taxonomy: argument/state/policy errors are
  `std::invalid_argument`, `std::logic_error`, or `std::runtime_error` as
  documented; MPI failures surface as `mp1::mpi_error` with location.
- Do not add an exception or a hidden allocation to the common path without a
  strong reason.
- Public headers must be self-contained and listed in `mp1/headers.txt` when
  appropriate. The generated `mp1/version.hpp` is the exception: it is produced
  by CMake and installed, never tracked.
- Keep include order/style; do not introduce a new dependency into
  `fwd.hpp`/`vocabulary.hpp` casually.
- `mp1::tag` is a strong type. Defaults that meant 0 are `{}`; probes default
  to `any_tag`. `status::tag()` returns `mp1::tag`; `source()` remains `int`.
- `mp1::empty_data` is the explicit empty P2P message. Empty collectives are
  not special-cased: call them normally on every rank. `blocks{}` is only a
  dummy on ignored sides; there is no used-side all-empty convenience.
- Do not reintroduce `layout<T>::empty()`, the seven removed duplicate
  functors, or a `blocks{}` used-side shortcut.
- ADL is supported for free functions with an mp1-typed argument; use explicit
  `mp1::` in headers and when `using namespace std` could add overloads.

## 5. MPI-specific rules

- Read MPI 4.1 before changing a call. Record the clause in the comment only
  when it prevents a plausible mistake.
- Never read `MPI_Status` fields that the relevant clause leaves undefined.
  Status provenance is explicit in `mp1/status.hpp`.
- `wait_all`/`test_all`: a per-slot `MPI_ERR_PENDING` is not completion; the
  slot stays active with its handle/payload. Never clear it as if completed.
- `MPI_PROC_NULL` statuses are normalized for blocking, nonblocking,
  persistent, probe, matched, and combined send-receive forms.
- `MPI_TAG_UB` is inclusive and cached once per communicator view; never query
  it per P2P call. `MPI_ANY_TAG` is accepted only by receive/probe paths.
- `MPI_Aint` arithmetic uses `MPI_Aint_diff`/`MPI_Aint_add`; relative maps across
  separate allocations are forbidden.
- Thread level must come from `MPI_Query_thread`/the actual `MPI_Init_thread`
  result, not an assumption.
- Do not introduce a collective that the user did not ask for, including a
  hidden agreement or error-checking collective.
- Provider deviations (MPICH status quirks, Open MPI ASan/MCA issues, zero-size
  datatype limits) are documented, not papered over.

## 6. Testing rules

- A new API form needs:
  - a runtime test at 2 and 4 ranks where meaningful;
  - a compile-pass control if it is a new overload/constraint;
  - a compile-fail probe if it must be rejected;
  - a docs/API-MATRIX row.
- Tests must not depend on a particular provider unless the behaviour is
  explicitly provider-specific and documented.
- Do not weaken or delete assertions. If a test encodes an old wrong
  assumption, fix the assumption and explain it in the commit.
- Prefer deterministic handshakes over sleeps. Avoid timing assumptions.
- Run ASan/UBSan on lifetime/ownership/callback changes.
- Compile-fail probes count only if they are registered and actually fail for
  the intended reason; use the self-check harness.
- Derive suite counts from the build (`ctest -N`, probe/control manifests), not
  from this file.

## 7. Docs and version

- `VERSION` is the sole version source. To release, edit it and reconfigure;
  never edit a generated header or add a literal elsewhere.
- `CHANGELOG.md`'s first `##` heading must begin with the `VERSION` string.
- Do not mention internal development phases, review artifacts, agent names, or
  deleted branches in product docs.
- The old library is mpl 0.5.0. RMA is new in mp1. `MPL_REFLECTION` is the
  historical macro; do not present the later fork's renamed reflection macro as
  a 0.5.0 name.
- Keep README/API-MATRIX/COMPATIBILITY/MIGRATION aligned with the headers.
- Do not add a second source of API counts; update the one documented census
  and its per-family table.

## 8. Release checklist

Before declaring a release candidate:

- [ ] `VERSION` and generated version agree; no hardcoded version in code.
- [ ] default CTest on MPICH, Open MPI and Intel MPI is green;
- [ ] extra-ranks CTest is green;
- [ ] compile-fail, compile-pass controls, compile-all, examples, wrap coverage;
- [ ] ASan/UBSan targeted, no diagnostics;
- [ ] `ci/static-checks.sh`, formatting, links/anchors, C++20/MPI assertions;
- [ ] install manifest and out-of-tree consumer `mp1 <VERSION>`;
- [ ] docs/API census/counts match the build;
- [ ] no old-fork-name / late-0.2x / internal references; mpl 0.5.0 facts correct;
- [ ] working tree clean; no accidental build artifacts;
- [ ] tag/push/history rewrite explicitly authorized by the maintainer.

## 9. Agent workflow

1. Read the relevant code and docs before editing. Do not guess an API from a
   name.
2. State the MPI/C++ reason for a behaviour change; if it is provider-dependent,
   say so.
3. Make the smallest coherent diff. Do not reformat or clean up unrelated code.
4. Add/adjust tests, compile probes, and docs in the same change.
5. Run focused tests first, then the full gate when the change is release-facing.
6. Report exact commands/results and any environment workaround.
7. Do not commit `agent-workspace/`, build directories, or generated
   `mp1/version.hpp`.
8. Do not tag, push, rewrite `main`, or amend the release commit without an
   explicit instruction.
9. If a request conflicts with the eight philosophy points, stop and ask.
