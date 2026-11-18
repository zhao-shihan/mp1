# Changelog

## 1.1.0 — 2026-11-18

Version 1.1.0 adds transparent reduction-functor spellings and aligns the
typed reduction and RMA operator mapping with the C++ datatype categories of
MPI 4.1 §6.9.2. The 1.0 public entry points are retained; callers that used
datatype categories outside the C++ typed surface (Fortran or multi-language
categories) now get the documented fail-closed rejection instead of a callback
mapping.

### Added

- `mp1::max<>`, `mp1::min<>` and `mp1::logical_xor<>` transparent spellings:
  the element type defaults to `void` and is deduced from the reduction
  element type, like the standard transparent functors.
- Transparent `mp1::op_traits` queries forward to the typed truth table from
  the reduction element type; a type-only transparent query remains `false`
  because it has no operand.

### Changed

- Typed collective reductions and RMA atomic operations use the MPI 4.1
  §6.9.2 C++ datatype rows directly: floating `max`/`min`, complex
  `plus`/`multiplies`, and C-integer logical operations map to predefined MPI
  operations; the same table validates raw RMA handles.
- An explicit `MP1_DATATYPE_TRAITS` on an enum is classified from its runtime
  handle; `fetch_and_op` and `compare_and_swap` are restricted to the C
  integer, Logical and Byte categories.
- The CI provider matrix is MPICH 4.3.2, Open MPI 6.0.0rc3 and Intel MPI
  2021.19. The documented CI-only skips cover the Intel holey allgather at
  four or more ranks, the Intel 2021.19.2 self-respawn, and the Open MPI 6+
  four-rank file-spawn entry; these are recorded as possible CI container
  environment limitations.

## 1.0.0 — 2026-11-16

Version 1.0.0 is the first stable release of mp1, the renamed and rewritten
successor to mpl 0.5.0. The project version, the installed CMake package and the
public `MP1_VERSION_*` macros all report `1.0.0`; they are derived from the
single top-level `VERSION` file and the generated `mp1/version.hpp` header.
The typed surface is the 1.0 surface described in [API-MATRIX.md](API-MATRIX.md)
and [README.md](README.md); the audited declaration count is **641** public
declarations (**620** across the sixteen family headers; **607** unique
signatures). The per-family counts and the counting rule are in
[API-MATRIX.md](API-MATRIX.md).

### Highlights

- **C++20 prerequisite.** mp1 requires C++20: GCC 13+, Clang 16+ (with
  libstdc++ 13+ or a conforming libc++), or MSVC 19.32+. The error layer uses
  `std::source_location` and `std::format`; buffer positions use concepts,
  `std::span` and designated initializers. The requirement is a toolchain
  prerequisite, not the migration subject; mp1 provides only the C++20 target.
- **One constrained position vocabulary.** Point-to-point, fixed collectives,
  reductions, RMA and file data access take a value, a contiguous range, or a
  `region<T>`. The predecessor's pointer + layout forms are gone; pointer +
  count was never part of mpl 0.5.0 and is not introduced.
- **Dense and typed variable descriptors.** The variable collectives take
  `mp1::blocks{.counts, .displacements}` or
  `std::span<const layout<T>>` + `std::span<const count_t>`.
- **Member-only files.** `mp1::file` exposes 129 declarations (5 lifecycle,
  7 control, 15 metadata/positioning, 6 atomicity/extent/handler and 96 data
  access), with no file view type and no free data-access wrappers.
- **Option-bag factories.** Window creation uses
  `window_create_options`/`window_allocate_options`; spawn uses
  `spawn_options`/`spawn_multiple_options`/`spawn_result`.
- **Error context.** `mp1::context`, `mp1::fail`, `mp1::require`,
  `mp1::throw_invalid` and `mpi_error::{operation,where}` carry the operation
  literal and the library check site. No public entry takes a `what` parameter.
- **Typed reduction callback image.** The mp1-managed reduction functor no
  longer sends an `MPI_BYTE` full-object image. Its callback datatype is the
  native typed map of `T` at extent `sizeof(T)`, so heterogeneous conversion
  applies to the data leaves; padding and unlisted members are not transmitted
  and a direct receive preserves them. The native true lower bound is remapped
  to 0 and compensated on the buffer side, so `T[len]` still addresses the
  listed fields at their C++ offsets. Callback functors must be stateless,
  nothrow-invocable for `T`, and must not inspect padding or unlisted members;
  `T` must be trivially copyable. `reduce_scatter`/`reduce_scatter_block`
  reject an image whose true extent does not cover `sizeof(T)` (a leading true
  lower bound or trailing unmapped bytes), because MPICH 4.3.2 (the floor
  provider) lays those callbacks out at the true extent; the other families
  accept it.
- **Reduction commutation traits.** The built-in functors no longer inherit a
  blanket `commutative_op`. `mp1::op_traits` marks only the operation/type
  pairs that are actually commutative: integral `mp1::max`/`min`, arithmetic
  `std::plus`/`std::multiplies`/`std::logical_and`/`std::logical_or` and
  `mp1::logical_xor`, floating `std::plus<std::complex<T>>`/
  `std::multiplies<std::complex<T>>`, and integral-or-`std::byte`
  `std::bit_*`. Floating-point `max`/`min` and user types are created with
  `commute = 0`; the `commutative_op` tag remains the user opt-in.

### Added

- `mp1::region<T>`, the non-owning single-layout occurrence
  (`{T* data, const layout<T>& map}`), and `mp1::blocks`, the dense variable
  descriptor (`{std::span<const count_t> counts, std::span<const count_t>
  displacements}`).
- `mp1::span_from(T* data, count_t count)`, the checked pointer + count ->
  `std::span` conversion; a negative count throws `std::invalid_argument` and
  `(nullptr, 0)` is an empty span.
- `mp1::tag`, the strong point-to-point tag value type (aggregate, public
  `int value`, default 0, all six comparisons, no implicit `int` conversion),
  plus `mp1::any_tag` (`MPI_ANY_TAG`) and `mp1::tag_ub_min` (32767).
- `mpi_communicator::tag_ub()` / `mp1::tag_ub(c)`, the cached inclusive
  `MPI_TAG_UB` bound.  It is queried once per communicator view with
  `MPI_Comm_get_attr` and falls back to `tag_ub_min` when the attribute is not
  exposed; tag validation never makes an MPI call.
- `mp1::empty_data`, an `inline constexpr std::span<std::byte>` explicit
  zero-count message; it is an ordinary zero-length buffer with no special
  send/receive/collective path.
- `mp1::layout<T>::vector(vector_options<T>)` and
  `strided_vector(strided_vector_options<T>)` option-bag factories.
- **One-sided communication (new in mp1; mpl 0.5.0 had no RMA).**
  `mp1::window`, `mp1::mpi_window`, `window_create_options`,
  `window_allocate_options`, the window factories, `put`/`get`/`accumulate`,
  `get_accumulate`, `fetch_and_op`/`compare_and_swap`, the request-based
  variants and the fence/post/start/complete/wait/test and lock/unlock
  synchronization functions.
- **Persistent collectives (new in mp1; mpl 0.5.0 had blocking/nonblocking
  collectives only).** All sixteen families have a persistent form; staged
  sends are re-packed at every start and staged receives are unpacked once per
  cycle.
- `mp1::spawn_options`, `mp1::spawn_multiple_options` and
  `mp1::spawn_result`.
- `mp1::context`, `mp1::fail`, `mp1::require`, `mp1::throw_invalid`, and
  `mpi_error::{operation,where,is_class}`. `mpi_error` keeps `code()`,
  `error_class()`, `class_name()`, `slot()` and `what()`.
- `mp1/version.hpp`, generated by CMake from the single top-level `VERSION`
  file, with `MP1_VERSION_MAJOR`, `MP1_VERSION_MINOR`, `MP1_VERSION_PATCH` and
  `MP1_VERSION_STRING` defined only there; the header is installed and included
  by the umbrella.
- `std::span<mp1::request>` as the batch completion input:
  `wait_all`, `test_all`, `wait_any`, `test_any`, `wait_some`, `test_some`,
  `get_status_all`, `get_status_any`, `get_status_some` and `start_all`.
- Five public layout parameter-builder aliases, each with a nested
  `entry_type`, named-range construction, `add()` overloads, and (subarray)
  `order()`.

### Changed

- **Point-to-point.** Twelve send modes (standard/buffered/synchronous/ready ×
  blocking/nonblocking/persistent) each expose exactly three forms: value,
  range, region. Receive exposes five blocking policies (value, fixed, region,
  auto-size, bounded `capacity`) and four nonblocking/persistent policies
  (no auto-size). Tag parameters take `mp1::tag` (source-breaking: an `int`
  literal no longer converts; spell `mp1::tag{n}` and the wildcard
  `mp1::any_tag`). Send tags must be in `[0, tag_ub]` and never `any_tag`;
  receive/probe tags also accept `any_tag`; the `create_group` and
  `intercomm_create` tags use the same cached bound; default tag arguments are
  `{}` and probes default to `any_tag`. `status::tag()` returns `mp1::tag`
  (source-breaking; use `.value` for a raw `int`). The explicit empty spelling
  is `mp1::empty_data`, with `std::span<T>{}` still accepted. Matched receive
  takes `message&` only.
- **Fixed collectives and reductions.** One declaration per family/lifetime
  over the shared position vocabulary, including `region` and `in_place`.
  `reduce`, `allreduce`, `scan`, `exscan`, `reduce_scatter` and
  `reduce_scatter_block` take the operator before the data; `reduce` also
  takes the root. Intercommunicator roots use `MPI_ROOT`/`MPI_PROC_NULL` as
  MPI defines them.
- **Variable collectives.** Dense `blocks` and typed layout-span descriptors,
  with pre-MPI length, non-negative and bounds validation. Blocking calls
  borrow; nonblocking and persistent calls copy the descriptor arrays and
  duplicate the layouts they stage.
- **Files.** Every data-access operation is a member with value, range and
  region forms. mpl 0.5.0 was already member-only and had no free file
  data-access wrappers or file view type; the change is the buffer spelling,
  the `set_view` argument order and the `request`/`status` return types. The
  split begin/end pair records the operation and buffer and remains a strict
  state machine. Reads are `const`; writes and split begin/end are non-`const`.
- **Spawn.** Four entry points over `spawn_options`/`spawn_multiple_options`;
  the `*_with_errcodes` forms return `spawn_result`. The result is a
  disconnect-policy intercommunicator.
- **Spawn command containers.** `command_line`, `command_lines` and `infos`
  are now standard-vector aliases; the predecessor's wrapper classes and
  their limited container APIs are gone.
- **Errors.** Diagnostics are formatted as
  `mp1: <operation> (<file>:<line>): <message>` using `std::format`, where the
  location is the library check site. The predecessor threw its own `error`
  hierarchy; 1.0 uses `std::invalid_argument`, `std::overflow_error`,
  `std::logic_error`, `std::runtime_error`, `std::bad_array_new_length`,
  `std::bad_alloc` and `mp1::mpi_error`. There is no common predecessor-style
  exception base; catch `std::exception` for a broad catch.
- **Layouts.** The five parameter builders are aliases of one template with a
  public nested `entry_type`; the positional `vector`/`strided_vector`
  factories are available alongside the option-bag forms.
- **Lifecycle.** mpl 0.5.0 initialized implicitly and finalized at static
  destruction; 1.0 requires an explicit `init`/`finalize` pair or one
  `mp1::environment`, does not finalize implicitly unless
  `finalize_at_exit()` is called, and requests `MPI_THREAD_SINGLE` by default.
  C++20 is a prerequisite, not a behavior change.
- **Info, groups and status.** `group` default state is invalid and
  `group::empty()` is valid; `status` carries `native`/`normalized`
  provenance, count/elements accessors, an `int` `source()` and an
  `mp1::tag` `tag()`. mpl 0.5.0 used a distinct tag class and a separate
  status wrapper; 1.0 uses the strong `mp1::tag` type and `mp1::status`.
- **Buffered sends.** The lower-level process and communicator attach
  functions use the large-count MPI 4.1 entry points and `count_t`;
  `bsend_buffer(int size)` remains the explicit-buffer RAII guard and keeps its
  `int` attachment ceiling. Both attach functions accept any size with
  `buffer_automatic`, and both detach functions report an undefined or
  unrepresentable detached size as `MPI_UNDEFINED` instead of throwing after
  the detach.
- **Reduction functors.** The typed built-in functor surface is
  `mp1::max`, `mp1::min` and `mp1::logical_xor` plus the seven
  standard-library functors `std::plus`, `std::multiplies`, `std::bit_and`,
  `std::bit_or`, `std::bit_xor`, `std::logical_and` and `std::logical_or`
  (including their transparent `void` specializations). The seven duplicate
  mp1 spellings are removed; see [Removed](#removed).

### Fixed

- **`MPI_PROC_NULL` receive and probe status.** Blocking, nonblocking,
  persistent and matched (`MPI_MESSAGE_NO_PROC` through `mrecv`/`imrecv`)
  receives, `probe`/`iprobe`, and the receive side of blocking and nonblocking
  combined send-receive (`sendrecv`/`sendrecv_replace` and
  `isendrecv`/`isendrecv_replace`) report the MPI 4.1 §3.10 status fields:
  source `MPI_PROC_NULL`, tag `mp1::any_tag`, count 0 and success. mp1
  synthesizes the status from the requested source (blocking receives, probes
  and the combined receive side) or recorded request metadata
  (nonblocking/persistent/matched receives and the nonblocking combined
  request), so pinned MPICH 4.3.2 (which leaves a persistent
  `MPI_PROC_NULL` receive's source at `MPI_ANY_SOURCE` and an
  `MPI_Iprobe(MPI_PROC_NULL)` status at `MPI_ANY_SOURCE`) reports the standard
  values too. The status has normalized provenance and must not be passed to
  an MPI status query.
- **Zero-size datatype wire counts.** Fixed collectives, variable collectives
  and reductions normalize the wire count to 0 when the addressed datatype
  size is 0 (MPI 4.1 §5.1.11), so a positive count of a zero-size datatype does
  not reach a provider that mishandles it (pinned MPICH 4.3.2). Dense
  descriptors skip the span-fit comparison for a zero-byte native datatype and
  accept a null base for a zero-byte repetition. P2P and file operations keep
  the caller's count.
- **File split controls and typed views.** `set_info`, `set_atomicity` and
  `seek_shared` remain legal during a split epoch (MPI 4.1 §14.2.8/§14.6.1/
  §14.4.4); `sync`, `set_view`, `resize` and `preallocate` still refuse while a
  split is active. A typed `set_view` validates the decoded filetype typemap
  (non-negative, non-descending displacements; overlapping filetype on a
  writable file); the hole-multiple-of-etype-extent rule remains the caller's
  obligation for an opaque or `from_raw` filetype.
- **`MPI_NO_OP` RMA origin.** Every origin under `MPI_NO_OP` is ignored, not
  inspected: a null or non-empty origin is accepted, its size is not checked,
  and it may alias the result (the atomic-read idiom). The native call receives
  the RMA dummy origin with count 0.
- **Spawn errcodes sizing.** The `*_with_errcodes` forms take an optional
  `errcodes_length` so a non-root rank that keeps its ignored default payload
  can still size the root's OUT array; without it the array follows the
  rank-local `maxprocs` and all ranks must pass the same lengths.
- **Info key/value length policy.** Keys and values are accepted only when
  shorter than `MPI_MAX_INFO_KEY`/`MPI_MAX_INFO_VAL` and free of embedded NUL,
  a conservative subset portable under either reading of the C binding's
  maximum length. The check runs before `MPI_Info_set`.
- **Zero-size layout true bounds.** A valid zero-size layout reports
  `true_byte_lower_bound() == true_byte_extent() ==
  true_byte_upper_bound() == 0` instead of exposing a provider sentinel, so
  `true_byte_upper_bound()` is total over valid layouts (MPI 4.1 §5.1.8 leaves
  the empty-typemap bounds undefined).
- **Release-path lifecycle cost (internal).** A settled lifecycle phase takes
  a single-guard fast path: release and destruction paths issue one
  `MPI_Finalized` query and no redundant `MPI_Initialized` pair, while still
  abandoning handles after an application-issued `MPI_Finalize`.
- **Foundational include graph (internal).** `vocabulary.hpp` no longer
  includes `<format>` or `<ranges>`; `span_from` keeps its diagnostic with
  `std::string`, and the buffer-element aliases live in
  `detail/concepts.hpp`. The four RMA request families share one transfer
  helper, and the window is validated once at entry instead of inside the
  target preparation. No API change.

### Removed

- The seven duplicate mp1 reduction functors `mp1::plus`, `multiplies`,
  `bit_and`, `bit_or`, `bit_xor`, `logical_and` and `logical_or`; the
  standard-library `std::plus`, `std::multiplies`, `std::bit_and`,
  `std::bit_or`, `std::bit_xor`, `std::logical_and` and `std::logical_or`
  functors are accepted and mapped identically. `mp1::max`, `mp1::min` and
  `mp1::logical_xor` remain (no standard-library object equivalent).
- The predecessor's layout classes, layout containers, null/empty sentinels,
  byte-unit displacement container, absolute-data helper, heterogeneous/void
  layout and iterator layout.
- `layout<T>::empty()` (source-breaking): `contiguous(0)` is the zero-size
  layout; `mp1::empty_data` is the explicit zero-count message.
- The predecessor's separate nonblocking/persistent request classes, request
  pools and completion enum.
- The predecessor's rich tag class (its `up()`/`any()` helpers, stream
  operators and implicit enum constructor), status and message wrapper types,
  and its size and offset aliases.  The strong `mp1::tag` value type replaces
  the predecessor's tag class, but intentionally exposes no implicit `int`
  conversion and no `up()` helper; query `mp1::tag_ub(c)` instead.
- The predecessor's exception hierarchy; 1.0 uses the standard exception types
  plus `mp1::mpi_error`.
- Pointer + layout operation families and iterator-pair point-to-point
  receives. Pointer + count was not an mpl 0.5.0 form.
- The predecessor's `absolute` constant (an `MPI_BOTTOM` spelling); use
  `MPI_BOTTOM` directly for raw MPI.
- The predecessor's buffer-size helper; use the MPI pack-size routine or
  automatic buffering.
- The predecessor's `MPL_DEBUG` macro and its `MPL_HOMOGENEOUS` byte opt-in
  (which did not compile and provided no working byte image); 1.0 validates
  unconditionally and uses `MP1_REFLECTION`/the explicit datatype annotations.
- Topology, Cartesian/graph/distributed-graph communicators, neighborhood
  collectives and grid types (already outside the typed surface; use raw MPI).

### Tests and verification

- **293** non-nightly CTest entries pass in the default configuration; **294**
  with `MP1_TEST_EXTRA_RANKS=ON`, which adds the 5-rank persistent-collective
  entry. The release build enables `MP1_BUILD_PERF`, which contributes the
  deterministic `perf_calls` gate. **97** of the default entries are
  compile-fail entries (**95** contract probes + 2 harness self-checks).
- **95** compile-fail probes, **63** compile-pass controls and **9**
  compile-all API sources are built by the same configure.
- **90** unit-test sources carry **509** Boost test cases; the full build
  compiles **279** C++20 translation units and has **416** build edges when
  tests, examples and perf are enabled.
- The wrap-coverage gate covers **177** wrapped MPI symbols across 34
  implementation files.
- `ci/static-checks.sh`, `clang-format --dry-run --Werror` and the public-header
  self-containment target pass.
- The suite is exercised with Open MPI and MPICH; the release evidence records
  the exact versions and configurations.

### Earlier releases

mpl 0.5.0 was the last release of the previous C++17 library. Version 1.0.0
renames the package to mp1 and replaces the old layout classes, pointer +
layout/iterator/container forms, implicit lifecycle and old request/status/tag
types with the typed surface above. File I/O was already member-only in 0.5.0:
there were no free data-access wrappers to remove. One-sided communication is
new in 1.0. C++20 is a prerequisite. There is no compatibility layer for the
mpl 0.5.0 package; see [MIGRATION.md](MIGRATION.md).
