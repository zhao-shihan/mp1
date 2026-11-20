# mp1

mp1 is a header-only C++20 typed wrapper for the **MPI 4.1 World Model**. It
wraps the MPI C API in a small set of RAII owners, borrowed views and free
functions: one communicator view and one owning communicator, one window view
and one owning window, one request type, one owning layout type, and free
point-to-point, collective, one-sided and file member operations. It is **not**
a complete MPI 4.1 binding; see [Scope](#scope) and
[COMPATIBILITY.md](COMPATIBILITY.md).

- Header-only, C++20, no compiler extensions.
- Requires an MPI 4.1 or newer implementation and a C++ compiler wrapper such
  as `mpicxx`.
- Free functions usually mirror MPI's C argument order, so porting an MPI call
  usually means moving the communicator to the first argument. The root-taking
  collectives take `(communicator, root, ...)` (`bcast`, `gather`, `scatter`,
  `gatherv`, `scatterv`); `reduce` takes `(communicator, op, root, ...)`;
  `reduce_local` takes `(op, send, recv)`; the spawn entry points take
  `(communicator, root, options)`, moving `root` ahead of the
  command/maxprocs/info payload; and `fetch_and_op` keeps MPI's operator-last
  order, `(window, origin, result, target_rank, target_disp, op)`. The other
  reductions take `(communicator, op, ...)`.
- Buffer positions are constrained C++20 forms, not overload families: a
  single value, a contiguous range (`std::span`, `std::vector`, `std::array`,
  a C array other than a built-in character array, `std::string`), or a
  `region<T>` (one occurrence of a committed layout). There are no pointer +
  count operation forms.
- No hidden threads, no generalized requests, no hidden collectives.

The headers are the API reference; `///` comments document the public
declarations. There is no generated API documentation. The audited surface is
**649** public declarations (**628** across the sixteen family headers;
**615** unique signatures, **594** across the sixteen). The count is a
scope-aware function/member declaration count, not a count of operation
forms: the per-family map and the exact counting rule are in
[API-MATRIX.md](API-MATRIX.md).

## Scope

mp1 requires an MPI implementation that advertises MPI 4.1 or newer and supports
the families listed in [COMPATIBILITY.md](COMPATIBILITY.md). "MPI 4.1 or newer"
is a requirement, not a coverage promise. For a per-family view of which
lifetimes and data spellings are declared, see the
[API matrix](API-MATRIX.md).

The typed surface is the World-Model core: lifecycle, thread and timer queries;
point-to-point (blocking, nonblocking, persistent, combined send-receive,
probes, matched messages); fixed, variable, persistent and reduction
collectives, including dense `blocks` descriptors, typed layout descriptors,
layout borrows and `reduce_local`; one-sided communication and shared-memory
windows; file I/O (core access, individual and shared file pointers, atomicity,
type-extent queries, member-based blocking/nonblocking/split data access);
communicator management (`dup`, `dup_with_info`, `idup`/`idup_with_info`
with `take_communicator`, `split`, `split_type`, `create`, `create_group`,
`create_from_group`, `merge`, `intercomm_create`,
`intercomm_create_from_groups`, `adopt`); process spawning; the MPI 4.1
process/communicator buffered-send buffers; groups (including `group_incl`/
`group_excl` over `std::span<const int>` and `group_range_incl`/
`group_range_excl`), info, layouts, status, request completion, and predefined
error-handler set/get.

The following are deliberately **not** wrapped:

- topology, Cartesian/graph/distributed-graph communicators, neighborhood
  collectives, and grid decomposition types;
- `MPI_Alltoallw`, `MPI_Type_create_darray`, runtime raw-struct/array datatype
  construction, and an owning/raw `MPI_Datatype` bridge;
- a public `MPI_Pack`/`MPI_Unpack` API (the wrapper's staging is internal);
- file `get_view` (future work; requires the owning raw-datatype bridge),
  user datareps/`register_datarep`, and raw heterogeneous views;
- MPI Sessions and partitioned point-to-point;
- attributes/keyvals, user error-handler callbacks, DPM connect/accept/join,
  object naming, generalized requests, and MPI_T.

Use raw MPI through `native_handle()` on an mp1 owner or view, or
`communicator::adopt()`, where the wrapper stops. A typed bulk-byte transfer
does not need a raw datatype bridge: `std::byte` maps to `MPI_BYTE`, so a
`region<std::byte>` or a `std::span<std::byte>` is the type-erasure route.

## Requirements

- A C++20 compiler: GCC 13 or newer, Clang 16 or newer (with libstdc++ 13+ or a
  conforming libc++), or MSVC 19.32 (Visual Studio 2022 17.2) or newer. The
  error layer uses `std::source_location` and `std::format`; the buffer forms
  use concepts, `std::span` and designated initializers.
- An MPI 4.1+ implementation with `mpicxx` and `mpiexec`.
- CMake 3.21 or newer to build examples or install the package. A consumer that
  only uses an installed package needs `find_package` support and a C++20
  compiler; mp1's imported target raises the consumer to C++20.
- Boost.Test is needed only when configuring the in-tree test suite. It is not
  a dependency of the installed package.

## Install

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /opt/mp1
```

`MP1_BUILD_EXAMPLES` and `MP1_INSTALL` default to `ON` when mp1 is the
top-level project. Set `-DMP1_BUILD_EXAMPLES=OFF` to skip the examples, and
`-DBUILD_TESTING=ON` to configure the test suite.

Downstream CMake:

```cmake
find_package(mp1 2.0 CONFIG REQUIRED)

add_executable(app main.cc)
target_link_libraries(app PRIVATE mp1::mp1)
```

The package installs `mp1Config.cmake`, `mp1ConfigVersion.cmake` and
`mp1Targets.cmake` under `lib/cmake/mp1`, plus the headers under
`include/mp1/`.

The version has one source: the top-level `VERSION` file. CMake derives the
project and package version from it and generates `mp1/version.hpp` from
`mp1/version.hpp.in`; the header is installed with the other public headers, so
the `MP1_VERSION_*` macros, the CMake package version and the installed header
always agree. Edit `VERSION`; do not add a version literal to CMake, a header
or a test.

## Initialize and finalize

```cpp
#include <mp1/mp1.hpp>

#include <iostream>

int main(int argc, char** argv) {
  mp1::init(argc, argv);

  auto world = mp1::comm_world();  // non-owning view; keep it, do not recreate per call
  std::cout << "rank " << world.rank() << " of " << world.size() << '\n';

  mp1::finalize();
}
```

`mp1::init()` calls `MPI_Init`; MPI 4.1 §11.2.1 states that a call to `MPI_INIT` has
the same effect as a call to `MPI_INIT_THREAD` with `required = MPI_THREAD_SINGLE`.
An implementation started with only a higher level available may install that level
instead (§11.2.1), so the accept-any form records the level MPI actually provides with
`MPI_Query_thread` and does not assume SINGLE. This is a lifecycle default, not a
thread-level promise. Use `mp1::init(mp1::threading_mode::multiple)`
(the `MPI_Init_thread` form) when the program calls mp1 from several threads.
The explicit form throws `std::runtime_error` if the level provided by MPI is
lower than the level requested; §11.2.1 allows an implementation to return a
lower level when the requirement cannot be satisfied, so the wrapper diagnoses
it instead of silently accepting it. If that happens after mp1 initialized MPI,
MPI stays initialized and mp1 still owns it: every rank must still reach
`mp1::finalize()`.

`mp1::init` is serialized with `mp1::finalize` and may be called from any
thread: exactly one `MPI_Init`/`MPI_Init_thread` is executed and concurrent
callers all return after it. A call while MPI is already active adopts the
environment (for example after the program called `MPI_Init` itself) without
taking ownership; subsequent calls are no-ops. `mp1::init` after finalization
throws `std::runtime_error`; the World Model cannot be restarted (MPI 4.1
§11.2.1, §11.2.3).

`mp1::finalize` finalizes only MPI that mp1 initialized; if the program
initialized MPI itself, it must also finalize it. `mp1::environment` is an
optional RAII wrapper for programs that prefer scope-based initialization; it
owns MPI only when it had to initialize it, and its destructor is `noexcept`.
If the constructor throws after it initialized MPI, no object is constructed,
MPI stays initialized, and mp1 still owns it: every rank must still reach
`mp1::finalize()`. There is no implicit finalize at exit. `mp1::finalize_at_exit()` is the opt-in
escape for programs whose `main` cannot reach `finalize()`; it is
single-threaded only and follows the same live-object precondition as
`finalize()`.

`mp1::comm_world()` and `mp1::comm_self()` throw before initialization and after
finalization. `mp1::comm_null()` is always available and returns an invalid
view by value. `comm_world()`/`comm_self()` return `const mpi_communicator&` to
a process-lifetime borrowed view; `auto world = mp1::comm_world()` still copies
the view if a copy is wanted.

### Finalization order

Before `mp1::finalize()` (or leaving an owning `mp1::environment` scope) the
process must complete its MPI involvement: all blocking operations returned,
every request complete/inactive/released, every RMA epoch closed, every window
freed, every file closed (including split epochs), every spawned
intercommunicator disconnected on both ends, every buffered-send buffer
detached, and every thread that used mp1 joined. This is the MPI 4.1 §11.2.2
precondition; `MPI_FINALIZE` is collective over all connected processes, and it
does not free objects created by MPI calls. mp1 does not enforce this
precondition: `finalize()` proceeds to `MPI_Finalize` even if an mp1 owner is
still live, and a live owner at that point is a caller precondition violation.
After finalize only the always-available functions of §11.4.1 (`info`,
`initialized`/`finalized`/`available`, version queries, `mpi_error` formatting)
and object destructors may run.

### Threads and lifecycle

The library is thread-compatible, not internally synchronized per object (MPI
4.1 §11.6.1 restates the same rule for MPI objects):

- `init`/`finalize` are serialized by the library; concurrent `finalize` losers
  get `std::runtime_error`. `MPI_FINALIZE` should still be called from the
  initializing thread after all other threads stopped making MPI calls.
- `is_thread_main()` wraps `MPI_Is_thread_main` and reports whether the calling
  thread is the thread that initialized MPI.
- Distinct mp1 objects may be used concurrently by distinct threads at
  `MPI_THREAD_MULTIPLE`. One request, communicator, window, file, info or
  layout object must be confined to one thread while it is being operated on.
- All wrapper process state is atomic, thread-local or a leaked one-time
  static; there are no per-object locks and no hidden threads.
- An external `MPI_Finalize` is observed by `initialized()`/`finalized()`/
  `available()`, by `finalize()`, by every destructor and by the release
  operations; ordinary operations do not query it, so do not call mp1 after an
  external finalize.
- `MPI_Info_*` calls are always available and may be called concurrently by the
  application (§11.4.1, Table 11.1). The wrapper's lazy handle creation in
  `info::set` is a write to that object: create the table before sharing it.

## Vocabulary and value model

The operation vocabulary lives in `mp1/impl/vocabulary.hpp`:

| Name | Meaning |
|---|---|
| `count_t` | `std::ptrdiff_t`; every public operation count, receive capacity and collective displacement (file offsets use `offset_t`). |
| `capacity` | Strong upper bound for a resizable-container receive: `capacity{64}`. It prevents a raw count from being read as a source rank. |
| `region<T>` | Two words: `T* data` and a reference to a `layout<T>`. One occurrence of the layout at `data`. A temporary layout is rejected; a rvalue `region` is not a form. |
| `blocks` | `{std::span<const count_t> counts; std::span<const count_t> displacements;}` for the four variable collectives. An empty `blocks{}` is the zero-block descriptor on an ignored side; a used side needs peer-length zero arrays or empty layouts. |
| `in_place` | The `MPI_IN_PLACE` sentinel, usable only in the buffer position MPI defines. |
| `span_from(p, n)` | Checked pointer + count -> `std::span`; a negative count throws `std::invalid_argument`, `(nullptr, 0)` is empty. |
| `layout<T>` | Owning committed `MPI_Datatype` plus construction facts. |

A used data position is empty only when its total byte size is zero (MPI 4.1
§5.1.11), not when its element count is zero: a count of zero, an empty
`layout<T>`, or an empty range is empty, and so is a positive count of a
zero-size datatype. In a collective or reduction the zero-byte position is
presented to MPI with a normalized count of 0 (the type signature is empty
either way), so a positive count of a zero-size datatype never reaches the
provider. A zero-byte position may be null; a non-empty position
with a null pointer is `std::invalid_argument` before any MPI data call or
managed `MPI_Op` creation. See [Null data positions](#null-data-positions) for
the exact preflight and the first-use datatype caveat. The empty spelling for
a typed position is the empty span: `std::span<T>{}`. There are no `*_empty`
function names; see [Empty messages](#empty-messages) for the exact spellings
and their native lowering.

A contiguous lvalue buffer is borrowed (zero-copy) for a blocking call and for
the lifetime of a request. A non-contiguous sequence of trivially copyable
elements is staged. A non-contiguous sequence of any other element type is
described without a value copy by a typed absolute-address datatype passed with
`MPI_BOTTOM` (MPI 4.1 §5.1.12); it borrows the caller's elements, so they must
stay alive and unmodified until the send completes. A blocking send
accepts an rvalue value or container (the call finishes before the temporary
dies); a nonblocking or persistent send requires a borrowable buffer
(`std::is_lvalue_reference_v` or a borrowed range). A borrowed buffer must stay
alive and must not be resized, reallocated or freed until the operation
completes.

A container element type needs an MPI datatype mapping: `native_datatype<T>()`,
`MP1_REFLECTION`, `MP1_DATATYPE_TRAITS`, or the built-in `std::pair`,
`std::tuple` and `std::array` mappings. C++ trivial copyability is not
required. A capacity receive also requires a default-constructible element
type, because the container is resized before MPI writes into it; a staged
buffer additionally requires default-constructible and copy-assignable element
types.

## Owners, views and release

A **view** (`mp1::mpi_communicator`, `mp1::mpi_window`) is a copyable borrowed
handle with an immutable metadata snapshot. It owns nothing: its destructor
never calls MPI. A view is valid only while the MPI object it borrows is live.
`free()`, `disconnect()`, the owner's destruction and `MPI_Finalize` invalidate
every view of the object. `release()` transfers the raw handle to the caller
without an MPI call, so views copied before it remain valid while the caller
keeps the MPI object alive. `is_valid()` only reports
`handle != MPI_COMM_NULL`/`MPI_WIN_NULL`; it is not a liveness check.

Constructing a view from a valid non-null raw handle (`mpi_communicator(MPI_Comm)`,
`mpi_communicator::borrow`, `mpi_window(MPI_Win)`, `mpi_window::borrow`)
requires active MPI. Default construction and construction from
`MPI_COMM_NULL`/`MPI_WIN_NULL` are MPI-free.

An **owner** (`mp1::communicator`, `mp1::window`) is move-only and contains a
private view. It exposes it by value through `view()` and an implicit
conversion, so an owner can be passed to any free function taking a view
(`mp1::send(world, ...)` works with `world` either view or owner). Owners do not
publicly derive from the view: binding a mutable view reference or a view
pointer to an owner is a compile error.

`mp1::communicator` carries one release policy fixed at construction
(`policy()`):

| Construction | Policy | Destructor | `free()` | `disconnect()` |
|---|---|---|---|---|
| `dup`/`dup_with_info`/`idup`/`idup_with_info`/`split`/`split_type`/`create`/`create_group`/`create_from_group`/`merge`/`intercomm_create`/`intercomm_create_from_groups`/`adopt` | `free` | `MPI_Comm_free` | allowed | allowed |
| `spawn`/`spawn_multiple`/`parent`/`adopt_connected` | `disconnect` | `MPI_Comm_disconnect` | rejected (`std::logic_error`) | required collective release |

`free()`/`disconnect()` make one attempt; on an MPI error the object becomes
terminal and `mp1::mpi_error` is thrown, because retrying a collective teardown
is unsafe. `release()` is the no-MPI-call escape. Both `free()` and
`disconnect()` are collective and may block. `communicator::abort(error_code)`
calls `MPI_Abort`: a successful `MPI_Abort` return is passed through, and a
non-success return throws `mp1::mpi_error`. Do not assume `abort()` never
returns; MPI 4.1 §11.4.2 describes the abort as a best attempt. A
`mp1::window` has the single `MPI_Win_free` policy plus `release()`.

| Type | Model | Copy / move | Teardown | No-MPI escape |
|---|---|---|---|---|
| `communicator` | Move-only owner of `MPI_Comm` with private view. | Move-only; `view()` copies. | Policy operation (`MPI_Comm_free`/`MPI_Comm_disconnect`). | `release()` |
| `mpi_communicator` | Copyable borrowed view; metadata snapshot. | Copyable. | Never calls MPI. | n/a |
| `window` | Move-only owner of `MPI_Win` with private view. | Move-only; `view()` copies. | `MPI_Win_free`; may block. | `release()` |
| `mpi_window` | Copyable borrowed view; metadata snapshot. | Copyable. | Never calls MPI. | n/a |
| `file` | Owner-only, no view; member data access over one shared state. | Move-only. | Finish split, then collective close; may block. | `release()` |
| `group` | Owner-only, move-only, no view. | Move-only. | `MPI_Group_free` (`MPI_GROUP_EMPTY` is not owned). | None public |
| `info` | Copyable owner. | Copy duplicates (`MPI_Info_dup`); move steals. | `MPI_Info_free`; always available. | None |
| `request` | Move-only owner; a completed `idup` request also owns its result until `take_communicator()`. | Move-only. | State-machine drop policy; may wait/cancel, or free a `result_pending` duplicate. | `release()` |
| `message` | Move-only owner of one matched message. | Move-only. | Discards a still-owned message; `noexcept`. | `release()` |
| `layout<T>` | Copyable owner of one committed `MPI_Datatype`. | Copy duplicates; move transfers. | `MPI_Type_free`. | None; `native_handle()` borrows |

`release()` is not one blanket contract. `communicator::release()` and
`window::release()` transfer the raw handle without an MPI call. `request` and
`message` have no raw-handle transfer: their `release()` is a no-MPI-call
abandon/leak of the handle and payload. `file::release()` transfers the
`MPI_File`, but it may call `MPI_Type_free` once if an active region split owns
a duplicated layout. `layout` has no public release; use `native_handle()`
only to borrow the datatype.

Move assignment over a live owner is a teardown with the same operation as
destruction: `MPI_Comm_free`/`MPI_Comm_disconnect`, `MPI_Win_free`, file
split/close, or the request drop policy. These operations may be collective
and/or blocking, so every rank must reach the same assignment in the same
order.

### Advanced communicator creation

`mp1::communicator` also wraps the MPI 4.1 source-less factories and the
nonblocking duplicate:

```cpp
// Collective over the members of gr; every member passes the same group and stringtag.
auto gr = mp1::group_range_incl(mp1::get_group(world),
                                std::array{mp1::rank_range{0, world.size() - 1, 1}});
mp1::communicator created = mp1::communicator::create_from_group(
    gr, "my-app-communicator", mp1::info{}, mp1::error_handler::returns);

// Collective over the union of local_group and remote_group.
mp1::communicator inter = mp1::communicator::intercomm_create_from_groups(
    local_group, 0, remote_group, 0, "my-app-intercomm", mp1::info{},
    mp1::error_handler::returns);

// Nonblocking duplicate: complete the request, then take the owning result.
mp1::request r = mp1::communicator::idup(world);
r.wait();
mp1::communicator duplicate = mp1::take_communicator(r);
```

Both from-group factories take MPI's argument order. `info` defaults to
`mp1::info{}` (which passes `MPI_INFO_NULL`, accepted by MPI) and the handler
defaults to `mp1::error_handler::are_fatal`, matching MPI's default initial
error handler for communicators. Pass `mp1::error_handler::returns` explicitly
when the application uses `install_recoverable_errors()` or wants creation
errors to surface as `mp1::mpi_error`; these source-less factories do not
inherit the `MPI_COMM_WORLD`/`MPI_COMM_SELF` handler. Under MPI 4.1 §9.3.1 the
handler must be a predefined handler or one created by
`MPI_Comm_create_errhandler`; `MPI_ERRHANDLER_NULL` is the null handle and is
**not** a valid handler. `mp1::error_handler` models only the predefined
handlers, so a custom handler is installed with raw MPI or with
`MPI_Comm_set_errhandler` on `native_handle()` after creation. `idup`
inherits the source communicator's handler exactly like `MPI_Comm_dup`.

A `stringtag` is a `std::string_view`, a distinct vocabulary from `mp1::tag`
(there is no conversion between them). It must contain no embedded NUL and be
shorter than `MPI_MAX_STRINGTAG_LEN` (the MPI macro includes the terminating
NUL); the empty string is legal. The view is borrowed for the blocking call
only, and all participating processes must pass the same stringtag.

Contract details:

- An empty `group`/`local_group` is a local operation returning an invalid
  owner without a communicator-creation call. An empty remote group at the
  leader is passed to MPI, which returns `MPI_COMM_NULL`.
- Non-membership of either from-group call is an MPI-erroneous precondition
  that mp1 deliberately does not diagnose locally; MPI is the authority.
- `local_group` and `remote_group` must be disjoint (MPI obligation). Within
  each group every process passes the same local group and leader, and all
  processes pass the same stringtag.
- `remote_group`/`remote_leader` are significant only at `local_leader`; a
  non-leader may pass `mp1::group{}` and an arbitrary non-wildcard remote
  leader value. Wildcards are not permitted for either leader. For
  portability, pass the same valid remote group and a valid remote leader on
  every local rank. Intel MPI 2021.19.2 validates `remote_group` on every
  local rank (see [COMPATIBILITY.md](COMPATIBILITY.md)).
- `idup`/`idup_with_info` are nonblocking collective operations: every rank
  must issue them in the same order on the same source context. The hints of
  `idup_with_info` are captured at initiation, so the `info` object may be
  destroyed after the call. Keep the source communicator valid until the
  request completes.
- A successful `idup` completion enters `request_state::result_pending`: the
  request owns the new `MPI_Comm` but no longer holds a live `MPI_Request`.
  `request::has_result()` is the result-axis query, while
  `is_valid()`/`operator bool` stay handle-only and are false in that state.
  `take_communicator(request&)` transfers the duplicate into an owning
  `mp1::communicator`; a second take is `std::invalid_argument`, and a take
  before completion is `std::logic_error`. `wait`/`test` and every bulk
  completion function complete the request normally and keep the result in the
  request.
- Destroying a `result_pending` request frees its duplicate with
  `MPI_Comm_free`, which is collective and can block (`drop_would_block()` is
  true); `release()` is the no-MPI-call escape that leaks it. Complete and take
  idup results before letting an exception unwind a batch.

## Namespaces and ADL

The public surface lives in namespace `mp1`. A free function is found by
argument-dependent lookup when a parameter's type is associated with `mp1`: a
communicator or window owner/view, a `tag`, or a standard request batch such as
`std::vector<request>`/`std::span<request>` whose element is `mp1::request`.
Local code may call such a function unqualified:

```cpp
#include <mp1/mp1.hpp>

#include <span>
#include <utility>
#include <vector>

using mp1::mpi_communicator;

void local(mpi_communicator world, int x, int y, int* a) {
  send(world, x, 1, mp1::tag{7});
  recv(world, y, 0, mp1::tag{7});
  bcast(world, 0, x);
  allgather(world, x, std::span<int>{a, 2});
  auto req = isend(world, x, 1);
  auto st = probe(world, 0, mp1::any_tag);
  int ub = tag_ub(world);
  std::vector<mp1::request> reqs;
  reqs.push_back(std::move(req));
  auto statuses = wait_all(reqs);
  (void)st;
  (void)ub;
  (void)statuses;
}
```

ADL covers names, not types or variables. `mp1::mpi_communicator`,
`mp1::request`, `mp1::status`, `mp1::tag` and `mp1::layout<T>` still need the
qualifier or a using-declaration. Helpers whose arguments are not
`mp1`-associated -- for example `span_from(int*, n)` and
`native_datatype<T>()` -- and public variables cannot be found by ADL: write
`mp1::span_from(p, n)`, `mp1::native_datatype<T>()`, `mp1::empty_data`,
`mp1::any_tag` and `mp1::tag_ub_min` (or introduce a using-declaration for the
name). Member factories are reached through the type:
`mp1::layout<T>::contiguous(n)` and `mp1::layout<T>::from_raw(...)`.

A `using namespace std;` can add standard names to the same unqualified
lookup. For example, with `<barrier>` in scope, `barrier(world)` names the
`std::barrier` class template (class-template argument deduction), not
`mp1::barrier`; write `mp1::barrier(world)`. `std::reduce` likewise joins the
overload set for `reduce` (and `std::scan`, where the standard library provides
it, for `scan`), so qualify the call if overload resolution is ambiguous. Use
explicit `mp1::` qualification in public headers; ADL is a convenience for
local code.

## Point-to-point

```cpp
#include <mp1/mp1.hpp>

#include <vector>

void exchange(const mp1::mpi_communicator& world) {
  if (world.rank() == 0) {
    std::vector<int> values{1, 2, 3, 4};
    mp1::send(world, values, 1, mp1::tag{7});
  } else if (world.rank() == 1) {
    std::vector<int> values;                       // resized from the exact message
    const auto status = mp1::recv(world, values, 0, mp1::tag{7});
    // status.get_count<int>() is the received element count
  }
}
```

There are twelve send modes: standard, buffered, synchronous and ready, each
blocking, nonblocking and persistent. Each has exactly three forms.

| Form | Send position | Notes |
|---|---|---|
| value | one object of its MPI datatype | `const T&` blocking; an lvalue for nonblocking/persistent. |
| range | `std::span`, `std::array`, C array (except a built-in character array), `std::vector`, `std::string`, `std::string_view`; `std::list`/`std::deque` on send only | Contiguous ranges are zero-copy. A built-in `char[N]`/`wchar_t[N]` is not a range: use `std::span<char>`, `std::array<char,N>`, `std::string` or `std::string_view`. A non-contiguous sequence of trivially copyable elements is staged; a non-contiguous sequence of other elements is described by an absolute-address datatype passed with `MPI_BOTTOM` and borrows the caller's elements until completion. Receive ranges must be contiguous and writable. |
| region | `region<T>` | One occurrence of a `layout<T>`. The layout is borrowed by a blocking call and duplicated at initiation for a request. |

Receive has five blocking policies and four nonblocking/persistent policies:

1. **value** — `T&`; one element. More than one is a truncation error.
2. **fixed** — `std::span<T>`, `std::array<T,N>` or `T[N]`; the extent is the
   capacity and nothing is resized.
3. **region** — `region<T>`; count 1 with the layout, and a
   construction-proven overlapping receive map is rejected.
4. **auto-size** — a contiguous resizable container (`std::vector`,
   `std::string`); blocking `recv` only. The wrapper matches with `MPI_Mprobe`,
   queries the count, and consumes with `MPI_Mrecv`.
5. **bounded** — a contiguous resizable container plus `capacity{n}`; the
   container is resized down to the received count.

`irecv` and `recv_init` provide policies 1, 2, 3 and 5; the auto-size policy is
blocking only because an `MPI_Irecv` needs a count. A raw `count_t` in the
third argument of a resizable receive is a deleted function:
`recv(c, v, n, source)` does not compile; spell the bound
`recv(c, v, capacity{n}, source)` or use the four-argument auto-size form
`recv(c, v, source, tag)`.

The auto-size receive is loss-free: a post-match type mismatch, count overflow,
resize failure or allocation failure consumes the matched message with an
`MPI_PACKED` receive before throwing, so a synchronous sender still completes.
The same mechanism backs `mp1::message::discard()`. A same-size wrong element
type is not detectable from the status; MPI's type matching rules make that
program erroneous. An auto-size receive is **not** a posted receive: it probes
first, so it cannot match a ready-mode (`rsend`) send, which requires the
matching receive to be posted before the send starts. Use a fixed, region or
bounded receive with a ready-mode send.

### Tags

Every point-to-point tag parameter, and the tag of `communicator::create_group`
and `communicator::intercomm_create`, is the strong `mp1::tag` value type:

| Name | Meaning |
|---|---|
| `mp1::tag{n}` | One tag value; aggregate with a public `int value`, declaration default `{}` (tag 0), all six comparisons. There is no implicit conversion from `int`. |
| `mp1::any_tag` | The `MPI_ANY_TAG` wildcard; accepted by receive/probe/`mprobe` and rejected by every send and by the communicator factories. |
| `mp1::tag_ub_min` | 32767, the smallest `MPI_TAG_UB` MPI guarantees (MPI 4.1 §3.2.3); every implementation accepts tags `0..32767`. |
| `mp1::tag_ub(c)`, `c.tag_ub()` | The communicator's cached `MPI_TAG_UB`, inclusive. |

This is a **source-breaking** change: an `int`/`MPI_ANY_TAG` no longer converts
implicitly, so `send(c, v, dest, 7)` and `probe(c, MPI_ANY_SOURCE, MPI_ANY_TAG)`
must be written `send(c, v, dest, mp1::tag{7})` and
`probe(c, MPI_ANY_SOURCE, mp1::any_tag)`.  The tag argument position is
unchanged. `status::tag()` also returns `mp1::tag` (source-breaking; use
`.value` where a raw `int` is required), while `status::source()` remains an
`int`.

A send tag is legal when `0 <= value <= tag_ub(c)`; a negative value, `any_tag`
or a value above the bound is `std::invalid_argument` before any MPI call.
Receive and probe tags accept the same range plus `any_tag`; any other negative
value is rejected the same way. The communicator-factory tags
(`create_group`, `intercomm_create`) use the same wildcard/range rule against
the source communicator's cached bound: `any_tag`, a negative value and a value
above the bound are rejected, and 0 is allowed. The default tag is 0, and
`probe`, `iprobe`, `mprobe` and `improbe` default to `any_tag`. The upper bound
is queried exactly once with `MPI_Comm_get_attr(MPI_TAG_UB)` whenever a
non-null `mpi_communicator` view is constructed (the process-lifetime
world/self views on first use and every owning factory or adoption path),
cached with the other communicator metadata, and never re-queried by tag
validation. If an implementation does not expose the attribute, the cache falls
back to `tag_ub_min`. The normalized and `MPI_PROC_NULL` statuses use
`any_tag` for their tag field.

### Empty messages

An empty point-to-point message is a real message: a zero-count send still has
an envelope, is matched, ordered, probed and completed like any other message,
and `status::get_count<T>()` is 0. `mp1::empty_data` is an
`inline constexpr std::span<std::byte>{}`: its size is zero, it allocates
nothing, and constructing it makes no MPI call. The ordinary positions are
also the empty spellings:

| Empty form | Spelling | Native call |
|---|---|---|
| zero-count message | `mp1::empty_data` (preferred explicit spelling), `std::span<T>{}` or `mp1::span_from(p, 0)`; the same spellings are valid receive targets | count 0, `native_datatype<T>()`; `empty_data` costs no MPI call to construct |
| zero-size datatype | `mp1::region<T>{p, mp1::layout<T>::contiguous(0)}` | count 1, the empty datatype |
| empty auto-size receive | `std::vector<T> v; recv(c, v, source, tag)` | `MPI_Mprobe`, count 0, `MPI_Mrecv` |

There is no pointer + count P2P overload and no `*_empty` name; `span_from` is
the checked pointer + count bridge. A zero-count receive accepts only an empty
message: a non-empty message arriving at a zero-capacity receive is an MPI
truncation error. `MPI_PROC_NULL` is a different mechanism: it is a no-op
sentinel that creates no message and is invisible to wildcard receives and
probes, whereas an empty real message is visible and matchable. A receive from
`MPI_PROC_NULL` (blocking, nonblocking, persistent, or `MPI_MESSAGE_NO_PROC`
through `mrecv`/`imrecv`), a `probe`/`iprobe` of `MPI_PROC_NULL`, and the
receive side of a combined send-receive with source `MPI_PROC_NULL` complete
with source `MPI_PROC_NULL`, tag `mp1::any_tag`, count 0 and success (MPI 4.1
§3.10); the wrapper synthesizes that status on every provider instead of
forwarding a provider-specific envelope, so the request accessors are portable.

A zero-byte position with a null pointer is accepted because no element is
addressed; a non-empty position with a null pointer is rejected before any MPI
data call (see [Null data positions](#null-data-positions)).
The "dummy pointer, count 0, base datatype" normalization applies to the
range/fixed-receive and collective lowerings. The `region<T>` path passes
count 1 with the empty datatype and the caller's pointer (possibly null), and
the auto-size container receive passes the container's `data()` with count 0.
Both transfer zero bytes and MPI does not dereference the pointer, but the
dummy wording is form-specific rather than a universal rule.

Empty collectives are meaningful in the same way. Every required rank must
still call in the collective order, an empty call may synchronize but cannot be
relied on to do so (`barrier` is the only collective with mandatory
synchronization), and empty fixed collectives accept empty spans, empty regions
and `layout<T>::contiguous(0)`.  There is no special empty-collective handling:
`mp1::empty_data` is an ordinary zero-count P2P message span, not a collective
mode or a `blocks{}` replacement. An empty base-repetition reduction layout is lowered
to count 0 with the base datatype, so no empty derived datatype is passed with a
predefined operation. Persistent collectives with empty data still require
every rank to start and complete a cycle once any rank starts it. The variable
descriptors have one separate shape limitation: see
[Variable collectives](#variable-collectives) for `blocks{}` on a used side and
the mandatory dense displacement array.

Matched receive:

```cpp
mp1::message matched = mp1::mprobe(world, 0, 7);
if (matched.probed_status().get_count<int>() == 4) {
  int values[4];
  (void)mp1::mrecv(values, matched);   // matched is invalid afterwards
} else {
  matched.discard();                   // consume without knowing the type
}
```

`mprobe` returns an owning, move-only `mp1::message`; `improbe` returns
`std::optional<mp1::message>`. The handle owns one message that only
`MPI_Mrecv`/`MPI_Imrecv` can consume, so every use consumes it exactly once.
`is_no_proc()` is true for `MPI_MESSAGE_NO_PROC`, the successful no-op handle
returned by a matching probe with `source == MPI_PROC_NULL`; it is not an
invalid handle. The destructor discards a still-owned message; `release()` is
the no-MPI escape and counts the handle as abandoned. `mrecv`/`imrecv` take
`message&` only; an rvalue message is a compile error.

Nonblocking and persistent sends use the same value/range/region forms:
`isend`, `ibsend`, `issend`, `irsend`, `send_init`, `bsend_init`, `ssend_init`
and `rsend_init`. A request retains the address of a value/range/region; an
rvalue container is rejected by the `borrowable` constraint. A persistent
container send keeps the container and revalidates its `data()` and `size()`
before every `start()`.

Probes: `probe` returns `status`, `iprobe` returns `std::optional<status>`;
`mprobe`/`improbe` return the matched `message`. Combined sends:
`sendrecv`, `sendrecv_replace`, `isendrecv` and `isendrecv_replace` each take
one send and one receive position and return the receive status (or a request).
The nonblocking combined forms are wait-only (MPI forbids cancelling them) and
borrow both buffers. A trivially identifiable overlap (same start address, or
two positive byte extents that intersect) is rejected before MPI; general
disjointness is not proven. Use `sendrecv_replace` for the in-place form. The
status returned by `isendrecv`/`isendrecv_replace` is the provider's receive
status (MPI 4.1 §3.7.2 does not require every provider to preserve its
tag/count, and for an ignored direction even source, so treat those fields as
provider-defined) unless the receive source is `MPI_PROC_NULL`: then the
wrapper returns the synthesized §3.10 status (source `MPI_PROC_NULL`, tag
`any_tag`, count 0, success) with normalized provenance.

A `mp1::request` is move-only and `[[nodiscard]]`. Assigning over a live
request performs the same drop-policy teardown as destroying it. `wait()`
returns a `mp1::status`; `test()` and `get_status()` return
`std::optional<mp1::status>`. `kind()`, `teardown_policy()`, `state()`,
`cancelable()`, `is_broken()`, `drop_would_block()` and `native_handle()`
expose the request state machine without an MPI call.

`cancel()` is legal only when `cancelable()` is true, which means an active
point-to-point send or receive. Collective, file, RMA, buffer-flush and
`sendrecv` requests are wait-only and throw `std::logic_error` from `cancel()`;
inactive persistent requests and broken requests also reject it.

The destructor follows one teardown table (MPI 4.1 §3.7.3, §3.8.4, §3.9,
§6.12, §6.13, §12.3.5):

| Request kind/state | Destructor action | Payload |
|---|---|---|
| invalid/released/moved-from/broken | none | destroy |
| inactive persistent | `MPI_Request_free` | destroy |
| active send | `MPI_Request_free` | abandon (MPI may still read staging) |
| active receive | `MPI_Cancel` + `MPI_Wait` (+ `MPI_Request_free` if persistent) | complete unless cancelled |
| active collective | `MPI_Wait` (persistent: then `MPI_Request_free`) | complete on success |
| active file/RMA/flush | `MPI_Wait` | none today |

An active data-carrying request **delivers** its payload from the destructor: a
dropped capacity receive is resized and a staged variable collective is
unpacked. Dropping can block; `drop_would_block()` reports it and `release()`
is the only no-MPI-call escape. A send is freed, never waited, and its owned
staging is leaked.

The batch completion functions take `std::span<request>`, so a mutable
`std::vector<mp1::request>`, `std::array` or C array converts implicitly.
`wait_all`, `test_all`, `wait_any`, `test_any`, `wait_some`, `test_some`,
`start_all` and the non-destructive `get_status_all`/`get_status_any`/
`get_status_some` queries are free functions. `wait_any`/`test_any` commit only
the slot MPI reports and throw `mp1::mpi_error` with `slot()` set to the
completed index, leaving every other slot untouched. A function-level
`wait_all` error abandons only the active slots; a function-level `test_all`
error with `flag == false` leaves every slot untouched. Inactive persistent
slots stay valid and restartable in either case.

### Batch completion

`wait_all`/`test_all` return one status per slot. `MPI_ERR_IN_STATUS` is not
thrown: a per-slot error is returned in `status::error()` (MPI 4.1 §3.7.5). A
slot whose per-slot error is `MPI_ERR_PENDING` "has neither failed nor
completed" (§3.7.5). mp1 returns a PENDING status for it but does **not** reap
the slot:

- the request stays active with its live MPI handle and payload (staged data
  stays owned and writable by MPI; a persistent handle has not returned to
  inactive);
- the PENDING status carries only `error() == MPI_ERR_PENDING`: `source()` and
  `tag()` are `MPI_ANY_SOURCE`/`mp1::any_tag` (`tag()` returns `mp1::tag`),
  `get_count()`/`get_elements()`
  are 0 and `is_cancelled()` is false, all without an MPI call;
- the caller must call the same completion function again (or wait/test the
  slot) to finish it. A returned status array therefore does not mean every
  slot was reaped, and `start()` on a pending persistent slot throws
  `std::logic_error` because that slot is still active.

A committed persistent slot becomes inactive and remains valid, so it can be
`start()`ed for the next cycle. `test_all` returns `nullopt` when not every
operation has completed, and MPI then modifies no request; the PENDING rule is
defensive there because `flag == true` means every operation completed.

A `result_pending` `MPI_Comm_idup` slot is finished but owns a result, so it is
inactive for every bulk completion function: `wait_all`/`test_all` report the
empty status for it and it is omitted (or `MPI_UNDEFINED`) in the any/some
forms. Its duplicate stays in the request; call `take_communicator()` to get
it. A PENDING idup slot stays active exactly like any other request and cannot
be taken until a later completion produces the result.

`wait_some`/`test_some` return only completed slots, as indices into the
original span. A provider-reported PENDING slot is omitted from the returned
list and left active, so call again exactly as for `wait_all`. A function-level
error that is not `MPI_ERR_IN_STATUS` is separate: `wait_all` abandons only the
active slots, while `test_all` with `flag == false` leaves every slot untouched
(a failure after `flag == true` abandons the active slots). Inactive persistent
slots stay valid and restartable.

### Buffered sends

`bsend_buffer` owns one process-specific buffered-send allocation and detaches
it on destruction. It allocates `size + mp1::bsend_overhead` bytes, where
`mp1::bsend_overhead` mirrors `MPI_BSEND_OVERHEAD`. The lower-level API is
`buffer_attach(void*, count_t)` + `buffer_detach()` with the large-count MPI 4.1
entry points, plus `buffer_flush()`, `buffer_iflush()`,
`attach_buffer(comm, ...)`, `detach_buffer(comm)`, `flush_buffer(comm)` and
`iflush_buffer(comm)`. `mp1::buffer_automatic` selects MPI-managed buffering.
`buffer_iflush()`/`iflush_buffer()` may return an inactive request; the result
is wait-only and `cancel()` throws `std::logic_error`. Only one
process-specific and one communicator-specific buffer may be attached at a
time; attaching while one is attached, or detaching something that is not
attached, is erroneous per MPI 4.1 §3.6 and mp1 does not track it. Detach is a
blocking flush, so complete every outstanding flush request first. A
`bsend_buffer` guard detaches unconditionally in its destructor, so do not call
`buffer_detach()` while the guard is alive: the destructor would detach
whatever buffer is attached at that moment, not necessarily the guard's own
allocation. The guard is non-copyable and non-movable.

## Collectives

Every rank must call each collective. The communicator is the first argument;
reductions take the operator before the data: `reduce(c, op, root, ...)`,
`allreduce(c, op, ...)`, `scan`/`exscan(c, op, ...)`,
`reduce_scatter`/`reduce_scatter_block(c, op, ...)`. `reduce_local(op, send, recv)` has no
communicator.

```cpp
int local = world.rank();

mp1::barrier(world);

std::vector<int> all(world.size());
mp1::allgather(world, local, all);                     // one int per rank

int total = 0;
mp1::reduce(world, std::plus<int>{}, 0, local, total); // root receives total
mp1::allreduce(world, MPI_SUM, local, total);          // native predefined op
mp1::allreduce(world, std::plus<int>{}, local, total); // mapped to MPI_SUM
```

### Families and lifetimes

| Family | Blocking | Nonblocking | Persistent | In-place | Intercommunicator |
|---|---|---|---|---|---|
| `barrier` | ✓ | ✓ | ✓ | n/a | ✓ |
| `bcast` | ✓ | ✓ | ✓ | n/a (intrinsic) | ✓ |
| `gather` | ✓ | ✓ | ✓ | ✓ (root) | ✓ |
| `scatter` | ✓ | ✓ | ✓ | ✓ (root) | ✓ |
| `allgather` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `alltoall` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gatherv` | ✓ | ✓ | ✓ | ✓ (root) | ✓ |
| `scatterv` | ✓ | ✓ | ✓ | ✓ (root) | ✓ |
| `allgatherv` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `alltoallv` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reduce` | ✓ | ✓ | ✓ | ✓ (root) | ✓ |
| `allreduce` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `scan` | ✓ | ✓ | ✓ | ✓ | — |
| `exscan` | ✓ | ✓ | ✓ | ✓ | — |
| `reduce_scatter` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reduce_scatter_block` | ✓ | ✓ | ✓ | ✓ | ✓ |

Nonblocking names are `i<family>`; persistent names are `<family>_init`
(including `barrier_init`). Persistent calls take a trailing
`const info& = {}` and return an inactive `request`.

### Buffer forms

Fixed-size families and reductions accept the same position vocabulary as
point-to-point: a value, a contiguous range (`std::span`, `std::array`, C
array, `std::vector`, `std::string`), a non-contiguous sequence on a send side
(staged), or a `region<T>`. There are no iterator, pointer + count, or
per-form duplicates. An in-place position is the `mp1::in_place` sentinel.

`bcast`, `gather`, `scatter`, `gatherv`, `scatterv` and `reduce` take an `int
root`; on an intracommunicator it is a local rank. On an intercommunicator the
root's group passes `MPI_ROOT` at the root and `MPI_PROC_NULL` at the other
members, while the remote group passes the root's rank in that remote group
(MPI 4.1 §6.2.2–§6.2.3). `reduce_scatter`/`reduce_scatter_block` have no root
and no ignored position: every rank passes the real receive buffer and
identical counts.

In-place is accepted only in the position MPI defines and only on an
intracommunicator: the send slot for `gather`, `reduce`, `allgather`,
`alltoall`, `allreduce`, `scan`, `exscan` and
`reduce_scatter[_block]`; the receive slot for `scatter`. `gather`, `scatter`
and `reduce` accept it only at the root. A wrong role is `std::invalid_argument`
before the native call. `allgather`, `alltoall`, `allgatherv`, `alltoallv`,
`allreduce` and `reduce_scatter_block` require every rank to choose the same
in-place or out-of-place form (MPI 4.1 §6.7, §6.8, §6.9.6, §6.10.1);
`reduce_scatter` is the one family where MPI permits mixing. mp1 validates only
the local choice.

An empty range/fixed-receive or collective position is passed as a dummy
pointer, count 0 and the base datatype. A P2P `region<T>` with
`layout<T>::contiguous(0)` is instead passed as count 1 with the empty datatype and
the caller's pointer (possibly null), and an auto-size container receive passes
the container's `data()` with count 0; those forms also transfer zero bytes.
See [Empty messages](#empty-messages). A used, out-of-place receive position
with a supplied layout rejects a construction-proven overlapping type map
before MPI.

A `region<T>` is always one occurrence of the layout over the whole buffer, not
a per-rank repetition. For a collective whose MPI signature repeats a block per
rank (for example `allgather`/`alltoall`), build one layout covering the whole
receive buffer (`layout<T>::contiguous(n, block)`) or use a flat range with the
family's counts.

### Variable collectives

`gatherv`, `scatterv`, `allgatherv` and `alltoallv` take a per-peer descriptor
in one of two spellings:

| Descriptor | Spelling | Notes |
|---|---|---|
| dense | `mp1::blocks{.counts = ..., .displacements = ...}` | Element counts and element displacements. Both spans must have the peer count; the displacement span is never omitted or empty for a used dense descriptor. |
| typed | `std::span<const layout<T>>` + `std::span<const count_t>` | One committed layout per peer plus optional element displacements. An empty displacement span means all-zero, except on the `scatterv` and `alltoallv` send sides, where it is mandatory because it precedes the receive position. A nonblocking/persistent call duplicates the layouts it stages. |

```cpp
std::vector<mp1::count_t> counts(world.size(), 1);
std::vector<mp1::count_t> displacements(world.size());
for (int i = 0; i < world.size(); ++i)
  displacements[i] = i;
std::vector<int> gathered(world.size());
mp1::gatherv(world, 0, local, gathered,
             mp1::blocks{.counts = counts, .displacements = displacements});
```

On every side that MPI uses on the calling rank, the descriptor arrays are
validated before the collective's MPI data call: counts and displacements must
have the peer length, dense counts and displacements (and a
non-empty typed block's displacement) must be non-negative, and the dense
`displacement + count` range (or the typed true-extent bound) must fit the
aggregate buffer. A descriptor on an ignored side (for example the root-only
`gatherv` receive or `scatterv` send on a non-root) is not inspected, and
`mp1::blocks{}` is the zero-block descriptor only on such an ignored side. On a
used side with a peer count greater than zero, `blocks{}` is rejected: spell
the empty used side as peer-length zero counts and zero displacements, or as
peer-length empty layouts. This is a descriptor-shape requirement, not a ban on
empty variable collectives.

Aggregate and single-block positions read their span differently. An aggregate
receive span (`gatherv`/`allgatherv`/`alltoallv`) is a capacity: the per-peer
descriptor decides how much is written, and the span is checked by the
descriptor fit. A single-block position (`gatherv` send, `scatterv` receive,
`allgatherv` send) passes its span length (or `layout.size()`) as the MPI block
count; for `scatterv` it must equal this rank's `sendcounts[rank]` at the root,
and for `allgatherv` it must match this rank's entry in every peer's
`recvcounts`. An over-sized receive span is a signature mismatch rather than
spare capacity. A block that addresses no location is exempt from
the fit comparison because MPI ignores its placement: a dense block with
`count == 0` skips only the fit comparison (its count and displacement are still
checked non-negative and its byte offset must be representable), and a typed
block whose layout `byte_size() == 0` skips the fit comparison but its
displacement must still be non-negative and representable. A blocking call
borrows the descriptor spans for the call; nonblocking and persistent calls
copy the arrays (and duplicate staged layouts) at initiation, so the caller's
descriptor arrays may be destroyed immediately after the call. The data buffers and communicator must still stay alive until
completion.

### Reductions and operators

Every reduction accepts either a native `MPI_Op` or a functor. A functor whose
operation and C++ element datatype are a §6.9.2 predefined-operation pair is
mapped directly to the corresponding `MPI_Op`; a pair outside that table uses
the stateless callback cache. The mapping is MPI's table itself, not a wrapper
policy: `std::plus`/`std::multiplies` map for C integer, floating and
`std::complex`, `mp1::max`/`mp1::min` for C integer and floating, the
`std::bit_*` functors for C integer and `std::byte`, and
`std::logical_*`/`mp1::logical_xor` for C integer and `bool` (the C++ datatypes
`MPI_CXX_BOOL` and `MPI_CXX_*_COMPLEX`; Fortran datatypes are out of scope).
The predefined operation's result semantics are MPI's and are passed through
unchanged: floating-point `max`/`min` NaN behavior is whatever `MPI_MAX`/
`MPI_MIN` does, and the integer logical result value is whatever `MPI_LAND`/
`MPI_LOR`/`MPI_LXOR` returns. A functor for a category outside the row
(`std::plus<MyType>`, `std::plus<std::string>`, character or `Logical`/`Byte`
max/min, floating-point logical functors, enums and user types) uses the
callback.

The standard-library functors are the general typed reduction surface:
`std::plus`, `std::multiplies`, `std::bit_and`, `std::bit_or`, `std::bit_xor`,
`std::logical_and` and `std::logical_or` (including the transparent `void`
specializations) are accepted and mapped to the same predefined `MPI_Op` where
the operation/type pair is mappable, and use the callback otherwise.
`mp1::max`, `mp1::min` and `mp1::logical_xor` are the only mp1 functors: they
have no standard-library object equivalent. Each takes its element type as a
defaulted parameter whose default is `void`, like the standard transparent
specializations, so `mp1::max{}` (equivalently `mp1::max<void>{}`) deduces its
operands at the call site and `mp1::max<int>{}` fixes them. The transparent
object is not tied to one element type: the reduction classifies it from the
element type, so `mp1::max{}` maps to `MPI_MAX` for a C integer or
floating-point element and keeps the callback for a character, `Logical` or
`Byte` element, an enum or a user-defined type. Its deduced `operator()`
returns a value of the common operand type, so a reduction (which always passes
two operands of the same element type) gets that element type back; on the
callback path the 0/1 result of `mp1::logical_xor` is normalized to that type
(for example the enum or floating-point element). The built-in functor surface
is those three mp1 names plus the seven standard-library functors; any other
stateless, nothrow-invocable functor uses the managed callback path.

`mp1::op_traits<F>::is_commutative` is true only where the C++ operation is
actually commutative: `mp1::max`/`mp1::min` for integral `T`,
`mp1::logical_xor` and `std::plus`/`std::multiplies`/`std::logical_and`/
`std::logical_or` for arithmetic `T`, `std::plus<std::complex<T>>`/
`std::multiplies<std::complex<T>>` for floating `T`, and `std::bit_and`/
`std::bit_or`/`std::bit_xor` for integral or `std::byte` `T`. Floating-point
`max`/`min` are **not** commutative (NaN defeats `(x < y) ? y : x`), and a
user-defined `operator+`/`operator<` is never assumed commutative. Users can
inherit `mp1::commutative_op` or specialize `mp1::op_traits` to opt in. The
trait is only the `commute` argument of `MPI_Op_create`, which is used only on
the callback path; a mapped §6.9.2 pair never creates an `MPI_Op` and never
consults the trait. For the callback-only pairs, the transparent spellings the library
classifies (the `std::plus`/`multiplies`/`equal_to`/`not_equal_to`/`logical_and`/
`logical_or`/`bit_*` `void` specializations and
`mp1::max<>`/`mp1::min<>`/`mp1::logical_xor<>`) decide the `commute` argument
by substituting the reduction element type into the same `op_traits` row as the
typed spelling, so the two spellings cannot disagree: `mp1::max{}` on `char`
(character is not a MAX category) creates its `MPI_Op` with `commute = 1`
exactly like `mp1::max<char>{}`, and `mp1::max{}` on `bool` (Logical is not a
MAX category) does the same. The transparent type itself cannot answer without
an operand: `op_traits<std::plus<void>>` and `op_traits<mp1::max<>>` are
`false` unless the user specializes them.

For an mp1-managed callback, one reduction element is a complete `T` object
described by a committed, process-lifetime **typed** image. The image is the
native map shifted to true lower bound 0 (a one-block struct at displacement
`-true_lb` when the native true lower bound is non-zero), resized to
`(lb 0, extent sizeof(T))`. Its type signature is exactly `T`'s native typed
components, so MPI 4.1 §3.3.2 representation conversion applies to the data
bytes on a heterogeneous run. mp1 shifts every callback-image buffer pointer
up by the native true lower bound, and the callback subtracts the same offset
before indexing `T[len]`.
There is no `MPI_BYTE` full-object image and no byte padding block; a T whose
native datatype is `MPI_BYTE` (`std::byte` or a byte trait) keeps that byte
leaf because it is the element's native type. Padding bytes and unlisted
members are not in the type map, so they are never transmitted, and a direct
receive writes only mapped fields (the caller's padding and unlisted members
are preserved even for an in-place reduction). A staged layout packs and
unpacks only the mapped positions.

The callback receives T arrays at `sizeof(T)` stride; `T[len]` indexing
addresses the listed fields at their C++ offsets for any native true lower
bound. A functor used on this path must be stateless (empty),
nothrow-invocable for `T`, and must read/write only listed members -- it must
not inspect padding or unlisted members. `T` must be trivially copyable. The
image is committed once per `T` and retained for the process lifetime. The
callback `MPI_Op` handles and the image are process-lifetime caches:
`MPI_Op_free`/`MPI_Type_free` are not called, because freeing after
`MPI_Finalize` is not allowed, so the caches are deliberately not destroyed. An over-aligned `T` (`alignof(T) >
alignof(std::max_align_t)`) is unsupported on this path because MPI may place
callback temporaries at only fundamental alignment. A custom derived `T`
whose native datatype cannot be decoded into the transmitted byte map
(subarray/darray/F90/`PACKED`/`VALUE_INDEX` constructors, or an overlapping
map) is rejected with `std::invalid_argument` before the reduction.
`reduce_scatter` and `reduce_scatter_block` additionally require the image's
true extent to equal `sizeof(T)` (no leading or trailing unmapped bytes):
MPICH 4.3.2 (the CI MPICH pin) lays those callbacks out at the true extent, so
mp1 rejects a gapped image for those two families before MPI. All other reduction
families accept the same gapped image.

`reduce_scatter` takes `std::span<const count_t>` receive counts and
`reduce_scatter_block` takes a scalar `count_t`. Both require the receive
buffer to have exactly the local result size (out-of-place) or the full
reduced size (in-place). `reduce_local(op, send, recv)` applies the operator
locally with no communicator and no in-place form.

### Persistent collectives

Persistent collectives are available for all sixteen families, including the
variable and reduction families. `*_init` is collective and follows the
collective ordering rules: every member must call the same initializations in
the same order. The returned request is inactive; each `start()` runs one
cycle and `wait()`/`test()` completes it. The caller's data buffers and
communicator must stay alive and unmodified until the request is destroyed.
Descriptor arrays and count values are copied into the plan at `*_init`, so
the caller's descriptor arrays may be destroyed afterwards. Each `start()`
re-packs a staged send and re-primes in-place staging from the caller's buffer;
each completion unpacks once.

Persistent collectives take fixed buffers only: a value, `std::span`,
`std::array`, C array, `region`, or a typed descriptor array. A resizable
container is not a persistent data-buffer form. Freeing an active persistent
collective request is erroneous in MPI; mp1 therefore waits for an active
request and then frees it (the destructor may block), while an inactive
request is freed with `MPI_Request_free`. `cancel()` throws `std::logic_error`
for a persistent collective request; `release()` is the no-MPI escape.

## Layouts and datatypes

`mp1::layout<T>` owns one committed `MPI_Datatype` and the facts proven when
its factory constructed it. A default-constructed layout is invalid;
`contiguous(0)` is the valid zero-size layout.  The former
`layout<T>::empty()` factory is **removed** (source-breaking): use
`contiguous(0)` wherever a zero-size datatype is needed and `mp1::empty_data`
for an explicit zero-count message.  A copy duplicates the committed datatype;
a move transfers it.

Factories:

| Factory | Forms |
|---|---|
| `contiguous(count)` | `count` copies of `native_datatype<T>()`. |
| `contiguous(count, block)` | `count` copies of an existing layout. |
| `vector(count, blocklength, stride)` | Adjacent blocks with an extent-unit stride. |
| `strided_vector(count, blocklength, byte_stride)` | Adjacent blocks with a byte stride. |
| `vector(vector_options<T>{.count=..., .blocklength=..., .stride=..., .block=...})` | Same, with an optional block layout. |
| `strided_vector(strided_vector_options<T>{...})` | Same, with an optional block layout. |
| `indexed(const indexed_parameter&)` / `indexed(parameter, block)` | Per-block lengths and element displacements. |
| `hindexed(const hindexed_parameter&)` / `hindexed(parameter, block)` | Per-block lengths and byte displacements. |
| `indexed_block(blocklength, parameter)` / `(..., block)` | Constant block length, per-block element displacements. |
| `hindexed_block(blocklength, parameter)` / `(..., block)` | Constant block length, per-block byte displacements. |
| `subarray(const subarray_parameter&)` / `subarray(parameter, block)` | Subarray tile; the parameter carries an `array_order` (row-major by default). |
| `from_raw(MPI_Datatype, elements, facts = {})` | Adopt one raw derived datatype (ownership transfers) with optional structural facts. |
| `contiguous(0)` | The valid zero-size layout (there is no separate `empty()` factory). |

The parameter builder names are `mp1::indexed_parameter`,
`hindexed_parameter`, `indexed_block_parameter`, `hindexed_block_parameter`
and `subarray_parameter`. Each owns its entries, validates them in `add()`,
has a nested `entry_type` alias, a default constructor, an explicit
`std::span<const entry_type>` constructor and the relevant `add()` overloads.
`subarray_parameter` additionally has `order(array_order)` and `order()`.
Builders are not initializer-list constructors: a braced listing is rejected;
pass a named contiguous range instead. All factories and parameter `add()`
calls reject a negative count, blocklength, stride or displacement with
`std::invalid_argument`, even where a particular MPI constructor would accept
a negative displacement; `byte_resize` permits a negative lower bound but
rejects a negative extent.

```cpp
auto contiguous = mp1::layout<int>::contiguous(16);
auto strided = mp1::layout<int>::vector(
    mp1::vector_options<int>{.count = 4, .blocklength = 2, .stride = 8});
auto holey = mp1::layout<int>::indexed(
    mp1::indexed_parameter{std::array{std::pair<mp1::count_t, mp1::count_t>{2, 0},
                                     std::pair<mp1::count_t, mp1::count_t>{2, 8}}});
```

The accessors are `size()`, `byte_size()`, `byte_extent()`,
`byte_lower_bound()`, `byte_upper_bound()`, `true_byte_lower_bound()`,
`true_byte_extent()`, `true_byte_upper_bound()`, `dense()`,
`is_base_repetition()`, `has_overlapping_entries()` and `is_valid()`.
MPI 4.1 §5.1.8 leaves the true bounds undefined for an empty typemap; a
zero-size layout reports the canonical empty geometry
`true_byte_lower_bound() == true_byte_extent() == true_byte_upper_bound() == 0`
instead of exposing a provider sentinel, so `true_byte_upper_bound()` is total
over every valid layout. `byte_lower_bound()`/`byte_extent()` (the extent
markers) are reported unchanged.
`byte_resize(lb, extent)` changes only the MPI extent markers and preserves
the map facts; `resized(lb, extent)` is the non-mutating form.
`native_handle()` borrows the committed datatype.

`layout<T>::from_raw(type, elements, facts = {})` transfers ownership of one
derived datatype into the layout. It is the only raw-datatype adoption escape;
there is no general raw-datatype bridge. The handle must be a derived datatype
(a predefined named, F90 or value-index handle throws `std::invalid_argument`
before any commit or free), `elements` is the application `size()` and never
an MPI count, and the handle's type map must be an occurrence of `T` (a caller
precondition; 1.0 does not validate the leaf basic type). `facts` may only
assert the three structural predicates MPI cannot query. Ownership transfers
on entry: on success the layout frees the handle exactly once and the caller
must not free or reuse it; if the call rejects the handle before adoption the
caller still owns it; if adoption fails after the commit/free stage the handle
has been freed and the caller must not free it again. `from_raw(t, n)`,
`from_raw(t, n, {})` and a designated facts aggregate are all unambiguous.

`dense()` is the construction-proven packed predicate, not
`byte_size() == true_byte_extent()`. `is_base_repetition()` is the predicate
that permits substituting a repeated `native_datatype<T>()` in a flat
reduction; its pointer adjustment is
`p + (true_lb(layout) - true_lb(native_datatype<T>()))`.

### Datatype mappings

`native_datatype<T>()` is the single mapping. It is predefined for the standard
scalar types, `std::byte` (`MPI_BYTE`), `std::complex`, and the reflected
aggregates. Customize with one of the global-scope macros:

- `MP1_DATATYPE_TRAITS(type, mpi_type)` — map a type to exactly one predefined
  datatype and mark `is_predefined_datatype_v<type>`. The handle is the data
  wire for send/receive, so it may name a datatype outside the C++ categories
  for a cross-language peer (for example `MPI_INTEGER`); the typed
  reduction/RMA operator mapping still covers the C++ datatype categories only.
- `MP1_DATATYPE_TRAITS_PACKED(type, mpi_type)` — map a type to a derived but
  byte-contiguous datatype; it does not mark it predefined.
- `MP1_DATATYPE_BASIC(T, U)` — assert that a custom non-predefined derived
  datatype has the single predefined basic component `U`. This is an RMA
  accumulating-family annotation only: it does not map a datatype, change the
  wire image, or replace the predecessor's byte shortcut.
- `MP1_REFLECTION(Struct, members...)` — synthesize a `STRUCT` datatype from
  the named members.
- `MP1_REFLECTION_EMPTY(Struct)` — a zero-member reflected aggregate.
- `MP1_REFLECTION_TEMPLATE(params, Struct, members...)` — the templated
  reflection form.

The reflection member list is limited to 119 members; a longer list fails a
compile-time assertion that names the limit. A member may be a scalar, a C
array, a `std::array`, or another reflected aggregate. A C array is supported
as a member only; a top-level C-array accumulating element type is rejected.
`MP1_REFLECTION` requires a default-constructible aggregate whose named
non-static data members are accessible at the macro's scope; use `_EMPTY` for a
zero-member type and `_TEMPLATE` for a generic template. A pointer member has
no datatype mapping; a `const` member is not guarded but is a poor receive
target.

`native_datatype<T>()` returns the trait handle directly; it is not the
validation point. The extent check (MPI extent must equal `sizeof(T)`) and the
true-bound check (the true extent must lie inside one `T` object) run when a
`layout<T>` or buffer consumes `datatype_info_of<T>()`. A padded reflected
struct or a non-zero true lower bound is legal. A custom trait returning a
derived datatype records `packed == false` (unknown map order and overlap)
unless it declares `byte_contiguous = true`; a `MP1_DATATYPE_TRAITS_PACKED`
specialization is the convenience macro.

The public traits are `has_datatype_v<T>`, `is_predefined_datatype_v<T>`,
`basic_component_t<T>`, `is_homogeneous_datatype_v<T>` and
`basic_component_traits<T>`. `basic_component_t<T>` is `T` for a predefined
`T`, the common predefined operand for a homogeneous derived `T` (`int` for
`std::pair<int,int>`), and `void` for a mixed, empty or opaque type.

## One-sided communication (RMA)

This surface is new in mp1. mpl 0.5.0 listed one-sided communication as not yet
supported and has no window, put/get/accumulate or epoch API; there is no
0.5.0 spelling to migrate. `mp1::mpi_window` is a copyable, non-owning view of
an `MPI_Win`; `mp1::window` is the move-only owner. Window factories take
option bags:

```cpp
auto win = mp1::window::allocate(
    world, {.size = 8 * static_cast<MPI_Aint>(sizeof(int)),
            .disp_unit = sizeof(int)});

auto exposed = mp1::window::create(
    world, {.base = memory, .size = bytes, .disp_unit = 1, .info = hints});
```

`window_create_options` is `{void* base; MPI_Aint size; int disp_unit;
mp1::info info;}`; `window_allocate_options` is `{MPI_Aint size; int
disp_unit; mp1::info info;}`. The other factories are
`allocate_shared(comm, window_allocate_options)`,
`create_dynamic(comm, info)` and `adopt(MPI_Win)`. The window accessors are
`base()`, `byte_size()`, `disp_unit()`, `create_flavor()`, `memory_model()`,
`rank()`, `group_size()`, `get_group()`, `get_info()`, `set_info()`,
`shared_query(rank)`, `attach(base, size)`, `detach(base)`, `address_of(obj)`,
`free()` and `release()`. The seven standard attributes (`base()`, `byte_size()`,
`disp_unit()`, `create_flavor()`, `memory_model()`, `rank()`, `group_size()`)
are cached at view construction and are MPI-free. `get_group()`, `get_info()`,
`shared_query()` and `set_info()` call MPI and require active MPI;
`attach()`/`detach()` are non-`const` because they mutate the dynamic-window
attachment set, while `set_info()` is `const` because constness protects the
view's cached state, not the shared MPI object. `adopt(MPI_Win)` is a local
handle adoption, not a collective factory; `create`/`allocate`/
`allocate_shared`/`create_dynamic` are the collective factories.

Fifteen free synchronization functions take `const mpi_window&`:

| Group | Operations |
|---|---|
| Active target | `fence`, `post`, `start`, `complete`, `wait`, `test` |
| Passive target | `lock`, `lock_all`, `unlock`, `unlock_all` |
| Flush / sync | `flush`, `flush_all`, `flush_local`, `flush_local_all`, `sync` |

`fence` is collective. `post` is local and does not wait for the matching
`start`. `lock`/`start` may block. `lock`, `unlock`, `flush` and `flush_local`
require a real rank; the ten data-movement calls accept `MPI_PROC_NULL` as a
normalized no-op. There are no RAII guards: the completion point is part of
the MPI contract and stays visible.

Data movement uses the same position vocabulary as point-to-point. The origin
or result is a `std::span<T>` (flat native elements) or a `region<T>` (one
layout occurrence); the target is either a `count_t` element count or a
`layout<T>`:

```cpp
const int send[2]{1000, 1001};
mp1::put(win, std::span<const int>{send, 2}, 1, 3, 2);  // target count 2

mp1::layout<int> target = mp1::layout<int>::vector(
    mp1::vector_options<int>{.count = 2, .blocklength = 1, .stride = 4});
mp1::put(win, std::span<const int>{send, 2}, 1, 0, target);  // target layout
```

There are eighteen data-movement declarations: `put`, `get`, `accumulate`,
`rput`, `rget`, `raccumulate` (two each: count target or layout target);
`get_accumulate`, `rget_accumulate` (two each: count or layout target, with
independent origin/result shapes); `compare_and_swap` and `fetch_and_op` (one
each, single element). That preserves all 42 origin/result/target shapes
(6 × 4 + 2 × 8 + 2).

The wrapper checks the origin/result size against the target count or
`layout.size()` before the MPI call. A zero target count is legal; a null
origin with a positive origin size (outside `MPI_NO_OP`), or a null result with
a positive target count, is rejected. Under `MPI_NO_OP` every origin argument
is ignored (MPI 4.1 §12.3.4): a null or non-empty origin is accepted, its size
is not checked, and it may alias the result (the atomic-read idiom); the
wrapper passes the RMA dummy origin with count 0. A zero-byte null origin or
result is otherwise accepted and normalized to the RMA dummy address (see
[Null data positions](#null-data-positions)). The both-flat origin/result pair
of `get_accumulate`/`rget_accumulate`, `fetch_and_op` and `compare_and_swap`
must be disjoint; the exact byte range is checked before MPI. A mixed
flat/layout or layout/layout pair leaves cross-object overlap to the caller.

A target layout must contain only relative displacements, so the only target
form is `layout<T>`. A target layout with a construction-proven overlapping
type map is rejected for `put`/`accumulate`/`get_accumulate` (target updated)
and for the result/origin position of `get`/`get_accumulate` (result updated);
unknown overlap provenance is accepted.

Blocking `put`/`get`/`accumulate`/`get_accumulate` return when the call has
been made, not when the data is complete. Origin buffers must stay alive, and
get/result buffers must not be read, until the next synchronization call:
`fence`, `complete`, `flush`, `flush_all`, `flush_local`, `flush_local_all`,
`unlock` or `unlock_all`. `get` returns `void`.

Request-based `rput`, `rget`, `raccumulate` and `rget_accumulate` are legal
only inside `lock`/`lock_all` epochs. They return `request` with kind `rma`;
`wait()`/`test()` reaps them and `cancel()` throws `std::logic_error`. The
origin/result and target layouts are duplicated into the request payload at
initiation, so the caller's `region`/layout objects may die when the call
returns, while the data buffers may not. Completion is origin-side; target
visibility still requires flush/unlock.

The accumulating families take `predefined_op<U>`, where `U` is
`basic_component_t<T>`: `T` for a predefined `T`, `int` for
`std::pair<int,int>`. `predefined_op<T>` requires a predefined `T`; the
aggregate spelling is a compile-time error. The typed functor mapping is the
same MPI 4.1 §6.9.2 C++ row as the collective mapping: `std::plus`/
`std::multiplies` map for C integer, floating and complex types; `mp1::max`/
`mp1::min` map for C integer and floating; `std::bit_*` map for C integer and
`std::byte`; `std::logical_and`/`std::logical_or` and `mp1::logical_xor` map
for C integer and `bool`. The transparent
`mp1::max<>`/`mp1::min<>`/`mp1::logical_xor<>` spellings are classified from
`U` with the same rule. A raw `MPI_Op` must be a standard predefined handle
valid for `U`'s datatype category; user-created handles, `MPI_OP_NULL`,
`MPI_MAXLOC`/`MPI_MINLOC` and category mismatches throw `std::invalid_argument`
before the MPI call. The raw handle path is validated against the same C++
category row; the C and Fortran datatype variants and the multi-language
categories are outside the C++ typed surface and are rejected before the MPI
call. Mixed, empty and opaque
types are compile-time errors. `MPI_REPLACE` is legal in every accumulating
family for a predefined or homogeneous-derived `T`; `MPI_NO_OP` is legal only
in `get_accumulate`/`rget_accumulate`/`fetch_and_op`.

`compare_and_swap` requires a predefined C++ type in the C integer, Logical or
Byte category. Atomicity is per basic element, not per C++ `T` object. The
`accumulate_ops` window info key defaults
to `same_op_no_op`.

Shared memory: use `window::allocate_shared` on a shared-memory communicator
(`communicator::split_type(world, mp1::comm_type_shared)`), then
`shared_query(rank)`. The returned `disp_unit` is the queried rank's. Direct
load/store requires a passive epoch plus `sync(w)` on the writer before and the
reader after process synchronization. `create_dynamic` windows use
`attach`/`detach` and `address_of`; a zero-size or null-base attach is rejected
because it is not portably detachable.

## Files

`mp1::file` is the only public file type: one owning, movable handle with
member data access. There is no view type and there are no free data-access
wrappers. Read data access is `const` and write data access is non-`const`;
split begin/end are non-`const` because they mutate the per-file split epoch.
Data access advances a file pointer in the pointer-advancing families:
`read`/`write` and `read_all`/`write_all` use the individual file pointer,
while `read_shared`/`write_shared` and `read_ordered`/`write_ordered` use the
shared file pointer. The `const` on a read protects the handle's cached state,
not the file position.

```cpp
mp1::file output(world, "mp1-demo.bin",
                 mp1::access_mode::create | mp1::access_mode::read_write);

auto block = mp1::layout<int>::contiguous(4);
output.set_view(0, block);              // collective; etype is int
std::vector<int> values(4, world.rank());
const mp1::offset_t offset =
    static_cast<mp1::offset_t>(world.rank()) * 4;  // etype units
output.write_at(offset, values);
std::vector<int> readback(4, 0);
output.read_at(offset, readback);
output.close();                          // collective; all ranks call it
```

The public members are:

- Lifecycle (5): default constructor, `file(comm, path, access_mode, info)`,
  move construction/assignment, destructor.
- Control (7): `open`, `adopt`, `close`, `release`, `is_open`,
  `native_handle`, `remove`.
- Metadata and positioning (15): `resize`, `preallocate`, `size`, `mode`,
  `sync`, `set_view` (byte and typed), `seek`, `seek_shared`, `position`,
  `get_position_shared`, `byte_offset`, `set_info`, `get_info`, `get_group`.
- Atomicity, extent and error handlers (6): `set_atomicity`,
  `get_atomicity`, `get_type_extent` (type and layout), `set_error_handler`,
  `get_error_handler`.
- Data access (96): the blocking read/write families, the nonblocking
  `i*` families, the split-begin families and the matching split ends.

Data access has three forms: a single value (`T&`/`const T&`), a contiguous
range of `T` (a borrowed `std::span`, `std::array`, `std::vector`,
`std::basic_string`, or `std::string_view` for writes), and a `region<T>` (one
application of a per-call layout). A nonblocking or split-begin write requires
a borrowable buffer, so an rvalue container or scalar is rejected. A per-call
region in a nonblocking/split request is duplicated at initiation, so the
caller's layout may die after the call while the data buffer may not.

Units follow MPI's mixed-unit model: `set_view` displacement is bytes from the
file start; `size`/`resize`/`preallocate` are bytes; `byte_offset` takes etype
units and returns absolute bytes; `seek`/`seek_shared`/`position`/
`get_position_shared` are etype units; data access offsets are etype units.
Before the first `set_view` the view is the initial byte view (displacement 0,
`MPI_BYTE` etype/filetype). The typed `set_view` makes `native_datatype<T>()`
the etype. A typed access is rejected unless the recorded view is unknown, the
view etype is `MPI_BYTE`, or the buffer element type has the same MPI datatype
identity as the view etype.

A typed `set_view` validates the filetype's decoded exact type map: a negative
or descending typemap displacement is rejected locally for every access mode,
and an overlapping filetype is rejected for a writable file (MPI 4.1 §14.3).
The rule that a hole's extent must be a multiple of the etype extent is left
to the caller for an opaque or `layout<T>::from_raw` filetype; the public
layout factories build extent-aligned maps.

Split collectives are a strict state machine (MPI 4.1 §14.4.5): at most one
active begin, same thread for begin and end, and an end that matches the
begun operation and buffer. A wrong, double or missing end, a mismatched
buffer, a collective I/O call during a split, and a close during a split throw
`std::runtime_error` locally. `sync`, `set_view`, `resize` and `preallocate`
also refuse during a split because their own clauses require split operations
to be complete. `set_info`, `set_atomicity` and `seek_shared` have no such
requirement and remain legal between begin and end. `set_info` rejects a null
`MPI_INFO_NULL` locally. A nonblocking or split request that outlives the
file keeps a lease on the shared file state; `close()` refuses while a
wrapper-owned request is outstanding. `~file()` is `noexcept`, finishes an
active split with the recorded operation and buffer, then closes collectively
when no request is outstanding; it abandons and counts the handle only when
the split state is unknown, a request is outstanding, or MPI is gone.

`open`, `close`, `resize`, `preallocate`, `sync`, `set_view`, `set_info`,
`set_atomicity`, `seek_shared` and every `*_at_all`/`*_all`/`*_ordered` family
are collective. `remove` is not collective.

Not implemented: `get_view`, `register_datarep`/user data representations, owning
or user file error-handler objects (the predefined handler set/get is
available), and a raw heterogeneous `set_view`. Use `native_handle()` for those
raw MPI operations.

## Spawning processes

`mp1::spawn`, `mp1::spawn_with_errcodes`, `mp1::spawn_multiple` and
`mp1::spawn_multiple_with_errcodes` wrap `MPI_Comm_spawn` and
`MPI_Comm_spawn_multiple`; `mp1::parent()` wraps `MPI_Comm_get_parent`. The
returned communicator is an intercommunicator with the `disconnect` policy:
both ends must disconnect, `free()` throws `std::logic_error`, the destructor
calls `MPI_Comm_disconnect` (collective, may block), and `release()` is the
no-MPI-call escape for a non-cooperating peer.

```cpp
// parent
auto children = mp1::spawn(world, 0,
                           mp1::spawn_options{.maxprocs = 2,
                                              .command = {"./child"},
                                              .info = {}});
children.disconnect();                 // collective; may block

// child
auto parent = mp1::parent();
parent.disconnect();
```

`spawn_options` is `{int maxprocs = 1; command_line command; mp1::info info;}`
where `command_line` is `std::vector<std::string>`; the first element is the
program and the rest are arguments. `spawn_multiple_options` is a builder:
`add(command, maxprocs, info = {})` appends one triple and keeps commands,
maxprocs and infos aligned; `size()`, `commands()`, `maxprocs()` and `infos()`
are the read surface. `spawn_result` is `{communicator intercomm;
std::vector<int> errcodes;}`.

Only the root's payload is significant; every rank must pass the same root and
the same communicator. mp1 validates the all-rank properties (valid
intracommunicator, in-range root) and adds no preflight collective. At the
root, command/argv/maxprocs/info content is passed through and is MPI's to
diagnose; non-root ranks ignore their copy and pass the portable empty form.
The default forms pass `MPI_ERRCODES_IGNORE`. The `*_with_errcodes` forms
fill `spawn_result::errcodes` with one entry per spawned process (`spawn`) or
`sum(maxprocs)` entries (`spawn_multiple`). The OUT array length is the root's
`maxprocs`, which a non-root rank cannot derive from its ignored payload, so
both calls take an optional `errcodes_length`: pass the root's length on every
rank when a non-root rank keeps the default `maxprocs = 1` (the safe
contract). Without it each rank sizes the array from its own `maxprocs`, so all
ranks must pass the same maxprocs lengths — a caller obligation the wrapper
cannot verify without a collective. A soft spawn is a success with a possibly
smaller `remote_size()` and a mix of error codes.

`mp1::parent()` is one-shot: a second `parent()` while the first owner is live
throws `std::logic_error`. `communicator::adopt_connected(MPI_Comm)` wraps a
raw handle from connect/accept/join with the disconnect policy; the rest of
DPM is not wrapped.

## Errors and lifetime

Diagnostics use `std::source_location` and `std::format`. The prefix is
`mp1: <operation> (<file>:<line>): <message>`, where `<file>:<line>` is the
library check site, captured with `std::source_location::current()` at the
check expression. No public entry takes a `source_location` or a `what`
parameter.

```cpp
struct context {
  std::string_view operation;                 // e.g. "mp1::gather"
  constexpr context(std::string_view) noexcept;
  constexpr context(const char*) noexcept;
};
```

`mp1::fail(ctx, where, format, args...)` and
`mp1::require(condition, ctx, where, format, args...)` throw
`std::invalid_argument`; `mp1::throw_invalid(operation, message)` is the
preformatted form. They are public helpers for code built on the same error
layer; ordinary users do not need them.

The exception taxonomy is closed:

| Situation | Type |
|---|---|
| locally checkable argument misuse | `std::invalid_argument`, before any MPI data call; a first-use reflected-datatype commit may precede the rejection (see [Null data positions](#null-data-positions)) |
| an MPI count/value cannot be represented internally | `std::overflow_error` |
| non-file state or policy misuse of a valid object | `std::logic_error`, before any MPI call |
| file state/epoch misuse (split begin/end, close, view/access mismatch, sequential mode) | `std::runtime_error`, before any MPI call |
| a `bsend_buffer` attachment size overflows `int` | `std::bad_array_new_length` |
| an auto-size receive allocation fails | `std::bad_alloc`, after the matched message is consumed |
| environment/lifecycle misuse | `std::runtime_error`, before any MPI call |
| an MPI call returned a non-success code, with a returning handler installed | `mp1::mpi_error` |

`mp1::mpi_error` derives from `std::runtime_error` and carries `code()`,
`error_class()`, `class_name()`, `operation()`, `slot()` (`-1` except for
multi-completion calls), `where()`, `is_class(int)` and `what()`. The
code/class/operation/slot/where accessors are MPI-free; `class_name()` and
`what()` may call `MPI_Error_string`, which is always available (MPI 4.1
§11.4.1, Table 11.1) and safe before init and after finalize.

Fatal-handler policy: by default an MPI error detected during execution aborts
the parallel computation inside MPI, except for file operations (MPI 4.1
§2.8), and no exception is thrown. `mp1::mpi_error` is reachable only when the
application selected a handler that returns. mp1 never installs a handler
implicitly and does not add user error-handler callbacks.

### Error handlers

Only the predefined handlers are exposed:

```cpp
mp1::set_error_handler(comm, mp1::error_handler::returns);
auto previous = mp1::get_error_handler(comm);   // optional<error_handler>

mp1::set_error_handler(window, mp1::error_handler::returns);
file.set_error_handler(mp1::error_handler::returns);  // file is a member-only surface

mp1::install_recoverable_errors();  // WORLD and SELF -> MPI_ERRORS_RETURN
```

`get_error_handler` classifies the returned handle against the three
predefined handlers and returns `std::nullopt` for a user-created handler
installed through raw MPI. The returned reference is freed with
`MPI_Errhandler_free` before the call returns. Windows do not inherit the
initial handler and start with `MPI_ERRORS_ARE_FATAL`; files start with
`MPI_ERRORS_RETURN`. `install_recoverable_errors()` is non-collective,
process-local and idempotent, and overwrites any handler the application
installed on `MPI_COMM_WORLD`/`MPI_COMM_SELF`.

There is no owning error-handler bridge and no callback registration API: the
three predefined values are the only `mp1::error_handler` values. A user
handler created with `MPI_Comm_create_errhandler` is used through raw MPI, or
attached to a communicator after creation with
`MPI_Comm_set_errhandler(c.native_handle(), handler)` /
`mp1::set_error_handler(c, predefined)`. The from-group factories default to
`MPI_ERRORS_ARE_FATAL` because a group has no parent communicator to inherit a
handler from; pass `mp1::error_handler::returns` for the recoverable path.
`MPI_ERRHANDLER_NULL` is the null handle, not a valid handler (MPI 4.1
§9.3.1).

MPI-backed objects may only be used while MPI is active. Operations that need
MPI throw `std::runtime_error` before `mp1::init` or after `mp1::finalize`;
destructors are `noexcept`, guard the same condition and abandon handles
instead of calling MPI. The one exception is `mp1::info`: every `MPI_Info_*`
routine is always available (MPI 4.1 §11.4.1, Table 11.1), so info works
before init and after finalize and its destructor frees the handle.

### Info, groups and status

`mp1::info` is one owning type. A default-constructed info is null;
`info::create()`/`info::create_env()` return a valid table that may be empty.
Key and value lengths are validated conservatively before any MPI call; a key
must be non-empty and shorter than `MPI_MAX_INFO_KEY`, a value shorter than
`MPI_MAX_INFO_VAL`, and neither may contain an embedded NUL. `info::remove`
returns `bool` and probes the key first.

`communicator::set_info` rejects a null info locally, but `dup_with_info`
passes the handle through; pass a valid `mp1::info` (or use `dup`) rather than
a default-constructed `info{}`.

`mp1::group` follows the other owning handles: the default state is invalid
(`MPI_GROUP_NULL`) and `group::empty()` is the valid predefined empty group.
Equality is `MPI_IDENT` and is MPI-free when either operand is invalid or both
native handles are equal. `compare()` returns `identical`/`similar`/`unequal`.
`translate` accepts `MPI_PROC_NULL`. `group_incl`/`group_excl` take
`std::span<const int>` and reject duplicate or invalid ranks locally; a braced
initializer list does not convert to a span, so spell the ranks as
`std::array{...}` or `std::vector{...}` (this is a deliberate source break).
`mp1::rank_range` is `{first, last, stride}` with MPI 4.1 §7.3.2 semantics:
`group_range_incl`/`group_range_excl` take `std::span<const rank_range>`,
validate stride/direction/bounds locally without enumerating the progression,
and leave duplicate/overlapping ranges as an MPI-erroneous program rather than
expanding them. Only the emitted ranks are required to be valid; a portable
caller keeps `last` within the group even when it is not emitted. A
communicator's group is obtained with the free
`mp1::get_group(c)`; `file` and `window` also expose a member `get_group()`,
and `mp1::get_group(w)` is the free window forwarder.

`mp1::status` records the provenance of the completion that produced it:
`kind()`, `source()` (`int`), `tag()` (`mp1::tag`), `error()`, `is_cancelled()`,
`get_count<T>()`/`get_count<T>(layout<T>)`,
`get_elements<T>()`/`get_elements<T>(layout<T>)` and `native_handle()`. The
accessors decide from provenance, never from the `MPI_SOURCE`/`MPI_TAG`/
`MPI_ERROR` field values: a conforming provider may leave an undefined field at
any value, including the values that look like an MPI empty status. The MPI
empty status (MPI 4.1 §3.7.3) is `MPI_ANY_SOURCE`/`mp1::any_tag`/`MPI_SUCCESS`
and is internally configured so `MPI_Get_count` returns 0; use `tag().value`
only when a raw `int` is required.

| Producing completion | `kind()` | `source()`/`tag()` | `get_count`/`get_elements` | `is_cancelled()` |
|---|---|---|---|---|
| P2P receive, `probe`/`iprobe`/`mprobe`, matched receive, combined-send receive side (real source) | native | provider-defined | query MPI; defined | query MPI |
| `MPI_PROC_NULL` receive, `probe`/`iprobe` or matched receive; combined-send receive side | normalized | source `MPI_PROC_NULL`, tag `any_tag` | 0 without MPI; standard-defined | false without MPI |
| file data access | native | normalized to ANY | query MPI; defined (MPI 4.1 §14.4.1) | query MPI |
| P2P send | native | normalized to ANY | 0, no MPI call; undefined for a send | query MPI |
| PENDING slot | native | normalized to ANY | 0, no MPI call | false, no MPI call |
| collective, RMA or buffer-flush completion; default status | normalized | normalized to ANY | 0, no MPI call | false, no MPI call |

`error()` is meaningful only for a status returned by an array completion that
reported `MPI_ERR_IN_STATUS`; a single-status completion error is thrown as
`mp1::mpi_error` and the status `MPI_ERROR` field is then undefined (MPI 4.1
§3.7.3). A PENDING status is the one exception: only
`error() == MPI_ERR_PENDING` is meaningful.

A normalized status is **not** an MPI empty status. Its `native_handle()` is a
synthetic `MPI_Status` that MPI did not configure (ANY/ANY/SUCCESS for the
collective/RMA/flush/default status, `MPI_PROC_NULL`/ANY/SUCCESS for the
`MPI_PROC_NULL` status), so it must not be passed to `MPI_Get_count`,
`MPI_Get_elements`, `MPI_Test_cancelled` or any other MPI status query. The
same caveat applies to a PENDING status, of which only `error()` is meaningful.
A native receive status keeps the provider's status. A send or file status
keeps the provider's count fields, but the wrapper overwrites source/tag with
ANY and forces `MPI_ERROR` to `MPI_SUCCESS` unless the completion reported
`MPI_ERR_IN_STATUS`; its `native_handle()` may be queried while MPI is active,
but only for the fields that family defines.

A native status query requires active MPI: `get_count`, `get_elements` and
`is_cancelled` call MPI and throw `std::runtime_error` after `mp1::finalize`
(a native status whose fields are the empty ones is the one case
`is_cancelled()` answers false without MPI). Normalized and PENDING accessors
make no MPI call. A cancelled receive may report ANY/ANY/SUCCESS, the same
field values as an empty native status; `is_cancelled()` (MPI 4.1 §3.8.4), not
the fields, is the way to distinguish them.

### Null data positions

One exact-byte rule applies to every used P2P, collective, reduction, RMA and
file position (MPI 4.1 §5.1.11): a position is empty only when its total byte
size is zero, not when its element count is zero. A zero count with a
positive-size datatype, and a positive count with a zero-size layout or native
datatype, are both empty.

| Position | Zero-byte null | Non-empty null |
|---|---|---|
| P2P value/range | accepted; per-slot dummy with the caller's count and base datatype | `std::invalid_argument` |
| fixed collective or reduction | accepted; per-slot dummy; the wire count is 0 when the base/single datatype size is 0 | `std::invalid_argument` |
| P2P `region<T>` with `layout<T>::contiguous(0)` | accepted; count 1 with the empty datatype and the caller's pointer (may be null) | `std::invalid_argument` |
| collective/reduction `region<T>` | accepted; per-slot dummy, count 0 and base datatype | `std::invalid_argument` |
| file region or dense range | accepted; stable dummy address | `std::invalid_argument` |
| RMA origin/result | accepted; RMA dummy address | `std::invalid_argument` for a positive origin/result size (an origin under `MPI_NO_OP` is ignored) |

The per-slot dummy differs for the send and receive positions so an ignored
send and an ignored receive never alias; MPI never dereferences it. Ignored
roles (a non-root position, an intercommunicator no-op), the in-place send slot
and `MPI_IN_PLACE` are never inspected.

A non-empty null position is rejected before any MPI **data** call and before a
managed `MPI_Op_create` (and its callback datatype). It is not a "before any
MPI call" guarantee: computing the exact byte size of a flat or sequence
position reads `native_datatype<T>()`, and the collective lowering resolves the
base datatype for every position (typed ones included). The first such use for
a reflected aggregate constructs and commits the synthesized datatype. That
commit is process-local, happens once per type and transfers no data, but it
can happen before the null check rejects the call. A typed reduction position
is measured by `layout::byte_size()` in the preflight and constructs no
datatype there. Staged reductions and `reduce_local` run the exact-byte
preflight before `reduction_op<T>()`, so a rejected call creates no managed
functor `MPI_Op` and stages no buffer. After a successful preflight, a zero-byte
collective or reduction position is presented to the native call as the per-slot
dummy with a wire count of 0 when the addressed datatype has size 0. A P2P
value/range position keeps the caller's count with the base datatype, and a P2P
empty `region<T>` uses count 1 with the empty datatype.

## Examples

`examples/` contains 28 mp1 examples and two raw-MPI comparison programs. They
are built by default when mp1 is the top-level project
(`-DMP1_BUILD_EXAMPLES=ON`). The C++ examples use the same public surface as
the installed package; the two C examples (`vibrating_string_mpi`,
`parallel_sort_mpi`) show the equivalent raw-MPI programs.

## Development checks

Formatting and static analysis are pinned to LLVM 21.1.8. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the clang-format and clang-tidy policy,
the NOLINT rules, and the local commands that match CI. The release checks are:

- configure and build **425** build edges, which compile **285** C++20
  translation units; the release build enables `MP1_BUILD_PERF`, so the
  deterministic `perf_calls` gate is one of the CTest entries;
- `ci/run-ctest.sh --test-dir build -LE nightly`: **304/304** non-nightly
  entries in the default configuration (including the 5-rank
  persistent-collective entry); **102** of them are compile-fail entries
  (**100** probes + 2 harness self-checks);
- **100** compile-fail probes, **66** compile-pass controls and **9**
  compile-all API sources, all built by the same configure;
- **93** unit-test sources carrying **533** Boost test cases;
- the wrap-coverage gate covers **177** wrapped MPI symbols across 34
  implementation files;
- `ci/static-checks.sh` and `clang-format --dry-run --Werror`;
- header self-containment for every public header.

## Migrating

mp1 1.0.0 is a rename and a source-breaking rewrite of mpl 0.5.0; it has no
compatibility aliases. mpl 0.5.0 was a C++17 library and C++20 is a
prerequisite of mp1, not the migration subject. mpl 0.5.0 had no one-sided
communication; RMA/windows are new in mp1 and have no old spelling. See
[MIGRATION.md](MIGRATION.md) for the verified mpl 0.5.0-to-mp1 map, the API
changes and a mechanical migration order. See
[CHANGELOG.md](CHANGELOG.md) for the release summary.

## License

The mp1 library is primarily licensed under the MIT License; see
[COPYING](COPYING). Files inherited from the 2014-2015 codebase remain
BSD-3-Clause; see [COPYING.BSD-3-Clause](COPYING.BSD-3-Clause). See
[LICENSING.md](LICENSING.md) for the exact scope.
