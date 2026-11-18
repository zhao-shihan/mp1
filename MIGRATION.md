# Migrating from mpl 0.5.0 to mp1 1.1

mp1 1.0.0 is a rename and a source-breaking rewrite of mpl 0.5.0.
The mpl 0.5.0 package, header and namespace are replaced by `mp1`; its operation
forms are replaced by the typed value/range/region surface. There are no
compatibility headers, forwarding headers, aliases, deprecated overloads or
shims. The compiler is the migration tool, and a compile error is the expected
signal.

C++20 is a prerequisite of mp1, not the migration subject: the source changes
below are API changes, not a language-standard port. The mpl 0.5.0 release was
a C++17 library; build the target with a supported C++20 compiler before
starting the source migration. The mpl 0.5.0 toolchain is not sufficient for
mp1.

The historical reference below translates mpl 0.5.0 code and build files. Do not
copy the left-hand column into new code.

## 1. Prerequisite: C++20

mp1 is a C++20 library. The requirement is:

- GCC 13 or newer, Clang 16 or newer (with libstdc++ 13+ or a conforming
  libc++), or MSVC 19.32 (Visual Studio 2022 17.2) or newer. The error layer
  uses `std::source_location` and `std::format`; the position vocabulary uses
  concepts, `std::span` and designated initializers.
- `find_package(mp1 1.1 CONFIG REQUIRED)` and
  `target_link_libraries(app PRIVATE mp1::mp1)`; the imported target raises the
  consumer to C++20.
- CMake 3.21 or newer to configure or install mp1. The configure step fails
  closed below the compiler floors; mp1 provides only the C++20 target, with no
  older-standard profile or fallback.

<!-- BEGIN HISTORICAL MIGRATION REFERENCE (exempt from ci/static-checks.sh) -->

The old library appears below under its mpl 0.5.0 spelling. The left-hand
column is historical and must not be copied into new code. The inventory and
every old identifier were checked against the upstream `v0.5.0` tag
(`rabauke/mpl`, commit `014f3efd`); the old README, headers and CMake files are
the authority for the left-hand column.

## 2. What mpl 0.5.0 actually provided

mpl 0.5.0 is a C++17, header-only, MPI 3.1 wrapper. Its public API is
declared across these headers:

`mpl/mpl.hpp`, `mpl/environment.hpp`, `mpl/error.hpp`, `mpl/status.hpp`,
`mpl/message.hpp`, `mpl/tag.hpp`, `mpl/ranks.hpp`, `mpl/info.hpp`,
`mpl/command_line.hpp`, `mpl/datatype.hpp`, `mpl/layout.hpp`,
`mpl/displacements.hpp`, `mpl/operator.hpp`, `mpl/request.hpp`,
`mpl/comm_group.hpp`, `mpl/topology_communicator.hpp`,
`mpl/cartesian_communicator.hpp`, `mpl/graph_communicator.hpp`,
`mpl/distributed_graph_communicator.hpp`, `mpl/distributed_grid.hpp` and
`mpl/file.hpp`. The release also installs `mpl/utility.hpp`, `mpl/vector.hpp`
and `mpl/flat_memory.hpp`, whose contents are detail-only in 0.5.0.

| Family | In mpl 0.5.0 | Verified public surface |
|---|---|---|
| lifecycle and environment | yes, implicit | `mpl::environment::comm_world`, `comm_self`, `processor_name`, `threading_mode`, `is_thread_main`, `wtime_is_global`, `wtime`, `wtick`, `buffer_attach`, `buffer_detach`; `mpl::bsend_buffer`; `mpl::threading_modes`. No public `init`, `finalize`, `initialized`, `finalized` or version query. |
| point-to-point | yes | `send`, `isend`, `send_init`; `bsend`, `ibsend`, `bsend_init`; `ssend`, `issend`, `ssend_init`; `rsend`, `irsend`, `rsend_init`; `recv`, `irecv`, `recv_init`; `sendrecv`, `sendrecv_replace`; `probe`, `iprobe`, `mprobe`, `improbe`, `mrecv`, `imrecv`; `bsend_size`. Scalar, pointer + layout, iterator and send-container forms. There are no pointer + count data-movement forms. |
| collectives | yes | blocking/nonblocking `barrier`, `bcast`, `gather`, `gatherv`, `allgather`, `allgatherv`, `scatter`, `scatterv`, `alltoall`, `alltoallv`, `reduce`, `allreduce`, `reduce_scatter_block`, `reduce_scatter`, `scan`, `exscan`. Variable forms use `mpl::layouts<T>` or `mpl::contiguous_layouts<T>` plus `mpl::displacements`, whose entries are **bytes**. No persistent collectives and no public count/displacement arrays. |
| reductions | yes | `mpl::max`, `min`, `plus`, `multiplies`, `logical_and`, `logical_or`, `logical_xor`, `bit_and`, `bit_or`, `bit_xor`, `op_traits`. |
| datatypes | yes | predefined integers, characters, floating point, complex, `bool`, `std::byte` and enums; automatic `std::array`/`std::pair`/`std::tuple`/C-array support; `base_struct_builder`, `struct_builder`, `struct_layout`. The public macros are `MPL_REFLECTION` and the `MPL_HOMOGENEOUS` opt-in, which is present but non-compiling; `MPL_DATATYPE_TRAITS` was internal and `#undef`ed, not a public macro. No public `native_datatype`. |
| layouts | yes | `layout<T>`, `null_layout<T>`, `empty_layout<T>`, `contiguous_layout<T>`, `vector_layout<T>`, `strided_vector_layout<T>`, `indexed_layout<T>`, `hindexed_layout<T>`, `indexed_block_layout<T>`, `hindexed_block_layout<T>`, `iterator_layout<T>`, `subarray_layout<T>`, `heterogeneous_layout`, `absolute_data`, `make_absolute`, `layouts<T>`, `contiguous_layouts<T>`, `displacements`, `array_orders`. |
| file I/O | yes, member-only | `mpl::file` with `access_mode` and `whence_mode`; constructors/`open`, `close`, `resize`, `preallocate`, `size`, `mode`, `sync`, `set_view`, `seek`, `position`, `byte_offset`, `set_info`, `get_info`, `native_handle`; blocking, nonblocking and split `read`/`write` families, including individual, shared and ordered file-pointer variants. All data access is a member call. There are no free data-access wrappers and no file view type. |
| communicators and groups | yes | `mpl::communicator`, `mpi_communicator`, `inter_communicator`, `mpi_inter_communicator`, `group`, `ranks`. Copying a communicator duplicates it; creation uses `comm_collective`, `group_collective`, `split`, `split_shared_memory` and the merge constructor; `info` is a setter/getter pair. `inter_communicator` is a separate type. |
| spawn | yes | `communicator::spawn` (root/maxprocs/command, optional info, and non-root forms), `communicator::spawn_multiple` (commands, optional infos, and non-root forms), `inter_communicator::parent`, `command_line`, `command_lines`, `infos`. No errcodes result type. |
| topology and grids | yes | `cartesian_communicator`, `graph_communicator`, `distributed_graph_communicator`, neighborhood collectives, `distributed_grid` and `local_grid`; the topology base is an implementation detail, not a public constructor. |
| requests | yes | `irequest`, `prequest`, `irequest_pool`, `prequest_pool`, `test_result`; pool `waitany`/`testany`/`waitsome`/`testsome`/`waitall`/`testall`. No span-based batch type and no polling helper; the pool is the batch container. |
| status, message and tags | yes | `status_t` (`source`, `tag`, `error`, `is_cancelled`/`is_canceled`, `get_count`), `message_t`, `mprobe_status`, `tag_t` and `tag_t::up()`/`tag_t::any()`. |
| errors | yes | `error` plus `invalid_rank`, `invalid_tag`, `invalid_size`, `invalid_count`, `invalid_displacement`, `invalid_layout`, `invalid_dim`, `invalid_datatype_bound`, `invalid_argument` and `io_failure`. No `mpi_error` and no error-handler API. |
| vocabulary | yes | `any_source`, `proc_null`, `undefined`, `root`, `absolute`, `bsend_overhead`, `size_t`, `ssize_t` in the umbrella header. |

### Absent from mpl 0.5.0

These have no left-hand spelling. They are new in mp1 (or remain outside the
typed surface), so there is nothing to migrate:

- **RMA/windows.** The old README explicitly lists one-sided communication as
  not yet supported; there is no `window` header, class or `MPI_Win` call in
  the release.
- **Persistent collectives.** The old collectives are blocking/nonblocking
  only; no `*_init` collective exists.
- **Dense `blocks` descriptors, `region<T>`, `in_place`, `capacity`,
  `predefined_op<T>`, `native_datatype<T>()`,
  `window_create_options`/`window_allocate_options`, `spawn_options`,
  `spawn_result`, `command_line`/`command_lines` as aliases, and the
  `*_with_errcodes` spawn forms.**
- **Span-based request batches and polling helpers, layout-view/borrow types,
  layout `from_range`, free file data access, file view types,
  `MPI_Pack`/`MPI_Unpack` public wrappers, a raw-datatype owner, and error
  handler set/get.**
- **Sessions, partitioned point-to-point and a public `alltoallw`.** The old
  implementation used `MPI_Alltoallw` internally for layout-based `alltoallv`,
  but exposed no `alltoallw` name.

## 3. Historical build and name map

| Area | Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|---|
| CMake package | `find_package(mpl)` | `find_package(mp1 1.1 CONFIG REQUIRED)` |
| CMake target | `mpl::mpl` | `mp1::mp1` |
| Header | `#include <mpl/mpl.hpp>` | `#include <mp1/mp1.hpp>` |
| Namespace | `mpl::` | `mp1::` |
| Language standard | C++17 (`target_compile_features(mpl INTERFACE cxx_std_17)`) | C++20 (`target_compile_features(mp1 INTERFACE cxx_std_20)`); a prerequisite, not an API migration subject |
| CMake options | `MPL_BUILD_EXAMPLES`, `MPL_INSTALL`, `MPL_BUILD_DOCUMENTATION` | `MP1_BUILD_EXAMPLES`, `MP1_INSTALL`; there is no documentation-build option in 1.0, and the version comes from the top-level `VERSION` file, not a cache variable |
| Installed CMake files | `mplConfig.cmake`, generated `mplTargets.cmake` | `mp1Config.cmake`, `mp1ConfigVersion.cmake`, `mp1Targets.cmake` only |
| Example helper | `add_mpl_executable` | `add_mp1_executable` |
| Example target suffix | `vibrating_string_mpl`, `parallel_sort_mpl` | `vibrating_string_mp1`, `parallel_sort_mp1` |

There is no `MPL_VERSION` cache option in mpl 0.5.0, and the current CI job
ids are unrelated to a consumer build; do not carry the old cache-variable name
into a build script.

## 4. Lifecycle, errors, status and vocabulary

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| No public `init`/`finalize`; a process-lifetime static environment initializes MPI on first use with `MPI_Init_thread(MPI_THREAD_MULTIPLE)` (unless the application initialized MPI itself) and finalizes at static destruction. | Explicit `mp1::init`/`mp1::finalize`, or one `mp1::environment`; no implicit finalize unless `mp1::finalize_at_exit()` is called. `init()` requests `MPI_THREAD_SINGLE` by default; use `init(mp1::threading_mode::multiple)` for threads. |
| `mpl::environment::threading_mode()` returns the provided level. | `mp1::provided()` returns the provided level; `mp1::query_thread()` queries the current level. `mpl::threading_modes` becomes `mp1::threading_mode`. |
| `mpl::environment::comm_world()` / `comm_self()` return process-lifetime references. | `mp1::comm_world()` / `comm_self()` return process-lifetime `mpi_communicator` views. |
| `mpl::environment::processor_name`, `wtime`, `wtick`, `wtime_is_global`, `is_thread_main` | free `mp1::` functions of the same names |
| `mpl::environment::buffer_attach(buffer, int)` / `buffer_detach()`; `mpl::bsend_buffer` | `mp1::buffer_attach(buffer, mp1::count_t)` / `buffer_detach()`; `mp1::bsend_buffer` remains the explicit-buffer RAII guard |
| `mpl::error` is the base class; `invalid_*` and `io_failure` are the specific old classes. `io_failure` formats an MPI I/O error. | There is no `mpl::error` equivalent base. Local contract violations use `std::invalid_argument`/`std::overflow_error`/`std::logic_error`; MPI failures use `mp1::mpi_error`; file state/epoch misuse uses `std::runtime_error` or `std::invalid_argument`. Catch `std::exception` for a broad catch and catch the specific new type when the old code caught a specific class. |
| `MPL_DEBUG` enables the old consistency checks; release builds may compile them out. | No debug-macro replacement. mp1 validates its public contracts unconditionally; `NDEBUG` does not disable those checks. |
| `mpl::status_t` with `source()`, `tag()` (`mpl::tag_t`), `error()`, `is_cancelled()`/`is_canceled()` and `get_count<T>()`. | `mp1::status` with `source()` (`int`), `tag()` (`mp1::tag`), `error()`, `is_cancelled()`, `get_count<T>()`, `get_elements<T>()` and `native_handle()`; the `is_canceled` spelling is gone. |
| `mpl::message_t` (an `MPI_Message`) and `mpl::mprobe_status`; matched receive takes `message_t&`. | `mp1::message` owns the matched handle; `mprobe` returns it and `improbe` returns `std::optional<mp1::message>`; matched receive takes `message&`. |
| `mpl::tag_t`, `tag_t::any()`, `tag_t::up()`, `any_source`, `root`, `proc_null`, `undefined`; `mpl::absolute` (an `MPI_BOTTOM` spelling); `mpl::bsend_overhead`. | `mp1::tag` plus `mp1::any_tag` for the wildcard; `mp1::tag_ub(c)`/`c.tag_ub()` for the cached `MPI_TAG_UB` (there is no `tag_t::up()`), or `mp1::tag_ub_min` for the guaranteed 32767 floor; `MPI_ANY_SOURCE`, `MPI_ROOT`, `MPI_PROC_NULL` and `MPI_UNDEFINED` remain raw constants; `mp1::bsend_overhead` remains. `mp1::absolute` does not exist; pass `MPI_BOTTOM` only to raw MPI. |
| `mpl::size_t` and `mpl::ssize_t`; `ssize_t` appears in layout sizes and file offsets. | Use `std::size_t`; use `mp1::count_t` (`std::ptrdiff_t`) for counts and `mp1::offset_t` (`MPI_Offset`) for file offsets. |
| `mpl::info`/`mpl::infos` key-value containers and `mpl::ranks` for rank lists. | `mp1::info`/`mp1::infos` keep the key/value operation names; `remove` returns `bool` (whether the key existed) instead of `void`, and `infos` is an alias for `std::vector<info>`. Rank lists use `std::vector<int>`/`std::span<const int>`; there is no `mp1::ranks` class. |

## 5. Point-to-point

Every old send is a member call; every new send is a free function with the
communicator first. Old sends come in scalar, pointer + layout and iterator
(and, for some modes, container) forms. There are **no** old pointer + count
data-movement calls, so do not translate a nonexistent `comm.send(p, n, ...)`.

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `comm.send(data, dest, tag)` (value or container) | `mp1::send(c, data, dest, mp1::tag{tag})` |
| `comm.send(ptr, layout, dest, tag)` | `mp1::send(c, mp1::region{ptr, layout}, dest, mp1::tag{tag})` |
| `comm.send(first, last, dest, tag)` | `mp1::send(c, range, dest, mp1::tag{tag})` with a contiguous range (`std::span`, `std::array`, C array, `std::vector`, `std::string`); a non-contiguous send of trivially copyable elements is staged, while any other element type is described by a typed absolute-address datatype passed with `MPI_BOTTOM` (MPI 4.1 §5.1.12) and borrows the caller's elements until the send completes |
| `comm.isend(...)` / `comm.send_init(...)` | `mp1::isend(...)` / `mp1::send_init(...)`; both return `mp1::request` |
| `comm.bsend` / `ibsend` / `bsend_init`; `ssend` / `issend` / `ssend_init`; `rsend` / `irsend` / `rsend_init` | same names as free functions with `c` first; nonblocking and persistent forms return `mp1::request` |
| `comm.recv(data, source, tag)` (value, or a resizable/other container that the old wrapper auto-sized) | `mp1::recv(c, data, source, mp1::tag{tag})` (value or container; the four-argument container form auto-sizes on a blocking receive) |
| `comm.recv(ptr, layout, source, tag)` | `mp1::recv(c, mp1::region{ptr, layout}, source, mp1::tag{tag})` |
| `comm.recv(first, last, source, tag)` | No iterator-pair receive. Use a contiguous destination (`std::span` over existing storage, `std::array`, or a fixed-size container) and `std::span`/`capacity`; a blocking container receive can auto-size, a nonblocking one cannot. |
| `comm.irecv(...)` / `comm.recv_init(...)` | `mp1::irecv(...)` / `mp1::recv_init(...)`; both return `mp1::request` |
| Nonblocking receive of a resizable container (old implementation used a background thread and a matched probe) | No auto-size `irecv`. Use the bounded `mp1::capacity{n}` form, a fixed `std::span`/`std::array`, or a blocking auto-size `recv`. |
| `comm.sendrecv(...)`, `comm.sendrecv_replace(...)` | `mp1::sendrecv(...)`, `mp1::sendrecv_replace(...)`; combined operations remain one-shot (no persistent `sendrecv`) |
| `comm.probe(...)`, `comm.iprobe(...)` | `mp1::probe(...)`, `mp1::iprobe(...)` returning `status`/`std::optional<status>` |
| `comm.mprobe(...)`/`comm.improbe(...)` returning `mprobe_status`; `comm.mrecv(data, message_t&)`/`imrecv` | `mp1::mprobe(...)` returns `mp1::message`; `improbe` returns `std::optional<mp1::message>`; `mrecv`/`imrecv` consume a `message&` |
| `comm.bsend_size<T>(n)` or `comm.bsend_size(layout, n)` | No direct helper. Size an explicit buffer with `MPI_Pack_size`/`MPI_Pack_size_c` and add `MPI_BSEND_OVERHEAD` as needed, or use `mp1::buffer_attach(mp1::buffer_automatic, 0)` for automatic buffering. `mp1::bsend_buffer` remains the explicit-buffer RAII guard. |

The old `mpl::tag_t` argument position is unchanged, but the value type is
source-breaking: an `int`/`tag_t` no longer converts implicitly.

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `mpl::tag_t{29}`, `mpl::tag_t(29)` | `mp1::tag{29}` |
| `mpl::tag_t{}` (tag 0) | `mp1::tag{}` (tag 0) |
| `mpl::tag_t::any()` | `mp1::any_tag` |
| `mpl::tag_t::up()` | `mp1::tag_ub(c)` or `c.tag_ub()` (cached, no MPI call); `mp1::tag_ub_min` is the guaranteed 32767 floor |
| `mpl::tag_t` from an enum (implicit constructor) | `mp1::tag{static_cast<int>(enum_value)}`; there is no implicit conversion |
| `status.tag()` as `mpl::tag_t` | `status.tag()` as `mp1::tag`; use `.value` for a raw `int`, `status.source()` remains `int` |

Every send tag must be in `[0, tag_ub]` and never `any_tag`; a receive, probe,
`mprobe` or `improbe` tag also accepts `any_tag`. The `create_group` and
`intercomm_create` tags use the same wildcard/range rule against the source
communicator's cached bound: `any_tag`, a negative value and a value above the
bound are rejected, and 0 is allowed. The checks are local.

mp1 has exactly three forms for each send mode (value, contiguous range,
region). Persistent requests are `mp1::request` with `start()`; starting a batch
is `mp1::start_all(std::span<request>)`.

## 6. Collectives

The old collectives are member calls; the new ones are free functions with the
communicator first and the root immediately after it (`mp1::gather(c, root,
send, recv)`), not in MPI's trailing root position. Fixed collectives used
value or pointer + layout forms; variable collectives used layout containers
plus `mpl::displacements`, whose entries are **bytes**.

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `comm.barrier()`, `comm.ibarrier()` | `mp1::barrier(c)`, `mp1::ibarrier(c)` |
| `comm.bcast(root, data)` / `comm.ibcast(root, data)` | `mp1::bcast(c, root, data)` / `mp1::ibcast(c, root, data)` |
| `comm.gather(root, send, recv)`, `comm.gather(root, send_ptr, send_layout, recv_ptr, recv_layout)` | `mp1::gather(c, root, send, recv)` with value/range/region positions |
| `comm.gatherv(root, send_ptr, send_layout, recv_ptr, recv_layouts, byte_displacements)` | `mp1::gatherv(c, root, send, recv, mp1::blocks{.counts = ..., .displacements = ...})`, or a layout-span descriptor plus counts; convert byte displacements to element units |
| `comm.scatter(root, send_ptr, recv)` or `comm.scatter(root, send_ptr, send_layout, recv_ptr, recv_layout)` | `mp1::scatter(c, root, send, recv)` |
| `comm.scatterv(root, send_ptr, send_layouts, byte_displacements, recv_ptr, recv_layout)` | `mp1::scatterv(c, root, send, mp1::blocks{...}, recv)` or the layout-span form |
| `comm.allgather(send, recv)`, `comm.allgather(send_ptr, send_layout, recv_ptr, recv_layout)` | `mp1::allgather(c, send, recv)` |
| `comm.allgatherv(send_ptr, send_layout, recv_ptr, recv_layouts, byte_displacements)` | `mp1::allgatherv(c, send, recv, mp1::blocks{...})` or layout spans |
| `comm.alltoall(send, recv)` / in-place form | `mp1::alltoall(c, send, recv)`; in-place uses `mp1::in_place` |
| `comm.alltoallv(send_ptr, send_layouts, send_byte_displacements, recv_ptr, recv_layouts, recv_byte_displacements)` | `mp1::alltoallv(c, send, send_blocks, recv, recv_blocks)`; convert both displacement arrays to element units |
| `comm.reduce(f, root, send, recv)` and its in-place overloads | `mp1::reduce(c, op, root, send, recv)`; one-buffer forms use `mp1::in_place` |
| `comm.allreduce(f, send, recv)` / `comm.allreduce(f, data)` | `mp1::allreduce(c, op, send, recv)`; one-buffer forms use `mp1::in_place` |
| `comm.reduce_scatter_block(f, send_ptr, recv, contiguous_layout<T>)`, `comm.reduce_scatter(f, send_ptr, recv_ptr, contiguous_layouts<T>)` | `mp1::reduce_scatter_block(c, op, send, recv)` / `mp1::reduce_scatter(c, op, send, recv, recvcounts)` |
| `comm.scan(f, data)`, `comm.exscan(f, data)` | `mp1::scan(c, op, send, recv)` / `mp1::exscan(c, op, send, recv)` |
| nonblocking `i`-prefixed families | same names as free functions; return `mp1::request` |
| persistent collectives | **new in mp1**; no mpl 0.5.0 spelling. The families use the `*_init` names and `mp1::request`. |

Old variable descriptors used `mpl::layouts<T>`/`mpl::contiguous_layouts<T>`
for block lengths and `mpl::displacements` in bytes. mp1's dense
`mp1::blocks` uses `mp1::count_t` **element** counts and displacements for the
element type; a layout-span descriptor uses `std::span<const mp1::layout<T>>`
and `std::span<const mp1::count_t>`. Convert an old byte displacement with
`byte_displacement / sizeof(T)` only when that divisibility is part of the old
program's contract; otherwise keep the layout form, which carries the true
extent. Do not pass the old byte values directly to `blocks`.

The old raw `MPI_Op`-or-functor argument remains a raw `MPI_Op` or functor in
mp1; the call position is the only change. The old `mpl::max`/`min`/
`logical_xor` functors and `op_traits` keep their names under `mp1::`. Their
element type now defaults to `void`, so `mp1::max{}` is the transparent
spelling that deduces its operands (the same operation as `mp1::max<T>{}`);
this is an addition, not a replacement. The duplicate mp1 spellings of the
other seven functors are gone in 1.0: replace the old
`mpl::plus`/`multiplies`/`logical_and`/`logical_or`/`bit_and`/
`bit_or`/`bit_xor` functors with the standard-library `std::plus`/
`std::multiplies`/`std::logical_and`/`std::logical_or`/`std::bit_and`/
`std::bit_or`/`std::bit_xor` (or the transparent `std::*<void>` form).
`mp1::predefined_op<T>` is new for the RMA accumulating families, not for
ordinary collectives. A functor that reaches the managed callback must now
be stateless, nothrow-invocable for `T`, and used with a trivially copyable
`T`; it must not inspect padding or unlisted members. A functor that maps to a
predefined operation does not reach the callback and is unaffected. The
built-in functors no longer inherit a blanket commutativity tag: `op_traits`
now reports conditional truth (integral `mp1::max`/`min`, arithmetic
`std::plus`/`std::multiplies`/`std::logical_and`/`std::logical_or` and
`mp1::logical_xor`, floating `std::plus<std::complex<T>>`/
`std::multiplies<std::complex<T>>`, integral-or-`std::byte` `std::bit_*`); the
trait is consulted only on the callback path, since a mapped MPI 4.1 §6.9.2
pair never creates an `MPI_Op`, and user types create their `MPI_Op` with
`commute = 0`. The transparent mp1 spellings are classified from the reduction
element type with the same truth table as the typed spelling;
`op_traits<mp1::max<>>` alone is `false` (a type-only query has no operand), as
for `std::plus<void>`. The `commutative_op` tag and user `op_traits`
specializations remain the opt-in.

## 7. Layouts, datatypes and operators

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `mpl::contiguous_layout<T>(n)` / `contiguous_layout<T>(n, block)` | `mp1::layout<T>::contiguous(n)` / `contiguous(n, block)` |
| `mpl::vector_layout<T>(count)` / `mpl::vector_layout<T>(count, block)` | `mp1::layout<T>::contiguous(count)` / `contiguous(count, block)` (or `vector(count, 1, 1)`); old `vector_layout` is a contiguous repetition, not a strided vector |
| `mpl::strided_vector_layout<T>(count, blocklength, element_stride[, block])` | `mp1::layout<T>::vector(count, blocklength, element_stride[, block])` (element stride) |
| no old byte-stride block layout | `mp1::layout<T>::strided_vector` is new; its stride is in bytes |
| `mpl::indexed_layout<T>(parameter)` | `mp1::layout<T>::indexed(indexed_parameter)` |
| `mpl::hindexed_layout<T>(parameter)` | `mp1::layout<T>::hindexed(hindexed_parameter)` |
| `mpl::indexed_block_layout<T>(blocklength, parameter)` | `mp1::layout<T>::indexed_block(blocklength, indexed_block_parameter)` |
| `mpl::hindexed_block_layout<T>(blocklength, parameter)` | `mp1::layout<T>::hindexed_block(blocklength, hindexed_block_parameter)` |
| `mpl::subarray_layout<T>(parameter)` | `mp1::layout<T>::subarray(subarray_parameter)` |
| `mpl::array_orders::C_order` / `Fortran_order` | `mp1::array_order::row_major` / `column_major` (the enum is singular and the enumerators are renamed) |
| `mpl::iterator_layout<T>(first, last)` | No direct factory; there is no `from_range` in 1.0. Use a contiguous/vector/subarray factory when the iteration is regular, or build a committed `MPI_Datatype` and adopt it with `mp1::layout<T>::from_raw` when it is not. |
| `mpl::null_layout<T>` | invalid `mp1::layout<T>{}` (`is_valid() == false`) |
| `mpl::empty_layout<T>` | valid `mp1::layout<T>::contiguous(0)` (zero elements); `layout<T>::empty()` is removed |
| `mpl::layouts<T>`, `mpl::contiguous_layouts<T>` | a contiguous range of `mp1::layout<T>` passed as `std::span<const mp1::layout<T>>` |
| `mpl::displacements` (byte entries) | `std::span<const mp1::count_t>` element displacements, or `mp1::blocks` |
| `mpl::heterogeneous_layout`, `layout<void>`, `absolute_data`, `make_absolute` | Removed. Use a typed `mp1::layout<T>`/`region<T>`, or raw MPI with `MPI_BOTTOM`; there is no absolute or heterogeneous layout type in 1.0. |
| old element-unit `layout::extent`/`lower_bound`/`upper_bound`/`true_extent`/`true_lower_bound`/`true_upper_bound`/`resize` accessors | `mp1::layout<T>::byte_extent`/`byte_lower_bound`/`byte_upper_bound`/`true_byte_extent`/`true_byte_lower_bound`/`true_byte_upper_bound`/`byte_resize` (or `resized`) report bytes; `size()` is the element count. Old byte accessors keep their byte meaning. |
| `MPL_REFLECTION(STRUCT, members...)` (the old release has only this reflection macro; the `MPL_` prefix is the 0.5.0 spelling, while later `MPLR_`-prefixed macros belong to a different fork lineage) | `MP1_REFLECTION(STRUCT, members...)`; `MP1_REFLECTION_EMPTY` and `MP1_REFLECTION_TEMPLATE` are additional new macros |
| `MPL_DATATYPE_TRAITS(type, mpi_type)` — internal: it expanded inside `mpl::detail` to specialize `mpl::detail::datatype_traits` and was `#undef`ed after the built-in mappings, so external code could not call it | `MP1_DATATYPE_TRAITS(type, mpi_type)` at global scope; it opens `namespace mp1` and specializes `mp1::datatype_traits`; this is a new public capability, not a rename of a public old macro |
| `MPL_HOMOGENEOUS` (a blanket opt-in for trivially copyable, copy-assignable non-enum, non-array types; it was intended to map the object to a byte image, but the specialization's `data_type_category` refers to itself and the configuration does not compile, so it never provided a working byte mode) | No equivalent and no working old behavior to preserve. Use `MP1_REFLECTION` when the members are visible. For type erasure use `std::byte` spans/`region<std::byte>`, or map a committed byte-contiguous derived datatype with `MP1_DATATYPE_TRAITS_PACKED`. `MP1_DATATYPE_BASIC(T,U)` is an RMA-only basic-component annotation, not a byte replacement. |
| `MPL_DEBUG` (consistency checks compiled in only when defined) | No macro; checks are unconditional. Do not define `NDEBUG` expecting to remove them. |
| `base_struct_builder`/`struct_builder` with `struct_layout<T>`, `register_struct`, `register_element` and `define_struct(layout)` | Manual builders now derive from `mp1::base_struct_builder<T>`, call `register_object`, then `register_element` for members, then `define_struct()` with no argument; `struct_layout` is gone. Prefer `MP1_REFLECTION`. |

`layout<T>::from_raw(type, elements, facts = {})` is the one raw-datatype
adoption escape (there is no general raw-datatype bridge). It adopts one
derived datatype: a predefined handle throws `std::invalid_argument`, the type
map must be an occurrence of `T` (caller precondition, not validated in 1.0),
`elements` is the application `size()`, and ownership transfers on entry (on
success the layout frees the handle once, while a pre-adoption rejection leaves
ownership with the caller). See the README layout section for the full
contract.

The naming trap is worth stating explicitly: 0.5.0 `vector_layout` is a
contiguous repetition and maps to `mp1::layout<T>::contiguous`, while the old
element-stride block layout is `strided_vector_layout` and maps to the new
element-stride `layout<T>::vector`. The new `strided_vector` factory takes a
**byte** stride and has no old counterpart. The old indexed/indexed-block
displacements are element units, while the old heterogeneous
indexed/indexed-block displacements are bytes; the new parameter types keep
those two unit systems. Where an old builder accepted an optional `layout<T>`
block, use the matching factory's block overload
(`indexed(parameter, block)`, `hindexed(parameter, block)`,
`subarray(parameter, block)`, and so on); the optional block argument is
otherwise unchanged in meaning.

The old builder classes are distinct types; the new five named parameter
builders are aliases of one shared template. Recompile code that names the old
builder types; do not rely on their old type identity. The nested `entry_type`
alias on the new builders is public.

## 8. Files

mpl 0.5.0 file I/O is already member-only: the class owns the `MPI_File` and
there are no free `read_at`/`write_at` forwarders and no file view type. The
migration is therefore a member-call signature change, not a free-function
removal.

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `file.set_view<T>(datarep, displacement, info)` (no layout; etype = filetype = `T`) | `file.set_view(displacement, mp1::layout<T>::contiguous(1), datarep, info)`; use the non-template `file.set_view(displacement, datarep, info)` only when the old `T` was `std::byte` (it sets `MPI_BYTE` etype/filetype) |
| `file.set_view<T>(datarep, layout, displacement, info)` | `file.set_view(displacement, layout, datarep, info)`; displacement is bytes, as before |
| `file.read(ptr, layout)` / `file.write(ptr, layout)` | `file.read(mp1::region{ptr, layout})` / `file.write(mp1::region{ptr, layout})` |
| `file.read_at(off, ptr, layout)` / `file.write_at(off, ptr, layout)` | `file.read_at(off, mp1::region{ptr, layout})` / `file.write_at(off, mp1::region{ptr, layout})` |
| pointer + layout forms of `read_shared`, `write_shared`, `read_all`, `write_all`, `read_at_all`, `write_at_all`, `read_ordered`, `write_ordered` | the same member names with `region{ptr, layout}` or a contiguous range; value forms still exist |
| nonblocking `iread*`/`iwrite*` returning `mpl::irequest` | the same member names returning `mp1::request` |
| split `read_at_all_begin`/`end`, `read_all_begin`/`end`, `read_ordered_begin`/`end`, and the write counterparts with scalar/layout buffers | the same split method names with value, range or region buffers; `begin` remains `void`, `end` returns `mp1::status` |
| `ssize_t` file offsets and sizes | `mp1::offset_t` file offsets and sizes; seek/position/data access remain etype units, `set_view`/`size`/`resize`/`preallocate` remain bytes |
| `mpl::io_failure` for an MPI I/O return code | `mp1::mpi_error` for the MPI failure; local file state/epoch misuse throws `std::runtime_error`/`std::invalid_argument` |
| The old compound-assignment operators for `file::access_mode` | No compound-assignment overloads; use the mp1 bitwise-or/bitwise-and access-mode operators and assign the result to `mode` |
| no `remove`, `set_atomicity`, `get_atomicity`, `seek_shared`, `get_position_shared` in 0.5.0 | those are new mp1 members; do not look for an old spelling |

The 0.5.0 `operator|=`/`operator&=` access-mode helpers have no mp1
compound-assignment overloads; assign the result of `mp1::operator|` or
`mp1::operator&`. The old file already had all data access as members; the old
class also had a `native_handle()` escape. mp1 keeps both properties. Some
0.5.0 comments label `read_at`/`write_at` offsets as bytes; MPI defines
explicit-offset data access in etype units, and both releases pass the value
through unchanged. Use etype units for data access and seek/position; only
`set_view` displacement and `size`/`resize`/`preallocate` are bytes.
`get_view`, `register_datarep`/user datareps and raw heterogeneous views remain
outside 1.0.

## 9. Communicators, groups, spawn and topology

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| Copying a `communicator` duplicates it; the copy constructor is the normal way to create a new context. | `communicator` is a move-only owner; call `mp1::communicator::dup(c)` or `dup_with_info(c, i)`. Passing a communicator by reference does not duplicate it. |
| `comm.info(i)` / `comm.info()` | `c.set_info(i)` / `c.get_info()` |
| `communicator::equality_type`/`identical`/`congruent`/`similar`/`unequal` (also nested in `group` and the old inter-communicator type) | one global `mp1::equality_type`; use `mp1::equality_type::identical` and the other enumerators |
| `communicator(comm_collective, other, group)`, `communicator(group_collective, other, group, tag)` | `mp1::communicator::create(other, group)`, `mp1::communicator::create_group(other, group, mp1::tag{tag})` |
| `communicator(split, other, color, key)` | `mp1::communicator::split(other, color, key)` |
| `communicator(split_shared_memory, other, key)` | `mp1::communicator::split_type(other, MPI_COMM_TYPE_SHARED, key)` |
| `communicator(inter_communicator, order)` with `communicator::order_low`/`order_high` | `mp1::communicator::merge(inter, high)`, passing `false` for low and `true` for high |
| `inter_communicator(local_comm, local_leader, peer_comm, remote_leader, tag)` | `mp1::communicator::intercomm_create(local, local_leader, peer, remote_leader, mp1::tag{tag})` |
| `inter_communicator` is a distinct type with its own `size`/`rank`/`remote_size`/`parent`; no `is_inter()` query. | one `mp1::communicator`/`mpi_communicator` type with runtime `is_inter()`, `remote_size()` and `remote_group()`; `parent()` returns an owning communicator |
| `group(comm)` / `group(inter_comm)` / `group(file)`; `group::Union`, `group::intersection`, `group::difference`, `group::include`, `group::exclude` constructors | `mp1::get_group(c)`; `mp1::group_union(a, b)`, `group_intersection`, `group_difference`, `group_incl(g, ranks)`, `group_excl(g, ranks)`. `group::translate` remains a member. |
| A default-constructed old `group` is the valid empty group. | A default-constructed `mp1::group` is invalid; `mp1::group::empty()` is the valid empty group. |
| `mpi_communicator(MPI_Comm)` / `mpi_inter_communicator(MPI_Comm)` wrapper classes | `mp1::mpi_communicator::borrow(MPI_Comm)` (a copyable view); `mp1::communicator::adopt(MPI_Comm)` for an owning adoption. Intercommunicators are views/owners with `is_inter() == true`. |
| `comm.spawn(root, maxprocs, command[, info])`, `comm.spawn(root)` | `mp1::spawn(c, root, mp1::spawn_options{.maxprocs = ..., .command = ..., .info = ...})`; the non-root form is the same call (non-root payload is not inspected) |
| `comm.spawn_multiple(root, commands[, infos])`, `comm.spawn_multiple(root)` | `mp1::spawn_multiple(c, root, options)` with `spawn_multiple_options::add(command, maxprocs, info)` |
| `inter_communicator::parent()` returning a borrowed static reference | `mp1::parent()` returning an owning `communicator`; still one-shot, with `disconnect()` as the last operation |
| no errcodes result | `mp1::spawn_with_errcodes` and `spawn_multiple_with_errcodes` return `mp1::spawn_result{communicator, std::vector<int>}` (new in mp1) |
| `mpl::command_line`, `mpl::command_lines`, `mpl::infos` are classes | `mp1::command_line` is `std::vector<std::string>`, `mp1::command_lines` is `std::vector<command_line>`, and `mp1::infos` is `std::vector<info>`; the names remain but the types are aliases |
| `cartesian_communicator`, `graph_communicator`, `distributed_graph_communicator`, their `dims_create`/`shift_ranks` helpers, neighborhood collectives, `distributed_grid`, `local_grid` | Removed in 1.0. Use raw MPI through `native_handle()` or `communicator::adopt()`; no grid decomposition type is provided. |

## 10. Requests

| Before: mpl 0.5.0 | After: mp1 1.1 |
|---|---|
| `irequest` (nonblocking) and `prequest` (persistent) are separate classes; `prequest` has `start()`. | One `mp1::request` covers nonblocking and persistent operations; `start()` starts a persistent request and `state()`/`kind()` expose the state machine. |
| `irequest_pool`/`prequest_pool` with `push`, indexed `wait`/`test`, `waitany`/`testany`/`waitsome`/`testsome`/`waitall`/`testall`, and pool-held status arrays. | Put requests in `std::vector<mp1::request>` and pass `std::span<mp1::request>` to `mp1::wait_all`, `test_all`, `wait_any`, `test_any`, `wait_some`, `test_some`, `get_status_all`, `get_status_any`, `get_status_some` and `start_all`. |
| `test_result` (`completed`, `no_completed`, `no_active_requests`) and `std::pair<test_result, index>` pool results. | `mp1::completion` is `std::pair<int, status>`; the any/some functions return `completion`/`completions` or `std::optional` of them, with `MPI_UNDEFINED` as the no-slot index. |
| A request's `get_status()` returns `std::optional<status_t>` and does not free the request. | `mp1::request::get_status()` returns `std::optional<status>` with the same non-destructive meaning. |
| pool `cancel(i)`/`cancelall()` | `mp1::request::cancel()` on the chosen request; there is no batch cancel-all helper, so loop over the span |
| Span-based request batches, polling helpers and pool index slicing | Not present in mpl 0.5.0 and not present in mp1; use `std::span`/`subspan` and the free completion functions. |

## 11. One-sided communication (RMA): new in mp1

mpl 0.5.0 has **no RMA surface**. Its README lists one-sided communication as
not yet supported, the release has no `window` header or class, and the source
contains no `MPI_Win` call. There is therefore no old `window::create`,
`window::allocate`, `put`, `get`, `accumulate`, `fence`, `lock` or
`create_dynamic` spelling to migrate. Any such left-hand column would be
invented.

mp1's RMA/window API is new: `mp1::window` (owner) and `mp1::mpi_window`
(view), the `window_create_options`/`window_allocate_options` factories,
`put`/`get`/`accumulate`/`get_accumulate` and the request-based variants,
`fetch_and_op`/`compare_and_swap`, and the fence/post/start/complete/wait/test
and lock/unlock synchronization functions. Read
[README.md](README.md) for that surface; it has no mpl 0.5.0 counterpart and is
not part of the source migration.

## 12. Removed names and shapes

The following old names and shapes have no 1.0 equivalent:

- the old layout class family and its containers, including the null/empty
  sentinels, the byte-unit displacement container, the absolute-data helper,
  the heterogeneous/void layout, and the iterator layout's iterator-pair
  constructor;
- the old request classes and pools, the old completion enum, and the
  pool-held status arrays;
- the old tag, status and message wrapper types and the old `size_t`/`ssize_t`
  aliases;
- the old exception class hierarchy (the 1.0 taxonomy is the standard library
  plus `mp1::mpi_error`);
- the old topology, Cartesian/graph/distributed-graph communicators and their
  helper types, neighborhood collectives and grid types;
- the old buffer-size helper; use the MPI pack-size routine or automatic
  buffering (see §5);
- the old debug and homogeneous-datatype macros; use unconditional validation
  and `MP1_REFLECTION`/the explicit datatype annotations.

Only the names in the verified inventory and historical tables above existed
in mpl 0.5.0. Do not infer an old spelling from a shape that was added for
mp1; if a name is not in those tables, it was not part of the old library.

## 13. Historical compile breaks

| Compile error | Migration |
|---|---|
| `std::format`/`std::source_location` not found | Use GCC 13+, Clang 16+ with a conforming library, or MSVC 19.32+ (C++20 prerequisite) |
| An old layout class is unknown | Use the matching `mp1::layout<T>` factory or `from_raw` |
| An old layout container or displacement type is unknown | Use `std::span<const mp1::layout<T>>` and `std::span<const mp1::count_t>`, or `mp1::blocks` |
| An old tag type is unknown | Use `mp1::tag{n}` for a literal tag and `mp1::any_tag` for the wildcard; `status::tag()` also returns `mp1::tag` (use `.value` for a raw `int`) |
| An old status or message type is unknown | Use `mp1::status` and `mp1::message` |
| An old request or pool type is unknown | Use `mp1::request`, `std::vector<mp1::request>` and `std::span<mp1::request>` |
| An old communicator copy or constructor is unknown | Use `dup`/`dup_with_info` or the static `create`/`split`/`merge`/`intercomm_create` factories |
| An old spawn call does not match | Use `spawn_options`/`spawn_multiple_options`; use the `*_with_errcodes` forms for errcodes |
| An old file data-access overload does not match | Keep the member call and pass a value, a contiguous range or `region<T>`; reorder `set_view` arguments |
| An old topology/grid type is unknown | Removed; use raw MPI through `native_handle()`/`communicator::adopt()` |
| An old exception catch no longer matches | Catch `std::exception` broadly or the new specific `std::...`/`mp1::mpi_error` type |

<!-- END HISTORICAL MIGRATION REFERENCE -->

## 14. Mechanical migration order

1. **Build files first.** Require a C++20 compiler, `find_package(mp1 1.1
   CONFIG REQUIRED)` and `mp1::mp1`. Include `<mp1/mp1.hpp>` and replace the
   old macro prefix with `MP1_`. Remove the old package names from
   `CMAKE_PREFIX_PATH` and from any exported install rules; there is no
   fallback package.
2. **Make the lifecycle explicit.** Add `mp1::init`/`mp1::finalize` or one
   `mp1::environment`; choose `init(mp1::threading_mode::multiple)` if several
   threads make mp1 calls.
3. **Compile and collect errors.** Work area by area: vocabulary, p2p,
   collectives, layouts/datatypes, files, spawn. RMA has no old code to port;
   add it only where new functionality is wanted.
4. **Replace buffer positions.** Pointer + layout becomes
   `mp1::region{p, layout}`. An iterator-pair send becomes a range
   (`std::span`, `std::array`, C array, `std::vector`); a non-contiguous send of
   trivially copyable elements is staged, while any other element type uses the
   typed `MPI_BOTTOM` absolute-address form and borrows the caller's elements
   until completion. There is no iterator-pair receive and no pointer + count
   form to preserve; use a contiguous destination for receives.
5. **Replace variable descriptors.** Layout containers plus byte displacements
   become `mp1::blocks` with element-unit counts/displacements, or a layout-span
   descriptor plus count span. Check the byte-to-element conversion explicitly.
6. **Replace layouts and datatypes.** Use the `mp1::layout<T>` factories,
   `MP1_REFLECTION` for visible members, and the explicit datatype annotations
   for manual byte-contiguous or RMA-homogeneous types. There is no blanket
   homogeneous mode.
7. **Replace file calls.** Keep the member call and switch the buffer position
   to value/range/region; reorder `set_view` to
   `(displacement, layout, datarep, info)`. An old no-layout typed view becomes
   `layout<T>::contiguous(1)`; the non-template byte `set_view` is only for the
   old `T = std::byte` case. Update nonblocking return types to `mp1::request`.
8. **Replace spawn.** Use the option bags and the builder for
   `spawn_multiple`; use `*_with_errcodes` when the errcodes array is needed.
9. **Replace request batches.** Put requests in a `std::vector`, pass
   `std::span`, and use the free completion functions; do not recreate a pool
   type.
10. **Rebuild from a fresh CMake cache.** An old cache can retain the previous
    compiler standard and package names.

Useful checks after each area:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/mp1
cmake --build build --parallel
ci/static-checks.sh
```

## 15. Leaving the old library behind

There is no compatibility header, forwarding alias, deprecated overload or
adapter for the mpl 0.5.0 spellings. The compiler is expected to reject the old
code; use the historical table above to translate it, then delete the old
package from the build. The public mp1 surface is documented in
[README.md](README.md), with per-family coverage in
[COMPATIBILITY.md](COMPATIBILITY.md) and [API-MATRIX.md](API-MATRIX.md).
