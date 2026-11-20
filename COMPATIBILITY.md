# Compatibility and scope

mp1 requires an MPI implementation that advertises MPI 4.1 or newer and a
C++20 compiler. Both are **requirements**, not coverage promises: mp1 is a
typed C++20 wrapper for the MPI 4.1 **World Model**, not a complete MPI 4.1
binding. This file is the authoritative list of what the typed surface covers;
anything not listed is outside the wrapper. The per-family coverage map,
including the available lifetimes and C++ data spellings, is the
[API matrix](API-MATRIX.md).

## Toolchain baseline

| Requirement | Minimum | Why |
|---|---|---|
| C++ standard | C++20 | concepts, `std::span`, designated initializers |
| GCC | 13 | `std::source_location` and conforming `<format>` |
| Clang | 16 | C++20 library facilities; pair with libstdc++ 13+ or a conforming libc++ |
| MSVC | 19.32 (Visual Studio 2022 17.2) | C++20 library facilities |
| MPI | 4.1 | large-count `_c` routines, matched probes, persistent collectives |
| CMake | 3.21 | `PROJECT_IS_TOP_LEVEL`, package config |

The C++20 toolchain requirement is a prerequisite of mp1, not the 1.0 migration
subject; the source migration from mpl 0.5.0 is covered in
[MIGRATION.md](MIGRATION.md). mpl 0.5.0 did not wrap one-sided communication;
the RMA/window support below is new in mp1 and has no 0.5.0 counterpart. The
configure step fails closed below these minimums instead of producing thousands
of diagnostics from an unsupported compiler.
The error layer uses `std::source_location` and `std::format`; no user-facing
function takes a `source_location` or a `what` parameter.

The examples are built by default when mp1 is the top-level project. A
consumer that only uses an installed package needs a C++20 compiler and
`find_package(mp1 2.0 CONFIG REQUIRED)`; the imported target raises the
consumer to C++20. Boost.Test is needed only for the in-tree test suite.

## Supported typed surface

- World-Model lifecycle: initialization/finalization, thread level
  (`query_thread`, `provided`, `is_thread_main`), timers, version and
  hardware-resource queries.
- Communicator management: `dup`/`dup_with_info`, nonblocking
  `idup`/`idup_with_info` with the `result_pending`/`take_communicator` result
  model, `split`/`split_type`, `create`/`create_group`, `create_from_group`,
  `merge`, `intercomm_create`/`intercomm_create_from_groups`, `adopt`/
  `adopt_connected`, `compare`, `get_info`/`set_info`, and
  `free`/`disconnect`/`release`, plus `view()`/`policy()` and the borrowed
  process-lifetime `comm_world()`/`comm_self()` views and the always-available
  `comm_null()` sentinel. The release policy is fixed at construction: `free`
  for the ordinary factories, `disconnect` for spawn/parent/adopted
  connections. Both from-group factories require an explicit `info` and
  predefined `mp1::error_handler`; `MPI_ERRHANDLER_NULL` is the null handle,
  not a valid handler (MPI 4.1 §9.3.1), so a provider that rejects it is
  conforming. mp1 has no owning error-handler bridge: a custom handler is raw
  MPI or `MPI_Comm_set_errhandler` on the created communicator. Non-membership
  of either from-group call is MPI-erroneous and is deliberately not diagnosed
  locally.
- Point-to-point: blocking, buffered, synchronous and ready sends, each in
  blocking, nonblocking and persistent form; blocking, nonblocking and
  persistent receives; matched probe/receive; blocking and nonblocking combined
  send-receive; non-destructive batch request-status queries. Each send mode
  has the same three forms: value, range, region. Receive has five blocking
  policies (value, fixed, region, auto-size, bounded) and four
  nonblocking/persistent policies (no auto-size). Tags are the strong
  `mp1::tag` value type (`any_tag` wildcard, `tag_ub_min` guaranteed minimum)
  and are validated against the communicator's cached `MPI_TAG_UB`; there is
  no implicit `int` conversion, and `status::tag()` also returns `mp1::tag`
  (source-breaking; `.value` gives the raw `int`). Empty point-to-point
  messages are supported
  through the explicit `mp1::empty_data` span and the ordinary empty
  span/region spellings; there is no separate `*_empty` P2P form and no
  pointer+count overload.
- Collectives: all sixteen families, blocking/nonblocking/persistent;
  fixed-size, variable and reduction forms; dense `blocks` and typed
  layout-span descriptors; in-place forms; intercommunicator support for the
  fourteen §6.2.2 families (`scan`/`exscan` are intracommunicator-only).
  Zero-count and empty-datatype calls are valid collective participations; all
  required ranks must still call them in the same order.
- One-sided communication (new in mp1; absent from mpl 0.5.0): window
  creation/allocation (including shared memory),
  fence/post/start/complete/wait/test, lock/unlock epochs, flush and
  synchronization, blocking and request-based data movement. The origin/result
  is a `std::span<T>` or `region<T>`; the target is a count or `layout<T>`.
  The accumulating families take `predefined_op<basic_component_t<T>>` for a
  predefined or homogeneous-derived `T`; `fetch_and_op` and
  `compare_and_swap` are single-element, predefined-only operations.
- File I/O: open/close, individual and shared file pointers, atomicity,
  positioning, type-extent queries, member-based data access for all blocking,
  nonblocking and split families (value, range, region), typed-view checks,
  and collective split operations.
- Buffered-send buffers: process-wide
  `buffer_attach`/`buffer_detach`/`buffer_flush`/`buffer_iflush` with the
  `bsend_buffer` guard, and communicator-specific
  `attach_buffer`/`detach_buffer`/`flush_buffer`/`iflush_buffer`.
- Process creation: `spawn`, `spawn_multiple`, `spawn_with_errcodes`,
  `spawn_multiple_with_errcodes` and `parent`.
- Support types: groups, info, layouts, status, requests, matched messages,
  and predefined error-handler set/get.

The ownership model is: `communicator` and `window` are move-only owners
exposing copyable views; `file`, `group`, `request` and `message` are
owner-only with no view; `info` and `layout<T>` are the only copyable
MPI-handle owners (a copy duplicates the handle). `request` and `message`
destructors perform the teardown/discard, and `release()` is the no-MPI-call
escape where the type provides it. The complete table is in the
[API matrix](API-MATRIX.md#1-owner-view-and-value-model).

## Typed wire and raw-MPI interoperability

`gatherv`, `scatterv`, `allgatherv`, `alltoallv` and their nonblocking and
persistent forms transfer typed values with
`(buffer, n, native_datatype<T>())`. The wire count is an element count of
`T`, and a displacement is an extent unit of `native_datatype<T>()`. mp1
requires the native datatype extent to equal `sizeof(T)`, so a displacement is
also the public element offset, and MPI applies the datatype's true lower
bound: a dense block does not shift the pointer by `true_lb` and no
displacement includes `true_lb`. `MPI_BYTE` is used only when
`native_datatype<T>()` is `MPI_BYTE` (`std::byte`, or a custom trait that maps
`T` to it); there is no byte-erased wire for typed `T`.

A dense `blocks` block is direct, and a typed block is direct only when its
layout is an identity repetition of `native_datatype<T>()`: the same true lower
bound and `byte_extent() == size() * extent(native_datatype<T>())`. A layout
that is a base repetition but has a marker-shifted or resized extent is staged,
because a per-block displacement cannot carry the marker shift. A direct block
is passed with no staging. Every other `layout<T>` map is staged to a
contiguous `T[n]` image locally with
`MPI_Pack_c`/`MPI_Unpack_c`; the collective itself transfers that image
natively, and a receive unpacks only the mapped positions, leaving storage
outside the type map untouched (MPI 4.1 §5.1.11). A padded or derived `T` --
including `size(native_datatype<T>()) < extent` -- is accepted in the dense
`blocks` spelling at any count; a count repeats the datatype at its extent
stride, so the element image is exact.

A raw-MPI peer that participates in the same collective must present the same
native type signature and the same amount of data (MPI 4.1 §3.3.1, §5.1.11,
§6.1). For a basic `T`, the peer uses the identical predefined datatype name;
for a reflected/derived `T`, it uses the same sequence of basic type
components (same leaf datatypes and extent). The peer's counts are element
counts and its displacements are extent units. A receiver may still take an
mp1 message as `MPI_PACKED` and unpack it (the `MPI_PACKED` receive count is a
byte count); mp1 never sends `MPI_PACKED`, and that relaxation does not make a
typed send's element count a byte count.

The datatype mapping is independent of the reduction/RMA operator mapping. A
custom `MP1_DATATYPE_TRAITS(T, mpi_type)` mapping may name a predefined
datatype outside the C++ categories (for example `MPI_INTEGER` or
`MPI_C_DOUBLE_COMPLEX`), and that handle is the data wire for send and receive:
a peer written in another language must present the same datatype, and mp1
neither rewrites nor reinterprets it. Such a `T` is outside the C++ datatype
categories used by the typed reduction and RMA operator mapping: a typed
reduction functor for it stays on the callback (no predefined mapping), the
typed `predefined_op` functor constructor rejects it, and a raw `MPI_Op` in a
collective is still forwarded unchanged with `native_datatype<T>()`. A raw
`MPI_Op` handle for an RMA operation is accepted only when the runtime datatype
category is one of the C++ categories (`MPI_INT`, for example, is the C++
integer category).

Because the wire is typed, MPI §3.3.2 representation conversion applies to
typed values across heterogeneous environments; it is specified by MPI and is
not exercised by homogeneous CI configurations. The same rule applies to the
mp1-managed reduction callback image: it is typed over `native_datatype<T>()`
and adds no byte component. The byte-level exception is unchanged:
`std::byte` maps to `MPI_BYTE` and its wire is the byte image with byte counts
(see [Type erasure](#type-erasure)).

## Reduction operator categories

Typed functors and raw `MPI_Op` handles are not interchangeable categories:

- **Typed functors** map to the predefined operation MPI 4.1 §6.9.2 defines
  for the C++ element datatype category: `std::plus`/`std::multiplies` use C
  integer, floating and complex; `mp1::max`/`mp1::min` use C integer and
  floating; `std::bit_*` use C integer and Byte (`std::byte`);
  `std::logical_and`/`std::logical_or` and `mp1::logical_xor` use C integer and
  Logical (`bool`). The C++ datatypes are `MPI_CXX_BOOL` and
  `MPI_CXX_*_COMPLEX`; Fortran datatypes are out of scope for the typed
  mapping and the predefined-operation table. The predefined
  operation's result semantics are MPI's (floating max/min NaN behavior, the
  integer logical value), not a C++ reimplementation. The mp1 functors default
  their element type to `void` (`mp1::max{}`): the same category rules apply,
  decided from the element type of the reduction, so the transparent spelling
  never changes a mapping.
- **Raw predefined `MPI_Op`** handles in the RMA families are accepted through
  `predefined_op<U>` according to MPI 4.1 §12.3.4 and the §6.9.2 datatype
  categories for `U`. The C++ typed surface maps the C++ datatypes only: a C or
  Fortran datatype variant (`MPI_C_BOOL`, `MPI_C_*_COMPLEX`, `MPI_LOGICAL`,
  `MPI_REAL`, `MPI_COMPLEX`, ...) or a multi-language category is rejected
  before the MPI call. A raw RMA handle must be a standard predefined
  operation; user-created handles, `MPI_OP_NULL`, `MPI_MAXLOC`/`MPI_MINLOC`,
  and category mismatches are rejected before MPI.
- **Raw `MPI_Op`** handles in the non-RMA collective reduction families are
  forwarded unchanged to MPI: any non-null handle is passed to the collective as
  the caller supplied it, and only `MPI_OP_NULL` is rejected locally. mp1 does
  not validate predefinedness, the §6.9.2 category, `MPI_MAXLOC`/`MPI_MINLOC`
  or caller-created handles on this path, so a caller-created callback is called
  with `native_datatype<T>()` as described below, and `MPI_MAXLOC`/`MPI_MINLOC`
  over a non-pair `T` is an erroneous MPI program rather than an mp1 rejection.

For the RMA accumulating families the category is checked against `U`, the
single basic component (`T` for a predefined `T`, `basic_component_t<T>` for a
homogeneous derived `T`). A homogeneous-derived `T` with a matching operand is
supported (for example `std::pair<int,int>` with `MPI_SUM` or `MPI_REPLACE`);
mixed, empty and opaque types are compile errors rather than category errors.
RMA has no user-defined callback fallback, so an unmappable functor is a
compile-time error and a raw `MPI_Op` that is not predefined is rejected before
MPI.

An mp1-managed functor defines one reduction element as a complete `T` object
described by a committed, process-lifetime image built from
`native_datatype<T>()`: the native map is placed at displacement `-true_lb`
when needed (a one-block struct) and resized to lower bound 0 and extent
`sizeof(T)`. The image has the native typed leaves and true lower bound 0; mp1
shifts the callback-image buffer pointer up by the native true lower bound and
the callback subtracts it again, so `T[len]` indexing addresses the listed
fields at their C++ offsets. There is no
`MPI_BYTE` full-object image and no byte padding block. Only a `T` whose native
datatype is `MPI_BYTE` (`std::byte` or a byte trait) has a byte leaf, because
that is the element's native type. Padding bytes and unlisted members are not
part of the type map: they are never transmitted, and a direct or staged
receive writes only mapped fields, preserving the caller's other bytes. The
functor must be stateless and nothrow-invocable for `T`, must not inspect
padding/unlisted members, and `T` must be trivially copyable. The image is
created once per `T` and retained for the process lifetime (freeing after
`MPI_Finalize` is not allowed), matching the `MPI_Op` cache; the cache is
deliberately not destroyed at process exit. mp1 builds a mapped-byte plan from
the native datatype to update only transmitted bytes. A reflected aggregate,
`std::pair`/`std::tuple`/`std::array`, and a custom datatype built from the
named, duplicated, contiguous, vector, indexed or struct constructors are
supported; a custom datatype using subarray, darray, F90, `PACKED` or
`VALUE_INDEX`, or a type map that overlaps itself, is rejected with
`std::invalid_argument` before the reduction. For `reduce_scatter` and
`reduce_scatter_block` the image must have true lower bound 0 and true extent
`sizeof(T)`: MPICH 4.3.2 (the CI MPICH pin) lays those callbacks out at the true
extent, so a `T` with a non-zero true lower bound or trailing unmapped bytes is
rejected for those two families before MPI (the other reduction families accept
it).

A raw caller-supplied `MPI_Op` receives
`native_datatype<T>()` and its callback must honor that datatype's true lower
bound and type map (MPI 4.1 §6.9.5). Over-aligned `T`
(`alignof(T) > alignof(std::max_align_t)`) is unsupported for user-functor
reductions because MPI may place callback temporaries at only fundamental
alignment.

The `commute` argument of `MPI_Op_create` follows
`mp1::op_traits<F>::is_commutative`: it is true only where the C++ operation is
actually commutative (integral `mp1::max`/`min`; arithmetic
`std::plus`/`std::multiplies`/`std::logical_and`/`std::logical_or` and
`mp1::logical_xor`; floating `std::plus<std::complex<T>>`/
`std::multiplies<std::complex<T>>`; integral-or-`std::byte` `std::bit_*`;
explicit user opt-in through `commutative_op` or an `op_traits`
specialization). The trait is consulted only on the callback path: a mapped
§6.9.2 pair never creates an `MPI_Op`. For a callback-only pair the value is the
`commute` argument of `MPI_Op_create` -- `commute = 0` where the C++ operation
is not commutative (for example a user-defined element type), which fixes the
operand order to ascending rank order (MPI 4.1 §6.9.5). The transparent
spellings the library classifies (the `std::plus`/`multiplies`/`equal_to`/
`not_equal_to`/`logical_and`/`logical_or`/`bit_*` `void` specializations and
`mp1::max<>`, `mp1::min<>`, `mp1::logical_xor<>`) are classified by substituting the reduction
element type into the same `op_traits` row as the typed spelling;
`op_traits<mp1::max<>>` and `op_traits<std::plus<void>>` alone are `false`,
because a type-only query has no operand to classify.

## Not supported

| Family | Raw-MPI path |
|---|---|
| Topology, Cartesian/graph/distributed-graph communicators, neighborhood collectives, grid decomposition | Not wrapped. Use raw MPI on `native_handle()` or `communicator::adopt()`. |
| `MPI_Alltoallw`, `MPI_Type_create_darray`, runtime raw-struct/array datatype construction | Not wrapped. HPF-style darray file views and heterogeneous datatypes require raw MPI. For homogeneous byte type erasure use `std::span<std::byte>` or `region<std::byte>`. There is no general raw-datatype bridge; `layout<T>::from_raw` is the one documented raw-adoption escape (derived handle, type map over `T`, ownership transfers on entry). |
| Public `MPI_Pack`/`MPI_Unpack` | Not wrapped; the wrapper's staging is internal. |
| RMA accumulating families with a heterogeneous/empty derived datatype, a custom derived trait without `MP1_DATATYPE_BASIC`, an unannotated opaque custom type, or a top-level C-array element type | Compile-time error (MPI 4.1 §12.3.4). A homogeneous-derived `T` is accepted with `predefined_op<U>`; an annotated custom trait is accepted after its one-time committed-datatype validation. A C array is supported as a member only. `put`/`get` are unaffected and still accept a mixed `region<T>`. |
| `MPI_MAXLOC`/`MPI_MINLOC`, CAS outside the C integer/Logical/Byte categories, or count/layout forms of CAS/FOP | Rejected before MPI. `fetch_and_op` and `compare_and_swap` are single-element and require a predefined C++ datatype; there is no homogeneous-derived fallback. |
| RMA container data forms | Origin/result are `std::span`/`region`; there is no container overload. Use `std::span{container}` or a layout. |
| Typed `MPI_MAXLOC`/`MPI_MINLOC` reductions | mp1 has no `MPI_2*` pair datatype model. Raw `MPI_Op` handles are accepted by the non-RMA reduction families, but using them is raw-MPI territory. |
| A raw `int` tag or an implicit `int` -> `mp1::tag` conversion | Source-breaking by design: spell tag literals as `mp1::tag{n}` and the wildcard as `mp1::any_tag`. |
| Point-to-point shape limits: persistent iterator forms, auto-size `irecv`, auto-size persistent receive, sequence-container receive targets, combined iterator forms, persistent `sendrecv` | Frozen out. Persistent operations capture a fixed address; a nonblocking container receive uses the bounded form, `std::array` or a region; combined send-receive is one-shot only. Use raw MPI where the exact shape is required. A non-contiguous sequence send is typed and never a byte image: trivially copyable elements are staged, other elements use the absolute-address `MPI_BOTTOM` form and borrow the caller's elements until completion. |
| Persistent collective container data buffers | Fixed buffers only. Dense and typed descriptor arrays are accepted because the plan copies them. |
| File `get_view`, `register_datarep`/user datareps, and raw heterogeneous `set_view` | `get_view` is future work (requires the owning raw-datatype bridge); the remaining items are not implemented. Typed views enforce only the locally provable MPI 4.1 §14.3 rules: a proven descending or negative etype displacement map is rejected for every access mode, and a proven overlapping filetype is rejected when the file is open for writing. A custom derived datatype map whose order or overlap is not recorded is left to MPI. |
| File `layout<T>` × count | A repeated layout is spelled `layout<T>::contiguous(n, block)`, or use the range/region form. |
| MPI Sessions | Not wrapped. Do not mix session and World-Model objects in mp1. |
| Partitioned point-to-point | Not wrapped. Use raw MPI requests. |
| Attributes/keyvals, object naming | Not wrapped. Use raw MPI through `native_handle()`. |
| User error-handler callbacks and `MPI_Add/Remove_error_*` | Not wrapped. Only the predefined handler set/get is exposed. |
| DPM connect/accept/join and name service | Not wrapped; only the spawn side is exposed. |
| Generalized requests and MPI_T | Not wrapped. |

## RMA homogeneous-derived element types

MPI 4.1 §12.3.4 permits a predefined datatype or a derived datatype whose
basic components are all the same predefined datatype for the accumulating RMA
families, and requires every datatype argument to be constructed from that
same predefined type. mp1 implements this at compile time:

- **Accepted**: a predefined `T`; `std::pair`, `std::tuple`, `std::array`, a
  nested homogeneous aggregate, a C-array member, an `MP1_REFLECTION` type,
  and a custom derived trait opted in with `MP1_DATATYPE_BASIC(T,U)`. The
  operator is `predefined_op<U>`.
- **Rejected at compile time**: a mixed (heterogeneous) or empty aggregate, an
  unannotated opaque custom type, a top-level C array, and
  `predefined_op<aggregate>`. A zero-element `layout<T>` of a non-empty
  homogeneous `T` remains a valid zero-element call.
- **Custom trait contract**: `MP1_DATATYPE_BASIC(T,U)` is an RMA-only contract.
  It does not create a datatype mapping or change the wire image, and it is not
  a replacement for the predecessor's byte shortcut. Because mp1 did not build
  the committed datatype from the C++ members, the handle is walked once
  against `native_datatype<U>()` before the first accumulating call, and a
  wrong annotation throws `std::invalid_argument` before the MPI call.
  The trusted pair/tuple/array/`MP1_REFLECTION` paths are compile-time only and
  perform no runtime walk. `component_types` is detail-only; the public traits
  are `basic_component_t<T>`, `is_homogeneous_datatype_v<T>` and
  `basic_component_traits<T>`.
- **A custom predefined C++ type mixed with a different C++ type that maps the
  same MPI handle** cannot be proven homogeneous; use a reflected `int` struct
  or a plain `int` pair.

`MPI_REPLACE` follows the same rule: accepted for a predefined or
homogeneous-derived `T` in all four accumulating families, rejected at compile
time for a heterogeneous-derived `T`. This is the current §12.3.4 rule: the
MPI 2.1 "predefined only" restriction was removed in MPI 2.2. `MPI_REPLACE` is
not a reduction operator: it is defined for the RMA accumulating families and
`fetch_and_op`, not for `reduce`/`allreduce` or the other collective
reductions.

Atomicity is per basic element, not per C++ `T` object (MPI 4.1 §12.7.1,
§12.3.4), so a `std::pair<int,int>` is two atomic elements and no whole-`T`
atomicity is promised. The `accumulate_ops` window info key defaults to
`same_op_no_op`: concurrent accumulates to the same target address must use the
same operator or `MPI_NO_OP`; changing the info key does not make different-op
mixing safe. `MPI_NO_OP` is legal only in
`get_accumulate`/`rget_accumulate`/`fetch_and_op`.

## Provider caveats

These are implementation support differences, not changes to mp1's contract.
The release suite is exercised with MPICH, Open MPI and Intel MPI; the exact
versions and configurations are recorded in the release evidence.

- **`MPI_REPLACE` with a derived datatype.** MPI 4.1 §12.3.4 permits a
  homogeneous-derived datatype for the accumulating families. Implementations
  differ in how they diagnose an invalid heterogeneous derived accumulate (an
  MPI error in some, a job abort in others, and in some epoch modes a silent
  execution of the invalid call), which is why mp1 rejects it before the
  accumulating MPI call rather than relying on provider diagnosis. The local
  classification and the one-time untrusted-annotation validation are the
  guarantee.
- **Homogeneous datatypes with a zero-blocklength entry or a non-contiguous
  map.** MPI permits a derived datatype whose zero-blocklength entries
  contribute no basic components; mp1's one-time validator accepts such a
  datatype when its positive entries are homogeneous. A provider may still
  reject the legal datatype in a data-movement call, and derived RMA datatypes
  with non-contiguous maps may have different performance or support across
  implementations. mp1 does not rewrite, pack or stage the committed datatype.
  If a provider rejects a type that MPI permits, use a contiguous homogeneous
  variant or a predefined datatype, or use raw MPI.
- **User-defined `MPI_Op` over a derived datatype.** Some providers can return
  a corrupted reduction result for this raw-MPI combination. The
  `raw_mpi_op_keeps_native_datatype` case is skipped on Intel MPI (observed
  with 2021.19). It is a provider limitation, not an mp1 contract; mp1 does not rewrite or pack
  the datatype to work around it. Use a predefined datatype/operator, pack the
  elements yourself, or use another provider/configuration.
- **`MPI_ERR_PENDING` from a multiple completion.** MPI 4.1 §3.7.5 lets
  `MPI_Waitall` return `MPI_ERR_IN_STATUS` as soon as one slot fails and report
  `MPI_ERR_PENDING` for a slot that "has neither failed nor completed"; a
  provider may instead wait for every slot. Open MPI can return the PENDING
  state, while pinned MPICH 4.3.2 and Intel MPI 2021.19 wait for every slot and
  cannot (both were verified by a direct incomplete-slot probe); the end-to-end
  PENDING cases are skipped on those providers. mp1
  accepts both: a PENDING slot stays active with its handle and payload, and a
  later completion call reaps it. Portable code must not assume that one
  `wait_all`/`test_all` return means every slot is complete; check
  `status::error()` for `MPI_ERR_PENDING` and call again. `wait_some`/
  `test_some` omit a PENDING slot from the returned list for the same reason.
- **Persistent receive from `MPI_PROC_NULL`.** MPI 4.1 §3.10 specifies
  source `MPI_PROC_NULL`, tag `mp1::any_tag`, count 0 and success for a
  receive from `MPI_PROC_NULL`, including a started persistent receive.
  Pinned MPICH 4.3.2 leaves `SOURCE` at `MPI_ANY_SOURCE` for a persistent
  receive initialized with `MPI_PROC_NULL`, while Open MPI returns the
  standard values. mp1 records the expected source in the request and
  synthesizes the §3.10 status for every receive form (blocking,
  nonblocking, persistent and matched/`MPI_MESSAGE_NO_PROC`), for
  `probe`/`iprobe` and for the receive side of a combined send-receive, so
  the status is provider-independent; the data buffer is untouched either
  way. Pinned MPICH 4.3.2 also leaves the `MPI_Iprobe(MPI_PROC_NULL)` source
  at `MPI_ANY_SOURCE`, which the wrapper normalizes the same way.
- **`MPI_TAG_UB` value.** MPI 4.1 §3.2.3 requires `MPI_TAG_UB >= 32767` and
  makes the bound inclusive. mp1 queries the attribute once per communicator
  view with `MPI_Comm_get_attr` and caches it; if an implementation does not
  expose the predefined attribute, the cache falls back to `tag_ub_min`
  (32767). A provider that reports a smaller nonconforming bound is trusted
  as reported, so tags above it are rejected locally; use raw MPI for a
  provider that violates the minimum.
- **`MPI_MAX_INFO_KEY`/`MPI_MAX_INFO_VAL` exact lengths.** MPI 4.1 §10 gives
  keys/values an implementation-defined maximum length, but the C binding
  stores them as null-terminated strings and implementations differ on whether
  the maximum counts the terminator. mp1 therefore accepts only the
  conservative subset (`key.size() < MPI_MAX_INFO_KEY`,
  `value.size() < MPI_MAX_INFO_VAL`) and rejects a key/value of exactly the
  maximum length locally before any `MPI_Info_set`; a program that is portable
  under the strictest reading uses at most `MPI_MAX_INFO_KEY - 1` /
  `MPI_MAX_INFO_VAL - 1` characters.
- **Zero-size datatypes on MPICH 4.3.2.** A standard-conforming positive
  count of a zero-size datatype can make pinned MPICH 4.3.2 abort with SIGFPE
  in some collective calls. mp1 normalizes the wire count to 0 for the fixed
  and variable collective and reduction families when the addressed datatype
  size is 0 (the type signature is empty either way, MPI 4.1 §5.1.11), so
  those paths are provider-independent. A custom zero-size trait used in a
  point-to-point or file operation still passes the caller's count; pass a
  zero count, or use raw MPI, when targeting that provider configuration.
- **Shared-memory RMA availability.** Some providers or configurations do not
  expose a multi-process shared-memory domain, so a shared window over the
  world cannot be allocated. The MPI call then fails (fatal under the default
  handler); use a non-shared window or raw MPI when the capability is absent.
- **Intel MPI `MPI_Comm_spawn` fabric.** Intel MPI 2021.19's default `shm`
  fabric aborts `MPI_Comm_spawn` with "shm fabric does not support Spawns".
  Use `I_MPI_FABRICS=shm:ofi`; the spawn tests export it. Intel MPI's own
  `vars.sh` does not add the bundled libfabric to `FI_PROVIDER_PATH` or the
  loader path, so `ci/mpi-env.sh` sources the oneAPI libfabric `vars.sh` too;
  without it `MPI_Init` under `shm:ofi` fails. mp1 does not set the fabric
  itself (it is a launch-environment choice).
- **Intel MPI 2021.19 two-executable spawn.** The dedicated
  `test_spawn_contract` scenario hangs in Intel's two-executable
  `MPI_Comm_spawn` path even with `I_MPI_FABRICS=shm:ofi`; those entries are
  skipped on Intel after the provider-independent option-payload checks. The
  self-respawn `test_file_spawn_contract` scenario exercises `MPI_Comm_spawn`
  on Intel with `shm:ofi`. Verify a two-executable spawn on this provider before
  relying on it.
- **Open MPI 6+ four-rank file-spawn test.** The four-rank
  `test_file_spawn_contract` entry is skipped on Open MPI 6+. The failure was
  observed only in GitHub CI; it is possibly caused by the CI container
  environment. The two-rank entry still runs, and the dedicated
  `test_spawn_contract.4` covers four-rank spawn with an external child.
- **Intel MPI 2021.19.2 self-respawn.** The self-respawn success case in
  `test_file_spawn_contract` is skipped on Intel MPI 2021.19.2. The failure was
  observed only in GitHub CI; it is possibly caused by the CI container
  environment. The file tests and local spawn validation still run.
- **Intel MPI holey allgather at four or more ranks.** The non-contiguous
  aggregate receive in `fixed_allgather_alltoall_contract` is skipped on Intel
  at four or more ranks. The failure was observed only in GitHub CI; it is
  possibly caused by the CI container environment. The dense forms and the
  two-rank case still run.
- **Intel MPI 2021.19 large subarray commit.** A subarray whose byte size
  exceeds 2 GiB (for example `sizes={INT_MAX,2}`, `subsizes={INT_MAX,1}`) makes
  Intel abort inside `MPI_Type_commit` (dataloop assertion). The layout test
  skips that case on Intel; avoid such subarrays on this provider.
- **F90 datatypes on C-only builds.** `MPI_Type_create_f90_*` is exercised only
  when the provider exposes the mapping. MPICH built with `--disable-fortran`
  returns `MPI_DATATYPE_NULL`; Open MPI built with `--disable-mpi-fortran` can
  return a non-null handle whose `MPI_Type_size_c` is `MPI_UNDEFINED`. That is
  not a standard size result (MPI 4.1 §5.1.5 reserves `MPI_UNDEFINED` for an
  OUT parameter that cannot express the size), but the C-only build does not
  provide the F90 mapping; the layout rejection test skips only the positive
  size sanity check and still checks the rejection and combiner contract. A
  custom C++ trait mapped to such a null or zero-extent Fortran datatype is
  rejected by the `native_datatype<T>()` validation; the allreduce test skips
  its data-movement assertions and asserts that the reduction throws
  `std::invalid_argument` instead of sending, and the RMA test skips its
  usable-wire/category assertion.
- **Intel MPI 2021.19 persistent collectives.** Reusing persistent requests
  across cycles can return stale or undefined payloads. The reuse-heavy
  gather/scatter, allgather/alltoall and intercommunicator cases (including the
  variable/scatter form) are skipped on Intel (previous-cycle values or `-1`);
  barrier/broadcast, variable, reduction and local-validation cases still run.
  Use the non-persistent forms or another provider when these persistent
  collectives are required.
- **Intel MPI 2021.19 `MPI_Win_shared_query` on non-shared windows.** MPI 4.1
  permits `MPI_Win_shared_query` on ALLOCATE, ALLOCATE_SHARED and CREATE
  windows (only DYNAMIC is excluded). Intel 2021.19 aborts with "Incorrect
  window flavor ... expected !MPI_WIN_FLAVOR_DYNAMIC" even for ALLOCATE and
  CREATE windows; the RMA contract test skips only the success queries on Intel,
  while its local rank rejection and DYNAMIC rejection still run. Use
  ALLOCATE_SHARED or another provider when shared-querying a non-shared
  window.
- **`MPI_File_get_byte_offset` on a sequential file.** MPI 4.1 §14.4.3 permits
  this call with `MPI_MODE_SEQUENTIAL`, but provider support varies. A provider
  may reject it (the wrapper surfaces that as `mp1::mpi_error`), and pinned
  MPICH 4.3.2 has been observed to return an incorrect offset (0) instead of an
  error after `set_view(MPI_DISPLACEMENT_CURRENT, layout<char>::contiguous(1))`,
  while Open MPI returns the standard-conforming value. The call is
  standard-conforming and mp1 does not reject it locally; verify it on the
  target provider and prefer a non-sequential mode when the byte offset is
  required.
- **Intel MPI 2021.19 non-leader `remote_group`.** MPI 4.1 §7.6.2 makes
  `remote_group` and `remote_leader` significant only at `local_leader`, but
  Intel MPI 2021.19.2 validates `remote_group` on every local rank and rejects
  a non-leader `MPI_GROUP_NULL` with `MPI_ERR_GROUP` ("Null group"). It ignores
  an invalid `remote_leader` at non-leaders. mp1 follows the standard and keeps
  the leader-only validation; portable code passes the same valid remote group
  on every local rank and a valid remote leader at the leader.
- **Large-count routines.** mp1 consistently uses the MPI 4.1 `_c` large-count
  entry points; a provider that advertises MPI 4.1 but has an incomplete `_c`
  implementation is not supported.

## Error-handling limits

Matched-message recovery is loss-free for every failure the wrapper can
detect: type signature, count overflow, a container-resize failure (including
the resize's `std::bad_alloc`), because MPI 4.1 §5.2 permits receiving a
message sent with any type as `MPI_PACKED`. The one remaining case is
exhaustion of memory while allocating the exact-size packed scratch buffer
needed for the recovery itself: mp1 then attempts a deliberately truncated
`MPI_PACKED` receive (§3.2.4 treats an overflow as an erroneous program) so a
synchronous sender is not stranded. That receive is handled by the error
handler of the communicator used for the matching probe (§3.8.3), not
necessarily `MPI_COMM_WORLD`; `mp1::install_recoverable_errors()` sets only
WORLD and SELF. A communicator duplicated from WORLD after that install
inherits its returning handler at duplication time (MPI 4.1 §7.4.2); one
created or duplicated before it, or duplicated from a communicator that is
still fatal, must be set explicitly with
`mp1::set_error_handler(c, mp1::error_handler::returns)` for recovery to throw
instead of aborting. With a returning handler, `message::discard()` throws
`mpi_error` for the truncation, while the auto-size `recv` guard suppresses it
and propagates the original `std::invalid_argument`/`std::bad_alloc`; the
`message` destructor is `noexcept` and abandons/counts the handle. A wrong
element type of the same size is not detectable from the status; MPI 4.1
§3.3.1 makes such a receive erroneous and mp1 behaves like the underlying MPI
call. The closed exception taxonomy, including `std::logic_error` for non-file
state/policy misuse, `std::runtime_error` for file state/epoch misuse,
`std::bad_array_new_length` and the auto-size receive's `std::bad_alloc`, is in
README's [Errors and lifetime](README.md#errors-and-lifetime).

## Type erasure

`mp1::native_datatype<std::byte>()` is `MPI_BYTE`. Use a
`std::span<std::byte>` or `region<std::byte>` when an application manages its
own byte-level layout. The `std::byte` variable-collective wire is the byte
image with byte counts; for the typed default see
[wire and raw-MPI interop](#typed-wire-and-raw-mpi-interoperability). There is
no absolute or heterogeneous `layout` type and no public `from_range` factory.
There is no general raw `MPI_Datatype` bridge; `layout<T>::from_raw` is the one
documented raw-adoption escape (derived handle, caller-asserted type map over
`T`, ownership transfers on entry; see the README layout section).

## Threading

The library is thread-compatible, not internally synchronized per object (MPI
4.1 §11.6.1). Distinct objects may be used concurrently by distinct threads at
`MPI_THREAD_MULTIPLE`; one request, communicator, window, file, info or layout
object must be confined to one thread while it is being operated on. `init` and
`finalize` are serialized by the library. `MPI_FINALIZE` should be called from
the initializing thread after all other threads stopped making MPI calls. The
one always-available exception is `info`: every `MPI_Info_*` routine may be
called concurrently, but the wrapper's lazy handle creation in `info::set` is
a write to that object, so create the table before sharing it.
