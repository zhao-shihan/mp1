# mp1 2.0.0 API matrix

mp1 is a header-only C++20 typed wrapper for the MPI 4.1 World Model. This page
maps the public typed surface of mp1 2.0.0: which operation families exist,
which C++ data spellings each family accepts, which lifetimes are covered, and
where the typed surface stops. It is a coverage map, not a signature reference:
exact declarations, parameter order, defaults and constraints are in the
headers under `mp1/impl/`; the umbrella header is `mp1/mp1.hpp`.

Public headers: `p2p.hpp`, `collective.hpp`, `file.hpp`, `rma.hpp`,
`window.hpp`, `layout.hpp`, `datatype.hpp`, `operator.hpp`, `info.hpp`,
`communicator.hpp`, `group.hpp`, `request.hpp`, `environment.hpp`,
`status.hpp`, `spawn.hpp`, `buffer.hpp`. `error.hpp`, `vocabulary.hpp` and
`fwd.hpp` are the supporting public headers outside the 16-header inventory.

## 0. Surface summary and counting

The sixteen public family headers are listed below. `error.hpp` and
`vocabulary.hpp` are public supporting headers outside that sixteen-header
inventory; `fwd.hpp` is the forward-declaration header.

One auditable count is reported: **measured public declarations** — every
public function/member declaration in namespace `mp1`, counted with a
scope-aware scanner. Private and protected class members and `namespace detail`
are excluded; out-of-line definitions are included; comments, string/char
literals and preprocessor lines are removed before counting. Explicit
constructors, the destructor and the move-assignment operator are counted;
deleted copy/move special members are excluded like the implicit ones. Data
members, aliases, `using` declarations, enumerators, static data members,
variable templates and macros are not function/member declarations and are not
counted; other overloaded operators are not counted as named declarations (the
move-assignment special member is the exception). "Unique" applies one more filter: duplicate signatures (for example a
declaration and its out-of-line definition, modulo a leading `inline`) are
counted once.

The scanner measures **628** public declarations in the sixteen family headers
and **649** across all eighteen (unique signatures: **594** and **615**). This
is the declaration count; it is not a feature count, because one constrained
template, alias, option-bag member or family-wide declaration can serve several
operation forms. The per-family operation coverage is the rest of this
document.

| Header | Measured declarations | Unique signatures | Notes |
|---|---:|---:|---|
| `p2p.hpp` | 79 | 79 | |
| `collective.hpp` | 61 | 61 | |
| `file.hpp` | 129 | 129 | 5 lifecycle + 7 control + 15 metadata/positioning + 6 atomicity/extent/handler + 96 data access |
| `rma.hpp` | 33 | 33 | 33 function declarations; `lock_type` is an enum and is not counted |
| `layout.hpp` | 38 | 38 | zero-size layout is `contiguous(0)`; the `empty()` factory is removed |
| `request.hpp` | 52 | 50 | owner/state machine, eleven named factories, `has_result` and the completion functions |
| `communicator.hpp` | 63 | 51 | owner/view lifecycle, factories (including `idup`/`idup_with_info`, the from-group factories and `take_communicator`), the cached `tag_ub` accessors and free functions |
| `window.hpp` | 51 | 36 | owner/view lifecycle, factories, metadata and synchronization |
| `datatype.hpp` | 29 | 24 | traits/helpers; the six public macros are not declarations |
| `operator.hpp` | 6 | 6 | `predefined_op<T>`'s constructors and accessors; the three mp1 functors (`max`, `min`, `logical_xor`), `commutative_op` and `op_traits<F>` are class/trait surface and are listed in §7.3 |
| `info.hpp` | 15 | 15 | |
| `group.hpp` | 19 | 19 | `rank_range` is an aggregate, not a counted declaration |
| `status.hpp` | 10 | 10 | |
| `environment.hpp` | 28 | 28 | |
| `spawn.hpp` | 9 | 9 | |
| `buffer.hpp` | 6 | 6 | |
| **Sixteen-header total** | **628** | **594** | |
| `error.hpp` | 14 | 14 | |
| `vocabulary.hpp` | 7 | 7 | |
| **All-header total** | **649** | **615** | |

`vocabulary.hpp` provides `count_t`, `capacity`, `region<T>`, `blocks`,
`span_from` and `empty_data`; `status.hpp` also provides the `tag`/`any_tag`/
`tag_ub_min` tag vocabulary; `error.hpp` provides the `context`/`mpi_error`
surface. The
private/detail-inclusive census (which also counts private helpers, for example
the `window.hpp` private members and the `error.hpp` formatting helpers) is a
different tracking metric and is not reported here.

Some coverage dimensions are stated directly in the tables and are
reproducible from them: 4 send modes × 3 lifetimes × 3 forms; 16 collective
families × 3 lifetimes; 5 blocking and 4 nonblocking/persistent receive
policies; 42 RMA origin/result/target shapes; 129 file declarations, of which
96 are data access; and the per-family descriptors in §§2–7.

The complete verification suite is **303** non-nightly CTest entries in the
default configuration (**304** with `MP1_TEST_EXTRA_RANKS=ON`, which adds the
5-rank persistent-collective entry), **102** of which are compile-fail entries
(**100** contract probes + 2 harness self-checks), plus **66** compile-pass
controls and **9** compile-all API sources. The tree has **93** unit-test
sources with **533** Boost test cases; the full build compiles **285** C++20
translation units and has **425** build edges when tests, examples and perf are
enabled (the release build enables `MP1_BUILD_PERF`, whose `perf_calls` gate is
one of the 303 entries); the wrap-coverage gate covers **177** MPI symbols
across 34 implementation files.

## How to read this matrix

Every operation has three independent dimensions:

1. **Family** — the MPI operation (`send`, `allgatherv`, `get_accumulate`,
   `read_at`, ...).
2. **Lifetime** — blocking, nonblocking (`i*`) or persistent (`*_init` +
   `start`/`wait`).
3. **Data form** — the mp1 spelling of the buffer position.

The position vocabulary is four spellings, not four predicates:

| Position | Accepted buffer | Used by |
|---|---|---|
| value | one object of its MPI datatype | p2p sends/value receives, fixed collectives, reductions, file value access |
| range | contiguous sized range (`std::span`, `std::array`, C array, `std::vector`, `std::string`); a non-contiguous sized sequence on a send side | p2p, fixed collectives, reductions, file range access |
| region | `region<T>`: one occurrence of a `layout<T>` | p2p, fixed collectives, reductions, RMA, file |
| in-place | `mp1::in_place` | the MPI-legal side of a collective |

The public predicate layer is `element`, `element_value`, `element_buffer`,
`element_sequence`, `resizable_element_buffer`, `borrowable` and
`buffer_at<Role>`. The four position spellings have named predicates
`send_source`, `recv_sink`, `in_place_send_source` and `in_place_recv_sink`;
they are the predicate form of the same four spellings, not a fifth position.
Public free functions live in namespace `mp1` and are found by ADL from an
`mp1`-associated argument; types, variables and helpers without such an
argument need `mp1::` qualification (see README's
[Namespaces and ADL](README.md#namespaces-and-adl)).

A cell marked ✓ means a declaration exists; it does not mean every MPI argument
combination is expressible. `B/NB/P` abbreviates blocking / nonblocking /
persistent.

Common notation:

- `count_t` is `std::ptrdiff_t` for operation counts and collective displacements; file offsets use `offset_t`. `capacity` is a distinct strong upper bound for
  a resizable-container receive.
- `region<T>` is `{T* data, const layout<T>& map}`; `blocks` is
  `{std::span<const count_t> counts, std::span<const count_t> displacements}`.
- `layout<T>` is a committed derived MPI datatype over `T`; a typed variable
  descriptor is `std::span<const layout<T>>` plus `std::span<const count_t>`
  element displacements.
- `predefined_op<U>` is the typed RMA operator; `U` is `T` for a predefined
  `T` and `basic_component_t<T>` for a homogeneous derived `T`
  (`predefined_op<int>` for `std::pair<int,int>`).
- A used **position is empty only when its total byte size is zero** (MPI 4.1
  §5.1.11), not when its element count is zero: a zero count with a
  positive-size datatype and a positive count with a zero-size layout or native
  datatype are both empty. A zero-byte collective/reduction position may have
  a null pointer and is lowered to the per-slot dummy; its wire count is 0 when
  the base/single datatype size is 0, so a positive count of a zero-size
  datatype never reaches the provider. A P2P value/range keeps the caller's
  count, a P2P `region<T>` with `layout<T>::contiguous(0)` keeps count 1 with the
  empty datatype and the nullable caller pointer, and file/RMA access uses its
  stable dummy address. A non-empty position with a null pointer is
  `std::invalid_argument` before any MPI data call or managed `MPI_Op_create`
  (an RMA origin under `MPI_NO_OP` is ignored, not inspected). The exact-byte
  preflight of a flat position, and the collective lowering of any position,
  resolves `native_datatype<T>()`; for a reflected aggregate on first use this
  constructs and commits the synthesized datatype before the null check can
  reject the call. The commit is process-local and one-time and transfers no
  data, so the guarantee is "before any MPI data call", not "before any MPI
  call". A typed reduction position is measured by `layout::byte_size()` in
  that preflight and constructs no datatype there. Ignored roles,
  `MPI_IN_PLACE` and the in-place send slot are never inspected. §3.4 covers
  the reduction preflight.
- A contiguous lvalue buffer is borrowed (zero-copy). A non-contiguous
  sequence of trivially copyable elements is staged; any other element type is
  described by a typed absolute-address datatype passed with `MPI_BOTTOM`
  (MPI 4.1 §5.1.12) and borrows the caller's elements until the send completes.
  A nonblocking or persistent send requires a borrowable
  buffer; a blocking call may borrow a temporary for the duration of the call.
- A borrowed buffer must stay alive and must not be resized, reallocated or
  freed until the request completes (or, for a blocking call, until the call
  returns).

## 1. Owner, view and value model

| Type | Model | Copy / move | Teardown | No-MPI escape |
|---|---|---|---|---|
| `communicator` | Move-only owner of `MPI_Comm` with a private `mpi_communicator` view. | Move-only; `view()` returns a copy. | `free`/`disconnect` per fixed policy. | `release()` |
| `mpi_communicator` | Copyable borrowed view with cached size/rank/inter/remote-size. | Copyable. | Never calls MPI. | n/a |
| `window` | Move-only owner of `MPI_Win` with a private `mpi_window` view. | Move-only; `view()` returns a copy. | `MPI_Win_free`; may block. | `release()` |
| `mpi_window` | Copyable borrowed view with cached window attributes. | Copyable. | Never calls MPI. | n/a |
| `file` | Owner-only, no view; member data access over one shared state. | Move-only. | Finish split, then collective close; may block. | `release()` |
| `group` | Owner-only, move-only, no view. | Move-only. | `MPI_Group_free` (`MPI_GROUP_EMPTY` not owned). | None public |
| `info` | Copyable owner. | Copy duplicates (`MPI_Info_dup`); move steals. | `MPI_Info_free`; always available. | None |
| `request` | Move-only owner; a completed `idup` request owns its `MPI_Comm` result until `take_communicator`. | Move-only. | State-machine drop policy; may wait/cancel, or free a `result_pending` duplicate. | `release()` |
| `message` | Move-only owner of a matched message. | Move-only. | Discards a still-owned message; `noexcept`. | `release()` |
| `layout<T>` | Copyable owner of one committed datatype. | Copy duplicates; move transfers. | `MPI_Type_free`. | None; `native_handle()` borrows |

- A view is valid only while the object it borrows is live. `free()`,
  `disconnect()`, the owner's destruction and `MPI_Finalize` invalidate every
  view. `is_valid()` is a null-sentinel test, not a liveness check.
- Constructing a view from a valid non-null raw handle requires active MPI;
  default and null-handle construction are MPI-free.
- `release()` is the no-MPI-call escape where the type provides one. For
  `communicator`/`window` it transfers the raw handle (and every raw-MPI
  obligation) to the caller; views copied before it stay valid while the caller
  keeps the handle alive. `request::release()` and `message::release()` leak or
  abandon their handle/payload rather than transferring it. `file::release()`
  transfers the `MPI_File` but may call `MPI_Type_free` once when an active split
  owns a duplicated region layout. `layout` has no public release;
  `native_handle()` borrows the committed datatype.
- A `request` exposes two independent ownership axes: `is_valid()`/`operator
  bool` report whether a live `MPI_Request` handle is held (a
  `result_pending` idup request has none, so both are false), while
  `state()`/`has_result()` report the retained-result axis. `take_communicator`
  is the only transfer; destroying a `result_pending` request disposes of the
  duplicate with a collective `MPI_Comm_free`.
- An owner's move assignment over a live target is a teardown with the same
  policy as destruction; it may be collective and/or blocking.
- `info` is the one handle owner whose operations are always available (MPI
  4.1 §11.4.1, Table 11.1). All other MPI-backed objects throw
  `std::runtime_error` when used before init or after finalize; their
  destructors abandon (and count) a handle instead of calling MPI.

## 2. Point-to-point

### 2.0 Tag vocabulary

Every P2P tag parameter and the tag of `communicator::create_group` /
`intercomm_create` is `mp1::tag` (an aggregate with public `int value`, default
0, and all six comparisons; there is no implicit conversion from `int`).  This
is **source-breaking**: spell `mp1::tag{n}` and `mp1::any_tag`.
`status::tag()` also returns `mp1::tag` (source-breaking; use `.value` for a
raw `int`), while `status::source()` remains `int`.

| Name | Type | Meaning |
|---|---|---|
| `mp1::tag{n}` | `tag` | One tag value; the declaration default is `{}` (tag 0). |
| `mp1::any_tag` | `tag` | `MPI_ANY_TAG`; receive/probe accepted, send/factory rejected. |
| `mp1::tag_ub_min` | `tag` | 32767; MPI's guaranteed inclusive minimum `MPI_TAG_UB` (MPI 4.1 §3.2.3). |
| `mp1::tag_ub(c)` / `c.tag_ub()` | `int` | The communicator's cached inclusive `MPI_TAG_UB`; no MPI call. |

`stringtag` is a distinct vocabulary: `communicator::create_from_group` and
`communicator::intercomm_create_from_groups` take a `std::string_view`, not
`mp1::tag`, and there is no conversion between them. The view must contain no
embedded NUL and be shorter than `MPI_MAX_STRINGTAG_LEN` (the macro includes
the terminating NUL); an empty string is legal. It is borrowed only for the
blocking call, and every participating process must pass the same value.

Send tags must satisfy `0 <= value <= tag_ub(c)`; receive and probe tags also
accept `any_tag`. The `create_group` and `intercomm_create` tags use the same
wildcard/range rule against the source communicator's cached bound, so
`any_tag`, a negative value and a value above the bound are rejected while 0 is
allowed. Validation is local (no MPI call) and throws `std::invalid_argument`
for a negative value, `any_tag` on a send or a factory, or a value above the
cached bound. The bound is queried once per non-null `mpi_communicator` view
with `MPI_Comm_get_attr(MPI_TAG_UB)` at construction (world/self on first use
and every owning factory or adoption path), cached with the other metadata, and
falls back to `tag_ub_min` when the attribute is unavailable.

### 2.1 Send modes

| Mode | Blocking | Nonblocking | Persistent |
|---|---|---|---|
| standard | `send` | `isend` | `send_init` |
| buffered | `bsend` | `ibsend` | `bsend_init` |
| synchronous | `ssend` | `issend` | `ssend_init` |
| ready | `rsend` | `irsend` | `rsend_init` |

Each mode has exactly three forms: value, range, region. There are no
pointer + count, iterator-pair, `*_empty`, or per-form duplicate names.
Buffered sends require an attached process or communicator buffer; ready sends
carry MPI's usual matched-receive precondition. A nonblocking/persistent value
or range send requires an lvalue (or a borrowed range); a region is duplicated
into request-owned storage at initiation. A contiguous range is zero-copy. A
non-contiguous sequence of trivially copyable elements is staged; any other
element type is described by a typed absolute-address datatype passed with
`MPI_BOTTOM` (MPI 4.1 §5.1.12) and borrows the caller's elements until the
send completes. The non-contiguous sequence send is never a byte image.

Two documented exclusions: a built-in character array (`char[N]`, `wchar_t[N]`)
is not a range send form (use `std::span<char>`, `std::array<char,N>`,
`std::string`, or `std::string_view`), and an auto-size receive is not a
pre-posted receive: it uses `MPI_Mprobe`/`MPI_Mrecv`, so a ready-mode send
(`rsend`) has no posted receive to match and the pair is an erroneous program.
Use a fixed, region or bounded receive with a ready-mode send.

### 2.2 Receive policies

| Policy | Blocking `recv` | Nonblocking `irecv` | Persistent `recv_init` | Spelling |
|---|---|---|---|---|
| value | ✓ | ✓ | ✓ | `T&`, exactly one element |
| fixed | ✓ | ✓ | ✓ | `std::span<T>`, `std::array<T,N>` or `T[N]`; the extent is the capacity |
| region | ✓ | ✓ | ✓ | `region<T>`; count 1 and a non-overlapping receive map |
| auto-size | ✓ | — | — | contiguous resizable container; `MPI_Mprobe` + count + `MPI_Mrecv` |
| bounded | ✓ | ✓ | ✓ | contiguous resizable container + `capacity{n}` |

A raw `count_t` in the third argument of a resizable receive is a deleted
function, so `recv(c, v, n, source)` never silently reads `n` as a source rank.
`std::vector<bool>` is not a container form, and sequence/associative
containers are not receive targets. The auto-size receive consumes a matched
message with `MPI_PACKED` if a post-match type/size/resize/allocation failure
would otherwise strand it.

### 2.3 Matched receive and probes

| Operation | Result | Notes |
|---|---|---|
| `probe(c, source, tag = any_tag)` | `status` | Blocking; `MPI_ANY_SOURCE`/`any_tag` allowed. |
| `iprobe(c, source, tag = any_tag)` | `std::optional<status>` | `nullopt` = no match; a successful `MPI_PROC_NULL` probe returns the synthesized §3.10 status. |
| `mprobe(c, source, tag = any_tag)` | `message` | Owning matched message plus probed status. |
| `improbe(c, source, tag = any_tag)` | `std::optional<message>` | `nullopt` = no match; an engaged `MPI_MESSAGE_NO_PROC` is a match. |

`message` is move-only: `is_valid()`, `is_no_proc()`, `probed_status()`,
`native_handle()`, `discard()` and `release()`; the destructor discards.
`mrecv`/`imrecv` take `message&` and consume it exactly once. `mrecv` has five
policies and `imrecv` four (no auto-size).

### 2.4 Combined send-receive and buffers

| Family | Blocking | Nonblocking | Send/receive forms |
|---|---|---|---|
| `sendrecv` | ✓ | — | value/range/region on each side |
| `sendrecv_replace` | ✓ | — | one in-place buffer |
| `isendrecv` | — | ✓ | value/range/region on each side |
| `isendrecv_replace` | — | ✓ | one in-place buffer |

The nonblocking combined forms are wait-only and borrow both buffers. A
trivially identifiable overlap (the same start address, or two positive byte
extents that intersect) is rejected before MPI; general disjointness is not
proven. There is no persistent combined form (MPI defines none). The status
returned by `isendrecv`/`isendrecv_replace` is the provider's receive status
(MPI 4.1 §3.7.2 does not require every provider to preserve tag/count, and for
an ignored direction even source, so treat those fields as provider-defined)
unless the receive source is `MPI_PROC_NULL`: then the wrapper returns the
synthesized §3.10 status (source `MPI_PROC_NULL`, tag `any_tag`, count 0,
success) with normalized provenance.

| Buffered-send surface | Declaration shape |
|---|---|
| process attach/detach | `buffer_attach(void*, count_t)`, `buffer_detach() -> std::pair<void*, count_t>` |
| process flush | `buffer_flush()`, `buffer_iflush() -> request` |
| communicator buffer | `attach_buffer(c, void*, count_t)`, `detach_buffer(c)`, `flush_buffer(c)`, `iflush_buffer(c)` |
| RAII guard | `bsend_buffer(int size)` |

`mp1::buffer_automatic` selects MPI-managed buffering. Buffer-flush requests
are wait-only; `cancel()` throws `std::logic_error`.

### 2.5 Explicit empty message

`mp1::empty_data` is an `inline constexpr std::span<std::byte>` with size 0
(vocabulary header, no MPI call to construct).  Sending it is an ordinary
zero-count byte message: it has a normal envelope, is matchable/probable, and
matches any zero-count receive; `probe(...).get_count<std::byte>()` is 0.  It
is also accepted as a receive target because the span's elements are writable.
There is no special empty send/receive/collective path and no `blocks{}`
replacement: collectives still require every rank to call in order.

## 3. Collectives

Every rank must call each collective. Persistent `*_init` is nonlocal and
follows the collective ordering rules. Nonblocking names are `i<family>`;
persistent names are `<family>_init`.

### 3.1 All 16 families × lifetime

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

`reduce_local(op, send, recv)` is the one local reduction: no communicator and
no in-place form.

### 3.2 Forms by family group

| Group | value | range | region | dense `blocks` | typed layout spans |
|---|---|---|---|---|---|
| Fixed: `bcast`, `gather`, `scatter`, `allgather`, `alltoall` | ✓ | ✓ | ✓ | n/a | n/a |
| Reductions: `reduce`, `allreduce`, `scan`, `exscan` | ✓ | ✓ | ✓ | n/a | n/a |
| Variable: `gatherv`, `scatterv`, `allgatherv`, `alltoallv` | single-block positions only | single-block positions only | single-block positions only | ✓ | ✓ |
| `reduce_scatter` | ✓ | ✓ | ✓ | counts span | n/a |
| `reduce_scatter_block` | ✓ | ✓ | ✓ | scalar count | n/a |
| `barrier` | n/a | n/a | n/a | n/a | n/a |

- A fixed-family or reduction send position also accepts a non-contiguous sized
  sequence (staged). Receive positions require a contiguous writable range, a
  value, a region or `in_place`.
- In the variable families, an aggregate position must be a flat contiguous
  element range because the per-block descriptor owns placement; its span is a
  capacity checked by the descriptor fit. A single-block position (`gatherv`
  send, `scatterv` recv, `allgatherv` send) accepts a value, a contiguous
  range, or a region; its span length (or `layout.size()`) is the MPI block
  count, not a capacity. For `scatterv` it must equal this rank's
  `sendcounts[rank]` at the root; for `allgatherv` it must match this rank's
  entry in every peer's `recvcounts`.
- Persistent families take fixed buffers only: a value, `std::span`,
  `std::array`, C array, `region`, or a typed descriptor array. Resizable and
  non-contiguous containers are rejected at compile time.
- `reduce_scatter` takes `std::span<const count_t>` receive counts.
  `reduce_scatter_block` takes a scalar `count_t`. Both check the exact
  receive size (out-of-place) or full reduced size (in-place) before MPI. The
  value form of either family is one element, so it is usable only when the
  total input size is one: `sum(recvcounts) == 1` for `reduce_scatter`, or
  `recvcount * comm.size() == 1` for `reduce_scatter_block` (that is, a
  one-rank communicator with `recvcount == 1`). Use a range or region for any
  larger block.
- A `region<T>` position is always one occurrence of the layout over the whole
  buffer. It is not repeated per rank: for a collective whose MPI signature
  repeats a block per rank (for example `allgather`/`alltoall`), build one
  layout that covers the whole receive buffer (`layout<T>::contiguous(n,
  block)`), or use a flat range with the family's counts.

### 3.3 Variable descriptors

| Descriptor | Spelling | Lifetime |
|---|---|---|
| dense | `blocks{.counts = counts_span, .displacements = displacements_span}` | Blocking borrows. Nonblocking/persistent copy both arrays at initiation. |
| typed | `std::span<const layout<T>> layouts` + `std::span<const count_t> displacements` | Blocking borrows. Nonblocking/persistent duplicate the layouts they stage. |

The predecessor's layout-container and byte-displacement types are not part of
the typed surface. The dense descriptor is `blocks`; the typed descriptor is the two spans.
Dense `blocks` always requires **both** spans, each with the peer count; an
empty displacement span is not the all-zero shortcut for the dense form. A
typed displacement span defaults to empty (all-zero element displacements)
except on the `scatterv` and `alltoallv` send sides, where it is mandatory
because it precedes the receive position. On every side that MPI uses on the
calling rank, both descriptors are validated before MPI: equal array lengths
(peer count), dense
counts and displacements (and a non-empty typed block's displacement)
non-negative, dense `displacement + count` fit, and typed true-extent fit. A
descriptor on an ignored side (for example the root-only `gatherv` receive or
`scatterv` send on a non-root) is not inspected.

`blocks{}` is the **ignored-side** zero-block descriptor. It is not accepted on
a used variable-collective side when the peer count is greater than zero; spell
an empty used side as peer-length zero `counts` and zero `displacements`
(dense), or as peer-length empty `layout<T>` entries (typed). This is a
descriptor-shape rule, not an empty-message restriction: a zero-total used side
is a valid collective participation.
A block that addresses no location is exempt from the fit comparison because
MPI ignores its placement: a dense block with `count == 0` skips only the fit
comparison (its displacement must still be non-negative and representable as a
byte offset), and a typed block whose layout `byte_size() == 0` skips the fit
comparison, but its displacement must still be non-negative and representable.

### 3.4 Reductions and in-place

| Operator form | Datatype passed to MPI | Element image |
|---|---|---|
| functor mapped to a predefined operation | `native_datatype<T>()` | MPI applies the predefined operation; no callback |
| mp1-managed functor | committed resized image of the native typed map shifted to true lb 0; typed leaves, extent `sizeof(T)`; buffer pointer shifted by the native true lb | one complete `T` per element; only the native type-map bytes are transmitted; padding and unlisted members are not; `reduce_scatter[_block]` requires true lower bound 0 and true extent `sizeof(T)` |
| raw `MPI_Op` (predefined or caller-created) | `native_datatype<T>()` | the callback must honor the datatype's true lower bound and type map |

The mapping is MPI 4.1 §6.9.2 itself: `std::plus`/`std::multiplies` map for C
integer, floating and `std::complex`; `mp1::max`/`mp1::min` for C integer and
floating; `std::bit_*` for C integer and `std::byte`; `std::logical_*` and
`mp1::logical_xor` for C integer and `bool`. The C++ datatypes are
`MPI_CXX_BOOL` and `MPI_CXX_*_COMPLEX`; the Fortran integer and multi-language
categories are not reachable from a C++ element type. The predefined
operation's result semantics are MPI's: floating `max`/`min` NaN behavior and
the integer logical result value are whatever MPI's operation does. A pair
outside the row uses the managed callback, which requires a stateless,
nothrow-invocable functor and a trivially copyable `T`; the functor must not
inspect padding or unlisted members, and an over-aligned `T` (`alignof(T) >
alignof(std::max_align_t)`) is unsupported on that path. The `mp1::op_traits`
truth table is integral-only for `mp1::max`/`min`, arithmetic for
`std::plus`/`std::multiplies`/`std::logical_and`/`std::logical_or` and
`mp1::logical_xor`, float-only for the explicit `std::complex`
specializations, and integral-or-`std::byte` for `std::bit_*`; the user
`commutative_op` tag remains the opt-in. The trait is consulted only on the
callback path; a mapped §6.9.2 pair never creates an `MPI_Op`.

The standard-library functors are the general reduction surface: `std::plus`,
`std::multiplies`, `std::bit_and`, `std::bit_or`, `std::bit_xor`,
`std::logical_and` and `std::logical_or` (including the transparent `void`
specializations) are accepted as reduction operators and select the same
predefined `MPI_Op` where the operation/type pair is mappable, and the callback
otherwise. `mp1::max`, `mp1::min` and `mp1::logical_xor` are the only mp1
functor names; they have no standard-library object equivalent. Their element
type defaults to `void`, so the transparent spellings `mp1::max{}`,
`mp1::min{}` and `mp1::logical_xor{}` deduce the operand from the reduction
element type (the typed `mp1::max<T>{}` spelling is unchanged). The built-in
functor surface is those three mp1 names plus the seven standard-library
functors; any other stateless, nothrow-invocable `T` functor uses the managed
callback path.
`mp1::op_traits<F>::is_commutative` is used only as the `commute` argument of
`MPI_Op_create` on the callback path: the predefined mapping does not consult
it. The trait is conditional on the operand type (integral for
`mp1::max`/`min`, arithmetic for `std::plus`/`std::multiplies`/
`std::logical_and`/`std::logical_or` and `mp1::logical_xor`, float-only for the
explicit `std::complex` specializations, and integral or `std::byte` for
`std::bit_*`); a callback-only pair -- `mp1::max`/`min` on a character or
`Logical` element, a floating-point logical functor, a user-defined operand
type -- is created with that value (`commute = 0` where the C++ operation is
not commutative) unless the user opts in with `commutative_op` or an
`op_traits` specialization. For a callback-only pair, the transparent spellings the
library classifies (the `std::plus`/`multiplies`/`equal_to`/`not_equal_to`/
`logical_and`/`logical_or`/`bit_*` `void` specializations and
`mp1::max<>`/`mp1::min<>`/`mp1::logical_xor<>`) are classified by substituting
the reduction element type into the same `op_traits` row as the typed spelling,
so the two spellings cannot disagree; a mapped pair never consults the trait.
The transparent type alone is `false` (`op_traits<mp1::max<>>`,
`op_traits<std::plus<void>>`) unless the user specializes it.

In-place rules:

| Topic | Rule |
|---|---|
| Position | Receive slot for `scatter`/`scatterv`; send/single-buffer slot for the other families. |
| Communicator | Intracommunicator only; root-only for `gather`, `scatter`, `reduce`, `gatherv`, `scatterv`. |
| Agreement | `allgather`, `alltoall`, `allgatherv`, `alltoallv`, `allreduce` and `reduce_scatter_block` require every rank to choose the same form (MPI 4.1 §6.7, §6.8, §6.9.6, §6.10.1); `reduce_scatter` is the one family where MPI permits mixing. mp1 validates only the local argument. |
| Root | `bcast`, `gather`, `scatter`, `gatherv`, `scatterv` and `reduce` take an `int root`. On an intercommunicator the root's group passes `MPI_ROOT` at the root and `MPI_PROC_NULL` at the other members; the remote group passes the root's rank. |
| Empty data | A zero-total position may use null pointers, including an all-zero counts array and an empty layout. |

For reductions, a managed functor's `MPI_Op` and callback datatype are created
only after both used positions pass the exact-byte preflight; `reduce_local`
applies the same preflight to its two positions. A rejected non-empty null
therefore creates no managed operation and stages no buffer. The preflight
itself may commit a first-use reflected datatype while measuring a flat
position; see **Null positions** in "How to read this matrix". After a
successful preflight a zero-byte reduction position is presented as the
per-slot dummy with count 0 and the base datatype.

`exscan` keeps rank 0's receive buffer untouched in both forms. The out-of-place
form uses an uninitialized rank-0 scratch owned by the plan/request; the
in-place form uses `MPI_IN_PLACE` directly.

### 3.5 Persistent collectives

- `*_init` is nonlocal and ordering-sensitive; the returned request is
  inactive. Each `start()` runs one cycle; `wait()`/`test()` completes it.
- The caller's data buffers and communicator must stay alive and unmodified
  until the request is destroyed. Descriptor arrays and counts are copied at
  `*_init`, so the caller's descriptor arrays may die immediately.
- Each start re-packs a staged send and re-primes in-place staging from the
  caller's data buffer; each completion unpacks once.
- An active persistent collective request is waited for and then freed, never
  freed while active; an inactive request is freed with `MPI_Request_free`.
  `cancel()` throws `std::logic_error`.
- All sixteen families have a persistent form; only `scan`/`exscan` are
  intracommunicator-only.

## 4. RMA

### 4.1 Windows

| Operation | Spelling |
|---|---|
| create | `window::create(comm, window_create_options{.base, .size, .disp_unit, .info})` |
| allocate | `window::allocate(comm, window_allocate_options{.size, .disp_unit, .info})` |
| allocate shared | `window::allocate_shared(comm, window_allocate_options)` |
| create dynamic | `window::create_dynamic(comm, info = {})` |
| adopt / free / release | `window::adopt(MPI_Win)`, `free()`, `release()` |
| view | `mpi_window`; `window` converts implicitly and exposes `view()` |

The `create`/`allocate`/`allocate_shared`/`create_dynamic` factories are
collective over the communicator and require an intracommunicator; `adopt`
is a local adoption of an existing raw handle, not a collective. `size` is
bytes; `disp_unit` must be positive; `size == 0` is legal (and a null base is
accepted for size 0). The view caches `base()`, `byte_size()`, `disp_unit()`,
`create_flavor()`, `memory_model()`, `rank()` and `group_size()` at
construction, so those seven are MPI-free. `get_group()`, `get_info()`,
`shared_query()` and `set_info()` call MPI and require active MPI;
`attach()`/`detach()` are non-`const` because they mutate the dynamic-window
attachment set, while `set_info()` is `const` because constness protects the
view's cached state, not the shared MPI object.

| Metadata / shared-memory / attach | Notes |
|---|---|
| `get_group()`, `get_info()`, `set_info(info)` | `set_info` is collective. |
| `shared_query(rank) -> shared_segment` | `{void* base; MPI_Aint size; int disp_unit;}`; valid on create/allocate/allocate_shared, rejected on dynamic. `size == 0` is legal. |
| `attach(base, size)`, `detach(base)` | Dynamic windows only; `size > 0` and a non-null base are required. |
| `address_of(object) -> MPI_Aint` | Target address helper. |
| `set_error_handler`, `get_error_handler` | Predefined handlers only. |

### 4.2 Synchronization

| Group | Operations |
|---|---|
| Active target | `fence`, `post`, `start`, `complete`, `wait`, `test` |
| Passive target | `lock`, `lock_all`, `unlock`, `unlock_all` |
| Flush / sync | `flush`, `flush_all`, `flush_local`, `flush_local_all`, `sync` |

All fifteen are free functions taking `const mpi_window&`. `fence` is
collective. `lock` and `start` may block. `lock`/`unlock`/`flush`/`flush_local`
require a real rank; the ten data-movement calls accept `MPI_PROC_NULL` as a
normalized no-op. There is no implicit flush or epoch management.

### 4.3 Data movement (42/42 shapes)

The origin/result is a `std::span<T>` (flat native elements) or a `region<T>`
(one layout occurrence). The target is a `count_t` element count or a
`layout<T>`.

| Family | Declarations | Shapes |
|---|---:|---|
| `put`, `get`, `accumulate`, `rput`, `rget`, `raccumulate` | 2 each | origin {span, region} × target {count, layout} = 4 each |
| `get_accumulate`, `rget_accumulate` | 2 each | origin {span, region} × result {span, region} × target {count, layout} = 8 each |
| `compare_and_swap` | 1 | one element; `const T* origin`, `const T* compare`, `T* result` |
| `fetch_and_op` | 1 | one element; `const T* origin`, `T* result`, `predefined_op<T>` |

That is 18 declarations and 6 × 4 + 2 × 8 + 2 = **42** origin/result/target
shapes. The wrapper checks origin/result size against the target count or
`layout.size()` before MPI. The both-flat origin/result pair of
`get_accumulate`/`rget_accumulate`, `fetch_and_op` and `compare_and_swap` must
be disjoint; the exact byte range is checked before the call. A mixed
flat/layout or layout/layout pair leaves cross-object overlap to the caller.

A target layout must contain only relative displacements, so `layout<T>` is the
only target form. A construction-proven overlapping target layout is rejected
where the target is updated; a construction-proven overlapping result/origin
layout is rejected where the result is written. Unknown overlap provenance is
accepted.

Blocking `put`/`get`/`accumulate`/`get_accumulate` return when the call has been
made, not when the data is complete. The origin/result buffer must stay alive,
and a get result must not be read, until the next synchronization call
(`fence`, `complete`, `flush*`, `unlock*`). `get` returns `void`.

Request-based RMA (`rput`, `rget`, `raccumulate`, `rget_accumulate`) is legal
only inside `lock`/`lock_all` epochs. Completion is origin-side and reported by
`wait()`/`test()`; target visibility still requires flush/unlock. The
origin/result and target layouts are duplicated into the request payload at
initiation. `cancel()` throws `std::logic_error`; dropping waits.

### 4.4 Accumulating operators

The accumulating families take `predefined_op<U>`, with `U =
basic_component_t<T>`: `T` for a predefined `T`, `int` for
`std::pair<int,int>`. `predefined_op<T>` requires a predefined `T`; the
aggregate spelling is a compile-time error. The typed functor mapping is:

| Functor | Mapped MPI_Op | Element categories |
|---|---|---|
| `std::plus`/`std::multiplies` | `MPI_SUM`/`MPI_PROD` | C integer, floating, complex |
| `mp1::max`/`mp1::min` | `MPI_MAX`/`MPI_MIN` | C integer, floating |
| `std::bit_and`/`std::bit_or`/`std::bit_xor` | `MPI_BAND`/`MPI_BOR`/`MPI_BXOR` | C integer, `std::byte` |
| `std::logical_and`/`std::logical_or`/`mp1::logical_xor` | `MPI_LAND`/`MPI_LOR`/`MPI_LXOR` | C integer, `bool` |

The typed functor gate covers the C++ datatype categories only: `bool` is
`MPI_CXX_BOOL` and `std::complex<T>` is `MPI_CXX_*_COMPLEX`; the Fortran
integer and multi-language categories are not reachable from a C++ element
type. A raw predefined `MPI_Op` is validated against the same C++ category
row; a C or Fortran datatype variant or a multi-language category is rejected
before the MPI call (see the raw-handle rule). The mp1 functors default their element type to
`void`, so `mp1::max{}`, `mp1::min{}` and `mp1::logical_xor{}` are accepted in
the same positions and classified from `U`; `predefined_op<U>`'s functor
constructor applies the same gate.

A raw `MPI_Op` must be a standard predefined handle valid for `U`'s datatype
category; user handles, `MPI_OP_NULL`, `MPI_MAXLOC`/`MPI_MINLOC` and category
mismatches throw `std::invalid_argument` before MPI. Mixed, empty and opaque
types are compile-time errors. `MPI_REPLACE` is accepted in every accumulating
family for a predefined or homogeneous-derived `T`; `MPI_NO_OP` is accepted
only in `get_accumulate`, `rget_accumulate` and `fetch_and_op`.
`compare_and_swap` requires a predefined C++ type in the C integer, Logical or
Byte category. Atomicity is per basic element. The `accumulate_ops` window info key defaults
to `same_op_no_op`, so concurrent accumulates to the same target address must
use the same operator or `MPI_NO_OP`.

### 4.5 Deliberate RMA limits

| Limit | Effect |
|---|---|
| One target count | The origin/result size must equal the target count or `layout.size()`. |
| Single layout occurrence | Use `layout<T>::contiguous(n, block)` for a repeated block. |
| CAS/FOP | Single element only; no count, layout or range form. |
| `MAXLOC`/`MINLOC` | No `MPI_2*` pair model; rejected before MPI for RMA. |
| Heterogeneous/empty accumulating `T` | Compile-time error. |
| Unannotated opaque custom derived `T` | Compile-time error; opt in with `MP1_DATATYPE_BASIC(T,U)`. |
| Raw non-layout datatype | Not part of the typed surface; use `native_handle()`. |
| Generic window attributes, naming, `MPI_Win_call_errhandler` | Not wrapped. |

## 5. File

`mp1::file` is one owning movable handle. There is no view type and no free
data-access wrapper; every data-access operation is a member. Read operations
are `const` and write operations are non-`const`; both move the individual file
pointer when they are one of the pointer-advancing families (`read`/`write`,
`read_all`/`write_all`, `read_ordered`/`write_ordered`, and the `_shared`
families move the shared pointer). The `const` on a read protects the handle's
cached state, not the file position. Split begin/end are non-`const` because
they mutate the per-file split epoch. The 129 declarations are:

| Group | Count | Surface |
|---|---:|---|
| Lifecycle | 5 | default constructor, `file(comm, path, access_mode, info)`, move construction/assignment, destructor |
| Control | 7 | `open`, `adopt`, `close`, `release`, `is_open`, `native_handle`, `remove` |
| Metadata/positioning | 15 | `resize`, `preallocate`, `size`, `mode`, `sync`, `set_view` (byte/typed), `seek`, `seek_shared`, `position`, `get_position_shared`, `byte_offset`, `set_info`, `get_info`, `get_group` |
| Atomicity/extent/handler | 6 | `set_atomicity`, `get_atomicity`, `get_type_extent` (type/layout), `set_error_handler`, `get_error_handler` |
| Data access | 96 | 48 read + 48 write |

The 96 data-access declarations are:

| Direction | Blocking families | Nonblocking families | Split-begin families | Split-end families |
|---|---|---|---|---|
| read | `read`, `read_at`, `read_shared`, `read_all`, `read_at_all`, `read_ordered` (6) | `iread`, `iread_at`, `iread_shared`, `iread_all`, `iread_at_all` (5) | `read_at_all_begin`, `read_all_begin`, `read_ordered_begin` (3) | `read_at_all_end`, `read_all_end`, `read_ordered_end` (3) |
| write | `write`, `write_at`, `write_shared`, `write_all`, `write_at_all`, `write_ordered` (6) | `iwrite`, `iwrite_at`, `iwrite_shared`, `iwrite_all`, `iwrite_at_all` (5) | `write_at_all_begin`, `write_all_begin`, `write_ordered_begin` (3) | `write_at_all_end`, `write_all_end`, `write_ordered_end` (3) |

Every blocking/nonblocking/split-begin family has three forms: value
(`T&`/`const T&`), contiguous range (`std::span`, `std::array`, C array,
`std::vector`, `std::basic_string`, `std::string_view` for writes) and
`region<T>`. Every split-end family has two forms: value and range. A
nonblocking/split-begin write requires a borrowable buffer. A per-call region
in a nonblocking/split request is duplicated at initiation.

Units: `set_view` displacement is bytes from the file start;
`size`/`resize`/`preallocate` are bytes; `byte_offset` takes etype units and
returns absolute bytes; `seek`/`seek_shared`/`position`/`get_position_shared`
are etype units; data-access offsets are etype units. Before the first
`set_view` the view is the initial byte view (displacement 0, `MPI_BYTE`
etype/filetype). A typed `set_view` makes `native_datatype<T>()` the etype; a
typed access is rejected unless the view is unknown, the etype is `MPI_BYTE`,
or the element datatype identity matches the view etype. A typed `set_view`
rejects a negative or descending filetype typemap displacement for every
access mode and an overlapping filetype on a writable file (MPI 4.1 §14.3);
the hole-multiple-of-etype-extent rule remains a caller obligation for an
opaque/`from_raw` filetype.

Collective calls: `open`, `close`, `resize`, `preallocate`, `sync`, `set_view`,
`set_info`, `set_atomicity`, `seek_shared` and every `*_at_all`/`*_all`/
`*_ordered` family. `remove` is not collective. `set_info` rejects a null
`MPI_INFO_NULL` locally. Split collectives are a strict state machine: at most
one active begin, same-thread begin/end, matching operation and buffer. A
wrong/double/missing end, a mismatched buffer, a collective I/O call during a
split, and a close during a split throw `std::runtime_error` locally. `sync`,
`set_view`, `resize` and `preallocate` also refuse during a split because
their clauses require split operations to be complete. `set_info`,
`set_atomicity` and `seek_shared` have no completion requirement in MPI 4.1
§14.2.8/§14.6.1/§14.4.4 and remain legal between begin and end. `close()`
refuses while a wrapper-owned request is outstanding.

## 6. Lifecycle, communicator, group, info, status, error, request, spawn

| Area | Measured declarations | Surface |
|---|---:|---|
| Environment | 28 | `init`/`finalize` forms, `finalize_at_exit`, `initialized`, `finalized`, `available`, `get_version`, `get_library_version`, `get_hw_resource_info`, `query_thread`, `provided`, `is_thread_main`, `processor_name`, `wtime`, `wtick`, `wtime_is_global`, `buffer_attach`/`buffer_detach`, `bsend_buffer`, `environment`, `threading_mode` |
| Communicator | 63 | owner/view lifecycle, `dup`, `dup_with_info`, `idup`, `idup_with_info`, `split`, `split_type`, `create`, `create_group`, `create_from_group`, `merge`, `intercomm_create`, `intercomm_create_from_groups`, `adopt`, `adopt_connected`, `take_communicator`, `free`, `disconnect`, `release`, `view`, `policy`, `compare`, `abort`, `get_info`/`set_info`, the cached `tag_ub` accessors and free `tag_ub(c)`, free `get_group(c)`, `remote_group`, `remote_size`, `is_inter`, `parent`, `comm_world`, `comm_self`, `comm_null`, `set_error_handler`/`get_error_handler`, `install_recoverable_errors`; `abort` calls `MPI_Abort`; a successful return is passed through and a non-success return throws `mpi_error` |
| Group | 19 | owner lifecycle, `empty`, `is_empty`, `is_valid`, `size`, `rank`, `translate`, `compare`, `native_handle`, `get_group`, `group_union`, `group_intersection`, `group_difference`, `group_incl`/`group_excl` over `std::span<const int>`, `rank_range`, `group_range_incl`, `group_range_excl` |
| Info | 15 | owner lifecycle, `create`, `create_env`, `set`, `value`, `remove`, `size`, `key`, `is_valid`, `empty`, `native_handle` |
| Status | 10 | lifecycle, `kind`, `source` (`int`), `tag` (`mp1::tag`), `error`, `is_cancelled`, `get_count<T>()`/`get_count<T>(layout<T>)`, `get_elements<T>()`/`get_elements<T>(layout<T>)`, `native_handle` |
| Error | 14 | `context`, `mpi_error` constructor/accessors, `fail`, `require`, `throw_invalid`, `error_handler`, `set_error_handler`/`get_error_handler`, `install_recoverable_errors` |
| Request | 52 | owner/state machine, `has_result`, eleven named factories (`from_send`, `from_recv`, `from_persistent_send`, `from_persistent_recv`, `from_collective`, `from_persistent_collective`, `from_file`, `from_matched_recv`, `from_sendrecv`, `from_flush`, `from_rma`), `wait`, `test`, `get_status`, `cancel`, `start`, `release`, and the free `wait_all`/`test_all`/`wait_any`/`test_any`/`wait_some`/`test_some`/`get_status_*`/`start_all` functions over `std::span<request>` |
| Spawn | 9 | `spawn`, `spawn_with_errcodes`, `spawn_multiple`, `spawn_multiple_with_errcodes`, plus `spawn_multiple_options::{add,size,commands,maxprocs,infos}` |

`error.hpp` is outside the 16-header inventory but part of the public surface.
Its 14 measured declarations are the constructor and accessors of
`mpi_error`, the `context` constructors, the three error-layer helpers,
`error_handler` and the three error-handler functions. An all-scope scan that
also counts private/detail formatting helpers is a different metric and is not
reported here.

### 6.1 Request completion and PENDING

`wait_all`/`test_all` return one status per slot and do not throw
`MPI_ERR_IN_STATUS`; a per-slot error is reported in `status::error()` (MPI 4.1
§3.7.5). `MPI_ERR_PENDING` is the per-slot code for an operation that "has
neither failed nor completed" (§3.7.5). mp1 returns a PENDING status for that
slot but does not reap it: the request stays active with its handle and payload,
and the PENDING status carries only `error()` (source/tag ANY, count/elements 0
and cancellation false, all without an MPI call). Call the completion function
again.

| Call | Completion result | PENDING slot |
|---|---|---|
| `wait_all` | one status per slot; per-slot errors in `error()` | PENDING status returned; slot stays active; call again |
| `test_all` | `nullopt` when not all complete (no request modified); otherwise statuses | defensive: `flag == true` means all completed; same shared transaction |
| `wait_some`/`test_some` | completed `(index, status)` list; `nullopt`/empty when none | omitted from the list and left active |
| `wait_any`/`test_any` | one completion, or `MPI_UNDEFINED`/`nullopt` | no PENDING status is produced |

A committed persistent slot becomes inactive and valid for the next `start()`.
A PENDING persistent slot was not committed, so it is still active and
`start()` throws `std::logic_error` until a later completion reaps it. A
function-level error that is not `MPI_ERR_IN_STATUS` is separate: `wait_all`
and `wait_some`/`test_some` abandon the active slots and leave inactive
persistent slots valid, while `test_all` with `flag == false` leaves every slot
untouched (a failure after `flag == true` abandons the active slots).

A successful `communicator::idup`/`idup_with_info` slot is the one
result-producing completion: it enters `request_state::result_pending`, holds
no live `MPI_Request` handle, and keeps the new `MPI_Comm` in its payload.
`request::has_result()` reports that state (no MPI call), and
`mp1::take_communicator(request&)` transfers the duplicate into an owning
`communicator`. The single and bulk completion signatures are unchanged and
return their normal statuses; a `result_pending` slot is inactive for every
bulk function (empty status in `wait_all`/`test_all`, omitted or
`MPI_UNDEFINED` in the any/some forms), and it cannot be in the PENDING state.
`is_valid()`/`operator bool` remain handle-only and are false in
`result_pending`; the result axis is separate. Destroying the request without
taking the result performs `MPI_Comm_free` (collective, so
`drop_would_block()` is true); `release()` leaks it without an MPI call. No new
`request_kind` and no new request factory exist: the idup request is
`request_kind::collective`.

### 6.2 Status provenance

The accessors decide from the producing operation family, never from the
`MPI_SOURCE`/`MPI_TAG`/`MPI_ERROR` field values. MPI 4.1 §3.7.3 defines the
empty status as ANY/ANY/SUCCESS and configures it internally so count queries
return 0; a provider may leave the undefined fields of a file or send status at
those same values.

| Provenance | `kind()` | `source()`/`tag()` | count/elements | cancellation | `native_handle()` |
|---|---|---|---|---|---|
| receive, `probe`/`iprobe`/`mprobe`, matched receive, combined send-receive receive side (real source) | native | provider-defined | query MPI; defined | query MPI | provider status; query while MPI active |
| `MPI_PROC_NULL` receive, `probe`/`iprobe` or matched receive; combined send-receive receive side | normalized | source `MPI_PROC_NULL`, tag `any_tag` | 0 without MPI; standard-defined | false without MPI | synthetic; must not be queried |
| file data access | native | normalized to ANY | query MPI; defined (MPI 4.1 §14.4.1) | query MPI | provider status with ANY envelope; query while MPI active |
| send | native | normalized to ANY | 0 without MPI; undefined for a send | query MPI | provider status with ANY envelope; count not meaningful |
| PENDING | native | normalized to ANY | 0 without MPI | false without MPI | must not be queried |
| collective, RMA or flush completion; default status | normalized | normalized to ANY | 0 without MPI | false without MPI | synthetic; must not be queried |

A normalized status is not an MPI empty status: its `native_handle()` is a
synthetic handle that MPI did not configure (ANY/ANY/SUCCESS for a
collective/RMA/flush/default status, `MPI_PROC_NULL`/ANY/SUCCESS for the
`MPI_PROC_NULL` status), and it must not be passed to `MPI_Get_count`,
`MPI_Get_elements`, `MPI_Test_cancelled` or another MPI status query. A native
count/elements/cancellation query requires active MPI and throws
`std::runtime_error` after finalize; normalized and PENDING accessors make no
MPI call. `error()` is meaningful for an `MPI_ERR_IN_STATUS` array completion;
a PENDING status is the exception, carrying only `MPI_ERR_PENDING`.

### 6.3 Spawn types

`spawn_options` is `{int maxprocs = 1; command_line command; mp1::info info;}`;
`command_line` is `std::vector<std::string>` (program first). `spawn_result` is
`{communicator intercomm; std::vector<int> errcodes;}`. The `_with_errcodes`
forms take an optional `errcodes_length`: pass the root's `maxprocs` (or
`sum(maxprocs)`) on every rank when a non-root payload does not repeat it;
otherwise each rank sizes the array from its own `maxprocs`, so all ranks must
pass the same lengths. `spawn_multiple_options` keeps
`commands`/`maxprocs`/`infos` aligned through `add`; `size()`, `commands()`,
`maxprocs()` and `infos()` are the read surface. Only the root's
payload is significant; mp1 validates the all-rank properties (valid
intracommunicator, in-range root) and adds no preflight agreement collective.
The returned intercommunicator has `release_policy::disconnect`.

## 7. Layout, datatype and operator

### 7.1 layout<T>

`layout<T>` is a copyable owner of one committed `MPI_Datatype` and its
construction facts. A default-constructed layout is invalid; `contiguous(0)` is
the valid zero-size layout.  The former `layout<T>::empty()` factory is removed
(**source-breaking**); the explicit empty message is `mp1::empty_data`.  A copy
duplicates the datatype; a move transfers it.

Factories: `contiguous(count)`, `contiguous(count, block)`,
`vector(count, blocklength, stride)`, `strided_vector(count, blocklength,
byte_stride)`, the `vector_options<T>`/`strided_vector_options<T>` option-bag
forms, `indexed`, `hindexed`, `indexed_block`, `hindexed_block`, `subarray`
(each with and without a block layout), `from_raw`, and `contiguous(0)` for the
zero-size layout.  There is no `empty()` factory.

Parameter builders: `indexed_parameter`, `hindexed_parameter`,
`indexed_block_parameter`, `hindexed_block_parameter`, `subarray_parameter`.
Each is an alias of one shared template and exposes:

- `entry_type` — `std::pair<count_t, count_t>` for indexed,
  `std::pair<count_t, MPI_Aint>` for hindexed, `count_t`/`MPI_Aint` for the
  block forms, `std::array<count_t, 3>` for subarray;
- default construction, explicit `std::span<const entry_type>` construction,
  the relevant `add()` overloads, and (subarray only) `order(array_order)` /
  `order()`.

The builders are not initializer-list constructors; a braced listing is
rejected. All factories and parameter `add()` calls reject a negative count,
blocklength, stride or displacement with `std::invalid_argument`, even where
MPI itself would accept a negative displacement in a particular constructor;
`byte_resize` permits a negative lower bound but rejects a negative extent.
Accessors: `is_valid`, `size`, `byte_size`, `byte_extent`,
`byte_lower_bound`, `byte_upper_bound`, `true_byte_lower_bound`,
`true_byte_extent`, `true_byte_upper_bound`, `dense`, `is_base_repetition`,
`has_overlapping_entries`, `byte_resize`, `resized`, `native_handle`, `swap`.
For a zero-size layout the true bounds are the canonical empty geometry
`true_lb == true_extent == true_ub == 0` (MPI 4.1 §5.1.8 leaves them undefined
for an empty typemap); the extent markers are unchanged, so
`true_byte_upper_bound` cannot throw for a valid layout.

`layout_facts` is the optional structural-fact payload of `from_raw`
(`{bool packed; bool base_repetition; bool has_overlapping_entries;}`). The
`vector_options<T>` and `strided_vector_options<T>` aggregates are
`{count, blocklength, stride/byte_stride, const layout<T>* block}`.

`layout<T>::from_raw(type, elements, facts = {})` is the one raw-datatype
adoption escape; there is no general raw-datatype bridge. The handle must be a
derived datatype (a predefined named, F90 or value-index handle throws
`std::invalid_argument` before any commit or free), `elements` is the
application `size()` and never an MPI count, and the type map must be an
occurrence of `T` (caller precondition, not validated). The facts
argument may only assert the three structural predicates MPI cannot query.
Ownership transfers on entry; on success the layout frees the handle exactly
once, while a pre-adoption rejection leaves ownership with the caller. The
spellings `from_raw(t, n)`, `from_raw(t, n, {})` and a designated facts
aggregate all compile.

### 7.2 Datatype

Public names: `datatype_traits<T>` (customization), `native_datatype<T>()`,
`struct_builder<T>`, `base_struct_builder<T>`, `has_datatype_v<T>`,
`is_predefined_datatype_v<T>`, `basic_component_t<T>`,
`is_homogeneous_datatype_v<T>`, `basic_component_traits<T>`.

Macros (six): `MP1_DATATYPE_TRAITS`, `MP1_DATATYPE_TRAITS_PACKED`,
`MP1_DATATYPE_BASIC`, `MP1_REFLECTION`, `MP1_REFLECTION_EMPTY`,
`MP1_REFLECTION_TEMPLATE`. The reflection member limit is 119; a longer list
fails a compile-time assertion that names the limit. `MP1_REFLECTION` requires
a default-constructible aggregate whose named non-static data members are
accessible at the macro's scope; `MP1_REFLECTION_EMPTY` covers a zero-member
type and `MP1_REFLECTION_TEMPLATE` the generic template. A pointer member has
no datatype mapping; a `const` member is not guarded but is a poor receive
target.

`native_datatype<T>()` returns the trait handle directly. The extent and
true-bound validation runs when a `layout<T>`/buffer consumes
`datatype_info_of<T>()`, so a direct `native_datatype<T>()` call is
unchecked. `MP1_DATATYPE_BASIC(T,U)` is an RMA accumulating-family annotation
only: it does not map a datatype, alter the wire image, or replace the
predecessor's byte shortcut. `MP1_DATATYPE_TRAITS_PACKED` is the custom
byte-contiguous mapping macro.

### 7.3 Operator

Public mp1 reduction functors: `max`, `min` and `logical_xor` (each template on
the element type, defaulted to `void`; `mp1::max{}` is the transparent spelling
that deduces its operands and `mp1::max<T>{}` fixes them). The standard-library
`plus`, `multiplies`, `bit_and`, `bit_or`, `bit_xor`, `logical_and` and
`logical_or` functors (including `void` specializations) are accepted and map
identically, so they are the general typed reduction surface; `max`, `min` and
`logical_xor` have no standard-library object equivalent and remain mp1-only.
The operator header also provides the `commutative_op` user opt-in tag,
`op_traits<F>` with the explicit built-in truth table, and `predefined_op<T>`
with `element_type`, `native_handle()` and `is_no_op()`. The managed callback
functor contract is: empty/stateless, nothrow-invocable for `T`, immutable, no
MPI calls, and no access to padding or unlisted members.
`op_traits<F>::is_commutative` is consulted only for the `commute` argument of
`MPI_Op_create` on that callback path.

### 7.4 Public constants

| Name | Type | Meaning |
|---|---|---|
| `comm_type_shared`, `comm_type_hw_guided`, `comm_type_hw_unguided`, `comm_type_resource_guided` | `int` | `MPI_Comm_split_type` selectors (MPI 4.1 §7.4.2); `split_type` also accepts implementation-specific positive values. |
| `bsend_overhead` | `int` | `MPI_BSEND_OVERHEAD`; a buffered-send buffer needs this many bytes beyond the payload size, and `bsend_buffer(n)` allocates `n + bsend_overhead` bytes. |
| `buffer_automatic` | `void* const` | `MPI_BUFFER_AUTOMATIC`; selects MPI-managed buffering for `buffer_attach`/`attach_buffer` (§2.4). Any size is accepted with it. |
| `any_tag`, `tag_ub_min` | `tag` | The `MPI_ANY_TAG` wildcard and the guaranteed minimum tag bound (§2.0). |
| `empty_data` | `std::span<std::byte>` | Explicit zero-count message; `constexpr`, no MPI call, no special path (§2.5). |

`buffer_automatic` is not `constexpr` because a provider may define the sentinel
as a cast from an integer pointer value. The `comm_type_*` and `bsend_overhead`
values are `constexpr int` mirrors of the corresponding MPI integer constants.

## 8. Deliberate limits

| Item | Status |
|---|---|
| Topology, neighborhood collectives, grids | Not wrapped. |
| `MPI_Alltoallw`, `MPI_Type_create_darray` | Not wrapped. |
| Public `MPI_Pack`/`MPI_Unpack`, general raw-datatype owner/transfer | Not wrapped; staging is internal. `layout<T>::from_raw` is the single documented raw-adoption escape (derived handle, type map over `T`, ownership transfers on entry). |
| `MPI_MAXLOC`/`MPI_MINLOC` typed reductions | No `MPI_2*` pair model; raw `MPI_Op` only for non-RMA reductions. |
| RMA container data forms | Origin/result are `std::span`/`region`; there is no container overload. |
| Persistent container data buffers | Fixed buffers only; use a `std::span` or `region`. |
| File `get_view`, user datareps, raw heterogeneous `set_view` | Not implemented. |
| File free data-access wrappers | Not present in 0.5.0 or 1.0; 0.5.0 file access was already member-only. |
| Predecessor layout-container and displacement types | Not in 1.0; use `region<T>`, `std::span<const layout<T>>` and `blocks`. |
| Predecessor request pools | Not in 1.0; the batch functions take `std::span<request>`. |
| MPI Sessions, partitioned point-to-point | Not wrapped. |
| Attributes/keyvals, object naming, user error-handler callbacks | Not wrapped. |
| DPM connect/accept/join, generalized requests, MPI_T | Not wrapped. |
| Owning `MPI_Errhandler` bridge / user error-handler callbacks | Not wrapped. `mp1::error_handler` models the three predefined handlers only; a custom handler is raw MPI or `MPI_Comm_set_errhandler` on `native_handle()` after creation. `MPI_ERRHANDLER_NULL` is not a valid handler (MPI 4.1 §9.3.1). |
| From-group non-membership | MPI-erroneous and deliberately not diagnosed locally; MPI is the authority. Cross-rank agreement (same group/leader/stringtag, disjoint groups) is the application's obligation. |
| Duplicate/overlapping `rank_range` triplets | Not expanded or diagnosed locally; `group_range_incl`/`group_range_excl` validate stride, direction and endpoints only (MPI 4.1 §7.3.2 makes distinct computed ranks an erroneous-program obligation). |
| `result_pending` destruction during unwinding | The request destructor performs the collective `MPI_Comm_free`; take completed idup results before rethrowing, or use `release()` and accept the documented leak. |
