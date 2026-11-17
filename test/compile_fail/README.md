# Compile-fail probes

Every probe is registered with three facts:

* `SOURCE` — the file that must fail to compile;
* `EXPECT` — a CMake regular expression for the **library-owned, stable** part
  of the intended diagnostic (a `static_assert` token such as
  `mp1: RMA only accepts a predefined MPI operation`, or a compiler-independent
  identifier/verb pair such as `deleted function.*isend`);
* `CONTROL` — a positive control in `test/compile_pass/` that uses the valid
  API shape and must compile with exactly the same command line.

`cmake/CompileFail.cmake` passes only when the probe runs, exits nonzero, does
not report `fatal error:` (a missing header is not the intended property),
attributes a diagnostic to the probe or an `mp1/` header, matches `EXPECT`, and
compiles the control successfully.  A successful compile, an unmatched
diagnostic, a broken control, or a missing expectation fails the check.  For a
concept-constrained lifetime guard the expectation requires the compiler's
constraint-failure context (`constraints not satisfied`) as well as the concept
name, so a wrong-arity call cannot pass just because a candidate note elsewhere
mentions `borrowable`.

`ci/check-compile-fail-selfcheck.cmake` is the inverse control.  It runs the
runner on fixtures in `self_check/` and asserts:

| Case | Expected |
|---|---|
| intended diagnostic + valid control | pass |
| wrong diagnostic (different `#error`) | fail |
| typo / wrong API name | fail |
| source compiles | fail |
| missing header | fail (`fatal error`) |
| positive control does not compile | fail |
| no `EXPECT` supplied | fail |

The set is **94** contract probes. They register **96** CTest entries: the 94
probes plus the runner self-check probe and the inverse runner self-check. The
probes are:

| Probe | Area | Basis |
|---|---|---|
| `scalar_rvalue_isend.cc` | p2p | §3.7.2: the send buffer must stay valid until the nonblocking send completes; a temporary scalar has no stable address. |
| `no_member_collectives.cc` | collectives | Local API shape: collectives are free functions (MPI 4.1 §6 defines procedures, not C++ members). |
| `no_inter_communicator.cc` | communicators | Local API shape: one `mpi_communicator` type handles intra and inter handles (MPI 4.1 §7.6 defines inter-communication operations). |
| `nodiscard_request.cc` | requests | §3.7.3 / §11.2.2: an initiated request must be completed (or freed) before finalize; the attribute makes dropping it explicit. |
| `rma_custom_functor.cc` | RMA operator adapter | §12.3.4: only predefined reduction operators may be used; user-defined operators cannot be used. |
| `throwing_reduction_functor.cc` | reduction operator adapter | MPI 4.1 §6.9.5: a C++ exception cannot unwind through the MPI user-op C callback; a functor whose `operator()` is not `noexcept` is a compile-time error. |
| `rma_cas_double.cc` | RMA typed boundary | §12.3.4 (Compare and Swap): the datatype must be a predefined C++ datatype in the C integer, Logical or Byte category; `double` is Floating point. |
| `rma_cas_character.cc` | RMA typed boundary | §12.3.4 (Compare and Swap) and §6.9.3: `MPI_CHAR`/`MPI_WCHAR` are not in the C++ CAS categories or reduction categories. |
| `rma_fop_derived.cc` | RMA typed boundary | §12.3.4 (Fetch and Op): the datatype argument must be a predefined datatype; a reflected aggregate is derived. |
| `rma_accumulate_mixed.cc` | RMA typed boundary | §12.3.4 (Accumulate): all basic components must be the same predefined datatype; an int+double reflected struct violates the rule. |
| `rma_accumulate_mixed_pair.cc` | RMA typed boundary | §12.3.4 (Accumulate): `std::pair<int,double>` has two different known basic components. |
| `rma_accumulate_mixed_replace.cc` | RMA typed boundary | MPI 4.1 §12.3.4 (REPLACE policy): homogeneity is a datatype rule, so a heterogeneous derived type is a compile-time error even for `MPI_REPLACE` (the historical Annex B "predefined only" sentence is the withdrawn MPI 2.1 rule). |
| `rma_accumulate_mixed_nested.cc` | RMA typed boundary | §12.3.4 (Accumulate): nesting does not hide a mixed outer sequence; `pair<pair<int,int>,double>` has int and double components. |
| `rma_accumulate_mixed_widths.cc` | RMA typed boundary | §12.3.4 and §6.9.2: different C integer widths are different predefined basic types. |
| `rma_accumulate_empty.cc` | RMA typed boundary | §12.3.4 (Accumulate): a datatype with no basic components has no reduction category; an empty reflected aggregate is rejected. |
| `rma_accumulate_opaque.cc` | RMA typed boundary | §12.3.4 (Accumulate): a custom non-predefined datatype without `MP1_DATATYPE_BASIC` has no compile-time basic component (no general runtime classifier in 1.0). |
| `rma_accumulate_custom_predefined_mixed.cc` | RMA typed boundary | §12.3.4 (Accumulate): a custom predefined trait has an opaque compile-time identity, so combining it with another C++ type that maps the same MPI handle cannot be proven homogeneous. |
| `rma_accumulate_predefined_op_aggregate.cc` | RMA operator adapter | §12.3.4: the operator operand is the predefined basic component, not the aggregate; `predefined_op<T>` requires a predefined `T`. |
| `rma_accumulate_functor_operand_mismatch.cc` | RMA operator adapter | §12.3.4 and §6.9.2: the selected `predefined_op<int>` rejects a functor declared for another operand (`std::plus<double>`). |
| `rma_pointer_origin_removed.cc` | RMA buffer forms | The raw pointer+count data-movement spelling is not part of the API; the span/region form is the only origin/result shape, and the buffer size is a checked fact. |
| `rma_pointer_layout_removed.cc` | RMA buffer forms | The pointer+layout spelling is not part of the API; a layout origin is a `region<T>` (data + layout), so `map().size()` is available to the size check. |
| `rma_const_result_removed.cc` | RMA buffer forms | §12.3.2: a result position is writable, so `span<const T>`/`region<const T>` is rejected; get writes the origin buffer. |
| `rma_container_origin_removed.cc` | RMA buffer forms | The RMA data-movement forms are span/region only (no container data forms); the caller passes `std::span{container}`. |
| `rma_origin_result_type_mismatch.cc` | RMA typed boundary | §12.3.4: every datatype argument must be constructed from the same predefined basic datatype; an `int` origin and a `double` result are rejected. |
| `rma_accumulate_c_array_toplevel.cc` | RMA typed boundary | §12.3.4: a top-level C array is not supported as the accumulating element type (C arrays are classified as members only); use `std::array` or a reflected member. |
| `invalid_send_mode.cc` | enum hardening | §3.4: the communication modes are exactly standard, buffered, synchronous, and ready. |
| `scatter_raw_count.cc` | fixed collectives | §6.6: the non-root receive block is the receive span itself; a raw integer does not bind in that position. |
| `block_count_removed.cc` | fixed collectives | `mp1::block_count` is not part of the API; the per-peer scatter block is the receive span. |
| `capacity_raw_count.cc` | p2p / request | §3.2.4/§3.2.5: the resizable-container receive bound is an upper bound, spelled `mp1::capacity`; a raw count in that position is rejected so it can never be read as a source rank. |
| `capacity_count_t_4arg.cc` | p2p / request | §3.2.4/§3.2.5: the four-argument container form with a `count_t` third argument is the removed untyped capacity spelling (deleted, not silently read as a source rank). |
| `owner_view_base_assignment.cc` | owners/views | MPI 4.1 §7.4.3: the owner is the only code allowed to replace or free the handle; binding a mutable view reference would allow `view = comm_world()` to overwrite it (an owner converts only to a view value). |
| `owner_view_pointer.cc` | owners/views | An owner's address must not become a view pointer (deleting through a non-owning base is the hazard); owners convert only to a view value. |
| `file_nonblocking_write_rvalue.cc` | file | §14.4.1: a nonblocking write keeps the caller's buffer address until the request completes; a temporary scalar would dangle. |
| `file_split_write_begin_rvalue.cc` | file | §14.4.5: a split-collective write begin keeps the buffer address until the matching end; a temporary scalar would dangle. |
| `igatherv_scalar_rvalue.cc` | collectives | §6.12: a nonblocking collective keeps the send buffer address until completion; the scalar send must be an lvalue. |
| `iallgatherv_scalar_rvalue.cc` | collectives | §6.12: a nonblocking collective keeps the send buffer address until completion; the scalar send must be an lvalue. |
| `dense_igatherv_scalar_rvalue.cc` | dense collectives | §6.12: the dense `blocks` nonblocking `gatherv` keeps the scalar send address until completion. |
| `dense_gatherv_init_scalar_rvalue.cc` | dense collectives | §6.13: a persistent `gatherv` captures the scalar send address at init; a temporary cannot outlive the request. |
| `dense_iallgatherv_scalar_rvalue.cc` | dense collectives | §6.12: the dense `blocks` nonblocking `allgatherv` keeps the scalar send address until completion. |
| `dense_allgatherv_init_scalar_rvalue.cc` | dense collectives | §6.13: a persistent `allgatherv` captures the scalar send address at init. |
| `typed_igatherv_scalar_rvalue.cc` | collectives | §6.12: the typed (layout-span) nonblocking `gatherv` keeps the scalar send address until completion. |
| `typed_allgatherv_init_scalar_rvalue.cc` | collectives | §6.13: the typed persistent `allgatherv` captures the scalar send address at init. |
| `counts_removed.cc` | vocabulary | The transitional `mp1::counts` alias is not part of the API; the dense descriptor is `mp1::blocks`. |
| `displacements_removed.cc` | vocabulary | The transitional `mp1::displacements` alias is not part of the API; typed displacements are `std::span<const count_t>`. |
| `layouts_removed.cc` | vocabulary | The transitional `mp1::layouts<T>` alias is not part of the API; typed descriptors are `std::span<const layout<T>>`. |
| `layout_view_removed.cc` | layout | `mp1::layout_view<T>` is deleted; a single-block position takes `region<T>` or a contiguous range. |
| `make_layout_span_removed.cc` | layout | `mp1::make_layout_span` is deleted; a layout vector converts to `std::span<const layout<T>>` directly. |
| `variable_gatherv_recv_in_place.cc` | collectives | Gatherv's in-place sentinel belongs to the send position; the aggregate receive position is `recv_sink`. |
| `variable_scatterv_send_in_place.cc` | collectives | Scatterv's in-place sentinel belongs to the receive position; the aggregate send position is `send_source`. |
| `collective_gather_recv_in_place.cc` | fixed collectives | MPI 4.1 §6.5: the in-place sentinel is legal in gather's send slot only; the receive position is `recv_sink`. |
| `collective_scatter_send_in_place.cc` | fixed collectives | MPI 4.1 §6.6: the in-place sentinel is legal in scatter's receive slot only; the send position is `send_source`. |
| `collective_bcast_in_place.cc` | fixed collectives | §6.4: bcast has no in-place form ("not meaningful"); the single buffer is `recv_sink`. |
| `collective_alltoall_recv_in_place.cc` | fixed collectives | MPI 4.1 §6.8: alltoall's sentinel occupies the send position; the receive position is `recv_sink`. |
| `collective_reduce_local_in_place.cc` | fixed collectives | §6.9.7: reduce_local has no in-place form ("not allowed"). |
| `collective_igather_rvalue_send.cc` | fixed collectives | §6.12: a nonblocking collective borrows the send buffer address until completion (`borrowable<S&&>`). |
| `collective_init_container_send.cc` | fixed collectives | MPI 4.1 §6.13: a persistent collective accepts fixed buffers only; a resizable container has no per-start revalidation. |
| `iscatter_container_in_place_rvalue.cc` | collectives | §3.7.2: the in-place nonblocking `scatter` borrows the caller's container storage until completion. |
| `file_non_element_type.cc` | file | `template<element T>` rejects a type without an MPI datatype mapping as a constraint failure, not an in-body assertion. |
| `isendrecv_scalar_rvalue.cc` | p2p | §3.7.2: the combined nonblocking send-receive keeps the scalar send address until completion. |
| `isendrecv_container_rvalue.cc` | p2p | §3.7.2: the container send form of `isendrecv` borrows the container storage until completion. |
| `isend_container_rvalue.cc` | p2p | §3.7.2: the container form of a nonblocking send borrows the caller's storage; `borrowable<C&&>` rejects an rvalue container. |
| `ibsend_scalar_rvalue.cc` | p2p | §3.7.2: a buffered nonblocking send keeps the scalar address until completion; `borrowable<T&&>` rejects an rvalue scalar. |
| `issend_scalar_rvalue.cc` | p2p | §3.7.2: a synchronous nonblocking send keeps the scalar address until completion; `borrowable<T&&>` rejects an rvalue scalar. |
| `irsend_scalar_rvalue.cc` | p2p | §3.7.2: a ready nonblocking send keeps the scalar address until completion; `borrowable<T&&>` rejects an rvalue scalar. |
| `send_init_container_rvalue.cc` | p2p | §6.13: a persistent send captures the container address across restarts; `borrowable<C&&>` rejects an rvalue container. |
| `bsend_init_scalar_rvalue.cc` | p2p | §6.13: a persistent buffered send captures the scalar address; `borrowable<T&&>` rejects an rvalue scalar. |
| `ssend_init_scalar_rvalue.cc` | p2p | §6.13: a persistent synchronous send captures the scalar address; `borrowable<T&&>` rejects an rvalue scalar. |
| `rsend_init_scalar_rvalue.cc` | p2p | §6.13: a persistent ready send captures the scalar address; `borrowable<T&&>` rejects an rvalue scalar. |
| `recv_init_raw_count.cc` | p2p / request | §3.2.4/§6.13: the persistent capacity receive spells the bound with `mp1::capacity`; a raw integer is rejected. |
| `persistent_collective_container.cc` | collectives | §6.13: persistent collectives are raw forms only; a container could reallocate between restarts. |
| `container_element_datatype.cc` | layout/datatype | A contiguous container whose element has no MPI datatype mapping is rejected with the actionable element-datatype static assertion, not "no matching function". |
| `file_dense_vector_bool.cc` | file | §14.9: `std::vector<bool>` is a proxy specialization without contiguous bool storage; the dense container forms are disabled for it, so the call fails the element-datatype check instead of reading proxy storage as bools. |
| `file_nonblocking_container_rvalue.cc` | file | §14.4.1: a nonblocking file write borrows the container's `data()` address until the request completes; a temporary container would dangle. |
| `file_write_sequence.cc` | file | §14.4.1: dense file access passes one contiguous (pointer, count) pair, so the write range forms require `element_buffer`; a non-contiguous sequence has no `data()` and is rejected at overload resolution instead of failing inside `<ranges>`. |
| `region_rvalue_layout.cc` | vocabulary | `region`'s deleted `(T*, const layout<...>&&)` constructor rejects a prvalue layout with an explicit element type; the reference member must never bind a temporary. |
| `region_ctad_rvalue_layout.cc` | vocabulary | CTAD deduces `T` from the pointer and then selects the same deleted rvalue-layout constructor, so `region{p, layout<T>{...}}` is rejected as well. |
| `region_ctad_nullptr.cc` | vocabulary | `region{nullptr, l}` is a deduction failure because `T` cannot be deduced from `nullptr`; the supported null spelling is `region<int>{nullptr, l}`. |
| `region_array_temporary_layout.cc` | vocabulary | One `std::array` object as one datatype requires a *named* layout; the prvalue spelling is intentionally rejected by the deleted rvalue-layout constructor. |
| `p2p_char_literal_send.cc` | p2p | A string literal is a built-in character array; it would send its NUL. Use `std::string_view`/`std::string`. |
| `p2p_char_array_recv.cc` | p2p | `char[N]` is a byte buffer, not a string, and is excluded on receive. Use `std::span<char>`/`std::array<char,N>`. |
| `p2p_sequence_recv.cc` | p2p | A non-contiguous sequence is send-only; receive targets are contiguous. |
| `p2p_map_send.cc` | p2p | Associative containers are keyed, not flat; materialize to `std::vector<std::pair<K,V>>`. |
| `p2p_set_recv.cc` | p2p | Associative containers are excluded on receive too. |
| `p2p_vector_bool_send.cc` | p2p | `std::vector<bool>` is bit-packed with proxy references, not a `bool` buffer. |
| `p2p_forward_list_send.cc` | p2p | Unsized ranges are excluded; `std::list` (sized) stays accepted. |
| `p2p_span_const_recv.cc` | p2p | A const buffer is not writable and cannot be a receive target. |
| `p2p_region_const_recv.cc` | p2p | `region<const T>` is send-only; a receive region must be mutable. |
| `p2p_mrecv_rvalue_message.cc` | p2p | Matched receives take `message&` only; the one-shot rvalue spelling is not part of the API. |
| `p2p_pointer_count_send.cc` | p2p | The pointer+count send form is not part of the API; `span_from` is the checked spelling. |
| `p2p_recv_rvalue_container.cc` | p2p | A receive into a temporary container is excluded. |
| `request_span_removed.cc` | request | The `request_span` container type is not part of the API; request batches are `std::span<request>`. |
| `rma_reflection_120_members.cc` | layout/reflection | Reflection expands at most 119 named members; the 120-member probe fails the static_assert (`supports at most 119 members`). |
| `layout_builder_braced_init.cc` | layout builders | The five parameter builders take a named contiguous range (`std::array`/`std::vector`/`std::span`) or `.add()`, not a braced list (a braced-init-list does not convert to `std::span`); the positive control is `test/compile_pass/layout_builder_named_ranges.cc`. |
| `layout_empty_removed.cc` | layout | `layout<T>::empty()` is removed (source-breaking); `contiguous(0)` is the zero-size layout and `mp1::empty_data` the explicit empty message. The positive control is `test/compile_pass/layout_zero_control.cc`. |

There is no absolute/heterogeneous target-layout form to reject (`layout<void>`
is not part of 1.0; MPI 4.1 §12.3.1), and `layout<T>` is the only target form.

`test/harness/compile_fail_self_check.cc` is a real `#error` probe with an
expectation and a control; `harness_compile_fail_runner_selfcheck` runs the
matrix above.  The diagnostics are compiler-independent where the standard
leaves wording to the compiler: `EXPECT` intentionally matches the identifier
plus one of the accepted compiler phrasings, while `static_assert` messages are
owned by mp1.

`harness_compile_pass_controls` compiles every source in `test/compile_pass/`
with the same command line as a probe control, so a control whose probe was
removed or renamed cannot become an inert file.  Probe-bound controls are still
compiled by their probe; this check covers the whole directory (64 sources).
