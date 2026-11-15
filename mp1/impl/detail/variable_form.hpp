#if !(defined MP1_DETAIL_VARIABLE_FORM_HPP)

#define MP1_DETAIL_VARIABLE_FORM_HPP

// Variable-collective form adapters.
//
// The public surface has exactly two declarations per variable family per lifetime: a dense
// descriptor (`blocks`, element counts + displacements) and a typed descriptor
// (`std::span<const layout<T>>` + `std::span<const count_t>`).  The two buffer positions are
// the same compile-time forms as the fixed collectives: a single-block position is a value, a
// contiguous element range, or a `region` (one layout occurrence); an aggregate position is a
// flat contiguous element range, because the per-block placement is owned by the descriptor.
// The compile-time in-place side is the family's fixed side; a non-contiguous sequence is not
// an advertised variable-family form.
//
// This header only converts the classified positions and descriptors into the engine's
// `variable_options`; `make_variable_plan` owns the role resolution, the per-block bounds and
// disjointness checks, the staging and the in-place priming.  The descriptor spans
// are borrowed for the call; the plan copies every array and staged layout it reads after the
// initiating call returns, so a nonblocking/persistent call may outlive the caller's
// descriptor arrays.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/collective_engine.hpp"
#include "mp1/impl/detail/collective_form.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <cstddef>
#include <span>
#include <type_traits>

namespace mp1::detail {

  // The one single-block source of a variable family.  An in-place position carries neither a
  // pointer nor a block, so the source is the empty element form and the engine ignores it.
  template<class T>
  [[nodiscard]] single_block_source<T> single_block_of(const form_position<T>& position) {
    if (position.typed)
      return single_block_source<T>::of_layout(position.map);
    return single_block_source<T>::of_elements(position.element_count());
  }

  // The four family adapters only fill the named `variable_options` sides; the engine owns the
  // role resolution, bounds/alias checks, staging, priming and self-block checks exactly once.

  // gatherv: send is a single block (in-place at the root); recv is the aggregate and carries
  // the per-peer descriptor on the root only.  A non-root caller supplies the documented
  // dummy (`std::span<T>{}` + `blocks{}`); the ignored positions are never inspected.
  template<class T>
  [[nodiscard]] variable_call<T> make_gatherv_variable_plan(
      const mpi_communicator& c, int root, const form_position<T>& send,
      const form_position<T>& recv, variable_block_source<T> recv_blocks, const context& ctx) {
    variable_options<T> options;
    options.family = variable_family::gatherv;
    options.root = root;
    options.place = place_state_of(send.in_place);
    options.send.data = send.in_place ? nullptr : send.send_data();
    options.send.elements = send.element_count();
    options.send.single = single_block_of(send);
    options.recv.data = recv.mutable_data;
    options.recv.elements = recv.element_count();
    options.recv.many = recv_blocks;
    return make_variable_plan<T>(c, options, ctx);
  }

  // scatterv: send is the aggregate + descriptor (root only); recv is a single block
  // (in-place at the root).  A non-root caller supplies the documented dummy send side.
  template<class T>
  [[nodiscard]] variable_call<T> make_scatterv_variable_plan(
      const mpi_communicator& c, int root, const form_position<T>& send,
      variable_block_source<T> send_blocks, const form_position<T>& recv, const context& ctx) {
    variable_options<T> options;
    options.family = variable_family::scatterv;
    options.root = root;
    options.place = place_state_of(recv.in_place);
    options.send.data = send.send_data();
    options.send.elements = send.element_count();
    options.send.many = send_blocks;
    options.recv.data = recv.mutable_data;
    options.recv.elements = recv.element_count();
    options.recv.single = single_block_of(recv);
    return make_variable_plan<T>(c, options, ctx);
  }

  // allgatherv: send is a single block (in-place at every rank); recv is the aggregate +
  // descriptor at every rank.
  template<class T>
  [[nodiscard]] variable_call<T> make_allgatherv_variable_plan(
      const mpi_communicator& c, const form_position<T>& send, const form_position<T>& recv,
      variable_block_source<T> recv_blocks, const context& ctx) {
    variable_options<T> options;
    options.family = variable_family::allgatherv;
    options.place = place_state_of(send.in_place);
    options.send.data = send.in_place ? nullptr : send.send_data();
    options.send.elements = send.element_count();
    options.send.single = single_block_of(send);
    options.recv.data = recv.mutable_data;
    options.recv.elements = recv.element_count();
    options.recv.many = recv_blocks;
    return make_variable_plan<T>(c, options, ctx);
  }

  // alltoallv: both positions are aggregates with their own descriptors.  With the send
  // sentinel the send counts/displacements are MPI-ignored; callers pass `blocks{}` there
  // (the receive descriptor is the exchange recipe).
  template<class T>
  [[nodiscard]] variable_call<T> make_alltoallv_variable_plan(
      const mpi_communicator& c, const form_position<T>& send,
      variable_block_source<T> send_blocks, const form_position<T>& recv,
      variable_block_source<T> recv_blocks, const context& ctx) {
    variable_options<T> options;
    options.family = variable_family::alltoallv;
    options.place = place_state_of(send.in_place);
    options.send.data = send.send_data();
    options.send.elements = send.element_count();
    options.send.many = send_blocks;
    options.recv.data = recv.mutable_data;
    options.recv.elements = recv.element_count();
    options.recv.many = recv_blocks;
    return make_variable_plan<T>(c, options, ctx);
  }

}  // namespace mp1::detail

#endif
