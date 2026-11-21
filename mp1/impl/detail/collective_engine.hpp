#if !(defined MP1_DETAIL_COLLECTIVE_ENGINE_HPP)

#define MP1_DETAIL_COLLECTIVE_ENGINE_HPP

// Collective engine: the one root-role resolver, peer-count helpers, the fixed-size native
// call adapters, the variable-family direction preparation (over xfer_plan), and the
// reduction flat/staged helpers.  Public overloads in mp1/impl/collective.hpp are thin
// adapters; this file owns no public name.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/detail/xfer_side.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/operator.hpp"
#include "mp1/impl/request.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1::detail {

  // ---------------------------------------------------------------------------------------
  // Root roles and peer counts.
  // ---------------------------------------------------------------------------------------

  enum class root_role { intra_root, intra_nonroot, inter_root, inter_noop, inter_remote };

  [[nodiscard]] inline root_role classify_root(const mpi_communicator& c, int root_arg,
                                               const context& ctx) {
    if (not c.is_inter()) {
      if (root_arg == MPI_ROOT or root_arg == MPI_PROC_NULL) [[unlikely]] {
        throw_invalid(ctx.operation,
                      "MPI_ROOT/MPI_PROC_NULL are not valid intracommunicator roots");
      }
      if (root_arg < 0 or root_arg >= c.size()) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "root {} is outside [0,{})", root_arg, c.size());
      }
      return c.rank() == root_arg ? root_role::intra_root : root_role::intra_nonroot;
    }
    if (root_arg == MPI_ROOT)
      return root_role::inter_root;
    if (root_arg == MPI_PROC_NULL)
      return root_role::inter_noop;
    if (root_arg < 0 or root_arg >= c.remote_size()) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "intercommunicator root {} is outside [0,{})",
                                             root_arg, c.remote_size());
    }
    return root_role::inter_remote;
  }

  [[nodiscard]] inline int peer_count(const mpi_communicator& c) {
    return c.is_inter() ? c.remote_size() : c.size();
  }

  inline void require_peers(std::size_t n, int expected, const context& ctx) {
    if (static_cast<std::size_t>(expected) != n) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "array length {} does not match the required {}",
                                             n, expected);
    }
  }

  inline void require_collective_communicator(const mpi_communicator& c, const context& ctx) {
    require_valid(c, ctx);
    require_mpi(ctx);
  }

  // One exact-byte null policy for a *used* data position.  A position is empty only when its
  // total byte size is zero, not when its element count is zero: a zero-size layout or a
  // zero-size native datatype with a positive count is empty (MPI 4.1 §5.1.11).  An empty
  // position may be null and is presented to MPI as the per-slot dummy with the caller's
  // count; a non-empty position with a null pointer is rejected locally, before any MPI call
  // or managed MPI_Op creation.  The slot selects the dummy and the diagnostic so an ignored
  // send and an ignored receive never alias (MPI 4.1 §6.5/§6.6).
  inline void require_used_bytes(buffer_slot slot, const void* p, MPI_Count bytes,
                                 const context& ctx) {
    if (p != nullptr or bytes == 0)
      return;
    if (slot == buffer_slot::send)
      throw_invalid(ctx.operation, "null send pointer for a non-empty buffer/layout");
    throw_invalid(ctx.operation, "null receive pointer for a non-empty buffer/layout");
  }

  // ---------------------------------------------------------------------------------------
  // One role/placement policy shared by every collective plan builder.
  //
  // MPI 4.1 §6.2.2-6.2.3: the root argument of a root-taking family is a local rank on an
  // intracommunicator; on an intercommunicator the root group passes the MPI_ROOT sentinel
  // at the root (MPI_PROC_NULL everywhere else in that group), while the remote group passes
  // the root's rank in the remote group.  A position the role marks ignored is never inspected
  // and is presented as {dummy(slot), 0, base datatype}.  In-place follows §6.2.1 and is never
  // valid on an intercommunicator (§6.2.3); the per-family position is fixed by §6.5-6.11.
  //
  // The named enumerations replace the previous positional boolean arguments:
  // every plan builder below takes a struct, and every engine decision is a named value.
  // ---------------------------------------------------------------------------------------

  enum class collective_family : unsigned char {
    bcast,
    gather,
    scatter,
    allgather,
    alltoall,
    reduce,
    allreduce,
    scan,
    exscan,
    reduce_scatter,
    reduce_scatter_block
  };

  enum class collective_role : unsigned char {
    all,
    intra_root,
    intra_nonroot,
    inter_root,
    inter_noop,
    inter_remote
  };

  // Whether the role reads/writes a position; an ignored position is never inspected.
  enum class side_use : unsigned char { ignored, used };

  // MPI_IN_PLACE occupies one fixed position per family (MPI 4.1 §6.2.1, §6.5-6.11).
  enum class place_state : unsigned char { data, mpi_in_place };

  // What the local array describes: one per-rank block or one entry per peer.
  enum class block_extent : unsigned char { single, aggregate };

  // MPI restricts a datatype that writes (or the scatterv root send that reads) to touching a
  // location once; a read-only direction skips the cross-block alias check.
  enum class alias_policy : unsigned char { read_only, restricted };

  // How long a request reads plan-owned layout metadata.
  enum class metadata_scope : unsigned char { completion, restart };

  // Whether a reduction request must carry a payload even when nothing is staged (the
  // reduce_scatter count arrays are read until completion).
  enum class payload_policy : unsigned char { on_demand, always };

  struct role_sides {
    collective_role role{collective_role::all};
    bool send_used{false};
    bool recv_used{false};
    bool in_place{false};
    bool send_in_place{false};
    bool recv_in_place{false};
  };

  [[nodiscard]] inline side_use side_use_of(bool used) noexcept {
    return used ? side_use::used : side_use::ignored;
  }

  [[nodiscard]] inline place_state place_state_of(bool in_place) noexcept {
    return in_place ? place_state::mpi_in_place : place_state::data;
  }

  [[nodiscard]] inline bool family_has_root(collective_family family) noexcept {
    switch (family) {
      case collective_family::bcast:
      case collective_family::gather:
      case collective_family::scatter:
      case collective_family::reduce:
        return true;
      default:
        return false;
    }
  }

  [[nodiscard]] inline const char* family_name(collective_family family) noexcept {
    switch (family) {
      case collective_family::bcast:
        return "bcast";
      case collective_family::gather:
        return "gather";
      case collective_family::scatter:
        return "scatter";
      case collective_family::allgather:
        return "allgather";
      case collective_family::alltoall:
        return "alltoall";
      case collective_family::reduce:
        return "reduce";
      case collective_family::allreduce:
        return "allreduce";
      case collective_family::scan:
        return "scan";
      case collective_family::exscan:
        return "exscan";
      case collective_family::reduce_scatter:
        return "reduce_scatter";
      case collective_family::reduce_scatter_block:
        return "reduce_scatter_block";
    }
    return "collective";
  }

  [[nodiscard]] inline role_sides resolve_sides(collective_family family,
                                                const mpi_communicator& c, int root_arg,
                                                place_state place, const context& ctx) {
    require_collective_communicator(c, ctx);
    role_sides sides;
    const bool inter = c.is_inter();
    const bool in_place_requested = place == place_state::mpi_in_place;

    if (family_has_root(family)) {
      switch (classify_root(c, root_arg, ctx)) {
        case root_role::intra_root:
          sides.role = collective_role::intra_root;
          break;
        case root_role::intra_nonroot:
          sides.role = collective_role::intra_nonroot;
          break;
        case root_role::inter_root:
          sides.role = collective_role::inter_root;
          break;
        case root_role::inter_noop:
          sides.role = collective_role::inter_noop;
          break;
        case root_role::inter_remote:
          sides.role = collective_role::inter_remote;
          break;
      }
    } else {
      if (inter and (family == collective_family::scan or family == collective_family::exscan))
          [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "{} is not defined on intercommunicators",
                                               family_name(family));
      }
      sides.role = collective_role::all;
    }

    const collective_role role = sides.role;
    const bool intra_root = role == collective_role::intra_root;
    const bool intra_nonroot = role == collective_role::intra_nonroot;
    const bool inter_root = role == collective_role::inter_root;
    const bool inter_remote = role == collective_role::inter_remote;
    const bool inter_noop = role == collective_role::inter_noop;

    if (in_place_requested) {
      bool legal = false;
      switch (family) {
        case collective_family::gather:
        case collective_family::scatter:
        case collective_family::reduce:
          legal = intra_root;
          break;
        case collective_family::allgather:
        case collective_family::alltoall:
        case collective_family::allreduce:
        case collective_family::reduce_scatter:
        case collective_family::reduce_scatter_block:
          legal = not inter;
          break;
        case collective_family::scan:
        case collective_family::exscan:
          legal = not inter;
          break;
        case collective_family::bcast:
          legal = false;
          break;
      }
      if (not legal) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "in-place {} is not valid for this role",
                                               family_name(family));
      }
    }
    sides.in_place = in_place_requested;
    // MPI_IN_PLACE occupies the send position of gather/reduce/all-to-all/scan families and
    // the receive position of scatter (MPI 4.1 §6.5, §6.6).
    sides.send_in_place = in_place_requested and family != collective_family::scatter;
    sides.recv_in_place = in_place_requested and family == collective_family::scatter;

    switch (family) {
      case collective_family::bcast:
        sides.send_used = not inter_noop;
        sides.recv_used = false;
        break;
      case collective_family::gather:
      case collective_family::reduce:
        sides.send_used = not sides.in_place and (intra_root or intra_nonroot or inter_remote);
        sides.recv_used = intra_root or inter_root;
        break;
      case collective_family::scatter:
        sides.send_used = intra_root or inter_root;
        sides.recv_used = not sides.in_place and (intra_root or intra_nonroot or inter_remote);
        break;
      case collective_family::allgather:
      case collective_family::alltoall:
      case collective_family::allreduce:
      case collective_family::scan:
      case collective_family::exscan:
      case collective_family::reduce_scatter:
      case collective_family::reduce_scatter_block:
        sides.send_used = true;
        sides.recv_used = true;
        break;
    }
    return sides;
  }

  // ---------------------------------------------------------------------------------------
  // One data-block contract: validate a used position, compute its exact byte size, and
  // normalize an empty position to {dummy(slot), 0, base datatype}.  A null pointer is only
  // rejected when the exact byte size is positive; a dummy never replaces a non-empty null
  // (that would hide caller misuse and write shared storage).  `layout` keeps the caller's
  // datatype at count 1; `elements` lowers a flat layout (or a pointer+count form) to the
  // base datatype at an element count.  MPI 4.1 §5.1.8 makes true_lb a span, never a pointer
  // offset; flat reduction callers do that lowering first through layout_lowering.hpp.
  // ---------------------------------------------------------------------------------------

  enum class block_kind : unsigned char { layout, elements };
  enum class block_state : unsigned char { absent, in_place, data };

  struct data_block {
    const void* ptr{nullptr};
    MPI_Count count{0};
    MPI_Datatype type{MPI_DATATYPE_NULL};
    MPI_Count bytes{0};
    block_state state{block_state::absent};

    [[nodiscard]] void* mutable_ptr() const noexcept {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API
      return const_cast<void*>(ptr);
    }
  };

  template<class T>
  [[nodiscard]] data_block make_block(buffer_slot slot, const T* user, count_t count,
                                      const layout<T>* l, side_use use, place_state place,
                                      const context& ctx,
                                      block_kind kind = block_kind::layout) {
    const MPI_Datatype base = native_datatype<T>();
    if (place == place_state::mpi_in_place)
      return data_block{MPI_IN_PLACE, 0, base, 0, block_state::in_place};
    if (use == side_use::ignored)
      return data_block{dummy_buffer(slot), 0, base, 0, block_state::absent};

    // MPI 4.1 §5.1.11 makes the datatype rule asymmetric: a datatype that only reads a buffer
    // may visit the same location more than once, but a datatype that updates one must not
    // write a location twice.  Reject a proven overlap only for a used, out-of-place receive
    // slot; send-side layouts (a read) and MPI_IN_PLACE are deliberately not inspected, and an
    // `unknown` overlap remains the caller's obligation.
    if (l != nullptr and slot == buffer_slot::recv)
      require_receive_layout(*l, ctx);

    MPI_Count mpi_count{0};
    MPI_Count bytes{0};
    MPI_Datatype type{base};
    if (l != nullptr) {
      if (not l->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      if (kind == block_kind::layout) {
        mpi_count = 1;
        type = l->native_handle();
      } else {
        if (not l->is_base_repetition()) [[unlikely]]
          detail::fail_with<std::logic_error>(ctx.operation, std::source_location::current(),
                                              "element lowering requires a flat layout");
        mpi_count = static_cast<MPI_Count>(l->size());
      }
      bytes = l->byte_size();
    } else {
      if (count < 0) [[unlikely]]
        throw_invalid(ctx.operation, "negative count");
      mpi_count = static_cast<MPI_Count>(count);
      bytes = checked_bytes(mpi_count, base_datatype<T>().size, ctx);
    }
    if (bytes != 0) {
      require_used_bytes(slot, user, bytes, ctx);
      return data_block{user, mpi_count, type, bytes, block_state::data};
    }
    return data_block{dummy_buffer(slot), 0, base, 0, block_state::absent};
  }

  template<class T>
  [[nodiscard]] data_block make_block(buffer_slot slot, const T* user, count_t count,
                                      std::nullptr_t, side_use use, place_state place,
                                      const context& ctx,
                                      block_kind kind = block_kind::layout) {
    return make_block(slot, user, count, static_cast<const layout<T>*>(nullptr), use, place,
                      ctx, kind);
  }

  [[nodiscard]] inline count_t container_block(count_t total, int peers, const context& ctx,
                                               const char* side) {
    if (total < 0) [[unlikely]]
      throw_invalid(ctx.operation, "negative count");
    if (peers <= 0) [[unlikely]]
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "empty peer group for a container {}", side);
    if (total % static_cast<count_t>(peers) != 0) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "container {} size is not a multiple of the peer count", side);
    }
    return total / static_cast<count_t>(peers);
  }

  // The local send/receive block equality is an intracommunicator invariant only.  On an
  // intercommunicator the two directions are in different groups and need not agree (MPI 4.1
  // §6.7 "Advice to users" and §6.8 "Advice to users"); MPI checks the pairwise type
  // signatures, which cannot be done locally.
  inline void require_same_blocks(const data_block& send, const data_block& recv,
                                  const context& ctx) {
    if (send.bytes != recv.bytes) [[unlikely]] {
      throw_invalid(ctx.operation, "send and receive block byte sizes differ");
    }
  }

  // Container sizes are std::size_t; public element counts are count_t.  A size that does not
  // fit is public input, so it throws std::invalid_argument.
  [[nodiscard]] inline count_t checked_size(std::size_t value, const context& ctx) {
    if (value > static_cast<std::size_t>(std::numeric_limits<count_t>::max())) [[unlikely]] {
      throw_invalid(ctx.operation, "container size does not fit count_t");
    }
    return static_cast<count_t>(value);
  }

  // ---------------------------------------------------------------------------------------
  // Fixed-size collectives: one options-driven plan builder for the five families, then
  // exactly one MPI call.  Only root arguments select a role; no hidden collective.
  //
  // A position carries its caller data pointer, element count, optional type map and the
  // local-extent classification.  `extent::aggregate` is a per-peer total that the engine
  // divides by the peer count when there is no caller layout; a `single` position is passed
  // as-is.  An in-place position carries no data and the engine emits MPI_IN_PLACE.
  // ---------------------------------------------------------------------------------------

  struct fixed_call {
    const void* send_ptr{nullptr};
    MPI_Count send_count{0};
    MPI_Datatype send_type{MPI_DATATYPE_NULL};
    void* recv_ptr{nullptr};
    MPI_Count recv_count{0};
    MPI_Datatype recv_type{MPI_DATATYPE_NULL};
  };

  [[nodiscard]] inline fixed_call make_fixed_call(const data_block& send,
                                                  const data_block& recv) {
    fixed_call call;
    call.send_ptr = send.ptr;
    call.send_count = send.count;
    call.send_type = send.type;
    call.recv_ptr = recv.mutable_ptr();
    call.recv_count = recv.count;
    call.recv_type = recv.type;
    return call;
  }

  template<class T>
  struct fixed_position {
    const T* data{nullptr};
    count_t elements{0};
    const layout<T>* map{nullptr};
    block_extent extent{block_extent::single};
    place_state place{place_state::data};
  };

  template<class T>
  struct fixed_options {
    collective_family family{collective_family::bcast};
    int root{0};
    fixed_position<T> send{};
    fixed_position<T> recv{};
  };

  // A used aggregate position is a per-peer total only while it carries no caller layout:
  // a typed position is already one datatype occurrence (or a staged/repetition pointer).
  template<class T>
  [[nodiscard]] inline count_t fixed_block_count(const fixed_position<T>& position,
                                                 const mpi_communicator& c, bool used,
                                                 const char* side, const context& ctx) {
    if (not used or position.place == place_state::mpi_in_place)
      return position.elements;
    if (position.map == nullptr and position.extent == block_extent::aggregate)
      return container_block(position.elements, peer_count(c), ctx, side);
    return position.elements;
  }

  template<class T>
  [[nodiscard]] fixed_call make_fixed_plan(const mpi_communicator& c, const fixed_options<T>& o,
                                           const context& ctx) {
    const place_state place =
        o.send.place == place_state::mpi_in_place ? o.send.place : o.recv.place;
    const role_sides sides = resolve_sides(o.family, c, o.root, place, ctx);
    if (o.family == collective_family::bcast) {
      // MPI_Bcast has one buffer argument, but only the root/remote-root reads it: a local
      // non-root (or the remote group of an intercommunicator) has its buffer written.  Use
      // the receive slot for those roles so make_block applies the written-position overlap
      // check; the root's send layout stays unchecked (reading an aliased map is legal).
      const bool receive_position = sides.role == collective_role::intra_nonroot or
                                    sides.role == collective_role::inter_remote;
      const data_block block = make_block(
          receive_position ? buffer_slot::recv : buffer_slot::send, o.send.data,
          o.send.elements, o.send.map, side_use_of(sides.send_used), place_state::data, ctx);
      fixed_call call;
      call.send_ptr = block.ptr;
      call.send_count = block.count;
      call.send_type = block.type;
      return call;
    }

    // Intracommunicator: the two local totals describe the same block count, so their
    // equality is a local invariant.  Intercommunicator: the two directions are different
    // groups and may carry different block sizes (MPI 4.1 §6.8).
    if (o.family == collective_family::alltoall and not sides.in_place and not c.is_inter() and
        o.send.elements != o.recv.elements) [[unlikely]] {
      throw_invalid(ctx.operation, "container send and receive sizes differ");
    }

    const count_t send_count = fixed_block_count(o.send, c, sides.send_used, "send", ctx);
    const count_t recv_count = fixed_block_count(o.recv, c, sides.recv_used, "receive", ctx);
    const data_block send_block =
        make_block(buffer_slot::send, o.send.data, send_count, o.send.map,
                   side_use_of(sides.send_used), place_state_of(sides.send_in_place), ctx);
    const data_block recv_block =
        make_block(buffer_slot::recv, o.recv.data, recv_count, o.recv.map,
                   side_use_of(sides.recv_used), place_state_of(sides.recv_in_place), ctx);

    switch (o.family) {
      case collective_family::gather:
      case collective_family::scatter:
        if (sides.role == collective_role::intra_root and sides.send_used and sides.recv_used) {
          require_same_blocks(send_block, recv_block, ctx);
        }
        break;
      case collective_family::allgather:
      case collective_family::alltoall:
        if (not sides.in_place and not c.is_inter())
          require_same_blocks(send_block, recv_block, ctx);
        break;
      default:
        break;
    }
    return make_fixed_call(send_block, recv_block);
  }

  // ---------------------------------------------------------------------------------------
  // Variable collectives: prepare both directions (xfer_plan), then one native call.
  // Validation is role-conditional: an ignored side is not inspected.  Displacements are
  // public T-element offsets, which are exactly the MPI extent units of the native datatype
  // because extent(native_datatype<T>()) == sizeof(T); the plan converts them once.  The
  // restricted side (Gatherv root receive, Scatterv root send, Allgatherv/Alltoallv receive)
  // rejects overlapping blocks before any MPI call (MPI 4.1 6.5/6.6/6.7/6.8: a location must
  // not be written/read more than once).
  // ---------------------------------------------------------------------------------------

  // The local send/receive block equality is an element-count equality: the wire type is
  // native_datatype<T>() on both sides, so equal element counts imply equal type signatures
  // (MPI 4.1 5.1.11) and equal logical byte sizes for every non-zero type size.  It applies
  // only where the intracommunicator makes the two directions locally comparable; an
  // intercommunicator's directions are in different groups and are left to MPI.
  inline void require_same_variable_elements(MPI_Count send_elements, MPI_Count recv_elements,
                                             const context& ctx) {
    if (send_elements != recv_elements) [[unlikely]] {
      throw_invalid(ctx.operation, "local send and receive block element counts differ");
    }
  }

  template<class T>
  struct variable_call {
    xfer_plan<T> send;
    xfer_plan<T> recv;

    // Copy-on-need descriptor ownership.  A nonblocking call owns only a staged receive (its
    // completion unpacks); a persistent call owns only a staged direction (prepare_start
    // re-packs it after init, MPI 4.1 6.13).  Dense, scalar-block and send-only nonblocking
    // directions never read a layout after construction and keep the borrow.  Two
    // directions built from the same caller array share one copy set, so MPI_Type_dup runs
    // once per distinct descriptor.
    void own_metadata(metadata_scope scope) {
      const bool for_restart = scope == metadata_scope::restart;
      const bool send_needed =
          for_restart ? send.needs_metadata_at_restart() : send.needs_metadata_at_completion();
      const bool recv_needed =
          for_restart ? recv.needs_metadata_at_restart() : recv.needs_metadata_at_completion();
      std::shared_ptr<const owned_layouts<T>> send_hold = send.metadata();
      if (send_needed and not send_hold)
        send_hold = send.copy_borrowed_metadata();
      std::shared_ptr<const owned_layouts<T>> recv_hold = recv.metadata();
      if (recv_needed and not recv_hold) {
        if (send_hold and send.shares_borrowed_metadata(recv))
          recv_hold = send_hold;
        else
          recv_hold = recv.copy_borrowed_metadata();
      }
      if (send_hold)
        send.bind_metadata(std::move(send_hold));
      if (recv_hold)
        recv.bind_metadata(std::move(recv_hold));
      send.clear_borrowed_metadata();
      recv.clear_borrowed_metadata();
    }
  };

  // ---------------------------------------------------------------------------------------
  // Descriptors for the four variable families.
  //
  // A single-block side is one caller layout (the `region` form) or a native-T repetition
  // count (the value/span form; a scalar is count 1).
  //
  // A multi-block side is either dense (per-peer element counts + displacements) or typed
  // (one layout per peer + element displacements).  Both are borrowed spans at the public
  // boundary: a blocking call borrows them for the call, and the plan copies the wire
  // counts/displacements it hands to MPI (and, for a staged direction, the per-block layouts
  // completion/restart reads) before the initiating call returns.  The caller's descriptor
  // arrays may therefore die after a nonblocking/persistent call.
  //
  // A dense block i is exactly layout<T>::contiguous(counts[i]) at the public element
  // displacement displs[i], so it is a native-T repetition: the plan converts it to
  // (base, counts[i], native_datatype<T>()) and the displacement is passed unchanged as an
  // extent displacement.  Any T with a native datatype is accepted, including a padded or
  // derived T: the count repeats at extent(native_datatype<T>()) == sizeof(T), the exact
  // image of a T array (MPI 4.1 5.1.11).  There is no `packed` or size == extent gate; only
  // count/unit/overflow, non-negative, buffer-bound, null and restricted-side disjointness
  // checks remain.
  // ---------------------------------------------------------------------------------------

  // One single-block side of gatherv/scatterv/allgatherv: either a caller layout or a base
  // datatype repetition count (the value/span dense form; the scalar form is count 1).
  template<class T>
  struct single_block_source {
    const layout<T>* view{nullptr};
    count_t elements{0};
    bool layout_form{false};

    [[nodiscard]] static single_block_source of_layout(const layout<T>* value) noexcept {
      single_block_source source;
      source.view = value;
      source.layout_form = true;
      return source;
    }

    [[nodiscard]] static single_block_source of_elements(count_t value) noexcept {
      single_block_source source;
      source.elements = value;
      return source;
    }
  };

  // One multi-block side: either a borrowed layout array or a dense counts array.  Both carry
  // the public displacement array (for the typed form an empty span means all-zero offsets).
  template<class T>
  struct variable_block_source {
    std::span<const layout<T>> layouts{};
    std::span<const count_t> values;
    std::span<const count_t> displs;
    bool dense_form{false};

    [[nodiscard]] bool dense() const noexcept {
      return dense_form;
    }

    [[nodiscard]] std::size_t size() const noexcept {
      return dense_form ? values.size() : layouts.size();
    }

    [[nodiscard]] static variable_block_source of_layouts(
        std::span<const layout<T>> layout_values,
        std::span<const count_t> displacement_values) noexcept {
      variable_block_source source;
      source.layouts = layout_values;
      source.displs = displacement_values;
      return source;
    }

    [[nodiscard]] static variable_block_source of_counts(
        std::span<const count_t> count_values,
        std::span<const count_t> displacement_values) noexcept {
      variable_block_source source;
      source.values = count_values;
      source.displs = displacement_values;
      source.dense_form = true;
      return source;
    }
  };

  // Every per-peer block of a *used* multi-block side must fit the flat caller buffer the
  // side addresses (a span makes the sparse bound checkable, which the old
  // raw pointer could not).  `elements` is the buffer's element count.  Negative counts and
  // displacements are rejected here, before any MPI call.  A block that addresses
  // no location is exempt from the fit comparison because MPI ignores its placement: a dense
  // count of 0 skips the fit comparison (its non-negative, representable displacement is still
  // checked), a dense block of a zero-size native datatype addresses no location and skips it
  // too, and a zero-byte-size typed layout skips both the fit and displacement checks.
  template<class T>
  void require_variable_blocks_fit(const variable_block_source<T>& source, int peers,
                                   count_t elements, const context& ctx, const char* side) {
    require_peers(source.size(), peers, ctx);
    if (source.dense()) {
      if (source.values.size() != source.displs.size()) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "counts length {} does not match the displacements length {}", source.values.size(),
            source.displs.size());
      }
      for (std::size_t i = 0; i < source.values.size(); ++i) {
        const count_t count = source.values[i];
        const count_t displacement = source.displs[i];
        if (count < 0) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                                 "negative count {} in counts block {}", count,
                                                 i);
        }
        if (displacement < 0) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "negative displacement {} in displacements block {}", displacement, i);
        }
        // The implied byte address d_i * extent(native_datatype<T>()) must be representable
        // (std::overflow_error, the 1.0 contract), then the sparse bound is checked against
        // the caller's span.
        (void)checked_count_mul(displacement, base_datatype<T>().extent, ctx);
        // A zero-byte native datatype has an empty type signature (MPI 4.1 §5.1.11): a
        // positive count addresses no location, so the caller's empty span is enough.  The
        // plan normalizes the wire count to 0; the non-negative/representability checks above
        // still run first, so a malformed descriptor is rejected before any datatype query.
        if (base_datatype<T>().size != 0 and count != 0 and
            (displacement > elements or count > elements - displacement)) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "{} block {} (displacement {}, count {}) exceeds the {} buffer of {} elements",
              side, i, displacement, count, side, elements);
        }
      }
      return;
    }
    if (not source.displs.empty() and source.displs.size() != source.layouts.size())
        [[unlikely]] {
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "displacement array length {} does not match the required {}", source.displs.size(),
          source.layouts.size());
    }
    const count_t buffer_bytes = checked_count_mul(elements, base_datatype<T>().extent, ctx);
    for (std::size_t i = 0; i < source.layouts.size(); ++i) {
      const layout<T>& block = source.layouts[i];
      if (not block.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid layout in a block array");
      }
      if (block.byte_size() == 0)
        continue;
      const count_t displacement = source.displs.empty() ? 0 : source.displs[i];
      if (displacement < 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "negative displacement {} in displacements block {}", displacement, i);
      }
      const count_t shift = checked_count_mul(displacement, base_datatype<T>().extent, ctx);
      const count_t lower = checked_count_add(shift, block.true_byte_lower_bound(), ctx);
      const count_t upper = checked_count_add(shift, block.true_byte_upper_bound(), ctx);
      if (lower < 0 or upper > buffer_bytes) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "{} block {} (displacement {}, layout bytes "
                                               "[{},{})) exceeds the {} buffer of {} elements",
                                               side, i, displacement, lower, upper, side,
                                               elements);
      }
    }
  }

  template<class T>
  [[nodiscard]] xfer_plan<T> make_single_plan(const T* base, single_block_source<T> source,
                                              transfer_direction direction, const context& ctx,
                                              const char* missing) {
    if (source.layout_form) {
      if (source.view == nullptr or not source.view->is_valid()) [[unlikely]]
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "missing {}", missing);
      return xfer_plan<T>::one(base, *source.view, direction);
    }
    return xfer_plan<T>::one_base(base, source.elements, direction);
  }

  // A restricted side is the side MPI restricts to touching each location at most once: every
  // receive side, plus scatterv's root send (MPI 4.1 6.5/6.6/6.7/6.8).  alltoallv's send side
  // is read-only, so aliasing send blocks are legal and skip the exact type-map merge.
  // `may_prime` reserves the scratch bound of an in-place local contribution that is copied
  // out of the caller's receive buffer before the call (gatherv/allgatherv rank block,
  // alltoallv every block); it never changes the wire form.
  template<class T>
  [[nodiscard]] xfer_plan<T> make_many_plan(const T* base, variable_block_source<T> source,
                                            int peers, transfer_direction direction,
                                            alias_policy aliasing, prime_policy priming,
                                            const context& ctx) {
    const std::size_t n = static_cast<std::size_t>(peers);
    xfer_plan<T> plan;
    if (source.dense()) {
      plan = xfer_plan<T>::many_counts(base, source.values, source.displs, n, direction);
    } else {
      require_peers(source.layouts.size(), peers, ctx);
      plan = xfer_plan<T>::many_layouts(base, source.layouts, source.displs, n, direction,
                                        priming);
    }
    if (aliasing == alias_policy::restricted)
      plan.require_disjoint_blocks(ctx);
    return plan;
  }

  enum class variable_family : unsigned char { gatherv, scatterv, allgatherv, alltoallv };

  // One direction of a variable family: the caller base pointer, its flat element count, and
  // exactly one descriptor form.  `single` is the one-block form (gatherv/allgatherv send,
  // scatterv receive); `many` is the per-peer form (scatterv/alltoallv send, gatherv/
  // allgatherv/alltoallv receive).
  template<class T>
  struct variable_side {
    const T* data{nullptr};
    count_t elements{0};
    single_block_source<T> single{};
    variable_block_source<T> many{};
  };

  template<class T>
  struct variable_options {
    variable_family family{variable_family::gatherv};
    int root{0};
    variable_side<T> send{};
    variable_side<T> recv{};
    place_state place{place_state::data};
  };

  // One plan builder for the four variable families.  Role resolution, in-place legality,
  // restricted-side bounds/alias checks, staging, priming and the local self-block
  // consistency checks live here exactly once; the family differences are the used sides and
  // the direction each descriptor belongs to.
  template<class T>
  [[nodiscard]] variable_call<T> make_variable_plan(const mpi_communicator& c,
                                                    const variable_options<T>& o,
                                                    const context& ctx) {
    require_collective_communicator(c, ctx);
    const bool in_place = o.place == place_state::mpi_in_place;
    const bool root_family =
        o.family == variable_family::gatherv or o.family == variable_family::scatterv;
    const int peers = peer_count(c);

    root_role role = root_role::intra_nonroot;
    if (root_family)
      role = classify_root(c, o.root, ctx);
    if (in_place) {
      if (not root_family) {
        if (c.is_inter()) [[unlikely]]
          throw_invalid(ctx.operation,
                        "in-place collectives are invalid on an intercommunicator");
      } else if (role != root_role::intra_root) [[unlikely]] {
        throw_invalid(ctx.operation,
                      o.family == variable_family::gatherv
                          ? "in-place gatherv requires an intracommunicator root"
                          : "in-place scatterv requires an intracommunicator root");
      }
    }
    const bool intra_root = role == root_role::intra_root;
    const bool intra_nonroot = role == root_role::intra_nonroot;
    const bool inter_remote = role == root_role::inter_remote;
    const bool root_side = intra_root or role == root_role::inter_root;

    bool send_used = false;
    bool recv_used = false;
    switch (o.family) {
      case variable_family::gatherv:
        send_used = not in_place and (intra_root or intra_nonroot or inter_remote);
        recv_used = root_side;
        break;
      case variable_family::scatterv:
        send_used = root_side;
        recv_used = not in_place and (intra_root or intra_nonroot or inter_remote);
        break;
      case variable_family::allgatherv:
      case variable_family::alltoallv:
        send_used = not in_place;
        recv_used = true;
        break;
    }

    // Validate every used multi-block descriptor before any plan is built, so a negative
    // count/displacement is rejected before the first MPI call.  The receive side
    // is validated first for the no-root families, matching the earlier rejection point.
    const auto validate_many = [&](const variable_side<T>& side, const char* name) {
      require_variable_blocks_fit<T>(side.many, peers, side.elements, ctx, name);
    };
    if (recv_used and o.family != variable_family::scatterv)
      validate_many(o.recv, "receive");
    if (send_used and
        (o.family == variable_family::scatterv or o.family == variable_family::alltoallv)) {
      validate_many(o.send, "send");
      if (o.family == variable_family::scatterv and not o.send.many.dense()) {
        // Scatterv's root send is the read-restricted side (MPI 4.1 6.6): a location read more
        // than once is erroneous, and the zero-displacement convenience must not alias
        // blocks.  A dense block is one packed interval by construction, so only layouts need
        // the per-descriptor map check.
        for (std::size_t i = 0; i < o.send.many.layouts.size(); ++i) {
          if (not o.send.many.layouts[i].is_valid()) [[unlikely]]
            throw_invalid(ctx.operation, "invalid layout in a block array");
          require_disjoint_layout(o.send.many.layouts[i], ctx, "root send layout");
        }
      }
    }

    variable_call<T> call;
    const auto build_send = [&] {
      // MPI_IN_PLACE occupies the send position of every variable family except scatterv,
      // whose in-place position is the single-block receive (MPI 4.1 §6.6).
      if (in_place and o.family != variable_family::scatterv) {
        call.send = xfer_plan<T>::in_place();
        return;
      }
      if (not send_used) {
        call.send = xfer_plan<T>::absent(transfer_direction::send);
        return;
      }
      if (o.family == variable_family::gatherv or o.family == variable_family::allgatherv) {
        call.send = make_single_plan(o.send.data, o.send.single, transfer_direction::send, ctx,
                                     "send layout for the block form");
        return;
      }
      call.send =
          make_many_plan<T>(o.send.data, o.send.many, peers, transfer_direction::send,
                            o.family == variable_family::scatterv ? alias_policy::restricted
                                                                  : alias_policy::read_only,
                            prime_policy::none, ctx);
    };

    const auto build_recv = [&] {
      if (in_place and o.family == variable_family::scatterv) {
        call.recv = xfer_plan<T>::in_place();
        return;
      }
      if (not recv_used) {
        call.recv = xfer_plan<T>::absent(transfer_direction::receive);
        return;
      }
      if (o.family == variable_family::scatterv) {
        call.recv = make_single_plan(o.recv.data, o.recv.single, transfer_direction::receive,
                                     ctx, "receive layout for the block form");
        return;
      }
      call.recv =
          make_many_plan<T>(o.recv.data, o.recv.many, peers, transfer_direction::receive,
                            alias_policy::restricted,
                            in_place ? prime_policy::may_prime : prime_policy::none, ctx);
      if (in_place) {
        // The local contribution is already in the caller's receive buffer.  gatherv and
        // allgatherv prime this rank's block; alltoallv's whole receive buffer is the input.
        if (o.family == variable_family::alltoallv)
          call.recv.prime_all(ctx);
        else
          call.recv.prime_block(static_cast<std::size_t>(c.rank()), ctx);
      }
    };

    // The root families stage the send side first; the no-root families build the receive
    // side first (the historical order, so local rejection points are unchanged).
    switch (o.family) {
      case variable_family::gatherv:
      case variable_family::scatterv:
        build_send();
        build_recv();
        break;
      case variable_family::allgatherv:
      case variable_family::alltoallv:
        build_recv();
        build_send();
        break;
    }

    // Intracommunicator self-block consistency: the local send block and the corresponding
    // receive block must agree (MPI 4.1 6.5/6.6/6.7/6.8).  An intercommunicator's directions
    // are in different groups; MPI checks the pairwise signatures instead.
    switch (o.family) {
      case variable_family::gatherv:
        if (intra_root and not in_place and send_used and recv_used) {
          require_same_variable_elements(
              call.send.block_elements(0),
              call.recv.block_elements(static_cast<std::size_t>(c.rank())), ctx);
        }
        break;
      case variable_family::scatterv:
        if (intra_root and not in_place and send_used and recv_used) {
          require_same_variable_elements(
              call.send.block_elements(static_cast<std::size_t>(c.rank())),
              call.recv.block_elements(0), ctx);
        }
        break;
      case variable_family::allgatherv:
        if (not in_place and not c.is_inter()) {
          require_same_variable_elements(
              call.send.block_elements(0),
              call.recv.block_elements(static_cast<std::size_t>(c.rank())), ctx);
        }
        break;
      case variable_family::alltoallv:
        if (not in_place and not c.is_inter()) {
          require_same_variable_elements(
              call.send.block_elements(static_cast<std::size_t>(c.rank())),
              call.recv.block_elements(static_cast<std::size_t>(c.rank())), ctx);
        }
        break;
    }
    return call;
  }

  // Reduction helpers.  A reduction resolves one reduction_operation.  A raw MPI_Op and a
  // functor mapped to a predefined operation keep native_datatype<T>() and an element count
  // of native T images; an mp1-managed functor callback is typed over T's native map
  // (reduction_image_datatype<T>(): the native typed leaves at extent sizeof(T), no MPI_BYTE
  // component) so its T[len] indexing addresses the listed fields at their C++ offsets.
  // Layouts are flat only under the exact base-repetition predicate, otherwise both
  // directions stage.  A staged managed-functor reduction zero-fills its full-T stage slots
  // before packing the layout's mapped positions for local determinism; padding and unlisted
  // members are not part of the image and are never transmitted.  Completion unpacks only the
  // mapped positions into the caller's buffer, so the caller's unmapped bytes are preserved.
  // ---------------------------------------------------------------------------------------

  template<class T>
  void require_reduction_base(const context& ctx) {
    const datatype_info& info = datatype_info_of<T>();
    if (info.overlap == overlap_state::overlap) [[unlikely]] {
      throw_invalid(ctx.operation,
                    "native_datatype<T>() has overlapping type-map entries; an overlapping "
                    "reduction receive is erroneous");
    }
  }

  // ---------------------------------------------------------------------------------------
  // Native call boundary: exactly one call per operation, always the _c spelling.
  // ---------------------------------------------------------------------------------------

  inline void native_barrier(MPI_Comm comm) {
    check_mpi(MPI_Barrier(comm), "MPI_Barrier");
  }

  [[nodiscard]] inline request native_ibarrier(MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Ibarrier(comm, &handle), "MPI_Ibarrier");
    return request::from_collective(handle);
  }

  inline void native_bcast(const fixed_call& call, int root, MPI_Comm comm) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI_Bcast writes its buffer
    check_mpi(MPI_Bcast_c(const_cast<void*>(call.send_ptr), call.send_count, call.send_type,
                          root, comm),
              "MPI_Bcast_c");
  }

  [[nodiscard]] inline request native_ibcast(const fixed_call& call, int root, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI_Ibcast writes its buffer
    check_mpi(MPI_Ibcast_c(const_cast<void*>(call.send_ptr), call.send_count, call.send_type,
                           root, comm, &handle),
              "MPI_Ibcast_c");
    return request::from_collective(handle);
  }

  inline void native_gather(const fixed_call& call, int root, MPI_Comm comm) {
    check_mpi(MPI_Gather_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                           call.recv_count, call.recv_type, root, comm),
              "MPI_Gather_c");
  }

  [[nodiscard]] inline request native_igather(const fixed_call& call, int root, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Igather_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                            call.recv_count, call.recv_type, root, comm, &handle),
              "MPI_Igather_c");
    return request::from_collective(handle);
  }

  inline void native_scatter(const fixed_call& call, int root, MPI_Comm comm) {
    check_mpi(MPI_Scatter_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                            call.recv_count, call.recv_type, root, comm),
              "MPI_Scatter_c");
  }

  [[nodiscard]] inline request native_iscatter(const fixed_call& call, int root,
                                               MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iscatter_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                             call.recv_count, call.recv_type, root, comm, &handle),
              "MPI_Iscatter_c");
    return request::from_collective(handle);
  }

  inline void native_allgather(const fixed_call& call, MPI_Comm comm) {
    check_mpi(MPI_Allgather_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                              call.recv_count, call.recv_type, comm),
              "MPI_Allgather_c");
  }

  [[nodiscard]] inline request native_iallgather(const fixed_call& call, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iallgather_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                               call.recv_count, call.recv_type, comm, &handle),
              "MPI_Iallgather_c");
    return request::from_collective(handle);
  }

  inline void native_alltoall(const fixed_call& call, MPI_Comm comm) {
    check_mpi(MPI_Alltoall_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                             call.recv_count, call.recv_type, comm),
              "MPI_Alltoall_c");
  }

  [[nodiscard]] inline request native_ialltoall(const fixed_call& call, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Ialltoall_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                              call.recv_count, call.recv_type, comm, &handle),
              "MPI_Ialltoall_c");
    return request::from_collective(handle);
  }

  // The twelve variable-family native entries (MPI 4.1 6.5-6.8 and their _init forms).  Each
  // passes native_datatype<T>() on both sides with the plan's element counts and extent-unit
  // displacements; the byte type is used only when T's own native datatype is the byte type
  // (std::byte or a custom trait), never for a typed T.  A single-block position is a scalar
  // MPI_Count; a many-block position is the plan's const MPI_Count[]/const MPI_Aint[] arrays.
  // In an ignored position the plan supplies a per-position dummy buffer and count 0
  // (MPI 4.1 6.2.3: the argument is still passed), and MPI_IN_PLACE is passed verbatim where
  // the family defines it.
  template<class T>
  inline void native_gatherv(const variable_call<T>& call, int root, MPI_Comm comm) {
    check_mpi(MPI_Gatherv_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                            call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                            native_datatype<T>(), root, comm),
              "MPI_Gatherv_c");
  }

  template<class T>
  [[nodiscard]] inline request native_igatherv(const variable_call<T>& call, int root,
                                               MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Igatherv_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                             call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                             native_datatype<T>(), root, comm, &handle),
              "MPI_Igatherv_c");
    return request::from_collective(handle);
  }

  template<class T>
  inline void native_scatterv(const variable_call<T>& call, int root, MPI_Comm comm) {
    check_mpi(MPI_Scatterv_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                             native_datatype<T>(), call.recv.mutable_ptr(), call.recv.count(),
                             native_datatype<T>(), root, comm),
              "MPI_Scatterv_c");
  }

  template<class T>
  [[nodiscard]] inline request native_iscatterv(const variable_call<T>& call, int root,
                                                MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iscatterv_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                              native_datatype<T>(), call.recv.mutable_ptr(), call.recv.count(),
                              native_datatype<T>(), root, comm, &handle),
              "MPI_Iscatterv_c");
    return request::from_collective(handle);
  }

  template<class T>
  inline void native_allgatherv(const variable_call<T>& call, MPI_Comm comm) {
    check_mpi(MPI_Allgatherv_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                               call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                               native_datatype<T>(), comm),
              "MPI_Allgatherv_c");
  }

  template<class T>
  [[nodiscard]] inline request native_iallgatherv(const variable_call<T>& call, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iallgatherv_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                                call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                                native_datatype<T>(), comm, &handle),
              "MPI_Iallgatherv_c");
    return request::from_collective(handle);
  }

  template<class T>
  inline void native_alltoallv(const variable_call<T>& call, MPI_Comm comm) {
    check_mpi(MPI_Alltoallv_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                              native_datatype<T>(), call.recv.mutable_ptr(), call.recv.counts(),
                              call.recv.displs(), native_datatype<T>(), comm),
              "MPI_Alltoallv_c");
  }

  template<class T>
  [[nodiscard]] inline request native_ialltoallv(const variable_call<T>& call, MPI_Comm comm) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Ialltoallv_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                         native_datatype<T>(), call.recv.mutable_ptr(), call.recv.counts(),
                         call.recv.displs(), native_datatype<T>(), comm, &handle),
        "MPI_Ialltoallv_c");
    return request::from_collective(handle);
  }

  // A managed callback image has true lower bound 0 because the native typed map was shifted
  // down by T's native true lower bound (operator.hpp).  The buffer pointer must therefore be
  // shifted up by the same amount: the mapped fields then land at their real T offsets.  The
  // sentinel and null pointers are never shifted.
  [[nodiscard]] inline const void* shift_reduction_buffer(const void* buffer,
                                                          count_t offset) noexcept {
    if (offset == 0 or buffer == nullptr or buffer == MPI_IN_PLACE)
      return buffer;
    return static_cast<const void*>(static_cast<const char*>(buffer) + offset);
  }

  [[nodiscard]] inline void* shift_reduction_buffer(void* buffer, count_t offset) noexcept {
    if (offset == 0 or buffer == nullptr or buffer == MPI_IN_PLACE)
      return buffer;
    return static_cast<void*>(static_cast<char*>(buffer) + offset);
  }

  struct reduction_buffers {
    const void* send;
    void* recv;

    reduction_buffers(const void* send_buffer, void* recv_buffer,
                      const reduction_operation& operation) noexcept
        : send(shift_reduction_buffer(send_buffer, operation.buffer_offset)),
          recv(shift_reduction_buffer(recv_buffer, operation.buffer_offset)) {
    }
  };

  inline void native_reduce(const void* send, void* recv, MPI_Count count,
                            const reduction_operation& operation, int root, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Reduce_c(buffers.send, buffers.recv, count, operation.datatype,
                           operation.handle, root, comm),
              "MPI_Reduce_c");
  }

  // MPI 4.1 §6.9.7: local reduction, no communicator and no MPI_IN_PLACE.
  inline void native_reduce_local(const void* send, void* recv, MPI_Count count,
                                  const reduction_operation& operation) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Reduce_local_c(buffers.send, buffers.recv, count, operation.datatype,
                                 operation.handle),
              "MPI_Reduce_local_c");
  }

  [[nodiscard]] inline request native_ireduce(const void* send, void* recv, MPI_Count count,
                                              const reduction_operation& operation, int root,
                                              MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Ireduce_c(buffers.send, buffers.recv, count, operation.datatype,
                            operation.handle, root, comm, &handle),
              "MPI_Ireduce_c");
    return request::from_collective(handle);
  }

  inline void native_allreduce(const void* send, void* recv, MPI_Count count,
                               const reduction_operation& operation, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Allreduce_c(buffers.send, buffers.recv, count, operation.datatype,
                              operation.handle, comm),
              "MPI_Allreduce_c");
  }

  [[nodiscard]] inline request native_iallreduce(const void* send, void* recv, MPI_Count count,
                                                 const reduction_operation& operation,
                                                 MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iallreduce_c(buffers.send, buffers.recv, count, operation.datatype,
                               operation.handle, comm, &handle),
              "MPI_Iallreduce_c");
    return request::from_collective(handle);
  }

  inline void native_reduce_scatter(const void* send, void* recv, const MPI_Count* counts,
                                    const reduction_operation& operation, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Reduce_scatter_c(buffers.send, buffers.recv, counts, operation.datatype,
                                   operation.handle, comm),
              "MPI_Reduce_scatter_c");
  }

  [[nodiscard]] inline request native_ireduce_scatter(const void* send, void* recv,
                                                      const MPI_Count* counts,
                                                      const reduction_operation& operation,
                                                      MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Ireduce_scatter_c(buffers.send, buffers.recv, counts, operation.datatype,
                                    operation.handle, comm, &handle),
              "MPI_Ireduce_scatter_c");
    return request::from_collective(handle);
  }

  inline void native_reduce_scatter_block(const void* send, void* recv, MPI_Count count,
                                          const reduction_operation& operation, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Reduce_scatter_block_c(buffers.send, buffers.recv, count, operation.datatype,
                                         operation.handle, comm),
              "MPI_Reduce_scatter_block_c");
  }

  [[nodiscard]] inline request native_ireduce_scatter_block(
      const void* send, void* recv, MPI_Count count, const reduction_operation& operation,
      MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Ireduce_scatter_block_c(buffers.send, buffers.recv, count, operation.datatype,
                                          operation.handle, comm, &handle),
              "MPI_Ireduce_scatter_block_c");
    return request::from_collective(handle);
  }

  inline void native_scan(const void* send, void* recv, MPI_Count count,
                          const reduction_operation& operation, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Scan_c(buffers.send, buffers.recv, count, operation.datatype,
                         operation.handle, comm),
              "MPI_Scan_c");
  }

  [[nodiscard]] inline request native_iscan(const void* send, void* recv, MPI_Count count,
                                            const reduction_operation& operation,
                                            MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iscan_c(buffers.send, buffers.recv, count, operation.datatype,
                          operation.handle, comm, &handle),
              "MPI_Iscan_c");
    return request::from_collective(handle);
  }

  inline void native_exscan(const void* send, void* recv, MPI_Count count,
                            const reduction_operation& operation, MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    check_mpi(MPI_Exscan_c(buffers.send, buffers.recv, count, operation.datatype,
                           operation.handle, comm),
              "MPI_Exscan_c");
  }

  [[nodiscard]] inline request native_iexscan(const void* send, void* recv, MPI_Count count,
                                              const reduction_operation& operation,
                                              MPI_Comm comm) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Iexscan_c(buffers.send, buffers.recv, count, operation.datatype,
                            operation.handle, comm, &handle),
              "MPI_Iexscan_c");
    return request::from_collective(handle);
  }

  // ---------------------------------------------------------------------------------------
  // Persistent collectives (MPI 4.0 section 6.13). The init call is the only new native
  // boundary: one MPI_*_init per family, always the _c spelling, and the returned handle is a
  // persistent collective request. Everything else (start/complete/free) goes through request.
  // ---------------------------------------------------------------------------------------

  inline request native_barrier_init(MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Barrier_init(comm, info, &handle), "MPI_Barrier_init");
    return request::from_persistent_collective(handle);
  }

  inline request native_bcast_init(const fixed_call& call, int root, MPI_Comm comm,
                                   MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI_Bcast_init writes its buffer
    check_mpi(MPI_Bcast_init_c(const_cast<void*>(call.send_ptr), call.send_count,
                               call.send_type, root, comm, info, &handle),
              "MPI_Bcast_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_gather_init(const fixed_call& call, int root, MPI_Comm comm,
                                    MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Gather_init_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                                call.recv_count, call.recv_type, root, comm, info, &handle),
              "MPI_Gather_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_scatter_init(const fixed_call& call, int root, MPI_Comm comm,
                                     MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Scatter_init_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                                 call.recv_count, call.recv_type, root, comm, info, &handle),
              "MPI_Scatter_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_allgather_init(const fixed_call& call, MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Allgather_init_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                             call.recv_count, call.recv_type, comm, info, &handle),
        "MPI_Allgather_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_alltoall_init(const fixed_call& call, MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Alltoall_init_c(call.send_ptr, call.send_count, call.send_type, call.recv_ptr,
                                  call.recv_count, call.recv_type, comm, info, &handle),
              "MPI_Alltoall_init_c");
    return request::from_persistent_collective(handle);
  }

  template<class T>
  [[nodiscard]] inline request native_gatherv_init(const variable_call<T>& call, int root,
                                                   MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Gatherv_init_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                           call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                           native_datatype<T>(), root, comm, info, &handle),
        "MPI_Gatherv_init_c");
    return request::from_persistent_collective(handle);
  }

  template<class T>
  [[nodiscard]] inline request native_scatterv_init(const variable_call<T>& call, int root,
                                                    MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Scatterv_init_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                            native_datatype<T>(), call.recv.mutable_ptr(), call.recv.count(),
                            native_datatype<T>(), root, comm, info, &handle),
        "MPI_Scatterv_init_c");
    return request::from_persistent_collective(handle);
  }

  template<class T>
  [[nodiscard]] inline request native_allgatherv_init(const variable_call<T>& call,
                                                      MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Allgatherv_init_c(call.send.ptr(), call.send.count(), native_datatype<T>(),
                              call.recv.mutable_ptr(), call.recv.counts(), call.recv.displs(),
                              native_datatype<T>(), comm, info, &handle),
        "MPI_Allgatherv_init_c");
    return request::from_persistent_collective(handle);
  }

  template<class T>
  [[nodiscard]] inline request native_alltoallv_init(const variable_call<T>& call,
                                                     MPI_Comm comm, MPI_Info info) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Alltoallv_init_c(call.send.ptr(), call.send.counts(), call.send.displs(),
                             native_datatype<T>(), call.recv.mutable_ptr(), call.recv.counts(),
                             call.recv.displs(), native_datatype<T>(), comm, info, &handle),
        "MPI_Alltoallv_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_reduce_init(const void* send, void* recv, MPI_Count count,
                                    const reduction_operation& operation, int root,
                                    MPI_Comm comm, MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Reduce_init_c(buffers.send, buffers.recv, count, operation.datatype,
                                operation.handle, root, comm, info, &handle),
              "MPI_Reduce_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_allreduce_init(const void* send, void* recv, MPI_Count count,
                                       const reduction_operation& operation, MPI_Comm comm,
                                       MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Allreduce_init_c(buffers.send, buffers.recv, count, operation.datatype,
                                   operation.handle, comm, info, &handle),
              "MPI_Allreduce_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_reduce_scatter_init(const void* send, void* recv,
                                            const MPI_Count* counts,
                                            const reduction_operation& operation, MPI_Comm comm,
                                            MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Reduce_scatter_init_c(buffers.send, buffers.recv, counts, operation.datatype,
                                        operation.handle, comm, info, &handle),
              "MPI_Reduce_scatter_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_reduce_scatter_block_init(const void* send, void* recv, MPI_Count count,
                                                  const reduction_operation& operation,
                                                  MPI_Comm comm, MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Reduce_scatter_block_init_c(buffers.send, buffers.recv, count, operation.datatype,
                                        operation.handle, comm, info, &handle),
        "MPI_Reduce_scatter_block_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_scan_init(const void* send, void* recv, MPI_Count count,
                                  const reduction_operation& operation, MPI_Comm comm,
                                  MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Scan_init_c(buffers.send, buffers.recv, count, operation.datatype,
                              operation.handle, comm, info, &handle),
              "MPI_Scan_init_c");
    return request::from_persistent_collective(handle);
  }

  inline request native_exscan_init(const void* send, void* recv, MPI_Count count,
                                    const reduction_operation& operation, MPI_Comm comm,
                                    MPI_Info info) {
    const reduction_buffers buffers{send, recv, operation};
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Exscan_init_c(buffers.send, buffers.recv, count, operation.datatype,
                                operation.handle, comm, info, &handle),
              "MPI_Exscan_init_c");
    return request::from_persistent_collective(handle);
  }

}  // namespace mp1::detail

#endif
