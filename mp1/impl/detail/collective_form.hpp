#if !(defined MP1_DETAIL_COLLECTIVE_FORM_HPP)

#define MP1_DETAIL_COLLECTIVE_FORM_HPP

// Collective buffer forms: the value/range/region forms, the aggregate/single-block
// classification, and the request-owned staging/type-map plan shared by the blocking,
// nonblocking and persistent fixed-collective adapters.
//
// The public surface has exactly one declaration per family per lifetime.  Inside it the two
// buffer positions are classified at compile time into
//
//   * value        a single T (count 1, base datatype)
//   * span/range   a contiguous element range (count = size, base datatype)
//   * sequence     a non-contiguous forward range (send only; staged to a flat vector)
//   * region       a pointer + layout (typed occurrence)
//
// and at run time into the MPI *aggregate* or *single-block* position of the family.  A flat
// aggregate (a span position, or a region whose layout records the base-repetition fact) lowers
// to base elements and is divided by the peer count by the engine's fixed plan builder.  A
// non-flat aggregate region is packed into a flat staging vector before MPI and unpacked after
// completion: MPI repeats a datatype per rank in an aggregate position, so one layout cannot
// describe the whole aggregate.  A non-flat single-block region is passed as one datatype
// occurrence; a blocking call borrows it and a nonblocking/persistent call owns a duplicate
// (`MPI_Type_dup`).
//
// Every plan is built before the native call.  Blocking calls finish (unpack) locally; the
// nonblocking and persistent adapters move the plan into `form_payload`, whose `finish()` runs
// at completion and whose `prepare_start()` replays the local pack for every persistent cycle.
// This header only converts the public forms into the named `fixed_position` options the one
// engine builder consumes; it contains no role, peer-count or native-call logic.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/collective_engine.hpp"
#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/request_payload.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <cstddef>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1::detail {

  // A staged reduction writes its receive block into the caller buffer at completion (or
  // reads it as the in-place contribution).  A null destination with a non-empty block is
  // locally detectable, so reject it before the collective call instead of silently
  // dropping the result (MPI 4.1 6.9 receive-buffer significance).  This lives here because
  // both the staged form aggregates and the reduction adapters need the pre-call check.
  template<class T>
  void require_staged_receive_pointer(const T* recv, const layout<T>& l, const context& ctx) {
    if (recv == nullptr and l.byte_size() != 0) [[unlikely]] {
      throw_invalid(ctx.operation, "null receive pointer for a non-empty layout");
    }
  }

  // ---------------------------------------------------------------------------------------
  // Element type selection and the same-element constraint.
  //
  // The public declarations constrain the two positions by the position concept only.  MPI
  // still requires matching type signatures, so a call whose two real positions disagree on
  // the element type is rejected at compile time by the wrapper's requires clause (the
  // in-place sentinel has no element type and takes it from the other position).
  // ---------------------------------------------------------------------------------------

  template<class B>
  inline constexpr bool is_in_place_arg_v = std::same_as<std::remove_cvref_t<B>, in_place_t>;

  enum class form_kind { in_place, region, value, range };

  template<class B>
  inline constexpr form_kind form_kind_of_v = [] {
    using plain = std::remove_cvref_t<B>;
    if constexpr (is_in_place_arg_v<B>)
      return form_kind::in_place;
    else if constexpr (is_region_v<plain>)
      return form_kind::region;
    else if constexpr (element_value<plain>)
      return form_kind::value;
    else
      return form_kind::range;
  }();

  template<class B, form_kind = form_kind_of_v<B>>
  struct form_element;

  template<class B>
  struct form_element<B, form_kind::in_place> {
    using type = void;
  };

  template<class B>
  struct form_element<B, form_kind::region> {
    using type = std::remove_cv_t<region_element_t<std::remove_cvref_t<B>>>;
  };

  template<class B>
  struct form_element<B, form_kind::value> {
    using type = std::remove_cv_t<std::remove_cvref_t<B>>;
  };

  template<class B>
  struct form_element<B, form_kind::range> {
    using type = std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<B>>>;
  };

  template<class B>
  using form_element_t = typename form_element<B>::type;

  // The element type of a two-position family: the first real position wins.
  template<class S, class R>
  using collective_element_t =
      std::conditional_t<is_in_place_arg_v<S>, form_element_t<R>, form_element_t<S>>;

  template<class S, class R>
  inline constexpr bool compatible_form_elements_v =
      is_in_place_arg_v<S> or is_in_place_arg_v<R> or
      std::same_as<form_element_t<S>, form_element_t<R>>;

  // ---------------------------------------------------------------------------------------
  // One classified buffer position.
  // ---------------------------------------------------------------------------------------

  template<class T>
  struct form_position {
    const T* data{nullptr};
    T* mutable_data{nullptr};
    const layout<T>* map{nullptr};
    count_t elements{0};
    stage_buffer<T> owned{};  // sequence staging (send only)
    bool typed{false};
    bool sequence{false};
    bool in_place{false};

    [[nodiscard]] count_t element_count() const noexcept {
      return sequence ? static_cast<count_t>(owned.size()) : elements;
    }

    [[nodiscard]] const T* send_data() const noexcept {
      return sequence ? owned.data() : data;
    }
  };

  // `T` is supplied explicitly so an `in_place_t` send never instantiates buffer_element_t.
  template<class T, class B>
  [[nodiscard]] form_position<T> make_form_position(B&& buffer, const context& ctx) {
    using plain = std::remove_cvref_t<B>;
    form_position<T> position;
    if constexpr (std::same_as<plain, in_place_t>) {
      position.in_place = true;
      return position;
    } else if constexpr (is_region_v<plain>) {
      position.map = std::addressof(buffer.map());
      position.typed = true;
      position.elements = buffer.map().size();
      position.data = buffer.data();
      if constexpr (not std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>)
        position.mutable_data = buffer.data();
      return position;
    } else if constexpr (element_value<plain>) {
      position.data = std::addressof(buffer);
      position.elements = 1;
      if constexpr (not std::is_const_v<std::remove_reference_t<B>>)
        position.mutable_data = std::addressof(buffer);
      return position;
    } else if constexpr (element_buffer<plain>) {
      position.data = std::ranges::data(buffer);
      position.elements =
          checked_size(static_cast<std::size_t>(std::ranges::ssize(buffer)), ctx);
      // Mutability is a property of the *elements*, not of the view object: a `const
      // std::span<int>` still yields `int&`, while a `const std::vector<int>&` yields
      // `const int&`.  `output_range` is the same criterion the receive concepts use.
      if constexpr (std::ranges::output_range<B, std::ranges::range_value_t<plain>>) {
        // The receive concepts already reject a const recv buffer, so a mutable view is only
        // used on a legal receive position.
        position.mutable_data = const_cast<T*>(position.data);
      }
      return position;
    } else {
      static_assert(element_sequence<plain>,
                    "mp1: a collective buffer must be a value, a contiguous element range, a "
                    "non-contiguous sized forward range (send only), or a region");
      position.sequence = true;
      position.elements =
          checked_size(static_cast<std::size_t>(std::ranges::ssize(buffer)), ctx);
      position.owned.resize(static_cast<std::size_t>(position.elements));
      std::ranges::copy(buffer, position.owned.data());
      position.data = position.owned.data();
      return position;
    }
  }

  // ---------------------------------------------------------------------------------------
  // The exact-byte preflight shared by every reduction plan and by reduce_local.
  //
  // MPI 4.1 §5.1.11: a used position is empty only when its total byte size is zero, not when
  // its element count is zero.  A typed position is measured by layout.byte_size(); a
  // flat/sequence position by checked n * size(native_datatype<T>()).  An empty position may
  // be null and is presented to MPI as the per-slot dummy; a non-empty position with a null
  // pointer throws std::invalid_argument before any MPI data call or managed MPI_Op creation.
  // Ignored roles (a non-root receive, a no-op intercommunicator position), the in-place send
  // slot and MPI_IN_PLACE are never inspected.
  // ---------------------------------------------------------------------------------------

  template<class T>
  void require_used_position(buffer_slot slot, const form_position<T>& position, bool used,
                             const context& ctx) {
    if (not used or position.in_place)
      return;
    MPI_Count bytes{0};
    if (position.typed) {
      if (not position.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      bytes = static_cast<MPI_Count>(position.map->byte_size());
    } else {
      bytes = checked_bytes(static_cast<MPI_Count>(position.element_count()),
                            base_datatype<T>().size, ctx);
    }
    const void* pointer = slot == buffer_slot::recv
                              ? static_cast<const void*>(position.mutable_data)
                              : static_cast<const void*>(position.send_data());
    require_used_bytes(slot, pointer, bytes, ctx);
  }

  // Resolve the role once and preflight the used positions of a fixed reduction family.  The
  // public adapters call this before reduction_op<T>(), so a managed functor's operation handle
  // (and its callback datatype) is not created for a call that is already invalid.
  template<class T>
  void validate_reduction_positions(const mpi_communicator& c, collective_family family,
                                    int root, const form_position<T>& send,
                                    const form_position<T>& recv, const context& ctx) {
    const role_sides sides = resolve_sides(family, c, root, place_state_of(send.in_place), ctx);
    require_used_position(buffer_slot::send, send, sides.send_used and not send.in_place, ctx);
    require_used_position(buffer_slot::recv, recv, sides.recv_used, ctx);
  }

  // ---------------------------------------------------------------------------------------
  // Per-call plan: the native arguments plus every locally-owned object the call needs.
  // Blocking calls use it on the stack; nonblocking/persistent calls move it into
  // form_payload.
  // ---------------------------------------------------------------------------------------

  // How long the plan must outlive this call.  A blocking call borrows every buffer and
  // layout for the native call; an asynchronous or persistent request owns the staged
  // buffers, the duplicated type maps and the restart recipe.
  enum class form_lifetime : unsigned char { blocking, asynchronous, persistent };

  template<class T>
  struct form_plan {
    fixed_call call{};
    stage_buffer<T> send_stage{};
    stage_buffer<T> recv_stage{};
    // One owning duplicate per distinct type map completion/restart reads.
    std::vector<std::unique_ptr<layout<T>>> owned_maps{};
    // The owning duplicate behind a typed single-block native datatype argument.
    std::unique_ptr<layout<T>> direct_layout{};

    // Completion unpack (a staged receive, including the in-place aggregate staging).
    bool unpack{false};
    T* unpack_to{nullptr};
    const layout<T>* unpack_map{nullptr};

    // Persistent re-pack of a staged send.
    bool repack{false};
    const T* repack_source{nullptr};
    const layout<T>* repack_map{nullptr};

    // Persistent re-pack of an in-place aggregate receive input.
    bool prime{false};
    const T* prime_source{nullptr};
    const layout<T>* prime_map{nullptr};

    context ctx{"mp1 collective"};

    // Duplicate exactly the type maps completion/restart reads after the call returns.  One
    // shared layout used by several roles costs one MPI_Type_dup.
    void own_metadata() {
      const layout<T>* originals[3] = {unpack_map, prime_map, repack_map};
      const layout<T>* resolved[3] = {nullptr, nullptr, nullptr};
      for (int i = 0; i < 3; ++i) {
        if (originals[i] == nullptr)
          continue;
        for (int j = 0; j < i; ++j) {
          if (originals[j] == originals[i]) {
            resolved[i] = resolved[j];
            break;
          }
        }
        if (resolved[i] == nullptr) {
          owned_maps.push_back(std::make_unique<layout<T>>(*originals[i]));
          resolved[i] = owned_maps.back().get();
        }
      }
      unpack_map = resolved[0];
      prime_map = resolved[1];
      repack_map = resolved[2];
    }

    void prepare_start() {
      if (repack)
        stage_pack_blocks(repack_source, *repack_map, 1, send_stage, 0, ctx);
      if (prime)
        stage_pack_blocks(prime_source, *prime_map, 1, recv_stage, 0, ctx);
    }

    void finish() {
      if (unpack)
        stage_unpack_blocks(recv_stage, 0, unpack_to, *unpack_map, 1, ctx);
    }
  };

  template<class T>
  class form_payload final : public request_payload {
  public:
    explicit form_payload(form_plan<T> plan) : plan_{std::move(plan)} {
    }

    form_payload(const form_payload&) = delete;
    form_payload& operator=(const form_payload&) = delete;

    void prepare_start() override {
      plan_.prepare_start();
    }

    void complete(const status&) override {
      plan_.finish();
    }

  private:
    form_plan<T> plan_;
  };

  // Attach a payload only when the request (or a later completion) reads something the plan
  // owns: staging vectors, an owned type map, or an unpack/repack recipe.  The staged send of
  // a non-contiguous sequence is the critical case: MPI may read the native send buffer until
  // the nonblocking collective completes (MPI 4.1 §3.7.2, §6.12), so `send_stage` must outlive
  // this call.  The plan is moved into the payload, which `request::wait`/`test` destroy only
  // after MPI completion, so the pointer captured in `call` stays valid for the whole request
  // lifetime.  Blocking calls never attach: their stack plan lives through the native call.
  template<class T>
  void attach_form_payload(request& result, form_plan<T> plan) {
    const bool owns_staging = not plan.send_stage.empty() or not plan.recv_stage.empty();
    const bool needs_payload = plan.unpack or plan.repack or plan.prime or
                               plan.direct_layout != nullptr or not plan.owned_maps.empty() or
                               owns_staging;
    if (needs_payload)
      attach_payload(result, std::make_unique<form_payload<T>>(std::move(plan)));
  }

  // ---------------------------------------------------------------------------------------
  // Small conversions.
  // ---------------------------------------------------------------------------------------

  template<class T>
  struct native_block {
    const T* ptr{nullptr};
    count_t count{0};
    const layout<T>* map{nullptr};
  };

  // A single-block position.  A region whose construction proved the base-repetition fact
  // lowers to base elements (no datatype, no copy); any other region stays one typed datatype
  // occurrence (count is unused and irrelevant then).
  //
  // A wire datatype with size 0 has an empty type signature (MPI 4.1 §5.1.11): count N and
  // count 0 describe the same no-data message.  Normalize the count to 0 so a positive count
  // of a zero-size type never reaches the provider (MPICH 4.3.2 has been recorded to abort on
  // that standard-conforming form in the variable collectives) and so the count matches the
  // zero-byte position the wrapper reports.
  template<class T>
  [[nodiscard]] native_block<T> single_block(const form_position<T>& position,
                                             const context& ctx) {
    if (position.typed) {
      if (not position.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      if (position.map->is_base_repetition())
        return native_block<T>{repetition_ptr(position.data, *position.map, ctx),
                               position.map->byte_size() == 0 ? 0 : position.map->size(),
                               nullptr};
      return native_block<T>{position.data, position.map->byte_size() == 0 ? 0 : 1,
                             position.map};
    }
    return native_block<T>{position.send_data(),
                           base_datatype<T>().size == 0 ? 0 : position.element_count(),
                           nullptr};
  }

  template<class T>
  [[nodiscard]] native_block<T> single_block_mutable(const form_position<T>& position,
                                                     const context& ctx) {
    if (position.typed) {
      if (not position.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      if (position.map->is_base_repetition())
        return native_block<T>{repetition_ptr(position.mutable_data, *position.map, ctx),
                               position.map->byte_size() == 0 ? 0 : position.map->size(),
                               nullptr};
      return native_block<T>{position.mutable_data, position.map->byte_size() == 0 ? 0 : 1,
                             position.map};
    }
    return native_block<T>{position.mutable_data,
                           base_datatype<T>().size == 0 ? 0 : position.element_count(),
                           nullptr};
  }

  // A staged aggregate send: pack the whole aggregate layout into a flat vector of its logical
  // element count.  Unlisted positions are zeroed so no indeterminate byte reaches MPI (the
  // receiving layout ignores them, but a deterministic wire image keeps sanitizer/valgrind
  // runs quiet).
  template<class T>
  void stage_aggregate_send(form_plan<T>& plan, const form_position<T>& position,
                            const context& ctx) {
    if (not position.map->is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid layout");
    const std::size_t elements = static_cast<std::size_t>(position.map->size());
    plan.send_stage.resize(elements);
    std::fill_n(plan.send_stage.data(), elements, T{});
    stage_pack_blocks(position.data, *position.map, 1, plan.send_stage, 0, ctx);
    plan.repack = true;
    plan.repack_source = position.data;
    plan.repack_map = position.map;
  }

  // A staged aggregate receive: MPI writes a flat aggregate into recv_stage and completion
  // unpacks it into the caller's aggregate layout.  For an in-place position the caller's
  // buffer is also the input, so it is packed into the staging vector before the call and
  // re-packed before every persistent start.
  template<class T>
  void stage_aggregate_recv(form_plan<T>& plan, const form_position<T>& position,
                            prime_policy priming, const context& ctx) {
    if (not position.map->is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid layout");
    require_receive_layout(*position.map, ctx);
    // Validate the caller's destination before the native collective is issued, not only when
    // completion runs `stage_unpack_blocks`: a null receive pointer with a non-empty type map
    // is locally detectable and must be a pre-call `std::invalid_argument`.
    require_staged_receive_pointer(position.mutable_data, *position.map, ctx);
    const std::size_t elements = static_cast<std::size_t>(position.map->size());
    plan.recv_stage.resize(elements);
    std::fill_n(plan.recv_stage.data(), elements, T{});
    if (priming == prime_policy::may_prime) {
      // The aggregate input is the caller's mapped data; packing it in map order gives the
      // flat rank-order image MPI expects in the receive buffer of an in-place call.
      stage_pack_blocks(position.mutable_data, *position.map, 1, plan.recv_stage, 0, ctx);
      plan.prime = true;
      plan.prime_source = position.mutable_data;
      plan.prime_map = position.map;
    }
    plan.unpack = true;
    plan.unpack_to = position.mutable_data;
    plan.unpack_map = position.map;
  }

  // ---------------------------------------------------------------------------------------
  // One classified position -> one engine position.  The four capture helpers are the only
  // place where the public forms are lowered; every family below shares them.
  // ---------------------------------------------------------------------------------------

  // A single-block send: a non-contiguous sequence is packed to a flat copy; a value, range
  // or region becomes a native block (a region may lower to a base repetition).
  template<class T>
  [[nodiscard]] fixed_position<T> capture_single_send(form_plan<T>& plan,
                                                      const form_position<T>& position,
                                                      place_state place, const context& ctx) {
    fixed_position<T> captured;
    captured.extent = block_extent::single;
    captured.place = place;
    if (place == place_state::mpi_in_place)
      return captured;
    if (position.sequence) {
      const count_t elements = position.element_count();
      plan.send_stage.resize(static_cast<std::size_t>(elements));
      std::ranges::copy_n(position.owned.data(), position.owned.size(), plan.send_stage.data());
      captured.data = plan.send_stage.data();
      captured.elements = elements;
      return captured;
    }
    const native_block<T> block = single_block(position, ctx);
    captured.data = block.ptr;
    captured.elements = block.count;
    captured.map = block.map;
    return captured;
  }

  // A single-block receive: only the in-place sentinel or a value/range/region form.  The
  // receive concepts reject a non-contiguous sequence, so no staging is needed here.
  template<class T>
  [[nodiscard]] fixed_position<T> capture_single_recv(const form_position<T>& position,
                                                      place_state place, const context& ctx) {
    fixed_position<T> captured;
    captured.extent = block_extent::single;
    captured.place = place;
    if (place == place_state::mpi_in_place)
      return captured;
    const native_block<T> block = single_block_mutable(position, ctx);
    captured.data = block.ptr;
    captured.elements = block.count;
    captured.map = block.map;
    return captured;
  }

  // An aggregate send (scatter/alltoall).  A sequence stages to a flat total; a typed
  // aggregate lowers a base repetition to elements and stages any other map (the caller's
  // datatype cannot describe the whole aggregate); an ignored typed position is not staged.
  template<class T>
  [[nodiscard]] fixed_position<T> capture_aggregate_send(form_plan<T>& plan,
                                                         const form_position<T>& position,
                                                         side_use use, place_state place,
                                                         const context& ctx) {
    fixed_position<T> captured;
    captured.extent = block_extent::aggregate;
    captured.place = place;
    if (place == place_state::mpi_in_place)
      return captured;
    if (position.sequence) {
      const count_t elements = position.element_count();
      if (use == side_use::used) {
        plan.send_stage.resize(static_cast<std::size_t>(elements));
        std::ranges::copy_n(position.owned.data(), position.owned.size(),
                            plan.send_stage.data());
        captured.data = plan.send_stage.data();
      } else {
        captured.data = position.send_data();
      }
      captured.elements = elements;
      return captured;
    }
    if (position.typed) {
      if (not position.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      captured.elements = position.map->size();
      if (position.map->is_base_repetition()) {
        captured.data = repetition_ptr(position.data, *position.map, ctx);
      } else if (use == side_use::used) {
        stage_aggregate_send(plan, position, ctx);
        captured.data = plan.send_stage.data();
      } else {
        captured.data = position.send_data();
        captured.map = position.map;
      }
      return captured;
    }
    captured.data = position.send_data();
    captured.elements = position.element_count();
    return captured;
  }

  // An aggregate receive (gather/allgather/alltoall).  A typed base repetition lowers to
  // elements; any other typed map is staged and unpacked at completion (and primed before the
  // call/restart for an in-place contribution).  An ignored typed position is not staged.
  template<class T>
  [[nodiscard]] fixed_position<T> capture_aggregate_recv(form_plan<T>& plan,
                                                         const form_position<T>& position,
                                                         side_use use, place_state place,
                                                         prime_policy priming,
                                                         const context& ctx) {
    fixed_position<T> captured;
    captured.extent = block_extent::aggregate;
    captured.place = place;
    if (place == place_state::mpi_in_place)
      return captured;
    if (position.typed) {
      if (not position.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid layout");
      captured.elements = position.map->size();
      if (position.map->is_base_repetition()) {
        captured.data = repetition_ptr(position.mutable_data, *position.map, ctx);
      } else if (use == side_use::used) {
        stage_aggregate_recv(plan, position, priming, ctx);
        captured.data = plan.recv_stage.data();
      } else {
        captured.data = position.mutable_data;
        captured.map = position.map;
      }
      return captured;
    }
    captured.data = position.mutable_data;
    captured.elements = position.element_count();
    return captured;
  }

  // Duplicate the one caller type map the native call keeps by reference for the lifetime of
  // an asynchronous or persistent request (MPI 4.1 §5.1.9/§3.7.2, §6.13).  A blocking call
  // borrows the map for the native call and never duplicates.
  template<class T>
  void own_direct_layout(form_plan<T>& plan, const layout<T>* map, MPI_Datatype& call_type) {
    if (map == nullptr)
      return;
    plan.direct_layout = std::make_unique<layout<T>>(*map);
    call_type = plan.direct_layout->native_handle();
  }

  // ---------------------------------------------------------------------------------------
  // The one fixed-family form plan builder.  `family` fixes which positions are single
  // blocks and which are aggregates; the capture helpers own the
  // form lowering and the engine owns the role, peer division and native block construction.
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] form_plan<T> make_fixed_form_plan(const mpi_communicator& c,
                                                  collective_family family, int root,
                                                  form_position<T>& send,
                                                  form_position<T>& recv,
                                                  form_lifetime lifetime, const context& ctx) {
    form_plan<T> plan;
    plan.ctx = ctx;
    const bool owns = lifetime != form_lifetime::blocking;

    if (family == collective_family::bcast) {
      // MPI_Bcast has one in/out buffer, no in-place (MPI 4.1 §6.4); the engine applies the
      // role-dependent send/receive slot.
      const native_block<T> block = single_block_mutable(send, ctx);
      fixed_options<T> options;
      options.family = family;
      options.root = root;
      options.send.data = block.ptr;
      options.send.elements = block.count;
      options.send.map = block.map;
      plan.call = make_fixed_plan(c, options, ctx);
      if (owns) {
        own_direct_layout(plan, block.map, plan.call.send_type);
        plan.own_metadata();
      }
      return plan;
    }

    const bool in_place = send.in_place or recv.in_place;
    const place_state place = place_state_of(in_place);
    const prime_policy priming = in_place ? prime_policy::may_prime : prime_policy::none;
    const role_sides sides = resolve_sides(family, c, root, place, ctx);

    fixed_options<T> options;
    options.family = family;
    options.root = root;
    switch (family) {
      case collective_family::gather:
      case collective_family::allgather:
        options.send = capture_single_send(plan, send, place_state_of(send.in_place), ctx);
        options.recv = capture_aggregate_recv(plan, recv, side_use_of(sides.recv_used),
                                              place_state_of(recv.in_place), priming, ctx);
        break;
      case collective_family::scatter:
        options.send = capture_aggregate_send(plan, send, side_use_of(sides.send_used),
                                              place_state_of(send.in_place), ctx);
        options.recv = capture_single_recv(recv, place_state_of(recv.in_place), ctx);
        break;
      case collective_family::alltoall:
        options.send = capture_aggregate_send(plan, send, side_use::used,
                                              place_state_of(send.in_place), ctx);
        options.recv = capture_aggregate_recv(plan, recv, side_use::used,
                                              place_state_of(recv.in_place), priming, ctx);
        break;
      default:
        break;
    }

    plan.call = make_fixed_plan(c, options, ctx);
    if (owns) {
      switch (family) {
        case collective_family::gather:
        case collective_family::allgather:
          own_direct_layout(plan, options.send.map, plan.call.send_type);
          break;
        case collective_family::scatter:
          own_direct_layout(plan, options.recv.map, plan.call.recv_type);
          break;
        default:
          break;
      }
      plan.own_metadata();
    }
    return plan;
  }

  // The public adapter's family spellings.  Each is a name over the one builder above.
  template<class T>
  [[nodiscard]] form_plan<T> make_bcast_form_plan(const mpi_communicator& c, int root,
                                                  form_position<T>& position,
                                                  form_lifetime lifetime, const context& ctx) {
    form_position<T> unused;
    return make_fixed_form_plan<T>(c, collective_family::bcast, root, position, unused,
                                   lifetime, ctx);
  }

  template<class T>
  [[nodiscard]] form_plan<T> make_gather_form_plan(const mpi_communicator& c, int root,
                                                   form_position<T>& send,
                                                   form_position<T>& recv,
                                                   form_lifetime lifetime, const context& ctx) {
    return make_fixed_form_plan<T>(c, collective_family::gather, root, send, recv, lifetime,
                                   ctx);
  }

  template<class T>
  [[nodiscard]] form_plan<T> make_scatter_form_plan(const mpi_communicator& c, int root,
                                                    form_position<T>& send,
                                                    form_position<T>& recv,
                                                    form_lifetime lifetime,
                                                    const context& ctx) {
    return make_fixed_form_plan<T>(c, collective_family::scatter, root, send, recv, lifetime,
                                   ctx);
  }

  template<class T>
  [[nodiscard]] form_plan<T> make_allgather_form_plan(const mpi_communicator& c,
                                                      form_position<T>& send,
                                                      form_position<T>& recv,
                                                      form_lifetime lifetime,
                                                      const context& ctx) {
    return make_fixed_form_plan<T>(c, collective_family::allgather, 0, send, recv, lifetime,
                                   ctx);
  }

  template<class T>
  [[nodiscard]] form_plan<T> make_alltoall_form_plan(const mpi_communicator& c,
                                                     form_position<T>& send,
                                                     form_position<T>& recv,
                                                     form_lifetime lifetime,
                                                     const context& ctx) {
    return make_fixed_form_plan<T>(c, collective_family::alltoall, 0, send, recv, lifetime,
                                   ctx);
  }

}  // namespace mp1::detail

#endif
