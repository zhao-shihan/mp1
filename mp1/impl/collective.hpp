#if !(defined MP1_IMPL_COLLECTIVE_HPP)

#define MP1_IMPL_COLLECTIVE_HPP

// Public collective overlay: fixed-size collectives, the four variable families (native MPI +
// local staging), and reductions.  Every function is free, communicator first, and is a thin
// adapter over detail/collective_engine.hpp and detail/xfer_side.hpp.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/collective_engine.hpp"
#include "mp1/impl/detail/collective_form.hpp"
#include "mp1/impl/detail/request_payload.hpp"
#include "mp1/impl/detail/variable_form.hpp"
#include "mp1/impl/detail/xfer_side.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/operator.hpp"
#include "mp1/impl/request.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mp1 {

  // The one in-place sentinel.  It is declared here because the collective
  // overloads are its first consumers; including this header directly also exposes it.
  struct in_place_t {
    explicit constexpr in_place_t() = default;
  };
  inline constexpr in_place_t in_place{};

  namespace detail {

    // -------------------------------------------------------------------------------------
    // Variable-family request payloads.  The payload owns the two direction plans: the pack
    // and the descriptor copies happen before MPI_I*; the completion unpacks staged receives.
    // Caller data buffers and the communicator are borrowed for the request lifetime; the
    // plan owns exactly the descriptors completion/restart reads (copy-on-need).
    // -------------------------------------------------------------------------------------

    template<class T>
    class variable_payload final : public request_payload {
    public:
      explicit variable_payload(variable_call<T> call) : call_{std::move(call)} {
        call_.own_metadata(metadata_scope::completion);
      }

      void complete(const status&) override {
        call_.recv.finish();
      }

      [[nodiscard]] const variable_call<T>& call() const noexcept {
        return call_;
      }

    private:
      variable_call<T> call_;
    };

    template<class T>
    [[nodiscard]] std::unique_ptr<variable_payload<T>> make_variable_payload(
        variable_call<T> call) {
      return std::make_unique<variable_payload<T>>(std::move(call));
    }

    // -------------------------------------------------------------------------------------
    // Persistent-family request payloads.  They live until the persistent request is freed
    // (not until the first completion): every start re-packs or re-primes the local staging
    // through prepare_start(), and every completion unpacks once.
    // -------------------------------------------------------------------------------------

    template<class T>
    class persistent_variable_payload final : public request_payload {
    public:
      persistent_variable_payload(variable_call<T> call, const context& ctx)
          : call_{std::move(call)}, ctx_{ctx} {
        call_.own_metadata(metadata_scope::restart);
      }

      void prepare_start() override {
        call_.send.prepare_start(ctx_);
        call_.recv.prepare_start(ctx_);
      }

      void complete(const status&) override {
        call_.recv.finish();
      }

      [[nodiscard]] const variable_call<T>& call() const noexcept {
        return call_;
      }

    private:
      variable_call<T> call_;
      context ctx_;
    };

    // -------------------------------------------------------------------------------------
    // Reduction plans.  Flat forms never allocate; a non-flat layout stages both directions
    // with the shared pack/unpack helpers.  reduce_scatter stages per-rank block repetitions.
    // -------------------------------------------------------------------------------------

    // Persistent reductions may change their send (or in-place local contribution) data between
    // starts; the init-time pack/copy must be replayed locally before every MPI_Start.  The
    // plan owns the staged type map (copy-on-need) and the count arrays; the caller's data
    // buffers remain caller-owned until the request is freed.
    enum class repack_mode : unsigned char { none, pack_single };

    // A staged managed-functor reduction submits reduction_image_datatype<T>(): the native
    // typed map of T at extent sizeof(T).  Bytes outside that map (unlisted members and
    // padding) are never transmitted.  The full-T stage slots are still zeroed before the
    // mapped positions are packed so the local staging buffer is deterministic; the zeroed
    // bytes are not part of the wire image.  Completion unpacks only the mapped positions, so
    // the caller's unmapped bytes are preserved.
    template<class T>
    void zero_full_element_stage(stage_buffer<T>& stage) {
      if (not stage.empty())
        std::memset(static_cast<void*>(stage.data()), 0, stage.size() * sizeof(T));
    }

    template<class T>
    struct reduction_plan {
      const void* send_ptr{nullptr};
      void* recv_ptr{nullptr};
      stage_buffer<T> send_stage;
      stage_buffer<T> recv_stage;
      // Out-of-place exscan's rank-0 receive scratch.  It is uninitialized (MPI leaves rank
      // 0's receive buffer undefined) and is owned by the payload for a nonblocking or
      // persistent request; no other path allocates it.
      stage_buffer<T> scratch_recv;
      std::vector<MPI_Count> counts;
      const T* user_recv{nullptr};
      const layout<T>* recv_layout{nullptr};
      count_t recv_blocks{0};
      MPI_Count count{0};
      bool staged{false};
      bool unpack{false};
      // In-place non-flat reduce_scatter / reduce_scatter_block: `send_stage` holds the whole
      // flat input image and `recv_stage` the rank's flat result block.  MPI 4.1 6.10 writes an
      // in-place result block at the start of the receive area, so completion lifts the result
      // prefix into the input image and unpacks the whole region map once.  Every mapped
      // position outside the result prefix keeps its input value; the in-place form leaves
      // those positions undefined (MPI 4.1 6.10.1/6.10.2).
      bool whole_image_unpack{false};
      // Re-pack recipe (see repack_mode).
      repack_mode repack{repack_mode::none};
      const T* repack_source{nullptr};
      const layout<T>* repack_layout{nullptr};
      // Copy-on-need descriptor ownership.  A staged
      // receive unpacks after the initiating call returns, and a persistent staged plan
      // re-packs at every start, so those two plan kinds own their type map.  Dense/flat
      // plans and blocking calls keep the borrow.
      std::shared_ptr<const layout<T>> owned_recv_layout;
      std::shared_ptr<const layout<T>> owned_repack_layout;

      // Copy exactly the layout metadata that completion/restart reads.  Called by the
      // nonblocking adapter (scope completion) and by the persistent payload (scope restart),
      // before the caller's layout can die.  Idempotent; one caller layout used by the repack
      // and the receive costs one MPI_Type_dup.
      void own_metadata(metadata_scope scope) {
        const bool for_restart = scope == metadata_scope::restart;
        const bool need_recv = unpack and user_recv != nullptr and recv_layout != nullptr;
        const bool need_repack =
            for_restart and repack != repack_mode::none and repack_layout != nullptr;
        const layout<T>* const borrowed_recv = recv_layout;
        const layout<T>* const borrowed_repack = repack_layout;
        if (need_recv and owned_recv_layout == nullptr)
          owned_recv_layout = std::make_shared<layout<T>>(*borrowed_recv);
        if (need_repack and owned_repack_layout == nullptr) {
          if (owned_recv_layout != nullptr and borrowed_repack == borrowed_recv)
            owned_repack_layout = owned_recv_layout;
          else
            owned_repack_layout = std::make_shared<layout<T>>(*borrowed_repack);
        }
        if (owned_recv_layout != nullptr)
          recv_layout = owned_recv_layout.get();
        if (owned_repack_layout != nullptr)
          repack_layout = owned_repack_layout.get();
      }

      void finish(const context& ctx) {
        if (whole_image_unpack) {
          // The native out-of-place call wrote the rank's result block into the start of
          // recv_stage.  `send_stage` still holds the whole flat input image; copy the result
          // prefix over it and unpack that image through the caller's whole-buffer map.  Only
          // the map's positions are written, and every position outside the prefix keeps the
          // value the caller supplied for this cycle.
          if (not recv_stage.empty())
            std::copy_n(recv_stage.data(), recv_stage.size(), send_stage.data());
          if (user_recv == nullptr or recv_layout == nullptr)
            return;
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API
          stage_unpack_blocks(send_stage, 0, const_cast<T*>(user_recv), *recv_layout, 1, ctx);
          return;
        }
        if (not unpack or user_recv == nullptr or recv_layout == nullptr)
          return;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API
        stage_unpack_blocks(recv_stage, 0, const_cast<T*>(user_recv), *recv_layout, recv_blocks,
                            ctx);
      }

      void prepare_start(const context& ctx) {
        switch (repack) {
          case repack_mode::none:
            return;
          case repack_mode::pack_single:
            stage_pack_blocks(repack_source, *repack_layout, 1, send_stage, 0, ctx);
            return;
          default:
            // The mode is internal and closed over the cases above; an unknown value means
            // the plan was corrupted, never a silently skipped re-pack.
            throw_invalid(ctx.operation, "invalid persistent-reduction repack mode");
        }
      }
    };

    template<class T>
    class reduction_payload final : public request_payload {
    public:
      reduction_payload(reduction_plan<T> plan, const context& ctx)
          : plan_{std::move(plan)}, ctx_{ctx} {
      }

      void complete(const status&) override {
        plan_.finish(ctx_);
      }

    private:
      reduction_plan<T> plan_;
      context ctx_;
    };

    template<class T>
    class persistent_reduction_payload final : public request_payload {
    public:
      persistent_reduction_payload(reduction_plan<T> plan, const context& ctx)
          : plan_{std::move(plan)}, ctx_{ctx} {
        // The init call has not happened yet; the plan copies the staged descriptors that
        // every start/completion will read, so the caller's layout may die after *_init
        // (MPI 4.1 6.13).
        plan_.own_metadata(metadata_scope::restart);
      }

      void prepare_start() override {
        plan_.prepare_start(ctx_);
      }

      void complete(const status&) override {
        plan_.finish(ctx_);
      }

      [[nodiscard]] const reduction_plan<T>& plan() const noexcept {
        return plan_;
      }

    private:
      reduction_plan<T> plan_;
      context ctx_;
    };

    // -------------------------------------------------------------------------------------
    // Persistent init adapters.  The payload is constructed before the native init (MPI may
    // keep pointers into its count/displacement/staging arrays), one MPI_*_init is issued,
    // and the handle receives the payload as its lifetime owner only after success.
    // -------------------------------------------------------------------------------------

    template<class T>
    [[nodiscard]] request gatherv_init_call(variable_call<T> call, int root, MPI_Comm comm,
                                            MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_variable_payload<T>>(std::move(call), ctx);
      request result = native_gatherv_init(payload->call(), root, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request scatterv_init_call(variable_call<T> call, int root, MPI_Comm comm,
                                             MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_variable_payload<T>>(std::move(call), ctx);
      request result = native_scatterv_init(payload->call(), root, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request allgatherv_init_call(variable_call<T> call, MPI_Comm comm,
                                               MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_variable_payload<T>>(std::move(call), ctx);
      request result = native_allgatherv_init(payload->call(), comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request alltoallv_init_call(variable_call<T> call, MPI_Comm comm,
                                              MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_variable_payload<T>>(std::move(call), ctx);
      request result = native_alltoallv_init(payload->call(), comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request reduce_init_call(reduction_plan<T> plan,
                                           const reduction_operation& op, int root,
                                           MPI_Comm comm, MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_reduce_init(prepared.send_ptr, prepared.recv_ptr, prepared.count,
                                          op, root, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request allreduce_init_call(reduction_plan<T> plan,
                                              const reduction_operation& op, MPI_Comm comm,
                                              MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_allreduce_init(prepared.send_ptr, prepared.recv_ptr,
                                             prepared.count, op, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request reduce_scatter_init_call(reduction_plan<T> plan,
                                                   const reduction_operation& op, MPI_Comm comm,
                                                   MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_reduce_scatter_init(prepared.send_ptr, prepared.recv_ptr,
                                                  prepared.counts.data(), op, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request reduce_scatter_block_init_call(reduction_plan<T> plan,
                                                         const reduction_operation& op,
                                                         MPI_Comm comm, MPI_Info info,
                                                         const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_reduce_scatter_block_init(prepared.send_ptr, prepared.recv_ptr,
                                                        prepared.count, op, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request scan_init_call(reduction_plan<T> plan, const reduction_operation& op,
                                         MPI_Comm comm, MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_scan_init(prepared.send_ptr, prepared.recv_ptr, prepared.count,
                                        op, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

    template<class T>
    [[nodiscard]] request exscan_init_call(reduction_plan<T> plan,
                                           const reduction_operation& op, MPI_Comm comm,
                                           MPI_Info info, const context& ctx) {
      auto payload = std::make_unique<persistent_reduction_payload<T>>(std::move(plan), ctx);
      const reduction_plan<T>& prepared = payload->plan();
      request result = native_exscan_init(prepared.send_ptr, prepared.recv_ptr, prepared.count,
                                          op, comm, info);
      attach_payload(result, std::move(payload));
      return result;
    }

  }  // namespace detail

  // =========================================================================================
  // Fixed-size collectives: one declaration per family per lifetime.
  //
  // The buffer positions are compile-time forms (value / span-range / sequence / region) and
  // compile-time in-place sides; the root, intercommunicator and in-place legality checks are
  // runtime and live in the engine's role resolver.  A blocking call borrows every buffer and
  // layout for the call; the i* form owns the staged buffers and duplicated type maps in the
  // request payload; the *_init form additionally re-packs staged sends at every start.
  // =========================================================================================

  inline void barrier(const mpi_communicator& c) {
    detail::require_collective_communicator(c, "mp1::barrier");
    detail::native_barrier(c.native_handle());
  }

  [[nodiscard]] inline request ibarrier(const mpi_communicator& c) {
    detail::require_collective_communicator(c, "mp1::ibarrier");
    return detail::native_ibarrier(c.native_handle());
  }

  // ---- bcast: one in/out buffer, no in-place ("not meaningful", MPI 4.1 §6.4) ----

  template<class B>
  requires(recv_sink<B>)
  void bcast(const mpi_communicator& c, int root, B&& buffer) {
    using T = detail::form_element_t<B>;
    const context ctx{"mp1::bcast"};
    detail::form_position<T> position =
        detail::make_form_position<T>(std::forward<B>(buffer), ctx);
    detail::form_plan<T> plan = detail::make_bcast_form_plan<T>(
        c, root, position, detail::form_lifetime::blocking, ctx);
    detail::native_bcast(plan.call, root, c.native_handle());
    plan.finish();
  }

  template<class B>
  requires(recv_sink<B> && borrowable<B &&>)
  [[nodiscard]] request ibcast(const mpi_communicator& c, int root, B&& buffer) {
    using T = detail::form_element_t<B>;
    const context ctx{"mp1::ibcast"};
    detail::form_position<T> position =
        detail::make_form_position<T>(std::forward<B>(buffer), ctx);
    detail::form_plan<T> plan = detail::make_bcast_form_plan<T>(
        c, root, position, detail::form_lifetime::asynchronous, ctx);
    request result = detail::native_ibcast(plan.call, root, c.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // ---- gather: send single block, recv aggregate at the root ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void gather(const mpi_communicator& c, int root, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gather"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_gather_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::blocking, ctx);
    detail::native_gather(plan.call, root, c.native_handle());
    plan.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request igather(const mpi_communicator& c, int root, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::igather"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_gather_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::asynchronous, ctx);
    request result = detail::native_igather(plan.call, root, c.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // ---- scatter: send aggregate at the root, recv single block ----

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void scatter(const mpi_communicator& c, int root, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatter"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_scatter_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::blocking, ctx);
    detail::native_scatter(plan.call, root, c.native_handle());
    plan.finish();
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iscatter(const mpi_communicator& c, int root, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iscatter"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_scatter_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::asynchronous, ctx);
    request result = detail::native_iscatter(plan.call, root, c.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // ---- allgather: send single block, recv aggregate at every rank ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void allgather(const mpi_communicator& c, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgather"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_allgather_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::blocking, ctx);
    detail::native_allgather(plan.call, c.native_handle());
    plan.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iallgather(const mpi_communicator& c, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iallgather"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_allgather_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::asynchronous, ctx);
    request result = detail::native_iallgather(plan.call, c.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // ---- alltoall: both positions are aggregates ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void alltoall(const mpi_communicator& c, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoall"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_alltoall_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::blocking, ctx);
    detail::native_alltoall(plan.call, c.native_handle());
    plan.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ialltoall(const mpi_communicator& c, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ialltoall"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_alltoall_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::asynchronous, ctx);
    request result = detail::native_ialltoall(plan.call, c.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // =========================================================================================
  // Variable collectives: exactly two declarations per family per lifetime.
  //
  //   dense  the per-peer descriptor is a `blocks` value (element counts + displacements);
  //   typed  the per-peer descriptor is `std::span<const layout<T>>` + `std::span<const
  //   count_t>`
  //          (one layout per peer, element displacements).
  //
  // The buffer positions use the same compile-time forms as the fixed collectives.  A
  // single-block position (gatherv send, scatterv recv) is a value, a contiguous element
  // range, or a `region` (one layout occurrence).  An aggregate position (gatherv recv,
  // scatterv send, allgatherv recv, alltoallv send/recv) is a flat contiguous element range:
  // the per-block placement belongs to the descriptor, and the descriptor is checked against
  // the span's element count before any MPI call.  The in-place side is the
  // family's fixed side and is compile-time; the root/inter/in-place role checks are
  // runtime.  A non-contiguous sequence is not a variable-family form.
  //
  // Blocking calls borrow the two descriptor spans for the call.  Nonblocking and persistent
  // calls copy every descriptor array and every staged layout they read after the call
  // returns, so the caller's `blocks`/layout arrays may be destroyed immediately after
  // initiation (the lifetime contract is enforced).  Persistent calls accept fixed
  // buffers only (no resizable container; no per-start data()/size() revalidation).
  // =========================================================================================

  // ---- gatherv: send is a single block (in-place at the root), recv is the aggregate; the
  // per-peer descriptor is the root's ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void gatherv(const mpi_communicator& c, int root, S&& send, R&& recv, blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    detail::native_gatherv(call, root, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void gatherv(const mpi_communicator& c, int root, S&& send, R&& recv,
               std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
               std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    detail::native_gatherv(call, root, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request igatherv(const mpi_communicator& c, int root, S&& send, R&& recv,
                                 blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::igatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_igatherv(payload->call(), root, c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request igatherv(
      const mpi_communicator& c, int root, S&& send, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::igatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_igatherv(payload->call(), root, c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request gatherv_init(const mpi_communicator& c, int root, S&& send, R&& recv,
                                     blocks recv_blocks, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gatherv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    return detail::gatherv_init_call(std::move(call), root, c.native_handle(),
                                     i.native_handle(), ctx);
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request gatherv_init(
      const mpi_communicator& c, int root, S&& send, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gatherv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_gatherv_variable_plan<T>(
        c, root, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    return detail::gatherv_init_call(std::move(call), root, c.native_handle(),
                                     i.native_handle(), ctx);
  }

  // ---- scatterv: send is the aggregate (root only), recv is a single block (in-place at the
  // root); the per-peer descriptor is the root's send side ----

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void scatterv(const mpi_communicator& c, int root, S&& send, blocks send_blocks, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatterv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position, ctx);
    detail::native_scatterv(call, root, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void scatterv(const mpi_communicator& c, int root, S&& send,
                std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
                std::span<const count_t> send_displacements, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatterv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position, ctx);
    detail::native_scatterv(call, root, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           borrowable<S &&> && borrowable<R &&> && detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iscatterv(const mpi_communicator& c, int root, S&& send,
                                  blocks send_blocks, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iscatterv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position, ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_iscatterv(payload->call(), root, c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           borrowable<S &&> && borrowable<R &&> && detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iscatterv(
      const mpi_communicator& c, int root, S&& send,
      std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
      std::span<const count_t> send_displacements, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iscatterv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position, ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_iscatterv(payload->call(), root, c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request scatterv_init(const mpi_communicator& c, int root, S&& send,
                                      blocks send_blocks, R&& recv, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatterv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position, ctx);
    return detail::scatterv_init_call(std::move(call), root, c.native_handle(),
                                      i.native_handle(), ctx);
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && element_buffer<std::remove_cvref_t<S>> &&
           borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request scatterv_init(
      const mpi_communicator& c, int root, S&& send,
      std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
      std::span<const count_t> send_displacements, R&& recv, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatterv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_scatterv_variable_plan<T>(
        c, root, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position, ctx);
    return detail::scatterv_init_call(std::move(call), root, c.native_handle(),
                                      i.native_handle(), ctx);
  }

  // ---- allgatherv: send is a single block (in-place at every rank), recv is the aggregate
  // plus the descriptor at every rank ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void allgatherv(const mpi_communicator& c, S&& send, R&& recv, blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    detail::native_allgatherv(call, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           detail::compatible_form_elements_v<S, R>)
  void allgatherv(const mpi_communicator& c, S&& send, R&& recv,
                  std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
                  std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    detail::native_allgatherv(call, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iallgatherv(const mpi_communicator& c, S&& send, R&& recv,
                                    blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iallgatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_iallgatherv(payload->call(), c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iallgatherv(
      const mpi_communicator& c, S&& send, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iallgatherv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_iallgatherv(payload->call(), c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request allgatherv_init(const mpi_communicator& c, S&& send, R&& recv,
                                        blocks recv_blocks, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgatherv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    return detail::allgatherv_init_call(std::move(call), c.native_handle(), i.native_handle(),
                                        ctx);
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && element_buffer<std::remove_cvref_t<R>> &&
           !element_sequence<std::remove_cvref_t<S>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request allgatherv_init(
      const mpi_communicator& c, S&& send, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgatherv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_allgatherv_variable_plan<T>(
        c, send_position, recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    return detail::allgatherv_init_call(std::move(call), c.native_handle(), i.native_handle(),
                                        ctx);
  }

  // ---- alltoallv: both positions are aggregates with their own per-peer descriptor; the
  // send side may be in-place at every rank, in which case its descriptor is MPI-ignored ----

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && detail::compatible_form_elements_v<S, R>)
  void alltoallv(const mpi_communicator& c, S&& send, blocks send_blocks, R&& recv,
                 blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoallv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    detail::native_alltoallv(call, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && detail::compatible_form_elements_v<S, R>)
  void alltoallv(const mpi_communicator& c, S&& send,
                 std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
                 std::span<const count_t> send_displacements, R&& recv,
                 std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
                 std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoallv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    detail::native_alltoallv(call, c.native_handle());
    call.recv.finish();
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ialltoallv(const mpi_communicator& c, S&& send, blocks send_blocks,
                                   R&& recv, blocks recv_blocks) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ialltoallv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_ialltoallv(payload->call(), c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && borrowable<S &&> && borrowable<R &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ialltoallv(
      const mpi_communicator& c, S&& send,
      std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
      std::span<const count_t> send_displacements, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ialltoallv"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    auto payload = detail::make_variable_payload(std::move(call));
    request result = detail::native_ialltoallv(payload->call(), c.native_handle());
    detail::attach_payload(result, std::move(payload));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request alltoallv_init(const mpi_communicator& c, S&& send, blocks send_blocks,
                                       R&& recv, blocks recv_blocks, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoallv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_counts(send_blocks.counts,
                                                    send_blocks.displacements),
        recv_position,
        detail::variable_block_source<T>::of_counts(recv_blocks.counts,
                                                    recv_blocks.displacements),
        ctx);
    return detail::alltoallv_init_call(std::move(call), c.native_handle(), i.native_handle(),
                                       ctx);
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> &&
           (std::same_as<std::remove_cvref_t<S>, in_place_t> ||
            element_buffer<std::remove_cvref_t<S>>) &&
           element_buffer<std::remove_cvref_t<R>> && borrowable<S &&> && borrowable<R &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request alltoallv_init(
      const mpi_communicator& c, S&& send,
      std::span<const layout<detail::collective_element_t<S, R>>> send_layouts,
      std::span<const count_t> send_displacements, R&& recv,
      std::span<const layout<detail::collective_element_t<S, R>>> recv_layouts,
      std::span<const count_t> recv_displacements = {}, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoallv_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::variable_call<T> call = detail::make_alltoallv_variable_plan<T>(
        c, send_position,
        detail::variable_block_source<T>::of_layouts(send_layouts, send_displacements),
        recv_position,
        detail::variable_block_source<T>::of_layouts(recv_layouts, recv_displacements), ctx);
    return detail::alltoallv_init_call(std::move(call), c.native_handle(), i.native_handle(),
                                       ctx);
  }


  // =========================================================================================
  // Reductions: one plan builder for the four single-block families, plus reduce_local and
  // the two reduce-scatter families.
  //
  // Every adapter follows one order: classify both positions locally, preflight the used ones
  // with the exact-byte rule (detail::validate_reduction_positions), and only then resolve
  // reduction_op<T>() -- so a managed functor's MPI_Op is never created for a call that is
  // already invalid, and a null non-empty position never reaches MPI.  The plan builders repeat
  // the preflight as defense in depth.  A zero-byte position is legal with a null pointer and
  // is presented to MPI as the per-slot dummy.
  //
  // A flat position (value, span, or a base-repetition region lowered to elements) goes to the
  // engine with the native datatype.  A non-flat region is one MPI datatype occurrence; because
  // a predefined reduction operator only accepts the predefined datatypes of MPI 4.1 §6.9.2, a
  // non-flat position is packed into a flat staging vector and the native call sees the base
  // datatype.  Each direction stages independently, so a region/span pair needs at most one
  // staged side.  The in-place typed form packs the receive buffer into the send stage and
  // stays an out-of-place MPI call (the established 1.0 staging contract).
  // =========================================================================================

  namespace detail {

    template<class T>
    void finish_blocking_reduction(reduction_plan<T>& plan, const context& ctx) {
      plan.finish(ctx);
    }

    template<class T>
    [[nodiscard]] request finish_nonblocking_reduction(
        request result, reduction_plan<T> plan, const context& ctx,
        payload_policy policy = payload_policy::on_demand) {
      if (policy == payload_policy::always or plan.staged or not plan.scratch_recv.empty()) {
        // The request outlives this call; own the staged receive type map before the caller's
        // layout can die (MPI 4.1 §6.13 for the persistent restart analogue).
        plan.own_metadata(metadata_scope::completion);
        attach_payload(result, std::make_unique<reduction_payload<T>>(std::move(plan), ctx));
      }
      return result;
    }

    [[nodiscard]] inline std::vector<MPI_Count> reduction_wire_counts(
        std::span<const count_t> recvcounts, int local_size, const context& ctx) {
      if (recvcounts.size() != static_cast<std::size_t>(local_size)) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "recvcounts length {} does not match the local size {}", recvcounts.size(),
            local_size);
      }
      std::vector<MPI_Count> result;
      result.reserve(recvcounts.size());
      for (const count_t value : recvcounts) {
        if (value < 0)
          throw_invalid(ctx.operation, "negative receive count");
        result.push_back(static_cast<MPI_Count>(value));
      }
      return result;
    }

    [[nodiscard]] inline count_t reduction_counts_total(std::span<const count_t> recvcounts,
                                                        const context& ctx) {
      count_t total{0};
      for (const count_t value : recvcounts)
        total = checked_count_add(total, value, ctx);
      return total;
    }

    inline void require_typed_layout(const auto& position, const context& ctx) {
      if (position.typed and not position.map->is_valid())
        throw_invalid(ctx.operation, "invalid layout");
    }

    // The reduction image a native operation requires: a raw/predefined op reduces the native
    // T image; an mp1-managed functor callback reduces the typed callback image over T's
    // native map (reduction_image_datatype<T>()), so its stage slots are zero-filled first for
    // local determinism.  Neither path transmits padding or unlisted members.
    enum class reduction_image : unsigned char { native, typed_callback };

    [[nodiscard]] inline reduction_image reduction_image_of(
        const reduction_operation& operation) noexcept {
      return operation.callback_image ? reduction_image::typed_callback
                                      : reduction_image::native;
    }

    template<class T>
    struct reduction_options {
      collective_family family{collective_family::reduce};
      int root{0};
      const form_position<T>* send{nullptr};
      const form_position<T>* recv{nullptr};
      reduction_image image{reduction_image::native};
    };

    template<class T>
    [[nodiscard]] reduction_options<T> reduction_request(collective_family family, int root,
                                                         const form_position<T>& send,
                                                         const form_position<T>& recv,
                                                         const reduction_operation& operation) {
      return reduction_options<T>{.family = family,
                                  .root = root,
                                  .send = std::addressof(send),
                                  .recv = std::addressof(recv),
                                  .image = reduction_image_of(operation)};
    }

    // One reduction plan for reduce/allreduce/scan/exscan, both positions single-block.
    template<class T>
    [[nodiscard]] reduction_plan<T> make_reduction_plan(const mpi_communicator& c,
                                                        const reduction_options<T>& options,
                                                        const context& ctx) {
      const form_position<T>& send = *options.send;
      const form_position<T>& recv = *options.recv;
      const collective_family family = options.family;
      const int root = options.root;
      const bool callback_image = options.image == reduction_image::typed_callback;
      require_reduction_base<T>(ctx);
      const bool in_place = send.in_place;
      const role_sides sides = resolve_sides(family, c, root, place_state_of(in_place), ctx);
      require_typed_layout(send, ctx);
      require_typed_layout(recv, ctx);
      // Defense in depth: the public adapters preflight before the managed MPI_Op exists, and
      // the plan builder repeats the exact-byte check so no internal caller can hand a null
      // non-empty position to the native call.
      require_used_position(buffer_slot::send, send, sides.send_used and not in_place, ctx);
      require_used_position(buffer_slot::recv, recv, sides.recv_used, ctx);

      const count_t send_elements = send.element_count();
      const count_t recv_elements = recv.element_count();
      count_t n{0};
      if (in_place) {
        n = recv_elements;
      } else if (sides.send_used and sides.recv_used) {
        if (send_elements != recv_elements) {
          throw_invalid(ctx.operation, "send and receive block sizes differ");
        }
        n = send_elements;
      } else if (sides.send_used) {
        n = send_elements;
      } else if (sides.recv_used) {
        n = recv_elements;
      } else {
        // Neither position is significant (an intercommunicator MPI_PROC_NULL member of
        // the root group): keep the caller's send count so the collective's count argument
        // stays uniform across the participating ranks instead of reading the dummy 0.
        n = send_elements;
      }
      if (n < 0)
        throw_invalid(ctx.operation, "negative count");

      reduction_plan<T> plan;
      // A zero-size native datatype has an empty type signature: count N and count 0 are the
      // same no-data reduction (MPI 4.1 §5.1.11), and MPICH has been recorded to abort on a
      // positive count of a zero-size type in related collectives.  Normalize the wire count;
      // the staging arithmetic above still works on the caller's element count.
      plan.count = base_datatype<T>().size == 0 ? 0 : static_cast<MPI_Count>(n);

      const bool send_nonflat = send.typed and not send.map->is_base_repetition();
      const bool recv_nonflat = recv.typed and not recv.map->is_base_repetition();
      const bool staged =
          (sides.send_used and not in_place and (send.sequence or send_nonflat)) or
          (sides.recv_used and recv_nonflat) or (in_place and recv_nonflat);

      if (not staged) {
        data_block send_block;
        if (in_place) {
          send_block = make_block(buffer_slot::send, static_cast<const T*>(nullptr), 0, nullptr,
                                  side_use::ignored, place_state::mpi_in_place, ctx);
        } else if (sides.send_used) {
          const T* pointer =
              send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
          send_block = make_block(buffer_slot::send, pointer, n, nullptr, side_use::used,
                                  place_state::data, ctx);
        } else {
          send_block = make_block(buffer_slot::send, static_cast<const T*>(nullptr), 0, nullptr,
                                  side_use::ignored, place_state::data, ctx);
        }

        data_block recv_block;
        if (sides.recv_used and family == collective_family::exscan and not in_place and
            c.rank() == 0) {
          // MPI leaves rank 0's out-of-place receive buffer undefined; the caller's buffer is
          // kept untouched through an uninitialized scratch buffer owned by the plan.
          if (n != 0)
            plan.scratch_recv.resize(static_cast<std::size_t>(n));
          T* pointer = plan.scratch_recv.empty() ? nullptr : plan.scratch_recv.data();
          recv_block = make_block(buffer_slot::recv, pointer, n, nullptr, side_use::used,
                                  place_state::data, ctx);
        } else if (sides.recv_used) {
          if (recv.typed)
            require_receive_layout(*recv.map, ctx);
          T* pointer = recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx)
                                  : recv.mutable_data;
          recv_block = make_block(buffer_slot::recv, pointer, n, nullptr, side_use::used,
                                  place_state::data, ctx);
        } else {
          recv_block = make_block(buffer_slot::recv, static_cast<T*>(nullptr), 0, nullptr,
                                  side_use::ignored, place_state::data, ctx);
        }
        plan.send_ptr = send_block.ptr;
        plan.recv_ptr = recv_block.mutable_ptr();
        return plan;
      }

      plan.staged = true;
      if (in_place) {
        // The receive buffer is the reduction input.  Pack it into the send stage and use an
        // out-of-place native call; the result is unpacked back into the same buffer.
        require_receive_layout(*recv.map, ctx);
        require_staged_receive_pointer(recv.mutable_data, *recv.map, ctx);
        const std::size_t size = static_cast<std::size_t>(n);
        plan.send_stage.resize(size);
        if (callback_image)
          zero_full_element_stage(plan.send_stage);
        stage_pack_blocks(recv.mutable_data, *recv.map, 1, plan.send_stage, 0, ctx);
        plan.send_ptr = n == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        plan.recv_stage.resize(size);
        if (callback_image)
          zero_full_element_stage(plan.recv_stage);
        plan.recv_ptr = n == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
        plan.repack = repack_mode::pack_single;
        plan.repack_source = recv.mutable_data;
        plan.repack_layout = std::addressof(*recv.map);
        if (family != collective_family::exscan or c.rank() != 0) {
          plan.user_recv = recv.mutable_data;
          plan.recv_layout = std::addressof(*recv.map);
          plan.recv_blocks = 1;
          plan.unpack = true;
        }
        return plan;
      }

      // Out of place, at least one staged direction.
      if (sides.send_used) {
        if (send.sequence) {
          plan.send_stage.resize(static_cast<std::size_t>(n));
          std::ranges::copy_n(send.owned.data(), send.owned.size(), plan.send_stage.data());
          plan.send_ptr = n == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        } else if (send_nonflat) {
          plan.send_stage.resize(static_cast<std::size_t>(n));
          if (callback_image)
            zero_full_element_stage(plan.send_stage);
          stage_pack_blocks(send.data, *send.map, 1, plan.send_stage, 0, ctx);
          plan.send_ptr = n == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
          plan.repack = repack_mode::pack_single;
          plan.repack_source = send.data;
          plan.repack_layout = std::addressof(*send.map);
        } else {
          plan.send_ptr =
              send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
        }
      } else {
        plan.send_ptr = dummy_buffer(buffer_slot::send);
      }

      if (sides.recv_used) {
        if (family == collective_family::exscan and c.rank() == 0) {
          // Rank 0 keeps its receive buffer unchanged; the staged receive is discarded.
          if (n != 0)
            plan.scratch_recv.resize(static_cast<std::size_t>(n));
          plan.recv_ptr = plan.scratch_recv.empty() ? dummy_buffer(buffer_slot::recv)
                                                    : plan.scratch_recv.data();
        } else if (recv_nonflat) {
          require_receive_layout(*recv.map, ctx);
          require_staged_receive_pointer(recv.mutable_data, *recv.map, ctx);
          plan.recv_stage.resize(static_cast<std::size_t>(n));
          if (callback_image)
            zero_full_element_stage(plan.recv_stage);
          plan.recv_ptr = n == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
          plan.user_recv = recv.mutable_data;
          plan.recv_layout = std::addressof(*recv.map);
          plan.recv_blocks = 1;
          plan.unpack = true;
        } else {
          if (recv.typed)
            require_receive_layout(*recv.map, ctx);
          plan.recv_ptr = recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx)
                                     : recv.mutable_data;
        }
      } else {
        plan.recv_ptr = dummy_buffer(buffer_slot::recv);
      }
      return plan;
    }

    // reduce_local (MPI 4.1 §6.9.7): no communicator and no MPI_IN_PLACE.  MPI has one datatype
    // argument, so at least one non-flat position forces both directions through flat staging;
    // the receive stage is primed from the caller buffer because reduce_local is inout.
    //
    // The exact-byte preflight runs before reduction_op(), so a null non-empty position is
    // rejected before a managed reduction operation (and its callback datatype) is created,
    // and a zero-byte position is presented to the one native call as the per-slot dummy.
    template<class T, class Op>
    void reduce_local_form(Op&& op, const form_position<T>& send, const form_position<T>& recv,
                           const context& ctx) {
      require_mpi(ctx);
      require_typed_layout(send, ctx);
      require_typed_layout(recv, ctx);
      require_used_position(buffer_slot::send, send, true, ctx);
      require_used_position(buffer_slot::recv, recv, true, ctx);
      require_reduction_base<T>(ctx);
      const reduction_operation native_op = reduction_op<T>(op);

      const count_t send_elements = send.element_count();
      const count_t recv_elements = recv.element_count();
      if (send_elements != recv_elements) {
        throw_invalid(ctx.operation, "send and receive block sizes differ");
      }
      const count_t n = send_elements;
      // Same zero-size normalization as the reduction plan: an empty type signature is a
      // no-data reduction whatever the caller's element count.
      const MPI_Count wire_count = base_datatype<T>().size == 0 ? 0 : static_cast<MPI_Count>(n);

      if (n == 0) {
        // A zero-element position is empty whatever its form: hand the native call the
        // per-slot dummy (an empty position may be null) and keep the no-op observable as one
        // local call, exactly as the flat spelling always did.
        native_reduce_local(static_cast<const T*>(dummy_buffer(buffer_slot::send)),
                            static_cast<T*>(dummy_buffer(buffer_slot::recv)), wire_count,
                            native_op);
        return;
      }

      const bool send_nonflat = send.typed and not send.map->is_base_repetition();
      const bool recv_nonflat = recv.typed and not recv.map->is_base_repetition();
      if (not send.sequence and not send_nonflat and not recv_nonflat) {
        if (recv.typed)
          require_receive_layout(*recv.map, ctx);
        const T* send_pointer = send.send_data();
        if (send.typed and send.data != nullptr)
          send_pointer = repetition_ptr(send.data, *send.map, ctx);
        T* recv_pointer = recv.mutable_data;
        if (recv.typed and recv.mutable_data != nullptr)
          recv_pointer = repetition_ptr(recv.mutable_data, *recv.map, ctx);
        if (send_pointer == nullptr)
          send_pointer = static_cast<const T*>(dummy_buffer(buffer_slot::send));
        if (recv_pointer == nullptr)
          recv_pointer = static_cast<T*>(dummy_buffer(buffer_slot::recv));
        native_reduce_local(send_pointer, recv_pointer, wire_count, native_op);
        return;
      }

      const std::size_t size = static_cast<std::size_t>(n);
      stage_buffer<T> send_stage;
      stage_buffer<T> recv_stage;
      const T* send_pointer = send.send_data();
      T* recv_pointer = recv.mutable_data;

      if (send.sequence) {
        send_stage.resize(size);
        std::ranges::copy_n(send.owned.data(), send.owned.size(), send_stage.data());
        send_pointer = send_stage.data();
      } else if (send_nonflat) {
        send_stage.resize(size);
        if (native_op.callback_image)
          zero_full_element_stage(send_stage);
        stage_pack_blocks(send.data, *send.map, 1, send_stage, 0, ctx);
        send_pointer = send_stage.data();
      } else if (send.typed and send.data != nullptr) {
        send_pointer = repetition_ptr(send.data, *send.map, ctx);
      }
      if (send_pointer == nullptr)
        send_pointer = static_cast<const T*>(dummy_buffer(buffer_slot::send));

      if (recv_nonflat) {
        require_receive_layout(*recv.map, ctx);
        recv_stage.resize(size);
        if (native_op.callback_image)
          zero_full_element_stage(recv_stage);
        stage_pack_blocks(recv.mutable_data, *recv.map, 1, recv_stage, 0, ctx);
        recv_pointer = recv_stage.data();
      } else if (recv.typed and recv.mutable_data != nullptr) {
        require_receive_layout(*recv.map, ctx);
        recv_pointer = repetition_ptr(recv.mutable_data, *recv.map, ctx);
      }
      if (recv_pointer == nullptr)
        recv_pointer = static_cast<T*>(dummy_buffer(buffer_slot::recv));

      native_reduce_local(send_pointer, recv_pointer, wire_count, native_op);
      if (recv_nonflat)
        stage_unpack_blocks(recv_stage, 0, recv.mutable_data, *recv.map, 1, ctx);
    }

    // reduce_scatter / reduce_scatter_block: counts-only descriptor, no root, no ignored
    // position.  Every rank passes the real receive buffer and identical counts.  A
    // region layout describes the whole buffer it is attached to, so the counts stay base
    // element counts on every form; a non-flat region is staged through a flat vector.  The
    // in-place receive buffer is the full input vector.
    template<class T>
    [[nodiscard]] reduction_plan<T> make_reduce_scatter_plan(
        const mpi_communicator& c, const form_position<T>& send, const form_position<T>& recv,
        std::span<const count_t> recvcounts, const context& ctx) {
      require_reduction_base<T>(ctx);
      require_typed_layout(send, ctx);
      require_typed_layout(recv, ctx);
      const bool in_place = send.in_place;
      (void)resolve_sides(collective_family::reduce_scatter, c, 0, place_state_of(in_place),
                          ctx);
      // Defense in depth: the adapter preflights before the managed MPI_Op exists.
      require_used_position(buffer_slot::send, send, not in_place, ctx);
      require_used_position(buffer_slot::recv, recv, true, ctx);

      reduction_plan<T> plan;
      plan.counts = reduction_wire_counts(recvcounts, c.size(), ctx);
      if (base_datatype<T>().size == 0) {
        // Zero-size native datatype: every wire count is 0 (see make_reduction_plan).
        for (MPI_Count& wire_count : plan.counts)
          wire_count = 0;
      }
      const count_t total = reduction_counts_total(recvcounts, ctx);
      const count_t local = recvcounts[static_cast<std::size_t>(c.rank())];

      const count_t send_elements = send.element_count();
      const count_t recv_elements = recv.element_count();
      const auto require_size = [ctx](count_t actual, count_t expected, const char* side) {
        if (actual != expected) {
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "{} buffer has {} elements but the counts require {}", side, actual, expected);
        }
      };
      if (in_place) {
        require_size(recv_elements, total, "receive");
      } else {
        require_size(send_elements, total, "send");
        require_size(recv_elements, local, "receive");
      }

      const bool send_nonflat = send.typed and not send.map->is_base_repetition();
      const bool recv_nonflat = recv.typed and not recv.map->is_base_repetition();

      if (in_place) {
        if (not recv_nonflat) {
          if (recv.typed)
            require_receive_layout(*recv.map, ctx);
          T* pointer = recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx)
                                  : recv.mutable_data;
          const data_block recv_block = make_block(buffer_slot::recv, pointer, total, nullptr,
                                                   side_use::used, place_state::data, ctx);
          plan.send_ptr = MPI_IN_PLACE;
          plan.recv_ptr = recv_block.mutable_ptr();
          return plan;
        }
        // A whole-buffer non-flat region cannot be the datatype of an MPI_IN_PLACE call: MPI
        // repeats the receive datatype `sum(recvcounts)` times at its extent stride, while the
        // region is one absolute map over the full input vector.  Stage the flat input image
        // and issue the native reduction out of place; MPI writes this rank's result block at
        // the start of the receive area (MPI 4.1 6.10.2), which completion lifts into the input
        // image before unpacking the whole map back into the caller's buffer.
        require_receive_layout(*recv.map, ctx);
        require_staged_receive_pointer(recv.mutable_data, *recv.map, ctx);
        plan.staged = true;
        plan.send_stage.resize(static_cast<std::size_t>(total));
        zero_full_element_stage(plan.send_stage);
        stage_pack_blocks(recv.mutable_data, *recv.map, 1, plan.send_stage, 0, ctx);
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        plan.recv_stage.resize(static_cast<std::size_t>(local));
        zero_full_element_stage(plan.recv_stage);
        plan.recv_ptr = local == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
        plan.repack = repack_mode::pack_single;
        plan.repack_source = recv.mutable_data;
        plan.repack_layout = std::addressof(*recv.map);
        plan.user_recv = recv.mutable_data;
        plan.recv_layout = std::addressof(*recv.map);
        plan.unpack = true;
        plan.whole_image_unpack = true;
        return plan;
      }

      if (not send.sequence and not send_nonflat and not recv_nonflat) {
        if (recv.typed)
          require_receive_layout(*recv.map, ctx);
        const T* send_pointer =
            send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
        T* recv_pointer =
            recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx) : recv.mutable_data;
        const data_block send_block =
            make_block(buffer_slot::send, send_pointer, total, nullptr, side_use::used,
                       place_state::data, ctx);
        const data_block recv_block =
            make_block(buffer_slot::recv, recv_pointer, local, nullptr, side_use::used,
                       place_state::data, ctx);
        plan.send_ptr = send_block.ptr;
        plan.recv_ptr = recv_block.mutable_ptr();
        return plan;
      }

      plan.staged = true;
      // Send side.
      if (send.sequence) {
        plan.send_stage.resize(static_cast<std::size_t>(total));
        std::ranges::copy_n(send.owned.data(), send.owned.size(), plan.send_stage.data());
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
      } else if (send_nonflat) {
        plan.send_stage.resize(static_cast<std::size_t>(total));
        zero_full_element_stage(plan.send_stage);
        stage_pack_blocks(send.data, *send.map, 1, plan.send_stage, 0, ctx);
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        plan.repack = repack_mode::pack_single;
        plan.repack_source = send.data;
        plan.repack_layout = std::addressof(*send.map);
      } else {
        plan.send_ptr =
            send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
      }

      // Receive side.
      if (recv_nonflat) {
        require_receive_layout(*recv.map, ctx);
        require_staged_receive_pointer(recv.mutable_data, *recv.map, ctx);
        plan.recv_stage.resize(static_cast<std::size_t>(local));
        zero_full_element_stage(plan.recv_stage);
        plan.recv_ptr = local == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
        plan.user_recv = recv.mutable_data;
        plan.recv_layout = std::addressof(*recv.map);
        plan.recv_blocks = 1;
        plan.unpack = true;
      } else {
        if (recv.typed)
          require_receive_layout(*recv.map, ctx);
        plan.recv_ptr =
            recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx) : recv.mutable_data;
      }
      return plan;
    }

    template<class T>
    [[nodiscard]] reduction_plan<T> make_reduce_scatter_block_plan(const mpi_communicator& c,
                                                                   const form_position<T>& send,
                                                                   const form_position<T>& recv,
                                                                   count_t recvcount,
                                                                   const context& ctx) {
      require_reduction_base<T>(ctx);
      require_typed_layout(send, ctx);
      require_typed_layout(recv, ctx);
      if (recvcount < 0)
        throw_invalid(ctx.operation, "negative count");
      const bool in_place = send.in_place;
      (void)resolve_sides(collective_family::reduce_scatter_block, c, 0,
                          place_state_of(in_place), ctx);
      // Defense in depth: the adapter preflights before the managed MPI_Op exists.
      require_used_position(buffer_slot::send, send, not in_place, ctx);
      require_used_position(buffer_slot::recv, recv, true, ctx);

      reduction_plan<T> plan;
      plan.count = base_datatype<T>().size == 0 ? 0 : static_cast<MPI_Count>(recvcount);
      const count_t total = checked_count_mul(recvcount, c.size(), ctx);
      const count_t local = recvcount;
      const count_t send_elements = send.element_count();
      const count_t recv_elements = recv.element_count();
      const auto require_size = [ctx](count_t actual, count_t expected, const char* side) {
        if (actual != expected) {
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "{} buffer has {} elements but the count requires {}", side, actual, expected);
        }
      };
      if (in_place) {
        require_size(recv_elements, total, "receive");
      } else {
        require_size(send_elements, total, "send");
        require_size(recv_elements, local, "receive");
      }

      const bool send_nonflat = send.typed and not send.map->is_base_repetition();
      const bool recv_nonflat = recv.typed and not recv.map->is_base_repetition();

      if (in_place) {
        if (not recv_nonflat) {
          if (recv.typed)
            require_receive_layout(*recv.map, ctx);
          T* pointer = recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx)
                                  : recv.mutable_data;
          const data_block recv_block = make_block(buffer_slot::recv, pointer, total, nullptr,
                                                   side_use::used, place_state::data, ctx);
          plan.send_ptr = MPI_IN_PLACE;
          plan.recv_ptr = recv_block.mutable_ptr();
          return plan;
        }
        // A whole-buffer non-flat region cannot be the datatype of an MPI_IN_PLACE call: MPI
        // repeats the receive datatype `sum(recvcounts)` times at its extent stride, while the
        // region is one absolute map over the full input vector.  Stage the flat input image
        // and issue the native reduction out of place; MPI writes this rank's result block at
        // the start of the receive area (MPI 4.1 6.10.2), which completion lifts into the input
        // image before unpacking the whole map back into the caller's buffer.
        require_receive_layout(*recv.map, ctx);
        require_staged_receive_pointer(recv.mutable_data, *recv.map, ctx);
        plan.staged = true;
        plan.send_stage.resize(static_cast<std::size_t>(total));
        zero_full_element_stage(plan.send_stage);
        stage_pack_blocks(recv.mutable_data, *recv.map, 1, plan.send_stage, 0, ctx);
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        plan.recv_stage.resize(static_cast<std::size_t>(local));
        zero_full_element_stage(plan.recv_stage);
        plan.recv_ptr = local == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
        plan.repack = repack_mode::pack_single;
        plan.repack_source = recv.mutable_data;
        plan.repack_layout = std::addressof(*recv.map);
        plan.user_recv = recv.mutable_data;
        plan.recv_layout = std::addressof(*recv.map);
        plan.unpack = true;
        plan.whole_image_unpack = true;
        return plan;
      }

      if (not send.sequence and not send_nonflat and not recv_nonflat) {
        if (recv.typed)
          require_receive_layout(*recv.map, ctx);
        const T* send_pointer =
            send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
        T* recv_pointer =
            recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx) : recv.mutable_data;
        const data_block send_block =
            make_block(buffer_slot::send, send_pointer, total, nullptr, side_use::used,
                       place_state::data, ctx);
        const data_block recv_block =
            make_block(buffer_slot::recv, recv_pointer, local, nullptr, side_use::used,
                       place_state::data, ctx);
        plan.send_ptr = send_block.ptr;
        plan.recv_ptr = recv_block.mutable_ptr();
        return plan;
      }

      plan.staged = true;
      if (send.sequence) {
        plan.send_stage.resize(static_cast<std::size_t>(total));
        std::ranges::copy_n(send.owned.data(), send.owned.size(), plan.send_stage.data());
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
      } else if (send_nonflat) {
        plan.send_stage.resize(static_cast<std::size_t>(total));
        zero_full_element_stage(plan.send_stage);
        stage_pack_blocks(send.data, *send.map, 1, plan.send_stage, 0, ctx);
        plan.send_ptr = total == 0 ? dummy_buffer(buffer_slot::send) : plan.send_stage.data();
        plan.repack = repack_mode::pack_single;
        plan.repack_source = send.data;
        plan.repack_layout = std::addressof(*send.map);
      } else {
        plan.send_ptr =
            send.typed ? repetition_ptr(send.data, *send.map, ctx) : send.send_data();
      }

      if (recv_nonflat) {
        require_receive_layout(*recv.map, ctx);
        plan.recv_stage.resize(static_cast<std::size_t>(local));
        zero_full_element_stage(plan.recv_stage);
        plan.recv_ptr = local == 0 ? dummy_buffer(buffer_slot::recv) : plan.recv_stage.data();
        plan.user_recv = recv.mutable_data;
        plan.recv_layout = std::addressof(*recv.map);
        plan.recv_blocks = 1;
        plan.unpack = true;
      } else {
        if (recv.typed)
          require_receive_layout(*recv.map, ctx);
        plan.recv_ptr =
            recv.typed ? repetition_ptr(recv.mutable_data, *recv.map, ctx) : recv.mutable_data;
      }
      return plan;
    }

  }  // namespace detail

  // ---- reduce: root final reduction, send may be in-place at the root ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void reduce(const mpi_communicator& c, Op&& op, int root, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce, root,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::reduce, root, send_position,
                                  recv_position, native_op),
        ctx);
    detail::native_reduce(plan.send_ptr, plan.recv_ptr, plan.count, native_op, root,
                          c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ireduce(const mpi_communicator& c, Op&& op, int root, S&& send,
                                R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ireduce"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce, root,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::reduce, root, send_position,
                                  recv_position, native_op),
        ctx);
    request result = detail::native_ireduce(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                                            root, c.native_handle());
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx);
  }

  // ---- reduce_local: local reduction, no communicator and no in-place ----

  template<class S, class R, class Op>
  requires(send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void reduce_local(Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_local"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::reduce_local_form<T>(std::forward<Op>(op), send_position, recv_position, ctx);
  }

  // ---- allreduce: every rank reduced vector ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void allreduce(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allreduce"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::allreduce, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::allreduce, 0, send_position,
                                  recv_position, native_op),
        ctx);
    detail::native_allreduce(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                             c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iallreduce(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iallreduce"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::allreduce, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::allreduce, 0, send_position,
                                  recv_position, native_op),
        ctx);
    request result = detail::native_iallreduce(plan.send_ptr, plan.recv_ptr, plan.count,
                                               native_op, c.native_handle());
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx);
  }

  // ---- scan / exscan: intracommunicator prefix reductions ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void scan(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scan"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::scan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::scan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    detail::native_scan(plan.send_ptr, plan.recv_ptr, plan.count, native_op, c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iscan(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iscan"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::scan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::scan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    request result = detail::native_iscan(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                                          c.native_handle());
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void exscan(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::exscan"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::exscan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::exscan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    detail::native_exscan(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                          c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request iexscan(const mpi_communicator& c, Op&& op, S&& send, R&& recv) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::iexscan"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::exscan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::exscan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    request result = detail::native_iexscan(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                                            c.native_handle());
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx);
  }

  // ---- reduce_scatter: counts-only descriptor, no root ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void reduce_scatter(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                      std::span<const count_t> recvcounts) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_scatter"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan =
        detail::make_reduce_scatter_plan<T>(c, send_position, recv_position, recvcounts, ctx);
    detail::native_reduce_scatter(plan.send_ptr, plan.recv_ptr, plan.counts.data(), native_op,
                                  c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ireduce_scatter(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                                        std::span<const count_t> recvcounts) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ireduce_scatter"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan =
        detail::make_reduce_scatter_plan<T>(c, send_position, recv_position, recvcounts, ctx);
    request result = detail::native_ireduce_scatter(
        plan.send_ptr, plan.recv_ptr, plan.counts.data(), native_op, c.native_handle());
    // The counts array is read until completion: the payload always owns it.
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx,
                                                detail::payload_policy::always);
  }

  // ---- reduce_scatter_block: scalar per-rank count, no root ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && detail::compatible_form_elements_v<S, R>)
  void reduce_scatter_block(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                            count_t recvcount) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_scatter_block"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter_block,
                                            0, send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan = detail::make_reduce_scatter_block_plan<T>(
        c, send_position, recv_position, recvcount, ctx);
    detail::native_reduce_scatter_block(plan.send_ptr, plan.recv_ptr, plan.count, native_op,
                                        c.native_handle());
    detail::finish_blocking_reduction(plan, ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request ireduce_scatter_block(const mpi_communicator& c, Op&& op, S&& send,
                                              R&& recv, count_t recvcount) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::ireduce_scatter_block"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter_block,
                                            0, send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan = detail::make_reduce_scatter_block_plan<T>(
        c, send_position, recv_position, recvcount, ctx);
    request result = detail::native_ireduce_scatter_block(
        plan.send_ptr, plan.recv_ptr, plan.count, native_op, c.native_handle());
    return detail::finish_nonblocking_reduction(std::move(result), std::move(plan), ctx,
                                                detail::payload_policy::always);
  }

  // =========================================================================================
  // Persistent collectives (MPI 4 section 6.13), raw forms only: scalar, pointer+count and
  // pointer+layout.  *_init is a nonlocal collective initialization (same ordering rules as
  // the blocking/nonblocking families) that returns an inactive persistent request.
  // start()/start_all() are local; wait/test (single or multi) complete one cycle and leave the
  // request inactive and reusable.  The caller's buffers, layouts, count and displacement
  // arrays must stay alive and unchanged until the request is freed.  Active destruction waits
  // then frees (it may block); cancel() is refused; release() is the non-MPI escape.
  // =========================================================================================

  inline request barrier_init(const mpi_communicator& c, const info& i = {}) {
    detail::require_collective_communicator(c, "mp1::barrier_init");
    return detail::native_barrier_init(c.native_handle(), i.native_handle());
  }

  // ---- persistent bcast / gather / scatter / allgather / alltoall ----
  //
  // Same shapes as the nonblocking forms, plus the trailing info and the fixed-buffer rule:
  // a resizable container (vector/string) is rejected because the plan has no per-start
  // data()/size() revalidation; a non-contiguous sequence is rejected
  // for the same reason.  Staged buffers and duplicated type maps are owned until the request
  // is freed; a staged send is re-packed before every start.

  template<class B>
  requires(recv_sink<B> && borrowable<B &&> &&
           !resizable_element_buffer<std::remove_cvref_t<B>>)
  [[nodiscard]] request bcast_init(const mpi_communicator& c, int root, B&& buffer,
                                   const info& i = {}) {
    using T = detail::form_element_t<B>;
    const context ctx{"mp1::bcast_init"};
    detail::form_position<T> position =
        detail::make_form_position<T>(std::forward<B>(buffer), ctx);
    detail::form_plan<T> plan = detail::make_bcast_form_plan<T>(
        c, root, position, detail::form_lifetime::persistent, ctx);
    request result =
        detail::native_bcast_init(plan.call, root, c.native_handle(), i.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request gather_init(const mpi_communicator& c, int root, S&& send, R&& recv,
                                    const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::gather_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_gather_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::persistent, ctx);
    request result =
        detail::native_gather_init(plan.call, root, c.native_handle(), i.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  template<class S, class R>
  requires(send_source<S> && in_place_recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request scatter_init(const mpi_communicator& c, int root, S&& send, R&& recv,
                                     const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scatter_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_scatter_form_plan<T>(
        c, root, send_position, recv_position, detail::form_lifetime::persistent, ctx);
    request result =
        detail::native_scatter_init(plan.call, root, c.native_handle(), i.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request allgather_init(const mpi_communicator& c, S&& send, R&& recv,
                                       const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allgather_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_allgather_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::persistent, ctx);
    request result =
        detail::native_allgather_init(plan.call, c.native_handle(), i.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  template<class S, class R>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request alltoall_init(const mpi_communicator& c, S&& send, R&& recv,
                                      const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::alltoall_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::form_plan<T> plan = detail::make_alltoall_form_plan<T>(
        c, send_position, recv_position, detail::form_lifetime::persistent, ctx);
    request result =
        detail::native_alltoall_init(plan.call, c.native_handle(), i.native_handle());
    detail::attach_form_payload(result, std::move(plan));
    return result;
  }

  // ---- persistent reduce / allreduce / scan / exscan ----
  //
  // Same shapes as the nonblocking forms, plus the trailing info and the fixed-buffer rule.
  // The plan (including a staged send or an in-place staging vector) is owned until the request
  // is freed; a staged send is re-packed before every start.

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request reduce_init(const mpi_communicator& c, Op&& op, int root, S&& send,
                                    R&& recv, const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce, root,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::reduce, root, send_position,
                                  recv_position, native_op),
        ctx);
    return detail::reduce_init_call(std::move(plan), native_op, root, c.native_handle(),
                                    i.native_handle(), ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request allreduce_init(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                                       const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::allreduce_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::allreduce, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::allreduce, 0, send_position,
                                  recv_position, native_op),
        ctx);
    return detail::allreduce_init_call(std::move(plan), native_op, c.native_handle(),
                                       i.native_handle(), ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request scan_init(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                                  const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::scan_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::scan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::scan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    return detail::scan_init_call(std::move(plan), native_op, c.native_handle(),
                                  i.native_handle(), ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request exscan_init(const mpi_communicator& c, Op&& op, S&& send, R&& recv,
                                    const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::exscan_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::exscan, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::reduction_plan<T> plan = detail::make_reduction_plan<T>(
        c,
        detail::reduction_request(detail::collective_family::exscan, 0, send_position,
                                  recv_position, native_op),
        ctx);
    return detail::exscan_init_call(std::move(plan), native_op, c.native_handle(),
                                    i.native_handle(), ctx);
  }

  // ---- persistent reduce_scatter / reduce_scatter_block: counts copied at initiation ----

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request reduce_scatter_init(const mpi_communicator& c, Op&& op, S&& send,
                                            R&& recv, std::span<const count_t> recvcounts,
                                            const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_scatter_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter, 0,
                                            send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan =
        detail::make_reduce_scatter_plan<T>(c, send_position, recv_position, recvcounts, ctx);
    return detail::reduce_scatter_init_call(std::move(plan), native_op, c.native_handle(),
                                            i.native_handle(), ctx);
  }

  template<class S, class R, class Op>
  requires(in_place_send_source<S> && recv_sink<R> && borrowable<S &&> &&
           !resizable_element_buffer<std::remove_cvref_t<S>> &&
           !element_sequence<std::remove_cvref_t<S>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>> &&
           detail::compatible_form_elements_v<S, R>)
  [[nodiscard]] request reduce_scatter_block_init(const mpi_communicator& c, Op&& op, S&& send,
                                                  R&& recv, count_t recvcount,
                                                  const info& i = {}) {
    using T = detail::collective_element_t<S, R>;
    const context ctx{"mp1::reduce_scatter_block_init"};
    detail::form_position<T> send_position =
        detail::make_form_position<T>(std::forward<S>(send), ctx);
    detail::form_position<T> recv_position =
        detail::make_form_position<T>(std::forward<R>(recv), ctx);
    detail::validate_reduction_positions<T>(c, detail::collective_family::reduce_scatter_block,
                                            0, send_position, recv_position, ctx);
    const detail::reduction_operation native_op = detail::reduction_op<T>(op);
    detail::require_reduce_scatter_callback_image<T>(native_op, ctx);
    detail::reduction_plan<T> plan = detail::make_reduce_scatter_block_plan<T>(
        c, send_position, recv_position, recvcount, ctx);
    return detail::reduce_scatter_block_init_call(std::move(plan), native_op, c.native_handle(),
                                                  i.native_handle(), ctx);
  }


}  // namespace mp1

#endif
