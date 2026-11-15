#if !(defined MP1_IMPL_RMA_HPP)

#define MP1_IMPL_RMA_HPP

// Public one-sided communication overlay: the fifteen synchronization calls and the eighteen
// data-movement declarations.  Every function is free, window first, and a thin adapter over
// exactly one MPI RMA call in detail/rma_engine.hpp; there is no implicit flush, barrier or
// epoch management.
//
// Buffer forms.  The six data-movement families take the
// origin/result as a `std::span<T>` (a native T array whose size is the element count) or a
// `region<T>` (one occurrence of a layout<T>); the target is a `count_t` element count or a
// `layout<T>`:
//
//   put/get/accumulate/rput/rget/raccumulate   2 declarations each = 4 shapes each
//   get_accumulate/rget_accumulate            2 declarations each = 8 shapes each
//   fetch_and_op/compare_and_swap             1 declaration each, pointer single element
//
// That is 18 declarations and preserves all 42 origin/result/target shapes (6x4 + 2x8 + 2).
// The eight get_accumulate shapes are origin {span, region} x result {span, region} x target
// {count, layout}; a region and a span are alternatives for the same position, so one
// declaration covers both.  There is no raw pointer+count data-movement form any more: a
// pointer carries no size, and the size is what the size rule checks.
//
// Count and disjointness rule.  The buffer size is a checked fact, never an inference:
//   * origin.size()/result.size() == target_count for a flat target, or == target.size() for a
//     layout target, before the MPI call; a region contributes map().size(), a span
//     contributes span.size().  A mismatch throws std::invalid_argument before MPI.
//   * target_count is never ignored or derived from a buffer size; a zero element count is
//     legal.
//   * the both-flat origin/result pair of get_accumulate/rget_accumulate (and of
//     fetch_and_op/compare_and_swap) must be disjoint, checked as an exact byte range before
//     the MPI call (MPI 4.1 §12.3.4: "The origin and result buffers must be disjoint").  A
//     mixed flat/layout or layout/layout pair leaves cross-object overlap to the caller: the
//     layout objects' addresses are not available to the plan (documented limitation).
//   * MPI_PROC_NULL skips every data-movement check (the whole call is a normalized no-op).
//     Every origin under MPI_NO_OP is ignored because MPI ignores origin_addr, origin_count
//     and origin_datatype (§12.3.4): the wrapper skips the origin size check, passes the RMA
//     dummy address with count 0 and MPI_BYTE, and skips the both-flat origin/result overlap
//     check, so a real origin buffer (even one aliasing the result) is the legal atomic-read
//     spelling.  A null origin without MPI_NO_OP or a null result is rejected before MPI.
//
// Completion semantics (MPI 4.1 §12.7 Semantics and Correctness, rule 1): a blocking
// put/get/accumulate returns when the call has been made, NOT when the data is complete.
// Origin buffers must stay alive (and get result buffers must not be read) until the next
// synchronization call: fence, complete, flush, flush_all, flush_local, flush_local_all,
// unlock or unlock_all.  get returns void; it deliberately does not return T.
//
// Accumulating families.  MPI 4.1 §12.3.4 requires each datatype argument to be a
// predefined datatype or a derived datatype whose basic components are all the same predefined
// datatype, and the operator is applied to elements of that predefined type.  The typed API
// enforces this at compile time for predefined and mp1-built/reflected T, and with a mandatory
// one-time validation of the committed datatype for an untrusted custom/manual annotation:
//
//   * T predefined: basic_component_t<T> = T, so the operator argument is
//     mp1::predefined_op<T> exactly as before (a built-in reduction functor mapped through the
//     §6.9.2 Predefined Reduction Operations table, or a raw predefined MPI_Op).  Unknown/
//     user-created handles, MPI_OP_NULL, MPI_MAXLOC and MPI_MINLOC are rejected (mp1 does not
//     model an MPI_2* pair datatype); they throw std::invalid_argument before the MPI call.
//   * T homogeneous-derived (std::pair/tuple/array, MP1_REFLECTION, or an annotated custom
//     trait): the operator argument is predefined_op<basic_component_t<T>>, e.g.
//     predefined_op<int> for std::pair<int,int>.  Typed functors work when the functor
//     operand matches that basic component.  predefined_op<T> for the aggregate is a
//     compile-time error.  The annotated custom/manual form is a runtime-checked contract
//     (one-time committed-datatype validation), not a compile-time proof.
//   * T mixed (heterogeneous), empty, or an opaque custom type without an annotation:
//     compile-time error (detail::accumulate_op_t<T>).
//
// MPI_REPLACE is legal in every accumulating family for a predefined or homogeneous-derived
// T (MPI 4.1 §12.3.4; the historical Annex B "predefined only" sentence is the withdrawn MPI
// 2.1 rule); a heterogeneous derived REPLACE is a compile-time error.  MPI_NO_OP is legal only
// in get_accumulate, rget_accumulate and fetch_and_op (§12.3.4 Get Accumulate), where a null
// origin is accepted because the origin arguments are ignored.  Get Accumulate, Fetch and Op
// and Compare and Swap require the origin and result buffers to be disjoint (§12.3.4);
// the both-flat form checks the byte range before the MPI call, while a mixed
// flat/layout or layout/layout pair leaves overlap across the distinct buffers/layout objects
// to the caller. compare_and_swap requires a predefined datatype in the C integer, Fortran
// integer, Logical, Multi-language or Byte category (§12.3.4 Compare and Swap); double, float,
// complex, char/wchar_t and derived types are rejected.  For an untrusted
// MP1_DATATYPE_BASIC(T,U) or a manual struct_builder<T>::component_types annotation, the
// committed datatype is validated once before the first accumulating call, because mp1 did not
// synthesize it from the listed members.  A span origin is native_datatype<T>(); a region
// origin/target is a derived datatype over the same basic type (for put/get it may be
// heterogeneous).  Atomicity is per basic element, not per C++ T object (§12.7.1).
//
// Request-based RMA is legal only inside a passive-target epoch (lock/lock_all); completion is
// origin-side and is reported by wait()/test() (or by the closing epoch/flush*).  The
// origin/result buffers must stay valid and unmodified until that completion, and target
// visibility still requires flush/unlock.  Each request initiation duplicates the caller's
// origin/result layout (the `region` map) and the target layout into a request-owned payload
// (MPI_Type_dup), so the caller's `region` object and layout may
// die as soon as the call returns; the payload is freed when the request completes.  The data
// buffers are copied, never owned.  MPI_Request_free is erroneous for RMA requests and
// request::cancel() rejects them; dropping or destroying an active request performs MPI_Wait,
// while release() is the no-MPI-call escape.  When the target rank is MPI_PROC_NULL every call
// is a normalized no-op (valid MPI_BYTE/count-0 descriptors) and still counts as the one MPI
// call for the operation; the epoch must still be closed.  All ten data-movement families
// accept MPI_PROC_NULL (§12.3); lock/unlock/flush/flush_local do not (§12.5.3, §12.5.4).
//
// A target layout must contain only relative displacements, so the only target form is
// layout<T>; there is no absolute/heterogeneous layout type (MPI 4.1 §12.3.1 Put).  A target
// layout with provably overlapping entries is rejected for put/accumulate/get_accumulate
// (target updated) and for the result/origin position of get/get_accumulate (result updated);
// `unknown` overlap provenance is accepted and remains the caller's responsibility.
//
// Threads (MPI 4.1 §11.6.1/§11.6.2): no mp1 object is internally synchronized; distinct
// windows may be used concurrently when the thread level allows it, and the same window and
// its synchronization calls must be ordered by the application.  A view must not be used
// concurrently with its owner's free().
#include "mp1/impl/detail/concepts.hpp"
#include "mp1/impl/detail/rma_engine.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/operator.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/window.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>


namespace mp1 {

  /// MPI_LOCK_EXCLUSIVE / MPI_LOCK_SHARED.
  enum class lock_type : int { exclusive = MPI_LOCK_EXCLUSIVE, shared = MPI_LOCK_SHARED };

  // -----------------------------------------------------------------------------------------
  // Synchronization
  // -----------------------------------------------------------------------------------------

  /// MPI_Win_fence; collective over the window group.  On return, the RMA operations this
  /// process started before the call are complete at the origin; they are complete at each
  /// target when that target's matching fence returns, so an origin's fence alone does not
  /// establish target-side completion (§12.5.1).  Opens/closes the access and exposure epochs;
  /// every member of the window group must call it with compatible assertions.  Usually
  /// synchronizing, but a fence known not to close an epoch need not synchronize.
  inline void fence(const mpi_window& w, int assert = 0) {
    detail::require_rma_window(w, "mp1::fence");
    detail::check_mpi(MPI_Win_fence(assert, w.native_handle()), {"MPI_Win_fence"});
  }

  /// MPI_Win_lock; opens a passive-target access epoch to one target.  May block.  The target
  /// rank must be a valid rank of the window group (MPI_PROC_NULL is not accepted by mp1).
  inline void lock(const mpi_window& w, lock_type type, int rank, int assert = 0) {
    detail::require_rma_window(w, "mp1::lock");
    if (type != lock_type::exclusive and type != lock_type::shared) {
      throw_invalid("mp1::lock", "invalid lock_type");
    }
    detail::require_rma_peer_rank(w, rank, "mp1::lock");
    detail::check_mpi(MPI_Win_lock(static_cast<int>(type), rank, assert, w.native_handle()),
                      {"MPI_Win_lock"});
  }

  /// MPI_Win_lock_all; passive shared lock on every target.  Not collective (ALL means all
  /// targets).  Must be closed with unlock_all.
  inline void lock_all(const mpi_window& w, int assert = 0) {
    detail::require_rma_window(w, "mp1::lock_all");
    detail::check_mpi(MPI_Win_lock_all(assert, w.native_handle()), {"MPI_Win_lock_all"});
  }

  /// MPI_Win_unlock; closes the passive epoch and completes origin and target side for rank.
  inline void unlock(const mpi_window& w, int rank) {
    detail::require_rma_window(w, "mp1::unlock");
    detail::require_rma_peer_rank(w, rank, "mp1::unlock");
    detail::check_mpi(MPI_Win_unlock(rank, w.native_handle()), {"MPI_Win_unlock"});
  }

  /// MPI_Win_unlock_all; closes the lock_all epoch and completes origin and target side.
  inline void unlock_all(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::unlock_all");
    detail::check_mpi(MPI_Win_unlock_all(w.native_handle()), {"MPI_Win_unlock_all"});
  }

  /// MPI_Win_flush; completes preceding RMA to rank on both origin and target side.  Legal
  /// only inside a passive-target epoch.
  inline void flush(const mpi_window& w, int rank) {
    detail::require_rma_window(w, "mp1::flush");
    detail::require_rma_peer_rank(w, rank, "mp1::flush");
    detail::check_mpi(MPI_Win_flush(rank, w.native_handle()), {"MPI_Win_flush"});
  }

  /// MPI_Win_flush_all; flush for every target.  Legal only inside a passive-target epoch.
  inline void flush_all(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::flush_all");
    detail::check_mpi(MPI_Win_flush_all(w.native_handle()), {"MPI_Win_flush_all"});
  }

  /// MPI_Win_flush_local; completes preceding RMA to rank on the origin side only.  The origin
  /// buffer may be reused; target visibility is not guaranteed.  Passive epoch only.
  inline void flush_local(const mpi_window& w, int rank) {
    detail::require_rma_window(w, "mp1::flush_local");
    detail::require_rma_peer_rank(w, rank, "mp1::flush_local");
    detail::check_mpi(MPI_Win_flush_local(rank, w.native_handle()), {"MPI_Win_flush_local"});
  }

  /// MPI_Win_flush_local_all; origin-side completion for every target.  Passive epoch only.
  inline void flush_local_all(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::flush_local_all");
    detail::check_mpi(MPI_Win_flush_local_all(w.native_handle()), {"MPI_Win_flush_local_all"});
  }

  /// MPI_Win_sync; synchronizes the private and public window copies (separate model) and
  /// makes shared-memory stores/loads visible to other processes (unified model).  It does
  /// not open or close an epoch and does not complete pending RMA operations.  Legal only
  /// inside a passive-target epoch.
  inline void sync(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::sync");
    detail::check_mpi(MPI_Win_sync(w.native_handle()), {"MPI_Win_sync"});
  }

  /// MPI_Win_post; local, opens an exposure epoch for the origins in gr (the group argument
  /// of MPI_Win_post is the set of origin processes that may access this target; the target
  /// set of an access epoch is MPI_Win_start's group).  MPI 4.1 §12.5.2 makes it a local
  /// procedure: it does not wait for the matching starts; MPI_Win_start may block until the
  /// matching posts have been made.
  inline void post(const mpi_window& w, const group& gr, int assert = 0) {
    detail::require_rma_window(w, "mp1::post");
    if (not gr.is_valid())
      throw_invalid("mp1::post", "invalid group");
    detail::check_mpi(MPI_Win_post(gr.native_handle(), assert, w.native_handle()),
                      {"MPI_Win_post"});
  }

  /// MPI_Win_start; local, starts an access epoch targeting gr.  May block until matching
  /// posts have been made.
  inline void start(const mpi_window& w, const group& gr, int assert = 0) {
    detail::require_rma_window(w, "mp1::start");
    if (not gr.is_valid())
      throw_invalid("mp1::start", "invalid group");
    detail::check_mpi(MPI_Win_start(gr.native_handle(), assert, w.native_handle()),
                      {"MPI_Win_start"});
  }

  /// MPI_Win_complete; closes an access epoch.  Guarantees origin-side completion only.
  inline void complete(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::complete");
    detail::check_mpi(MPI_Win_complete(w.native_handle()), {"MPI_Win_complete"});
  }

  /// MPI_Win_wait; closes an exposure epoch and waits for all matching completes.  Target-side
  /// completion is guaranteed when it returns.
  inline void wait(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::wait");
    detail::check_mpi(MPI_Win_wait(w.native_handle()), {"MPI_Win_wait"});
  }

  /// MPI_Win_test; nonblocking form of wait().  Returns true when the exposure epoch has
  /// completed; after a true return do not call test again until the next post.
  [[nodiscard]] inline bool test(const mpi_window& w) {
    detail::require_rma_window(w, "mp1::test");
    int flag{0};
    detail::check_mpi(MPI_Win_test(w.native_handle(), &flag), {"MPI_Win_test"});
    return flag != 0;
  }

  namespace detail {

    // The public span/region adapters.  A span contributes its native element count and a null
    // layout; a region contributes map().size() and the layout pointer.  The engine's
    // prepare_* functions implement the size rule once for every family and every shape.
    template<class B>
    requires rma_origin<B>
    [[nodiscard]] inline rma_origin_buffer<rma_element_t<B>> make_rma_origin_buffer(
        const B& buffer) {
      using U = std::remove_cvref_t<B>;
      if constexpr (is_region_v<U>) {
        return {buffer.data(), &buffer.map(), buffer.map().size()};
      } else {
        return {buffer.data(), nullptr, static_cast<count_t>(buffer.size())};
      }
    }

    template<class B>
    requires rma_result<B>
    [[nodiscard]] inline rma_result_buffer<rma_element_t<B>> make_rma_result_buffer(
        B&& buffer) {
      using U = std::remove_cvref_t<B>;
      if constexpr (is_region_v<U>) {
        return {buffer.data(), &buffer.map(), buffer.map().size()};
      } else {
        return {buffer.data(), nullptr, static_cast<count_t>(buffer.size())};
      }
    }

  }  // namespace detail

  // -----------------------------------------------------------------------------------------
  // put: target_disp is scaled by the TARGET process's disp_unit, not by sizeof(T).  The
  // origin (span or region) must describe exactly the target element count.
  // -----------------------------------------------------------------------------------------

  /// Origin (span or region) to a flat target of target_count elements of T.
  template<class O>
  requires(detail::rma_origin<O>)
  void put(const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
           count_t target_count) {
    detail::rma_put<detail::rma_element_t<O>>(w, detail::make_rma_origin_buffer(origin),
                                              target_rank, target_disp, target_count, nullptr,
                                              "mp1::put");
  }

  /// Origin (span or region) to a target layout (target.size() elements of T).
  template<class O>
  requires(detail::rma_origin<O>)
  void put(const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
           const layout<detail::rma_element_t<O>>& target) {
    detail::rma_put<detail::rma_element_t<O>>(w, detail::make_rma_origin_buffer(origin),
                                              target_rank, target_disp, 0, &target, "mp1::put");
  }

  // -----------------------------------------------------------------------------------------
  // get: data is valid at the origin only after the following synchronization call.  The
  // result (span or region) must describe exactly the target element count.
  // -----------------------------------------------------------------------------------------

  /// Flat target of target_count elements of T into the result (span or region).
  template<class R>
  requires(detail::rma_result<R>)
  void get(const mpi_window& w, R&& origin, int target_rank, MPI_Aint target_disp,
           count_t target_count) {
    detail::rma_get<detail::rma_element_t<R>>(w, detail::make_rma_result_buffer(origin),
                                              target_rank, target_disp, target_count, nullptr,
                                              "mp1::get");
  }

  /// Target layout (target.size() elements of T) into the result (span or region).
  template<class R>
  requires(detail::rma_result<R>)
  void get(const mpi_window& w, R&& origin, int target_rank, MPI_Aint target_disp,
           const layout<detail::rma_element_t<R>>& target) {
    detail::rma_get<detail::rma_element_t<R>>(w, detail::make_rma_result_buffer(origin),
                                              target_rank, target_disp, 0, &target, "mp1::get");
  }

  // -----------------------------------------------------------------------------------------
  // accumulate: op is a predefined MPI_Op or a functor that maps to one, typed as
  // predefined_op<basic_component_t<T>> (predefined_op<T> for a predefined T); MPI_NO_OP is
  // rejected (12.3.4 Get Accumulate) even for MPI_PROC_NULL.  The origin (span or region)
  // must describe exactly the target element count.
  // -----------------------------------------------------------------------------------------

  template<class O>
  requires(detail::rma_origin<O>)
  void accumulate(const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
                  count_t target_count,
                  std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    detail::rma_accumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), target_rank, target_disp, target_count,
        nullptr, op.native_handle(), "mp1::accumulate");
  }

  template<class O>
  requires(detail::rma_origin<O>)
  void accumulate(const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
                  const layout<detail::rma_element_t<O>>& target,
                  std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    detail::rma_accumulate<detail::rma_element_t<O>>(w, detail::make_rma_origin_buffer(origin),
                                                     target_rank, target_disp, 0, &target,
                                                     op.native_handle(), "mp1::accumulate");
  }

  // -----------------------------------------------------------------------------------------
  // get_accumulate: op is typed as predefined_op<basic_component_t<T>>; it returns the old
  // target value in result, and result data is valid after the following synchronization call.
  // The origin and result shapes are independent: each may be a span<T> (native datatype,
  // size equal to the target element count) or a region<T> (one layout occurrence, size equal
  // to the target element count), and the target is either a flat count_t or a layout<T>.  All
  // eight origin/result/target combinations are provided by the two declarations below.
  // Origin and result must be disjoint (§12.3.4): the both-flat form is checked before the
  // MPI call, while any mixed flat/layout or layout/layout pair leaves overlap across the
  // distinct buffers and layout objects to the caller.  MPI_NO_OP is legal and ignores every
  // origin argument (its size and overlap are not checked; the wire uses the dummy origin).
  // -----------------------------------------------------------------------------------------

  /// Flat target of target_count elements of T; the result-buffer size must equal it.
  template<class O, class Res>
  requires(detail::rma_origin<O> && detail::rma_result<Res> &&
           std::same_as<detail::rma_element_t<O>, detail::rma_element_t<Res>>)
  void get_accumulate(
      const mpi_window& w, const O& origin, Res&& result, int target_rank, MPI_Aint target_disp,
      count_t target_count,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    detail::rma_get_accumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), detail::make_rma_result_buffer(result),
        target_rank, target_disp, target_count, nullptr, op.native_handle(),
        "mp1::get_accumulate");
  }

  /// Target layout (target.size() elements of T); the origin and result sizes must equal it.
  template<class O, class Res>
  requires(detail::rma_origin<O> && detail::rma_result<Res> &&
           std::same_as<detail::rma_element_t<O>, detail::rma_element_t<Res>>)
  void get_accumulate(
      const mpi_window& w, const O& origin, Res&& result, int target_rank, MPI_Aint target_disp,
      const layout<detail::rma_element_t<O>>& target,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    detail::rma_get_accumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), detail::make_rma_result_buffer(result),
        target_rank, target_disp, 0, &target, op.native_handle(), "mp1::get_accumulate");
  }

  // -----------------------------------------------------------------------------------------
  // Single-element atomics
  // -----------------------------------------------------------------------------------------

  /// MPI_Compare_and_swap; single element (no count or layout overloads).  T must be a
  /// predefined datatype in the C integer, Fortran integer, Logical, Multi-language or Byte
  /// category (MPI 4.1 §12.3.4 Compare and Swap).  Origin and result must be disjoint; the
  /// compare buffer is a separate read-only input, not part of that pairwise rule.  The flat
  /// origin/result overlap is checked before the MPI call.
  template<class T>
  void compare_and_swap(const mpi_window& w, const T* origin, const T* compare, T* result,
                        int target_rank, MPI_Aint target_disp) {
    detail::rma_compare_and_swap<T>(w, origin, compare, result, target_rank, target_disp,
                                    "mp1::compare_and_swap");
  }

  /// MPI_Fetch_and_op; single element (no count or layout overloads).  T must be a predefined
  /// datatype.  op is a predefined MPI_Op or a mapping built-in functor; MPI_NO_OP and
  /// MPI_REPLACE are legal.  With MPI_NO_OP the origin is ignored (§12.3.4): null or real,
  /// and it may alias the result (the atomic-read idiom); otherwise the flat origin and result
  /// buffers must be disjoint (checked before the call).
  template<class T>
  void fetch_and_op(const mpi_window& w, const T* origin, T* result, int target_rank,
                    MPI_Aint target_disp, std::type_identity_t<predefined_op<T>> op) {
    detail::rma_fetch_and_op<T>(w, origin, result, target_rank, target_disp, op.native_handle(),
                                "mp1::fetch_and_op");
  }

  // -----------------------------------------------------------------------------------------
  // Request-based RMA: legal only inside lock/lock_all epochs.  Completion is origin-side and
  // is reported by wait()/test() (closing the epoch or a flush* also indicates completion,
  // §12.3.5).  Origin and result buffers must stay valid and unmodified until that origin-side
  // completion; get results must not be read before it, and target visibility still requires
  // flush/unlock.  The caller's `region` map and target layout are duplicated into a
  // request-owned payload at initiation (MPI_Type_dup), so those
  // layout objects may die when the call returns; the duplicate is freed on completion.
  // MPI_Request_free is erroneous for RMA requests and request::cancel() rejects them;
  // dropping or destroying an active request performs MPI_Wait, while release() is the
  // no-MPI-call escape.  The origin/result forms and the size checks are exactly those of the
  // blocking families; rget_accumulate's origin and result shapes are independent exactly as
  // in get_accumulate.
  // -----------------------------------------------------------------------------------------

  template<class O>
  requires(detail::rma_origin<O>)
  [[nodiscard]] request rput(const mpi_window& w, const O& origin, int target_rank,
                             MPI_Aint target_disp, count_t target_count) {
    return detail::rma_rput<detail::rma_element_t<O>>(w, detail::make_rma_origin_buffer(origin),
                                                      target_rank, target_disp, target_count,
                                                      nullptr, "mp1::rput");
  }

  template<class O>
  requires(detail::rma_origin<O>)
  [[nodiscard]] request rput(const mpi_window& w, const O& origin, int target_rank,
                             MPI_Aint target_disp,
                             const layout<detail::rma_element_t<O>>& target) {
    return detail::rma_rput<detail::rma_element_t<O>>(w, detail::make_rma_origin_buffer(origin),
                                                      target_rank, target_disp, 0, &target,
                                                      "mp1::rput");
  }

  template<class R>
  requires(detail::rma_result<R>)
  [[nodiscard]] request rget(const mpi_window& w, R&& origin, int target_rank,
                             MPI_Aint target_disp, count_t target_count) {
    return detail::rma_rget<detail::rma_element_t<R>>(w, detail::make_rma_result_buffer(origin),
                                                      target_rank, target_disp, target_count,
                                                      nullptr, "mp1::rget");
  }

  template<class R>
  requires(detail::rma_result<R>)
  [[nodiscard]] request rget(const mpi_window& w, R&& origin, int target_rank,
                             MPI_Aint target_disp,
                             const layout<detail::rma_element_t<R>>& target) {
    return detail::rma_rget<detail::rma_element_t<R>>(w, detail::make_rma_result_buffer(origin),
                                                      target_rank, target_disp, 0, &target,
                                                      "mp1::rget");
  }

  template<class O>
  requires(detail::rma_origin<O>)
  [[nodiscard]] request raccumulate(
      const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
      count_t target_count,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    return detail::rma_raccumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), target_rank, target_disp, target_count,
        nullptr, op.native_handle(), "mp1::raccumulate");
  }

  template<class O>
  requires(detail::rma_origin<O>)
  [[nodiscard]] request raccumulate(
      const mpi_window& w, const O& origin, int target_rank, MPI_Aint target_disp,
      const layout<detail::rma_element_t<O>>& target,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    return detail::rma_raccumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), target_rank, target_disp, 0, &target,
        op.native_handle(), "mp1::raccumulate");
  }

  template<class O, class Res>
  requires(detail::rma_origin<O> && detail::rma_result<Res> &&
           std::same_as<detail::rma_element_t<O>, detail::rma_element_t<Res>>)
  [[nodiscard]] request rget_accumulate(
      const mpi_window& w, const O& origin, Res&& result, int target_rank, MPI_Aint target_disp,
      count_t target_count,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    return detail::rma_rget_accumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), detail::make_rma_result_buffer(result),
        target_rank, target_disp, target_count, nullptr, op.native_handle(),
        "mp1::rget_accumulate");
  }

  template<class O, class Res>
  requires(detail::rma_origin<O> && detail::rma_result<Res> &&
           std::same_as<detail::rma_element_t<O>, detail::rma_element_t<Res>>)
  [[nodiscard]] request rget_accumulate(
      const mpi_window& w, const O& origin, Res&& result, int target_rank, MPI_Aint target_disp,
      const layout<detail::rma_element_t<O>>& target,
      std::type_identity_t<detail::accumulate_op_t<detail::rma_element_t<O>>> op) {
    return detail::rma_rget_accumulate<detail::rma_element_t<O>>(
        w, detail::make_rma_origin_buffer(origin), detail::make_rma_result_buffer(result),
        target_rank, target_disp, 0, &target, op.native_handle(), "mp1::rget_accumulate");
  }

}  // namespace mp1

#endif
