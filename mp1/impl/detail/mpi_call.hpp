#if !(defined MP1_DETAIL_MPI_CALL_HPP)

#define MP1_DETAIL_MPI_CALL_HPP

// The single MPI-version boundary of the library. Every other mp1 header includes this one
// instead of <mpi.h> (error.hpp is the sole exception because it must format MPI error codes).
#include <mpi.h>

#if !(defined MPI_VERSION) || MPI_VERSION < 4 || \
    (MPI_VERSION == 4 && (!defined(MPI_SUBVERSION) || MPI_SUBVERSION < 1))
#error "mp1 requires MPI 4.1 or newer"
#endif

#include "mp1/impl/detail/contract.hpp"
#include "mp1/impl/error.hpp"
#include "mp1/impl/fwd.hpp"

#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>


namespace mp1::detail {

  // ---------------------------------------------------------------------------------------
  // Process-global lifecycle (MPI 4.1 §11.2.1/§11.2.2:
  // exactly one initialization routine is called per World Model process and MPI cannot be
  // restarted after MPI_FINALIZE; §11.4.1/Table 11.1 lists the always-available queries).
  //
  // One atomic phase word is the sole authority for operations; transitions are serialized by
  // one leaked mutex and are monotone.  The fast path of every operation is a single acquire
  // load.  A release/destruction path adds exactly one Table 11.1 MPI_Finalized query once
  // the phase has settled (never the redundant MPI_Initialized pair) so it can still abandon
  // a handle after an application-issued MPI_Finalize.  All state is constant-initialized or
  // deliberately leaked, so it stays usable while static objects are destroyed (possibly
  // after MPI_Finalize).
  // ---------------------------------------------------------------------------------------

  enum class lifecycle_phase : unsigned char {
    idle,            // no MPI yet (or not observed yet)
    initializing,    // one thread inside MPI_Init/MPI_Init_thread
    active_owned,    // mp1 initialized MPI and must finalize it
    active_adopted,  // MPI was initialized outside mp1; the application finalizes it
    finalizing,      // one thread inside MPI_Finalize
    finalized,       // MPI_Finalize returned or was observed
    init_failed      // MPI_Init* returned an error (terminal; no retry)
  };

  inline std::atomic<lifecycle_phase>& lifecycle_phase_word() noexcept {
    static std::atomic<lifecycle_phase> phase{lifecycle_phase::idle};
    return phase;
  }

  /// Thread level recorded by the last successful init/adopt; MPI_THREAD_SINGLE before any
  /// init.  Stored as int because MPI's four constants are implementation-defined (only their
  /// order is guaranteed) and threading_mode is an opaque declaration in this header.
  inline std::atomic<int>& provided_thread_level() noexcept {
    static std::atomic<int> level{MPI_THREAD_SINGLE};
    return level;
  }

  /// The only lock in the library: init/finalize transitions (cold path, deliberately leaked).
  /// It is leaked on purpose: a function-local static object would be destroyed at exit while
  /// release paths may still take the lock.  The allocation cannot be retried or reported from
  /// a noexcept cold path, so an allocation failure here is fatal either way.
  inline std::mutex& lifecycle_mutex() noexcept {
    static std::mutex* mutex =
        new std::mutex;  // NOLINT(bugprone-unhandled-exception-at-new): immortal lock
    return *mutex;
  }

  inline bool mpi_initialized() noexcept {
    int flag{0};
    MPI_Initialized(&flag);
    return flag != 0;
  }

  inline bool mpi_finalized() noexcept {
    int flag{0};
    MPI_Finalized(&flag);
    return flag != 0;
  }

  inline bool mpi_available() noexcept {
    return mpi_initialized() and not mpi_finalized();
  }

  /// Publish the observed external MPI_Finalize.  Monotone: only the two terminal states are
  /// left alone.
  inline void publish_finalized() noexcept {
    auto& phase = lifecycle_phase_word();
    lifecycle_phase current = phase.load(std::memory_order_acquire);
    while (current != lifecycle_phase::finalized and current != lifecycle_phase::init_failed) {
      if (phase.compare_exchange_weak(current, lifecycle_phase::finalized,
                                      std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  /// Publish an externally initialized MPI environment (one CAS).  Returns true when MPI is
  /// active according to the published phase.  On CAS success `expected` still holds idle, so
  /// the result is returned explicitly rather than inferred from `expected`.
  ///
  /// The first adoption records the actual thread level with MPI_Query_thread so that
  /// provided() reports the level the application initialized MPI with instead of the
  /// conservative MPI_THREAD_SINGLE default.  The level is stored before the phase is
  /// published, so a reader that observes active_adopted also observes the recorded level.
  /// This is a noexcept hot-path adoption: an MPI_Query_thread failure is not reported and
  /// leaves the conservative MPI_THREAD_SINGLE value.  An explicit init() still queries the
  /// level on its own adoption path and records the result, as before.
  inline bool publish_adoption() noexcept {
    auto& phase = lifecycle_phase_word();
    lifecycle_phase expected = lifecycle_phase::idle;
    int level = MPI_THREAD_SINGLE;
    if (MPI_Query_thread(&level) != MPI_SUCCESS)
      level = MPI_THREAD_SINGLE;
    provided_thread_level().store(level, std::memory_order_release);
    if (phase.compare_exchange_strong(expected, lifecycle_phase::active_adopted,
                                      std::memory_order_acq_rel, std::memory_order_acquire)) {
      return true;
    }
    const lifecycle_phase current = phase.load(std::memory_order_acquire);
    return current == lifecycle_phase::active_owned or
           current == lifecycle_phase::active_adopted;
  }

  /// Hot-path predicate: no MPI call once a terminal/active phase is published.
  ///
  /// The idle phase performs the always-available MPI_Initialized/MPI_Finalized queries to
  /// adopt an externally initialized MPI (the documented external-init path) or to observe a
  /// finalize.  Hot operations deliberately do not observe an external MPI_Finalize once the
  /// phase is settled; calling them afterwards is an application error.
  inline bool mpi_active_fast() noexcept {
    const lifecycle_phase phase = lifecycle_phase_word().load(std::memory_order_acquire);
    if (phase == lifecycle_phase::active_owned or phase == lifecycle_phase::active_adopted)
      return true;
    if (phase != lifecycle_phase::idle)
      return false;
    if (mpi_finalized()) {
      publish_finalized();
      return false;
    }
    if (not mpi_initialized())
      return false;
    return publish_adoption();
  }

  /// Lifecycle-adjacent predicate: Table 11.1 queries, observes an external finalize and
  /// publishes adoption.  Used by destructors, release operations and lifecycle queries.
  ///
  /// Settled phases take the single-guard fast path: an active phase already proves MPI was
  /// initialized, and MPI cannot become uninitialized again (there is no restart after
  /// MPI_Finalize, MPI 4.1 §11.2.1), so only the always-available MPI_Finalized query can
  /// change the answer.  The redundant MPI_Initialized query is therefore skipped after the
  /// phase settled; the idle phase still performs both because `MPI_Finalized == false`
  /// alone does not prove that MPI was initialized.  The MPI_Finalized query is kept (one
  /// per release/destruction, not two) because a release path must not call MPI after an
  /// application-issued MPI_Finalize, even while mp1 owns the environment.
  inline bool mpi_active_checked() noexcept {
    const lifecycle_phase phase = lifecycle_phase_word().load(std::memory_order_acquire);
    switch (phase) {
      case lifecycle_phase::finalizing:
      case lifecycle_phase::finalized:
      case lifecycle_phase::init_failed:
        return false;
      case lifecycle_phase::active_owned:
      case lifecycle_phase::active_adopted:
        if (mpi_finalized()) {
          publish_finalized();
          return false;
        }
        return true;
      case lifecycle_phase::idle:
        if (mpi_finalized()) {
          publish_finalized();
          return false;
        }
        if (not mpi_initialized())
          return false;
        return publish_adoption();
      case lifecycle_phase::initializing:
        // The transition is still in progress, so the phase does not yet prove
        // initialization: keep observing both queries as before.
        if (mpi_finalized()) {
          publish_finalized();
          return false;
        }
        return mpi_initialized();
    }
    return false;
  }

  /// Full predicate for destructors and static-lifetime code; observes an external
  /// MPI_Finalize with the single-guard fast path of mpi_active_checked().  Public operations
  /// use require_mpi()'s phase-only fast path instead.
  inline bool mpi_active() noexcept {
    return mpi_active_checked();
  }

  inline std::atomic<std::size_t>& abandoned_handle_counter() noexcept {
    static std::atomic<std::size_t> counter{0};
    return counter;
  }

  inline void note_abandoned_handle() noexcept {
    abandoned_handle_counter().fetch_add(1, std::memory_order_relaxed);
  }

  inline std::size_t abandoned_handle_count() noexcept {
    return abandoned_handle_counter().load(std::memory_order_relaxed);
  }

  [[noreturn]] inline void throw_not_active(
      const context& ctx, std::source_location where = std::source_location::current()) {
    std::string_view reason;
    switch (lifecycle_phase_word().load(std::memory_order_acquire)) {
      case lifecycle_phase::initializing:
        reason = "MPI initialization is in progress";
        break;
      case lifecycle_phase::finalizing:
        reason = "MPI finalization is in progress";
        break;
      case lifecycle_phase::finalized:
        reason = "MPI is already finalized";
        break;
      case lifecycle_phase::init_failed:
        reason = "a previous MPI initialization failed";
        break;
      default:
        reason =
            "requires an active MPI environment (MPI_Init called, MPI_Finalize not called)";
        break;
    }
    detail::fail_with<std::runtime_error>(ctx, where, reason);
  }

  /// Operation precondition: MPI is active.  One acquire load on the settled fast path.
  inline void require_mpi(const context& ctx,
                          std::source_location where = std::source_location::current()) {
    if (not mpi_active_fast())
      throw_not_active(ctx, where);
  }

  /// Release-operation precondition (file::close, communicator free/disconnect, destructors):
  /// observes an external finalize before the MPI call.
  inline void require_mpi_checked(
      const context& ctx, std::source_location where = std::source_location::current()) {
    if (not mpi_active_checked())
      throw_not_active(ctx, where);
  }

  // ---------------------------------------------------------------------------------------
  // Checked conversions. Public inputs throw std::invalid_argument; internal arithmetic that
  // cannot be traced to one caller argument throws std::overflow_error (MPI 4.1 §2.5.8
  // defines MPI_Count, and §3.2.5 defines MPI_UNDEFINED for a count that does not fit the
  // int-sized output of MPI_GET_COUNT).
  //
  // std::in_range performs the exact bound check of a narrowing integer conversion, including
  // the mixed-signedness case a hand-written comparison gets wrong.  C++20 has no standard
  // checked add/subtract/multiply, so those three keep explicit overflow guards; std::cmp_*
  // is used where the operands have different types.  The dead checked conversions of
  // MPI_Aint and of an MPI_Count to int had no caller and are gone from the vocabulary.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline count_t checked_count(
      MPI_Count count, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (not std::in_range<count_t>(count)) {
      detail::fail_as<std::overflow_error>(ctx, where,
                                           "MPI count {} does not fit std::ptrdiff_t",
                                           static_cast<long long>(count));
    }
    return static_cast<count_t>(count);
  }

  [[nodiscard]] inline int checked_int(
      std::size_t size, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (not std::in_range<int>(size)) {
      detail::fail_as<std::invalid_argument>(ctx, where, "size {} does not fit int", size);
    }
    return static_cast<int>(size);
  }

  [[nodiscard]] inline MPI_Count checked_mpi_count(
      count_t count, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (count < 0) {
      detail::fail_as<std::invalid_argument>(ctx, where, "count {} is negative", count);
    }
    return static_cast<MPI_Count>(count);
  }

  /// MPI_UNDEFINED as an element count, tested with equality rather than a sign test.
  ///
  /// MPI 4.1 §3.2.5 defines MPI_UNDEFINED as the value MPI_GET_COUNT returns when the count
  /// cannot be expressed with the given datatype; the constant is implementation-defined and
  /// is not promised to be negative. A wrapper accessor that returns it as count_t must
  /// therefore be compared with the sentinel, never with zero.
  [[nodiscard]] inline constexpr bool is_undefined_count(count_t count) noexcept {
    return count == static_cast<count_t>(MPI_UNDEFINED);
  }

  // ---------------------------------------------------------------------------------------
  // MPI address arithmetic.  MPI 4.1 §5.1.5 requires MPI_AINT_ADD and MPI_AINT_DIFF for
  // arithmetic on MPI addresses: the MPI_Aint representation need not be a flat integer
  // address, so intrinsic `+`/`-` is not portable.  §5.1.12 additionally forbids a difference
  // of addresses from different allocations in a datatype displacement.
  //
  // The standard leaves the result of a non-representable difference undefined; the checked
  // helper makes that a local std::overflow_error by verifying the documented round trip.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline MPI_Aint checked_address_diff(
      MPI_Aint high, MPI_Aint low, const context& ctx,
      std::source_location where = std::source_location::current()) {
    const MPI_Aint difference = MPI_Aint_diff(high, low);
    if (MPI_Aint_add(low, difference) != high) {
      detail::fail_as<std::overflow_error>(ctx, where,
                                           "MPI address difference is not representable");
    }
    return difference;
  }

  /// A non-negative byte displacement added to an MPI address with MPI_Aint_add.
  [[nodiscard]] inline MPI_Aint checked_address_add(
      MPI_Aint base, count_t displacement, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (displacement < 0) {
      detail::fail_as<std::invalid_argument>(ctx, where, "negative MPI address displacement");
    }
    if (not std::in_range<MPI_Aint>(displacement)) {
      detail::fail_as<std::overflow_error>(ctx, where,
                                           "MPI address displacement does not fit MPI_Aint");
    }
    return MPI_Aint_add(base, static_cast<MPI_Aint>(displacement));
  }

  [[nodiscard]] inline count_t checked_count_add(
      count_t left, count_t right, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if ((right > 0 and left > std::numeric_limits<count_t>::max() - right) or
        (right < 0 and left < std::numeric_limits<count_t>::min() - right)) {
      detail::fail_as<std::overflow_error>(ctx, where, "count overflow");
    }
    return left + right;
  }

  [[nodiscard]] inline count_t checked_count_sub(
      count_t left, count_t right, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if ((right > 0 and left < std::numeric_limits<count_t>::min() + right) or
        (right < 0 and left > std::numeric_limits<count_t>::max() + right)) {
      detail::fail_as<std::overflow_error>(ctx, where, "count overflow");
    }
    return left - right;
  }

  // Element counts are non-negative; multiplication is checked for overflow and for an
  // accidental negative operand (an internal contract violation, not user input).
  [[nodiscard]] inline count_t checked_count_mul(
      count_t left, count_t right, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (left < 0 or right < 0) {
      detail::fail_as<std::invalid_argument>(ctx, where, "negative count in size computation");
    }
    if (left != 0 and right > std::numeric_limits<count_t>::max() / left) {
      detail::fail_as<std::overflow_error>(ctx, where, "element count overflow");
    }
    return left * right;
  }

  // ---------------------------------------------------------------------------------------
  // Owning-handle release used by every wrapper destructor. No MPI call when MPI is gone; the
  // handle is normalized and counted as abandoned instead.
  // ---------------------------------------------------------------------------------------

  template<class H, class Free>
  inline void release_handle(H& handle, H null_value, bool owns, Free free_fn) noexcept {
    if (handle == null_value)
      return;
    if (not owns) {
      handle = null_value;
      return;
    }
    if (not mpi_active()) {
      note_abandoned_handle();
      handle = null_value;
      return;
    }
    free_fn(handle);
    handle = null_value;
  }

  // ---------------------------------------------------------------------------------------
  // Noexcept finalization path (environment destructor and the opt-in atexit handler). The
  // caller has already established that the library owns MPI; the return code cannot be
  // reported. It never throws and never blocks on the lifecycle mutex.
  //
  // MPI 4.1 §11.2.2 describes MPI_FINALIZE as collective and as not freeing objects created by
  // MPI calls.  Live MPI objects (windows, files, requests, spawned/connected
  // intercommunicators) at this point are the application's MPI precondition, not an
  // mp1-managed resource: this path does not count, refuse, diagnose or skip them, and it does
  // not release them.  It simply calls MPI_Finalize when mp1 owns an MPI environment that is
  // still active.
  // ---------------------------------------------------------------------------------------

  inline void finalize_noexcept() noexcept {
    std::unique_lock<std::mutex> lock{lifecycle_mutex(), std::try_to_lock};
    if (not lock.owns_lock())
      return;
    auto& phase = lifecycle_phase_word();
    if (phase.load(std::memory_order_acquire) != lifecycle_phase::active_owned)
      return;
    if (mpi_finalized()) {
      phase.store(lifecycle_phase::finalized, std::memory_order_release);
      return;
    }
    if (not mpi_initialized()) {
      phase.store(lifecycle_phase::finalized, std::memory_order_release);
      return;
    }
    phase.store(lifecycle_phase::finalizing, std::memory_order_release);
    (void)MPI_Finalize();
    phase.store(lifecycle_phase::finalized, std::memory_order_release);
  }

  /// The handler registered by the opt-in mp1::finalize_at_exit().
  inline void finalize_at_exit() noexcept {
    finalize_noexcept();
  }

}  // namespace mp1::detail

#endif
