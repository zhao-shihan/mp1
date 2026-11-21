#if !(defined MP1_IMPL_WINDOW_HPP)

#define MP1_IMPL_WINDOW_HPP

// Window types and the four collective window-creation factories.  The data-movement and
// synchronization API lives in rma.hpp; this header only owns the MPI_Win handle and the
// standard window attributes.  The factories take designated-initializer option bags
// (window_create_options / window_allocate_options), so each creation form is one declaration;
// the metadata/validation helpers live in detail/window_engine.hpp, not in this public header.
//
// Ownership model (mirrors mpi_communicator / communicator):
//   mpi_window  copyable, borrowed, non-owning view; its destructor never calls MPI.
//   window      move-only owner that contains a private mpi_window and exposes it by value
//               through view()/an implicit conversion to mpi_window; free()/release()/the
//               destructor and move assignment over a live owner are the teardown paths.
//
// A live MPI_Win at MPI_Finalize is the caller's MPI precondition, not an mp1-managed resource:
// mp1 does not count windows, refuse finalization because of one, or free it for the caller
// (MPI 4.1 §11.2.2 says MPI_FINALIZE does not free objects created by MPI calls).
//
// MPI_Win_free is collective and (unless the window was created with the no_locks=true info
// hint) synchronizing: it delays its return until local passive-target accesses complete.  A
// destructor can therefore block.  Prefer an explicit free() on the normal path and use the
// destructor only as an exception/shutdown fallback; release() is the no-MPI-call escape.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/concepts.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/detail/window_engine.hpp"
#include "mp1/impl/fwd.hpp"
#include "mp1/impl/group.hpp"
#include "mp1/impl/info.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>


namespace mp1 {

  /// One segment returned by MPI_Win_shared_query.  base is the address at which the calling
  /// process can load/store the segment; size is in bytes and disp_unit is the target rank's
  /// disp_unit, not the calling process's.  A size of 0 is legal and means the segment is not
  /// directly accessible (or is empty); base is then implementation-defined (MPI_Alloc_mem(0)
  /// semantics) and must not be dereferenced.
  struct shared_segment {
    void* base{nullptr};
    MPI_Aint size{0};
    int disp_unit{0};
  };

  /// Payload of window::create (MPI_Win_create): expose caller-owned memory.  A
  /// designated-initializer aggregate; C++20 requires the designators in declaration order.
  /// size is in bytes, disp_unit must be positive, and size == 0 is legal (base may then be
  /// null).  The caller keeps ownership of base and must keep it alive until the window is
  /// freed; a positive size with a null base is rejected locally before MPI.
  struct window_create_options {
    void* base{nullptr};
    MPI_Aint size{0};
    int disp_unit{1};
    // The initializer is redundant for value-initialization, but -Wmissing-field-initializers
    // warns at designated-initializer call sites that omit .info when it is absent.
    // NOLINTNEXTLINE(readability-redundant-member-init): -Wmissing-field-initializers
    mp1::info info{};
  };

  /// Payload of window::allocate / window::allocate_shared (MPI_Win_allocate /
  /// MPI_Win_allocate_shared): MPI owns the segment.  A designated-initializer aggregate;
  /// C++20 requires the designators in declaration order.  size is in bytes, disp_unit must be
  /// positive, and size == 0 is legal.
  struct window_allocate_options {
    MPI_Aint size{0};
    int disp_unit{1};
    // The initializer is redundant for value-initialization, but -Wmissing-field-initializers
    // warns at designated-initializer call sites that omit .info when it is absent.
    // NOLINTNEXTLINE(readability-redundant-member-init): -Wmissing-field-initializers
    mp1::info info{};
  };

  namespace detail {
    void require_valid(const mpi_window& w, const context& ctx);
  }  // namespace detail

  /// Non-owning, copyable, borrowed view of an MPI_Win.
  ///
  /// A view holds a copied MPI handle, not a reference: it is valid only while the MPI object
  /// it borrows is live, so the owner must outlive every view.  free(), release(), the owner's
  /// destruction and MPI_Finalize (including an external one) invalidate every view of the
  /// object.  mp1 cannot detect use of an invalidated view (native_handle() is a documented raw
  /// escape), so that use is a precondition violation.  is_valid() only reports
  /// `handle != MPI_WIN_NULL`; it does not promise liveness.
  ///
  /// Constructing a view of a valid raw handle requires active MPI: it eagerly caches
  /// MPI_WIN_BASE/SIZE/DISP_UNIT/CREATE_FLAVOR/MODEL and the local rank and size of the window
  /// group, so every later accessor is MPI-free.  The cache is a snapshot and never refreshes.
  /// A valid handle constructed while MPI is inactive (before MPI_Init or after MPI_Finalize)
  /// throws std::runtime_error; there is no deferred view.  Default construction and
  /// mpi_window{MPI_WIN_NULL} are MPI-free and remain valid in every lifecycle state.  The
  /// destructor never calls MPI.
  ///
  /// A const view is a borrowed handle, not a const native object: operations that mutate the
  /// shared window through the handle (set_info() and the set_error_handler() free function) are
  /// const members or take a const view, because constness protects the view's own cached state
  /// and not the MPI object that every copy of the handle refers to.
  class mpi_window {
  public:
    /// MPI_WIN_NULL; no MPI call.
    mpi_window() noexcept = default;

    /// Borrow h and cache its metadata.  MPI_WIN_NULL is MPI-free; a non-null handle requires
    /// active MPI and throws std::runtime_error when MPI is inactive.
    explicit mpi_window(MPI_Win h) : handle_{h} {
      if (handle_ == MPI_WIN_NULL)
        return;
      const detail::window_metadata metadata = detail::query_window_metadata(handle_);
      base_ = metadata.base;
      byte_size_ = metadata.byte_size;
      disp_unit_ = metadata.disp_unit;
      create_flavor_ = metadata.create_flavor;
      memory_model_ = metadata.memory_model;
      rank_ = metadata.rank;
      group_size_ = metadata.group_size;
    }

    /// Named form of the borrowing constructor; see mpi_window(MPI_Win).
    [[nodiscard]] static mpi_window borrow(MPI_Win h) {
      return mpi_window{h};
    }

    mpi_window(const mpi_window&) noexcept = default;
    mpi_window& operator=(const mpi_window&) noexcept = default;
    mpi_window(mpi_window&&) noexcept = default;
    mpi_window& operator=(mpi_window&&) noexcept = default;
    ~mpi_window() noexcept = default;

    /// \return true if the handle is not MPI_WIN_NULL; no MPI call
    [[nodiscard]] bool is_valid() const noexcept {
      return handle_ != MPI_WIN_NULL;
    }

    /// \return non-owning native handle; safe after finalize
    [[nodiscard]] MPI_Win native_handle() const noexcept {
      return handle_;
    }

    /// \return cached MPI_WIN_BASE; for a dynamic window this is MPI_BOTTOM
    [[nodiscard]] void* base() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::base");
      return base_;
    }

    /// \return cached MPI_WIN_SIZE in bytes (0 for a dynamic window)
    [[nodiscard]] MPI_Aint byte_size() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::byte_size");
      return byte_size_;
    }

    /// \return cached local MPI_WIN_DISP_UNIT (1 for a dynamic window)
    [[nodiscard]] int disp_unit() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::disp_unit");
      return disp_unit_;
    }

    /// \return cached MPI_WIN_CREATE_FLAVOR (MPI_WIN_FLAVOR_*)
    [[nodiscard]] int create_flavor() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::create_flavor");
      return create_flavor_;
    }

    /// \return cached MPI_WIN_MODEL (MPI_WIN_UNIFIED or MPI_WIN_SEPARATE)
    [[nodiscard]] int memory_model() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::memory_model");
      return memory_model_;
    }

    /// \return cached local rank in the window group; no MPI call
    [[nodiscard]] int rank() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::rank");
      return rank_;
    }

    /// \return cached number of processes in the window group; no MPI call
    [[nodiscard]] int group_size() const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::group_size");
      return group_size_;
    }

    /// MPI_Win_get_group; returns a newly duplicated, owning group.
    [[nodiscard]] group get_group() const {
      detail::require_window_active(is_valid(), "mp1::mpi_window::get_group");
      MPI_Group native{MPI_GROUP_NULL};
      detail::check_mpi(MPI_Win_get_group(handle_, &native), "MPI_Win_get_group");
      return detail::group_factory::make(native);
    }

    /// MPI_Win_get_info; MPI allocates a fresh info table and ownership moves to the result.
    [[nodiscard]] info get_info() const {
      detail::require_window_active(is_valid(), "mp1::mpi_window::get_info");
      MPI_Info native{MPI_INFO_NULL};
      detail::check_mpi(MPI_Win_get_info(handle_, &native), "MPI_Win_get_info");
      return detail::info_access::adopt(native);
    }

    /// MPI_Win_set_info; collective over the window group and mutates the shared native window.
    /// The method is const on the view because a view is a borrowed handle: constness protects
    /// the view's own cached state, not the MPI object (the same rule as set_error_handler).
    /// A null info is rejected locally.
    void set_info(const info& i) const {
      detail::require_window_active(is_valid(), "mp1::mpi_window::set_info");
      if (i.native_handle() == MPI_INFO_NULL) [[unlikely]]
        throw_invalid("mp1::mpi_window::set_info", "null info");
      detail::check_mpi(MPI_Win_set_info(handle_, i.native_handle()), "MPI_Win_set_info");
    }

    /// MPI_Win_shared_query.
    ///
    /// MPI 4.1 allows this on CREATE, ALLOCATE and ALLOCATE_SHARED windows, rejecting only
    /// DYNAMIC; the returned disp_unit is the queried rank's, not the local one.  rank may be
    /// MPI_PROC_NULL (lowest rank with a non-zero segment).  A size of 0 is legal and means
    /// the segment is not directly accessible.  On an MPI 4.0 implementation this call fails
    /// for CREATE/ALLOCATE windows; mp1 requires MPI 4.1.
    [[nodiscard]] shared_segment shared_query(int rank) const {
      detail::require_window_valid(is_valid(), "mp1::mpi_window::shared_query");
      detail::require_mpi("mp1::mpi_window::shared_query");
      if (create_flavor_ == MPI_WIN_FLAVOR_DYNAMIC) [[unlikely]] {
        throw_invalid("mp1::mpi_window::shared_query", "not available for a dynamic window");
      }
      if (rank != MPI_PROC_NULL and (rank < 0 or rank >= group_size_)) [[unlikely]] {
        detail::fail_as<std::invalid_argument>({"mp1::mpi_window::shared_query"},
                                               std::source_location::current(),
                                               "rank {} is outside [0,{})", rank, group_size_);
      }
      MPI_Aint size{0};
      int unit{0};
      void* pointer{nullptr};
      detail::check_mpi(MPI_Win_shared_query(handle_, rank, &size, &unit, &pointer),
                        "MPI_Win_shared_query");
      return shared_segment{pointer, size, unit};
    }

    /// MPI_Win_attach; local, dynamic windows only.  The memory must stay alive until detach
    /// or MPI_Win_free.
    ///
    /// MPI 4.1 §12.2.4 (Window of Dynamically Attached Memory) defines size as a
    /// non-negative integer, so a zero-size attach is standard-conforming.  mp1 rejects it as
    /// a portability subset: Open MPI 6.0 cannot detach a zero-size region (MPI_Win_detach
    /// returns MPI_ERR_UNKNOWN) and MPICH cannot detach a null base, so the attach/detach pair
    /// the API would permit could not be closed on every supported implementation.  A null
    /// base is likewise rejected.  Detaching an unregistered base or attaching overlapping
    /// ranges is erroneous in MPI; mp1 does not keep a per-window attachment registry, so
    /// that misuse cannot be diagnosed locally.
    void attach(void* base, MPI_Aint size) {
      detail::require_window_dynamic(is_valid(), create_flavor_, "mp1::mpi_window::attach");
      if (size <= 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"mp1::mpi_window::attach"}, std::source_location::current(),
            "size {} is not positive; MPI 4.1 §12.2.4 permits a zero-size attach but it is not "
            "portably detachable (Open MPI 6.0 cannot detach it), so mp1 accepts only "
            "non-empty regions",
            static_cast<long long>(size));
      }
      if (base == nullptr) [[unlikely]] {
        throw_invalid("mp1::mpi_window::attach",
                      "null base; a null-base attach has no portable detach (MPI_Win_detach "
                      "requires the same base and MPICH rejects null)");
      }
      detail::check_mpi(MPI_Win_attach(handle_, base, size), "MPI_Win_attach");
    }

    /// MPI_Win_detach; local, dynamic windows only.  base must be a non-null address that was
    /// passed to a previous attach (MPI 4.1 §12.2.4: the arguments must match the previous
    /// MPI_Win_ATTACH).  mp1 cannot verify the recorded base without per-window bookkeeping;
    /// detaching an address that was never attached is erroneous in MPI.
    void detach(const void* base) {
      detail::require_window_dynamic(is_valid(), create_flavor_, "mp1::mpi_window::detach");
      if (base == nullptr) [[unlikely]] {
        throw_invalid("mp1::mpi_window::detach",
                      "null base; MPI 4.1 §12.2.4 requires the base passed to a previous "
                      "MPI_Win_attach, and a null-base attach is rejected by mp1");
      }
      detail::check_mpi(MPI_Win_detach(handle_, base), "MPI_Win_detach");
    }

  private:
    MPI_Win handle_{MPI_WIN_NULL};
    void* base_{nullptr};
    MPI_Aint byte_size_{0};
    int disp_unit_{0};
    int create_flavor_{0};
    int memory_model_{0};
    int rank_{0};
    int group_size_{0};
  };

  /// Unique owning MPI_Win.  Move-only.  It contains a private mpi_window view and exposes it by
  /// value through view()/an implicit conversion to mpi_window; it does not publicly derive from
  /// the view, so a mpi_window& cannot be bound to the owner and the owner's handle cannot be
  /// overwritten, sliced or deleted through a view.
  ///
  /// free() and the destructor call MPI_Win_free exactly once and are single-attempt:
  /// MPI_Win_free is collective over the window group and may block (MPI 4.1 §12.2.5 Window
  /// Destruction), and on an MPI error the handle is counted as abandoned and mpi_error is
  /// thrown.  release() is the no-MPI-call escape for shutdown or for raw-MPI takeover.
  ///
  /// Move construction transfers ownership without an MPI call.  Move assignment to a live
  /// target is a teardown path exactly like free()/the destructor: it calls MPI_Win_free, which
  /// is collective over the target window's group and may block, so every rank must reach the
  /// assignment in the same way as destruction.  It is noexcept and abandons the old handle on
  /// an MPI error.
  class window {
  public:
    /// Invalid, owning nothing; no MPI call.
    window() noexcept = default;

    window(window&& other) noexcept : view_{other.view_} {
      other.view_ = mpi_window{};
    }

    window& operator=(window&& other) noexcept {
      if (this != &other) {
        release_owner();
        view_ = other.view_;
        other.view_ = mpi_window{};
      }
      return *this;
    }

    window(const window&) = delete;
    window& operator=(const window&) = delete;

    /// MPI_Win_free iff valid and MPI is active; otherwise abandons and counts the handle.
    /// Collective and potentially blocking.  No MPI call after finalize.
    ~window() noexcept {
      release_owner();
    }

    /// \return the borrowed, non-owning view by value.  The result is a copied handle plus the
    /// metadata snapshot: it stays valid after this owner is moved, but only while the MPI
    /// window is live, so the (new) owner must outlive it.
    [[nodiscard]] mpi_window view() const noexcept {
      return view_;
    }

    /// Implicit borrow: pass an owner wherever a `mpi_window` or `const mpi_window&` is
    /// expected.  Returns a by-value copy of the borrowed view, not a reference into the movable
    /// owner, so it is unaffected by moving the owner.  The copy never transfers ownership.
    operator mpi_window() const noexcept {
      return view_;
    }

    /// \return true iff the owner currently holds a handle; not a liveness check for views
    [[nodiscard]] bool is_valid() const noexcept {
      return view_.is_valid();
    }

    /// \return non-owning native handle; safe after finalize
    [[nodiscard]] MPI_Win native_handle() const noexcept {
      return view_.native_handle();
    }

    /// \return cached MPI_WIN_BASE; for a dynamic window this is MPI_BOTTOM
    [[nodiscard]] void* base() const {
      return view_.base();
    }

    /// \return cached MPI_WIN_SIZE in bytes (0 for a dynamic window)
    [[nodiscard]] MPI_Aint byte_size() const {
      return view_.byte_size();
    }

    /// \return cached local MPI_WIN_DISP_UNIT (1 for a dynamic window)
    [[nodiscard]] int disp_unit() const {
      return view_.disp_unit();
    }

    /// \return cached MPI_WIN_CREATE_FLAVOR (MPI_WIN_FLAVOR_*)
    [[nodiscard]] int create_flavor() const {
      return view_.create_flavor();
    }

    /// \return cached MPI_WIN_MODEL (MPI_WIN_UNIFIED or MPI_WIN_SEPARATE)
    [[nodiscard]] int memory_model() const {
      return view_.memory_model();
    }

    /// \return cached local rank in the window group; no MPI call
    [[nodiscard]] int rank() const {
      return view_.rank();
    }

    /// \return cached number of processes in the window group; no MPI call
    [[nodiscard]] int group_size() const {
      return view_.group_size();
    }

    /// MPI_Win_get_group; returns a newly duplicated, owning group.
    [[nodiscard]] group get_group() const {
      return view_.get_group();
    }

    /// MPI_Win_get_info; MPI allocates a fresh info table and ownership moves to the result.
    [[nodiscard]] info get_info() const {
      return view_.get_info();
    }

    /// MPI_Win_set_info; collective over the window group and mutates the shared native window
    /// through the borrowed view.
    void set_info(const info& i) const {
      view_.set_info(i);
    }

    /// MPI_Win_shared_query.
    [[nodiscard]] shared_segment shared_query(int rank) const {
      return view_.shared_query(rank);
    }

    /// MPI_Win_attach; local, dynamic windows only.
    void attach(void* base, MPI_Aint size) {
      view_.attach(base, size);
    }

    /// MPI_Win_detach; local, dynamic windows only.
    void detach(const void* base) {
      view_.detach(base);
    }

    /// MPI_Win_create: expose user memory (`create(comm, {.base=..., .size=...,
    /// .disp_unit=..., .info=...})`).  The caller keeps ownership of the segment and must keep
    /// it alive until the window is freed.  Collective over comm; comm must be an
    /// intracommunicator.  size is in bytes; disp_unit must be positive; size == 0 is legal
    /// (base may then be null).
    [[nodiscard]] static window create(const mpi_communicator& comm,
                                       const window_create_options& options) {
      detail::require_window_factory_operand(comm, "mp1::window::create");
      detail::require_window_size(options.size, "mp1::window::create");
      detail::require_window_disp_unit(options.disp_unit, "mp1::window::create");
      if (options.size > 0 and options.base == nullptr) [[unlikely]]
        throw_invalid("mp1::window::create", "null base for positive size");
      MPI_Win result{MPI_WIN_NULL};
      detail::check_mpi(
          MPI_Win_create(options.base, options.size, options.disp_unit,
                         options.info.native_handle(), comm.native_handle(), &result),
          "MPI_Win_create");
      return take_result(result);
    }

    /// MPI_Win_allocate: MPI allocates and owns the bytes (`allocate(comm, {.size=...,
    /// .disp_unit=..., .info=...})`); MPI_Win_free releases them.  Collective.  size == 0 is
    /// legal.  The local segment is base(); use shared_query for other ranks.
    [[nodiscard]] static window allocate(const mpi_communicator& comm,
                                         const window_allocate_options& options) {
      detail::require_window_factory_operand(comm, "mp1::window::allocate");
      detail::require_window_size(options.size, "mp1::window::allocate");
      detail::require_window_disp_unit(options.disp_unit, "mp1::window::allocate");
      void* base{nullptr};
      MPI_Win result{MPI_WIN_NULL};
      detail::check_mpi(
          MPI_Win_allocate(options.size, options.disp_unit, options.info.native_handle(),
                           comm.native_handle(), &base, &result),
          "MPI_Win_allocate");
      return take_result(result);
    }

    /// MPI_Win_allocate_shared: MPI allocates shared memory (`allocate_shared(comm, {.size=...,
    /// .disp_unit=..., .info=...})`).  Every process in comm must be in the same shared-memory
    /// domain (use communicator::split_type(..., comm_type_shared) to build such a
    /// communicator).  Collective; size == 0 is legal.
    [[nodiscard]] static window allocate_shared(const mpi_communicator& comm,
                                                const window_allocate_options& options) {
      detail::require_window_factory_operand(comm, "mp1::window::allocate_shared");
      detail::require_window_size(options.size, "mp1::window::allocate_shared");
      detail::require_window_disp_unit(options.disp_unit, "mp1::window::allocate_shared");
      void* base{nullptr};
      MPI_Win result{MPI_WIN_NULL};
      detail::check_mpi(
          MPI_Win_allocate_shared(options.size, options.disp_unit, options.info.native_handle(),
                                  comm.native_handle(), &base, &result),
          "MPI_Win_allocate_shared");
      return take_result(result);
    }

    /// MPI_Win_create_dynamic: initially empty; attach()/detach() manage the memory.  For a
    /// dynamic window target_disp is the target address in bytes (base is MPI_BOTTOM,
    /// disp_unit is 1) and is not required to be non-negative.  Collective.
    [[nodiscard]] static window create_dynamic(const mpi_communicator& comm,
                                               const info& i = {}) {
      detail::require_window_factory_operand(comm, "mp1::window::create_dynamic");
      MPI_Win result{MPI_WIN_NULL};
      detail::check_mpi(
          MPI_Win_create_dynamic(i.native_handle(), comm.native_handle(), &result),
          "MPI_Win_create_dynamic");
      return take_result(result);
    }

    /// Take ownership of a raw window handle.  Requires active MPI and a non-null handle.
    /// Metadata is queried before ownership is transferred: if that query throws, the caller
    /// still owns the handle and no MPI_Win_free is called on it.
    [[nodiscard]] static window adopt(MPI_Win h) {
      detail::require_mpi("mp1::window::adopt");
      if (h == MPI_WIN_NULL) [[unlikely]]
        throw_invalid("mp1::window::adopt", "MPI_WIN_NULL");
      return window{h, adopt_handle_t{}};
    }

    /// MPI_Win_free; collective and possibly blocking (MPI 4.1 §12.2.5 Window Destruction).
    /// Idempotent: an invalid window is a no-op.  Throws std::runtime_error when MPI is not
    /// active.  All RMA requests and epochs on the window must be complete before this call.
    /// Single attempt: on an MPI error the handle is counted as abandoned and mpi_error is
    /// thrown; the object is terminal and the collective teardown is not retried.
    void free() {
      if (not is_valid())
        return;
      detail::require_mpi_checked("mp1::window::free");
      MPI_Win handle = native_handle();
      const int rc = MPI_Win_free(&handle);
      view_ = mpi_window{};
      if (rc != MPI_SUCCESS) [[unlikely]] {
        detail::note_abandoned_handle();
        detail::check_mpi(rc, "MPI_Win_free");
      }
    }

    /// Detach the raw MPI_Win and give up ownership without any MPI call.  The caller becomes
    /// responsible for MPI_Win_free (or for handing the window to raw MPI).  The object becomes
    /// invalid.
    [[nodiscard]] MPI_Win release() noexcept {
      MPI_Win handle = native_handle();
      view_ = mpi_window{};
      return handle;
    }

  private:
    struct adopt_handle_t {
      explicit adopt_handle_t() = default;
    };

    /// Take ownership of a raw handle produced by MPI and cache its metadata.  If caching
    /// throws, the window is not constructed and the caller keeps the handle.
    window(MPI_Win h, adopt_handle_t) : view_{h} {
    }

    [[nodiscard]] static window take_result(MPI_Win handle) {
      if (handle == MPI_WIN_NULL)
        return window{};
      try {
        return window{handle, adopt_handle_t{}};
      } catch (...) {
        // A rank-local MPI_Win_free here would deadlock the other ranks: the window was
        // created by a collective factory and the metadata failure is local.  Leave the
        // handle to process cleanup and count it as abandoned.
        detail::note_abandoned_handle();
        throw;
      }
    }

    /// Relinquish ownership without an MPI call.  Called by every path that stops owning the
    /// handle: free(), release(), move assignment and the destructor.  Unlike the communicator
    /// owner there is no process-global counter: live windows at finalize are the caller's MPI
    /// precondition.
    void release_owner() noexcept {
      if (not is_valid())
        return;
      MPI_Win handle = native_handle();
      view_ = mpi_window{};
      if (not detail::mpi_active()) [[unlikely]] {
        detail::note_abandoned_handle();
        return;
      }
      if (MPI_Win_free(&handle) != MPI_SUCCESS) [[unlikely]]
        detail::note_abandoned_handle();
    }

    mpi_window view_;
  };

  namespace detail {

    /// The one validity precondition used by every operation taking a window.
    inline void require_valid(const mpi_window& w, const context& ctx) {
      if (not w.is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid window");
    }

  }  // namespace detail

  /// MPI_Win_set_errhandler with a predefined handler (MPI 4.1 §9.3.2).  Non-collective,
  /// process-local; requires active MPI and a valid window.  Windows do not inherit the
  /// initial error handler (§9.3).  The window is logically mutated although the view is passed
  /// by const reference: a const view is a borrowed handle, and constness protects the view's
  /// cached state, not the shared native object (see mpi_window).
  inline void set_error_handler(const mpi_window& w, error_handler handler) {
    const context ctx{"mp1::set_error_handler"};
    detail::require_valid(w, ctx);
    detail::require_mpi(ctx);
    detail::check_mpi(
        MPI_Win_set_errhandler(w.native_handle(), detail::native_error_handler(ctx, handler)),
        ctx);
  }

  /// MPI_Win_get_errhandler plus predefined classification; the returned reference is freed
  /// with MPI_Errhandler_free before returning (MPI 4.1 §9.3, Table 11.1).
  [[nodiscard]] inline std::optional<error_handler> get_error_handler(const mpi_window& w) {
    detail::require_valid(w, "mp1::get_error_handler");
    detail::require_mpi("mp1::get_error_handler");
    MPI_Errhandler handler{MPI_ERRHANDLER_NULL};
    detail::check_mpi(MPI_Win_get_errhandler(w.native_handle(), &handler),
                      "MPI_Win_get_errhandler");
    const std::optional<error_handler> result = detail::classify_error_handler(handler);
    if (handler != MPI_ERRHANDLER_NULL)
      detail::check_mpi(MPI_Errhandler_free(&handler), "MPI_Errhandler_free");
    return result;
  }

  /// MPI_Win_get_group as a free function (member forwarder).
  [[nodiscard]] inline group get_group(const mpi_window& w) {
    return w.get_group();
  }

  /// MPI_Get_address.  Used by dynamic windows to compute a target address in the target
  /// process.  The object's lifetime is the caller's responsibility.
  template<class T>
  [[nodiscard]] MPI_Aint address_of(const T& object) {
    detail::require_mpi("mp1::address_of");
    MPI_Aint address{0};
    detail::check_mpi(MPI_Get_address(std::addressof(object), &address), "MPI_Get_address");
    return address;
  }

}  // namespace mp1

#endif
