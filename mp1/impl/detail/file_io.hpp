#if !(defined MP1_DETAIL_FILE_IO_HPP)

#define MP1_DETAIL_FILE_IO_HPP

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>


namespace mp1::detail {

  // ---------------------------------------------------------------------------------------
  // File state shared by the owning mp1::file and by every outstanding nonblocking request.
  //
  // The standards model has three state dimensions that a bool cannot represent:
  //   - the handle (open/null generation),
  //   - the mutable view (etype identity, MPI 4.1 section 14.3, type matching 14.6.6),
  //   - the split collective epoch (at most one active begin, matching end, same buffer and
  //     thread; MPI 4.1 section 14.4.5).
  // The state is reference counted so a request can outlive the file object without dangling
  // (14.4.5 requires the data buffer to live until the end/wait; the wrapper's own state must
  // live at least as long).
  // ---------------------------------------------------------------------------------------

  enum class split_phase : unsigned char { none, active, unknown };

  enum class split_kind : unsigned char {
    none,
    read_at_all,
    write_at_all,
    read_all,
    write_all,
    read_ordered,
    write_ordered
  };

  // Recorded view. `etype_token` is a non-owning comparison token (a predefined datatype
  // constant or the process-lifetime struct_builder<T> handle), never freed here and never
  // compared after MPI_Finalize. `known == false` means a raw/adopted view or a failed
  // set_view: typed access is then delegated to MPI's own type-signature rule.
  struct file_view_state {
    bool known{true};
    bool byte_etype{true};
    MPI_Datatype etype_token{MPI_BYTE};
  };

  // The address of a thread_local object is a stable per-thread identity. MPI 4.1 section
  // 14.4.5 requires a split begin/end pair to be called from the same thread, so the split
  // epoch records this token.
  [[nodiscard]] inline const void* current_thread_token() noexcept {
    thread_local const char token = 0;
    return &token;
  }

  struct file_split_state {
    split_phase phase{split_phase::none};
    split_kind kind{split_kind::none};
    const void* buffer{nullptr};
    const void* thread{nullptr};
    // A split begin that borrows a region duplicates its layout so the caller's layout may die
    // before the matching end (the layout must outlive the epoch).  The duplicate's handle is
    // the one passed to the begin; MPI_File_*_end takes no datatype, so the duplicate must stay
    // alive in the epoch until the end returns.
    // Released by reset() after the end / in the destructor's finish path.
    std::shared_ptr<void> layout_keepalive;

    void reset() noexcept {
      phase = split_phase::none;
      kind = split_kind::none;
      buffer = nullptr;
      thread = nullptr;
      layout_keepalive.reset();
    }
  };

  struct file_state {
    MPI_File handle{MPI_FILE_NULL};
    // Access mode recorded at open/adopt. MPI 4.1 section 14.2.1 requires the same amode on
    // every rank, so amode-derived local validation cannot reject a legal program on a subset
    // of ranks.
    int amode{0};
    file_view_state view{};
    file_split_state split{};
    // One lease per wrapper-owned nonblocking request. Incremented before the MPI call that
    // creates the request and decremented when the request object is reaped or destroyed.
    std::atomic<count_t> pending_requests{0};

    [[nodiscard]] bool is_open() const noexcept {
      return handle != MPI_FILE_NULL;
    }
  };

  // One checked-call helper for every MPI-IO return code. This replaces the per-method
  // `if (err != MPI_SUCCESS) throw ...` block; no file method writes its own error branch.
  inline void check_file(int rc, const context& ctx) {
    check_mpi(rc, ctx);
  }

  // MPI 4.1 section 14.2.1: exactly one access bit; CREATE/EXCL are erroneous with RDONLY;
  // SEQUENTIAL is erroneous with RDWR. amode is uniform, so a local throw is collective-safe.
  inline void validate_file_amode(int amode, const context& ctx) {
    const int access = amode & (MPI_MODE_RDONLY | MPI_MODE_RDWR | MPI_MODE_WRONLY);
    if (access == 0) {
      throw_invalid(ctx.operation,
                    "exactly one of read_only, read_write or write_only is required");
    }
    if ((access & (access - 1)) != 0) {
      throw_invalid(ctx.operation,
                    "exactly one of read_only, read_write or write_only may be set");
    }
    if ((amode & MPI_MODE_RDONLY) != 0 and (amode & (MPI_MODE_CREATE | MPI_MODE_EXCL)) != 0) {
      throw_invalid(ctx.operation, "create/no_replace are erroneous with read_only");
    }
    if ((amode & MPI_MODE_SEQUENTIAL) != 0 and (amode & MPI_MODE_RDWR) != 0) {
      throw_invalid(ctx.operation, "sequential is erroneous with read_write");
    }
  }

  [[nodiscard]] inline bool amode_reads(int amode) noexcept {
    return (amode & (MPI_MODE_RDONLY | MPI_MODE_RDWR)) != 0;
  }

  [[nodiscard]] inline bool amode_writes(int amode) noexcept {
    return (amode & (MPI_MODE_WRONLY | MPI_MODE_RDWR)) != 0;
  }

  [[nodiscard]] inline bool amode_sequential(int amode) noexcept {
    return (amode & MPI_MODE_SEQUENTIAL) != 0;
  }

  // The one validity precondition shared by every file operation: MPI is active (cached fast
  // path), the object holds a state, and the handle is open. Returns the handle by value; the
  // state may be updated by the caller and the handle must not be cached across an MPI call.
  [[nodiscard]] inline MPI_File require_file(const std::shared_ptr<file_state>& state,
                                             const context& ctx) {
    require_mpi(ctx);
    if (not state or state->handle == MPI_FILE_NULL)
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "file is not open");
    return state->handle;
  }

  // MPI 4.1 section 14.4.5: no collective I/O operation may overlap a split epoch. The
  // collective controls whose own clauses require outstanding split operations to be complete
  // (sync §14.6.1, set_view §14.3, set_size §14.2.4, preallocate §14.2.5) and close use this
  // guard too. set_info/set_atomicity/seek_shared carry no such completion requirement and use
  // require_collective_file_during_split below.
  inline void require_no_split(const file_state& state, const context& ctx) {
    switch (state.split.phase) {
      case split_phase::active:
        detail::fail_with<std::runtime_error>(
            ctx.operation, std::source_location::current(),
            "collective file operation during a split collective");
      case split_phase::unknown:
        detail::fail_with<std::runtime_error>(
            ctx.operation, std::source_location::current(),
            "split collective state is unknown after a failed begin/end");
      case split_phase::none:
        return;
    }
    detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                          "invalid split collective state");
  }

  // MPI 4.1 sections 14.2.4 (set_size), 14.2.5 (preallocate) and 14.3 (set_view) make these
  // operations erroneous while a nonblocking request or split collective on the handle is
  // outstanding. The split epoch has its own guard; this one covers the wrapper-owned
  // nonblocking requests, whose count is already tracked locally, so reject before any MPI
  // call. close()/sync() use the same policy and diagnostic.
  //
  // set_atomicity is deliberately not guarded: MPI 4.1 section 14.6.1 permits it with
  // outstanding nonblocking operations.
  inline void require_no_pending_requests(const file_state& state, const context& ctx) {
    const count_t pending = state.pending_requests.load(std::memory_order_relaxed);
    if (pending != 0) {
      detail::fail_as<std::runtime_error>(ctx, std::source_location::current(),
                                          "{} outstanding nonblocking file request(s); wait, "
                                          "test or release them before proceeding",
                                          pending);
    }
  }

  // MPI 4.1 sections 14.4.2/14.4.3/14.6.2: explicit-offset and individual-pointer data access
  // are erroneous on a sequential file; shared-pointer access is the only permitted family.
  inline void require_sequential_ok(const file_state& state, bool sequential_ok,
                                    const context& ctx) {
    if (not sequential_ok and amode_sequential(state.amode)) {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "operation is not permitted on a file opened with access_mode::sequential");
    }
  }

  [[nodiscard]] inline MPI_File require_collective_file(
      const std::shared_ptr<file_state>& state, bool sequential_ok, const context& ctx) {
    MPI_File handle = require_file(state, ctx);
    require_no_split(*state, ctx);
    require_sequential_ok(*state, sequential_ok, ctx);
    return handle;
  }

  // MPI 4.1 §14.4.5 forbids only "collective I/O operations" between a split begin and its
  // end.  MPI_File_set_info (§14.2.8), MPI_File_set_atomicity (§14.6.1) and
  // MPI_File_seek_shared (§14.4.4) are collective calls with no "split/outstanding operation
  // must be complete" clause, so a conforming program may call them during a split.  This
  // helper keeps the file-open and access-mode checks and deliberately skips require_no_split.
  [[nodiscard]] inline MPI_File require_collective_file_during_split(
      const std::shared_ptr<file_state>& state, bool sequential_ok, const context& ctx) {
    MPI_File handle = require_file(state, ctx);
    require_sequential_ok(*state, sequential_ok, ctx);
    return handle;
  }

  [[nodiscard]] inline MPI_File require_noncollective_file(
      const std::shared_ptr<file_state>& state, bool sequential_ok, const context& ctx) {
    MPI_File handle = require_file(state, ctx);
    require_sequential_ok(*state, sequential_ok, ctx);
    return handle;
  }

  [[nodiscard]] inline MPI_File require_data_access(const std::shared_ptr<file_state>& state,
                                                    bool collective, bool writes,
                                                    bool sequential_ok, const context& ctx) {
    MPI_File handle = require_file(state, ctx);
    if (writes ? not amode_writes(state->amode) : not amode_reads(state->amode)) {
      detail::fail_with<std::runtime_error>(
          ctx, std::source_location::current(),
          writes ? "the file access mode does not permit writing"
                 : "the file access mode does not permit reading");
    }
    require_sequential_ok(*state, sequential_ok, ctx);
    if (collective)
      require_no_split(*state, ctx);
    return handle;
  }

  // MPI 4.1 section 14.6.6: the buffer datatype's signature must match a whole number of
  // contiguous etype copies; MPI_BYTE matches any datatype. The wrapper can enforce the common
  // exact-token case without another MPI call, and deliberately does not accept a mixed
  // signature (for example `(int,int)` with an `int` etype): use a matching view or the byte
  // view. Unknown view (raw/adopted or a failed set_view) is delegated to MPI.
  template<class T>
  inline void check_etype(const file_state& state, const context& ctx) {
    if (not state.view.known or state.view.byte_etype)
      return;
    if (state.view.etype_token == native_datatype<T>())
      return;
    throw_invalid(ctx.operation,
                  "buffer element type does not match the file view etype; use a matching "
                  "set_view, the MPI_BYTE view, or native_handle()");
  }

  // Blocking collective/noncollective control call with no status (set_view/set_info/sync/
  // resize/preallocate/set_atomicity/seek_shared/seek).
  template<class Call>
  inline void file_call(const std::shared_ptr<file_state>& state, bool collective,
                        bool sequential_ok, const context& ctx, Call&& call) {
    MPI_File handle = collective ? require_collective_file(state, sequential_ok, ctx)
                                 : require_noncollective_file(state, sequential_ok, ctx);
    check_file(std::forward<Call>(call)(handle), ctx);
  }

  // Same call wrapper for the collective controls that are legal during a split epoch.
  template<class Call>
  inline void file_call_during_split(const std::shared_ptr<file_state>& state,
                                     bool sequential_ok, const context& ctx, Call&& call) {
    MPI_File handle = require_collective_file_during_split(state, sequential_ok, ctx);
    check_file(std::forward<Call>(call)(handle), ctx);
  }

  template<class T>
  inline void require_file_data_type() noexcept {
    // The file family uses the same datatype_traits gate as p2p/collective, not
    // is_trivially_copyable: std::pair and reflected aggregates are valid MPI datatypes.
    static_assert(has_datatype_v<T>,
                  "mp1::file: T has no MPI datatype; specialize datatype_traits<T> or use "
                  "MP1_REFLECTION");
  }

  // One lease per nonblocking file request. The payload owns a reference to the shared state,
  // so the state (and its pending count) survives a request that outlives the file object.
  // complete() is intentionally a no-op: the file kind is wait-only and carries no data
  // payload; destroying the payload is the decrement, and an abandoned payload is leaked
  // (never destroyed), keeping close() conservative.
  class file_request_lease final : public request_payload {
  public:
    explicit file_request_lease(std::shared_ptr<file_state> state,
                                std::shared_ptr<void> layout_keepalive = {})
        : state_{std::move(state)}, layout_keepalive_{std::move(layout_keepalive)} {
      state_->pending_requests.fetch_add(1, std::memory_order_relaxed);
    }

    file_request_lease(const file_request_lease&) = delete;
    file_request_lease& operator=(const file_request_lease&) = delete;

    ~file_request_lease() override {
      state_->pending_requests.fetch_sub(1, std::memory_order_relaxed);
    }

    void complete(const status&) override {
    }

  private:
    std::shared_ptr<file_state> state_;
    // Nonblocking region accesses own a duplicate of the per-call layout; the request outlives
    // the caller's region object.  Blocking calls leave this empty.
    std::shared_ptr<void> layout_keepalive_;
  };

  // MPI 4.1 section 14.4.5: begin is collective, incomplete (may block), and must record the
  // operation kind, the exact buffer and the calling thread so the matching end can be
  // validated locally. A failed begin poisons the epoch: MPI gives no portable guarantee that
  // no operation started, so close() must not be forwarded afterwards.
  template<class T, class Call>
  inline void file_begin(const std::shared_ptr<file_state>& state, split_kind kind,
                         const void* buffer, bool writes, bool sequential_ok,
                         const context& ctx, std::shared_ptr<void> keepalive, Call&& call) {
    require_file_data_type<T>();
    // The begin itself is the operation that requires phase == none; do not use the generic
    // collective guard, which would report "collective during a split" instead of "another
    // split is already active".
    MPI_File handle = require_data_access(state, false, writes, sequential_ok, ctx);
    if (state->split.phase == split_phase::active) {
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "another split collective is already active");
    }
    if (state->split.phase == split_phase::unknown) {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "split collective state is unknown after a failed begin/end");
    }
    check_etype<T>(*state, ctx);
    const int rc = std::forward<Call>(call)(handle);
    if (rc != MPI_SUCCESS) {
      state->split.reset();
      state->split.phase = split_phase::unknown;
      check_file(rc, ctx);
    }
    state->split.phase = split_phase::active;
    state->split.kind = kind;
    state->split.buffer = buffer;
    state->split.thread = current_thread_token();
    state->split.layout_keepalive = std::move(keepalive);
  }

  // A local end mismatch never reaches MPI and leaves the active epoch untouched, so a
  // caller can recover with the correct end. Only an MPI error poisons the epoch.
  //
  // Validate the end against the recorded epoch (MPI 4.1 section 14.4.5: one unmatched begin
  // for the same operation; the same buffer in begin and end; the same thread in a
  // multithreaded implementation). Throws std::runtime_error before any MPI call on mismatch.
  inline void require_split_end(const std::shared_ptr<file_state>& state, split_kind kind,
                                const void* buffer, const context& ctx) {
    if (not state or state->handle == MPI_FILE_NULL)
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "file is not open");
    switch (state->split.phase) {
      case split_phase::none:
        detail::fail_with<std::runtime_error>(
            ctx.operation, std::source_location::current(),
            "no split collective is active (end without a matching begin, or double end)");
      case split_phase::unknown:
        detail::fail_with<std::runtime_error>(
            ctx.operation, std::source_location::current(),
            "split collective state is unknown after a failed begin/end");
      case split_phase::active:
        break;
    }
    if (state->split.kind != kind) {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "split collective end does not match the begun operation");
    }
    if (state->split.buffer != buffer) {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "split collective end buffer differs from the begin buffer");
    }
    if (not(state->split.thread == current_thread_token())) {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "split collective begin and end must be called from the same thread");
    }
  }

  template<class EndCall>
  [[nodiscard]] inline status file_end(const std::shared_ptr<file_state>& state,
                                       split_kind kind, const void* buffer, const context& ctx,
                                       EndCall&& call) {
    require_mpi_checked(ctx);
    require_split_end(state, kind, buffer, ctx);
    MPI_Status native{};
    const MPI_File handle = state->handle;
    const int rc = std::forward<EndCall>(call)(handle, &native);
    if (rc != MPI_SUCCESS) {
      state->split.reset();
      state->split.phase = split_phase::unknown;
      check_file(rc, ctx);
    }
    state->split.reset();
    return status_access::make_file(native, false);
  }

  // Destructor-only split completion: no throw, no state mutation beyond the epoch. The
  // recorded buffer must still be alive; that is the documented abnormal-path obligation.
  [[nodiscard]] inline int finish_split_noexcept(MPI_File handle, split_kind kind,
                                                 const void* buffer) noexcept {
    MPI_Status native{};
    // Read ends take a mutable void*; write ends take const void*. The wrapper stored the
    // exact pointer the begin received, so a write end can pass it through unchanged.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI read-end C API
    void* mutable_buffer = const_cast<void*>(buffer);
    switch (kind) {
      case split_kind::read_at_all:
        return MPI_File_read_at_all_end(handle, mutable_buffer, &native);
      case split_kind::write_at_all:
        return MPI_File_write_at_all_end(handle, buffer, &native);
      case split_kind::read_all:
        return MPI_File_read_all_end(handle, mutable_buffer, &native);
      case split_kind::write_all:
        return MPI_File_write_all_end(handle, buffer, &native);
      case split_kind::read_ordered:
        return MPI_File_read_ordered_end(handle, mutable_buffer, &native);
      case split_kind::write_ordered:
        return MPI_File_write_ordered_end(handle, buffer, &native);
      case split_kind::none:
        return MPI_ERR_UNKNOWN;
    }
    return MPI_ERR_UNKNOWN;
  }

  // Validate the layout and the caller buffer before the MPI call. A null pointer is legal
  // only for a zero-byte layout (MPI 4.1 section 14.6.7 leaves the buffer semantics to the
  // access call; a zero-size layout accesses no bytes).
  template<class T>
  inline void require_file_layout(const layout<T>& filetype, const T* data,
                                  const context& ctx) {
    require_file_data_type<T>();
    if (not filetype.is_valid())
      throw_invalid(ctx.operation, "invalid layout");
    if (data == nullptr and filetype.byte_size() != 0) {
      throw_invalid(ctx.operation, "null data pointer for a non-empty layout");
    }
  }

  // MPI 4.1 §5.1.11: a datatype may specify overlapping entries, but using it in a
  // communication "in association with a buffer updated by the operation is erroneous".
  // A region is the memory datatype on read, so a construction-proven overlap is rejected
  // before MPI.  On write the layout describes a read-only buffer, so MPI does not require
  // overlap-free entries and the wrapper does not inspect it.
  template<class T>
  inline void require_file_read_layout(const layout<T>& filetype, const T* data,
                                       const context& ctx) {
    require_file_layout(filetype, data, ctx);
    if (filetype.has_overlapping_entries()) {
      throw_invalid(
          ctx.operation,
          "the per-call read layout contains overlapping typemap regions; MPI 4.1 §5.1.11 "
          "forbids using such a datatype with a buffer updated by the operation");
    }
  }

  // MPI 4.1 §14.3.1: a derived file-view etype may only have typemap displacements that are
  // nonnegative and monotonically nondecreasing. Only the construction-proven violation is
  // rejected; a custom derived handle whose map order is not recorded stays unknown and is
  // left to MPI's own validation.
  template<class T>
  inline void require_file_view_etype(const context& ctx) {
    const datatype_info& info = datatype_info_of<T>();
    if (info.order != type_map_order::non_monotone)
      return;
    detail::fail_as<std::invalid_argument>(
        ctx, std::source_location::current(),
        "file view etype native_datatype<{}>() has a descending or negative typemap "
        "displacement; MPI 4.1 §14.3.1 requires all etype displacements to be nonnegative and "
        "monotonically nondecreasing",
        datatype_type_name<T>());
  }

  // MPI 4.1 §14.3 filetype rules that the wrapper can prove locally:
  //  * every typemap displacement is nonnegative and monotonically nondecreasing in typemap
  //    order (the same rule as the etype), and
  //  * on a file opened for writing, neither etype nor filetype may contain overlapping
  //    regions.
  // The exact type map is decoded (the same decoder the collective disjointness checks use);
  // its output is reverse map order, so it is reversed before the monotonicity walk.  An
  // unmodelled constructor leaves the map unverified and delegates those rules to MPI.  The
  // remaining §14.3 rule — the extent of any hole must be a multiple of the etype's extent —
  // is not decidable from basic-entry intervals (an etype's internal holes are not filetype
  // holes), so it stays a caller obligation; the public layout factories produce
  // extent-aligned filetypes.
  template<class T>
  inline void require_file_view_filetype(const file_state& state, const layout<T>& filetype,
                                         const context& ctx) {
    // The true lower bound is the minimum typemap address: a negative value proves a negative
    // displacement even when the map cannot be decoded.
    if (filetype.true_byte_lower_bound() < 0) {
      throw_invalid(ctx.operation,
                    "file view filetype has a negative typemap displacement; MPI 4.1 §14.3 "
                    "requires all filetype displacements to be nonnegative and monotonically "
                    "nondecreasing");
    }
    std::vector<type_map_interval> intervals;
    if (decode_type_map(filetype.native_handle(), intervals, ctx)) {
      std::reverse(intervals.begin(), intervals.end());
      MPI_Aint previous_begin = 0;
      bool have_previous = false;
      for (const type_map_interval& interval : intervals) {
        if (interval.begin < 0) {
          throw_invalid(ctx.operation,
                        "file view filetype has a negative typemap displacement; MPI 4.1 "
                        "§14.3 requires all filetype displacements to be nonnegative and "
                        "monotonically nondecreasing");
        }
        if (have_previous and interval.begin < previous_begin) {
          throw_invalid(ctx.operation,
                        "file view filetype has a descending typemap displacement; MPI 4.1 "
                        "§14.3 requires all filetype displacements to be nonnegative and "
                        "monotonically nondecreasing");
        }
        previous_begin = interval.begin;
        have_previous = true;
      }
    }
    if (not amode_writes(state.amode) or not filetype.has_overlapping_entries())
      return;
    throw_invalid(ctx.operation,
                  "file view filetype contains overlapping typemap regions; MPI 4.1 §14.3.1 "
                  "forbids an overlapping etype or filetype on a file opened for writing; use "
                  "a disjoint filetype or open the file read-only");
  }

  // ---------------------------------------------------------------------------------------
  // Dense (pointer + count) data access helpers.
  //
  // MPI file data access is a (buffer, count, datatype) triple. The dense spelling passes the
  // caller's element count straight through as the exact MPI_Count and uses the predefined
  // native_datatype<T>() as the filetype, so no layout<T> is constructed, committed or freed.
  //
  // A null buffer is legal only when it addresses no data (count == 0). In that case the
  // stable dummy address used by the p2p/collective paths is substituted, so MPI never
  // receives a null data pointer; a null with a positive count is rejected before any MPI
  // call. Split begins record that substituted address and the split ends (which receive the
  // buffer but no count) apply the same substitution, so a zero-count null begin/end pair
  // passes the existing buffer identity check.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline count_t file_dense_size(std::size_t size, const context& ctx) {
    if (size > static_cast<std::size_t>(std::numeric_limits<count_t>::max())) {
      detail::fail_with<std::overflow_error>(
          ctx.operation, std::source_location::current(),
          "container size does not fit the element count type");
    }
    return static_cast<count_t>(size);
  }

  // The stable non-null address for a zero-count/empty access. The const and mutable
  // overloads preserve constness so a read buffer stays mutable and a write buffer stays
  // const.
  template<class T>
  [[nodiscard]] inline T* file_nonnull_buffer(T* data) noexcept {
    return data != nullptr ? data : static_cast<T*>(dummy_buffer());
  }

  template<class T>
  [[nodiscard]] inline const T* file_nonnull_buffer(const T* data) noexcept {
    return data != nullptr ? data : static_cast<const T*>(dummy_buffer());
  }

  template<class T>
  [[nodiscard]] inline T* file_dense_recv_buffer(T* data, MPI_Count count, const context& ctx) {
    if (data == nullptr and count != 0) {
      throw_invalid(ctx.operation, "null data pointer with a positive count");
    }
    return file_nonnull_buffer(data);
  }

  template<class T>
  [[nodiscard]] inline const T* file_dense_send_buffer(const T* data, MPI_Count count,
                                                       const context& ctx) {
    if (data == nullptr and count != 0) {
      throw_invalid(ctx.operation, "null data pointer with a positive count");
    }
    return file_nonnull_buffer(data);
  }

  // ---------------------------------------------------------------------------------------
  // Collapsed data-access boundary.
  //
  // Every public data-access family has exactly three forms: one value, one borrowed buffer
  // and one region.  These helpers turn each form into the one MPI (buffer, count, datatype)
  // triple; count is an MPI_Count of datatype applications, never bytes.  A range uses
  // count = size and native_datatype<T>(); a region uses count = 1 and the layout handle
  // A nonblocking/split region duplicates its layout into `layout_keepalive` and passes the
  // duplicate's handle to MPI, because the request/epoch outlives the caller's region object
  // and MPI then holds the duplicate for exactly as long as the keepalive lives.
  // ---------------------------------------------------------------------------------------

  template<class Pointer>
  struct file_transfer {
    Pointer data{nullptr};
    MPI_Count count{0};
    MPI_Datatype datatype{MPI_DATATYPE_NULL};
    std::shared_ptr<void> layout_keepalive;
  };

  using file_read_transfer = file_transfer<void*>;
  using file_send_transfer = file_transfer<const void*>;

  // A single value: one application of native_datatype<T>().
  template<class T>
  [[nodiscard]] inline file_read_transfer file_read_value(T& value) noexcept {
    return {static_cast<void*>(std::addressof(value)),
            MPI_Count{1},
            native_datatype<std::remove_cv_t<T>>(),
            {}};
  }

  template<class T>
  [[nodiscard]] inline file_send_transfer file_send_value(const T& value) noexcept {
    return {static_cast<const void*>(std::addressof(value)),
            MPI_Count{1},
            native_datatype<std::remove_cv_t<T>>(),
            {}};
  }

  // A contiguous borrowed range: count = element count, datatype = native element datatype.
  // The null + zero rules are the dense-access rules: a null data pointer is legal only when
  // the count is zero and is then replaced by the stable dummy address.
  template<class R>
  [[nodiscard]] inline file_read_transfer file_read_range(R&& buffer, const context& ctx) {
    using element_type = std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>;
    const MPI_Count count =
        checked_mpi_count(file_dense_size(std::ranges::size(buffer), ctx), ctx);
    return {file_dense_recv_buffer(std::ranges::data(buffer), count, ctx),
            count,
            native_datatype<element_type>(),
            {}};
  }

  template<class R>
  [[nodiscard]] inline file_send_transfer file_send_range(const R& buffer, const context& ctx) {
    using element_type = std::remove_cv_t<std::ranges::range_value_t<R>>;
    const MPI_Count count =
        checked_mpi_count(file_dense_size(std::ranges::size(buffer), ctx), ctx);
    return {file_dense_send_buffer(std::ranges::data(buffer), count, ctx),
            count,
            native_datatype<element_type>(),
            {}};
  }

  // One occurrence of a layout: count = 1, datatype = layout handle.  `own_layout` is true
  // for nonblocking and split-begin forms and false for the blocking forms, which only borrow
  // the layout for the duration of the call.  When the layout is owned, the duplicate (not the
  // caller's handle) is the datatype passed to MPI, so the keepalive guards the handle MPI
  // actually holds; the caller's layout may then die after initiation.
  template<class T>
  [[nodiscard]] inline file_read_transfer file_read_region(region<T> buffer, bool own_layout,
                                                           const context& ctx) {
    using element_type = std::remove_cv_t<T>;
    const layout<element_type>& map = buffer.map();
    require_file_read_layout(map, buffer.data(), ctx);
    std::shared_ptr<void> keepalive;
    MPI_Datatype datatype = map.native_handle();
    if (own_layout) {
      auto owned = std::make_shared<layout<element_type>>(map);
      datatype = owned->native_handle();
      keepalive = std::move(owned);
    }
    return {static_cast<void*>(file_nonnull_buffer(buffer.data())), MPI_Count{1}, datatype,
            std::move(keepalive)};
  }

  template<class T>
  [[nodiscard]] inline file_send_transfer file_send_region(region<T> buffer, bool own_layout,
                                                           const context& ctx) {
    using element_type = std::remove_cv_t<T>;
    const layout<element_type>& map = buffer.map();
    require_file_layout(map, buffer.data(), ctx);
    std::shared_ptr<void> keepalive;
    MPI_Datatype datatype = map.native_handle();
    if (own_layout) {
      auto owned = std::make_shared<layout<element_type>>(map);
      datatype = owned->native_handle();
      keepalive = std::move(owned);
    }
    return {static_cast<const void*>(file_nonnull_buffer(buffer.data())), MPI_Count{1},
            datatype, std::move(keepalive)};
  }

  // A split end takes no count/datatype: only the exact buffer identity recorded by the
  // matching begin matters (MPI 4.1 §14.4.5).  Null is normalized to the same stable dummy
  // address the matching begin used for an empty access.
  template<class R>
  [[nodiscard]] inline void* file_read_end_buffer(R&& buffer) noexcept {
    return file_nonnull_buffer(std::ranges::data(buffer));
  }

  template<class R>
  [[nodiscard]] inline const void* file_send_end_buffer(const R& buffer) noexcept {
    return file_nonnull_buffer(std::ranges::data(buffer));
  }

  // Blocking read/write of one (buffer, count, datatype) triple.
  template<class T, class Call>
  [[nodiscard]] inline status file_read_status(const std::shared_ptr<file_state>& state,
                                               bool collective, bool sequential_ok,
                                               const context& ctx,
                                               const file_read_transfer& transfer,
                                               Call&& call) {
    require_file_data_type<T>();
    MPI_File handle = require_data_access(state, collective, false, sequential_ok, ctx);
    check_etype<T>(*state, ctx);
    MPI_Status native{};
    check_file(std::forward<Call>(call)(handle, transfer.data, transfer.count,
                                        transfer.datatype, &native),
               ctx);
    return status_access::make_file(native, false);
  }

  template<class T, class Call>
  [[nodiscard]] inline status file_write_status(const std::shared_ptr<file_state>& state,
                                                bool collective, bool sequential_ok,
                                                const context& ctx,
                                                const file_send_transfer& transfer,
                                                Call&& call) {
    require_file_data_type<T>();
    MPI_File handle = require_data_access(state, collective, true, sequential_ok, ctx);
    check_etype<T>(*state, ctx);
    MPI_Status native{};
    check_file(std::forward<Call>(call)(handle, transfer.data, transfer.count,
                                        transfer.datatype, &native),
               ctx);
    return status_access::make_file(native, false);
  }

  // Nonblocking read/write; the lease owns the optional layout duplicate.
  template<class T, class Call>
  [[nodiscard]] inline request file_read_request(const std::shared_ptr<file_state>& state,
                                                 bool collective, bool sequential_ok,
                                                 const context& ctx,
                                                 file_read_transfer transfer, Call&& call) {
    require_file_data_type<T>();
    MPI_File handle = require_data_access(state, collective, false, sequential_ok, ctx);
    check_etype<T>(*state, ctx);
    auto lease =
        std::make_unique<file_request_lease>(state, std::move(transfer.layout_keepalive));
    MPI_Request native{MPI_REQUEST_NULL};
    check_file(std::forward<Call>(call)(handle, transfer.data, transfer.count,
                                        transfer.datatype, &native),
               ctx);
    request result = request::from_file(native);
    attach_payload(result, std::move(lease));
    return result;
  }

  template<class T, class Call>
  [[nodiscard]] inline request file_write_request(const std::shared_ptr<file_state>& state,
                                                  bool collective, bool sequential_ok,
                                                  const context& ctx,
                                                  file_send_transfer transfer, Call&& call) {
    require_file_data_type<T>();
    MPI_File handle = require_data_access(state, collective, true, sequential_ok, ctx);
    check_etype<T>(*state, ctx);
    auto lease =
        std::make_unique<file_request_lease>(state, std::move(transfer.layout_keepalive));
    MPI_Request native{MPI_REQUEST_NULL};
    check_file(std::forward<Call>(call)(handle, transfer.data, transfer.count,
                                        transfer.datatype, &native),
               ctx);
    request result = request::from_file(native);
    attach_payload(result, std::move(lease));
    return result;
  }

  // ---------------------------------------------------------------------------------------
  // Control-call validation moved out of the class (the class surface stays at 129
  // declarations).
  // ---------------------------------------------------------------------------------------

  inline void require_file_datarep(const char* datarep, const context& ctx) {
    if (datarep == nullptr)
      throw_invalid(ctx.operation, "null data representation");
  }

  // MPI 4.1 §14.3: the displacement is an absolute byte count from the start of the file; it
  // is passed to MPI_File_set_view unchanged (never scaled by the etype extent).  A sequential
  // file is the one exception and requires MPI_DISPLACEMENT_CURRENT.
  inline void require_file_displacement(const file_state& state, MPI_Offset disp,
                                        const context& ctx) {
    const bool sequential = amode_sequential(state.amode);
    if (sequential and disp != MPI_DISPLACEMENT_CURRENT) {
      throw_invalid(ctx.operation,
                    "a sequential file requires MPI_DISPLACEMENT_CURRENT as the displacement");
    }
    if (not sequential and disp == MPI_DISPLACEMENT_CURRENT) {
      throw_invalid(ctx.operation, "MPI_DISPLACEMENT_CURRENT requires access_mode::sequential");
    }
    // MPI_DISPLACEMENT_CURRENT is the one negative displacement (a sentinel, not a byte
    // position) and is legal only on a sequential file, handled above.
    if (not sequential and disp < 0) {
      throw_invalid(ctx.operation,
                    "the view displacement must be a nonnegative absolute byte offset");
    }
  }

  // MPI 4.1 §14.4.1: an explicit file offset is a nonnegative etype count relative to the view.
  inline void require_file_offset(MPI_Offset offset, const context& ctx) {
    if (offset < 0) {
      throw_invalid(ctx.operation, "the file offset must be a nonnegative etype count");
    }
  }

  inline void abandon_file_handle(std::shared_ptr<file_state>& state) noexcept {
    state->handle = MPI_FILE_NULL;
    state->split.reset();
    note_abandoned_handle();
    state.reset();
  }


  // The destructor/move-assignment teardown path, moved out of the class so the class keeps
  // exactly the 33 control declarations.  No throwing close(), no MPI after finalize.
  inline void close_file_noexcept(std::shared_ptr<file_state>& state) noexcept {
    if (not state)
      return;
    MPI_File handle = state->handle;
    if (handle == MPI_FILE_NULL) {
      state.reset();
      return;
    }
    if (not mpi_active()) {
      abandon_file_handle(state);
      return;
    }
    if (state->split.phase == split_phase::active) {
      // MPI 4.1 section 14.4.5: begin and end must be called from the same thread. A
      // destructor running elsewhere cannot legally finish the split; abandon instead.
      if (not(state->split.thread == current_thread_token())) {
        abandon_file_handle(state);
        return;
      }
      const int rc = finish_split_noexcept(handle, state->split.kind, state->split.buffer);
      if (rc != MPI_SUCCESS) {
        abandon_file_handle(state);
        return;
      }
      state->split.reset();
    } else if (state->split.phase == split_phase::unknown) {
      abandon_file_handle(state);
      return;
    }
    if (state->pending_requests.load(std::memory_order_relaxed) != 0) {
      abandon_file_handle(state);
      return;
    }
    state->handle = MPI_FILE_NULL;
    const int rc = MPI_File_close(&handle);
    state.reset();
    if (rc != MPI_SUCCESS)
      note_abandoned_handle();
  }

  // MPI_File_get_info hands ownership of a fresh MPI_Info table to the caller. Route it through
  // the single detail::info_access adoption hook (defined with mp1::info) so the owned table is
  // never copied key by key.
  [[nodiscard]] inline info copy_native_info(MPI_Info native, const char*) {
    return info_access::adopt(native);
  }

}  // namespace mp1::detail

#endif
