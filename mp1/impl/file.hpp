#if !(defined MP1_IMPL_FILE_HPP)

#define MP1_IMPL_FILE_HPP

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/concepts.hpp"
#include "mp1/impl/detail/file_io.hpp"
#include "mp1/impl/group.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/request.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1 {

  // File offsets and sizes are MPI_Offset (64-bit on all supported platforms).
  using offset_t = MPI_Offset;

  /// MPI-IO access mode bits; combine with operator| and mask with operator&.
  /// append sets the initial individual and shared file pointers to the initial end of file;
  /// it does not make later writes append automatically (MPI 4.1 section 14.2.1).
  enum class access_mode : int {
    read_only = MPI_MODE_RDONLY,
    read_write = MPI_MODE_RDWR,
    write_only = MPI_MODE_WRONLY,
    create = MPI_MODE_CREATE,
    no_replace = MPI_MODE_EXCL,
    delete_on_close = MPI_MODE_DELETE_ON_CLOSE,
    unique_open = MPI_MODE_UNIQUE_OPEN,
    sequential = MPI_MODE_SEQUENTIAL,
    append = MPI_MODE_APPEND
  };

  constexpr access_mode operator|(access_mode a, access_mode b) noexcept {
    return static_cast<access_mode>(static_cast<int>(a) | static_cast<int>(b));
  }

  constexpr access_mode operator&(access_mode a, access_mode b) noexcept {
    return static_cast<access_mode>(static_cast<int>(a) & static_cast<int>(b));
  }

  /// MPI_File_seek whence values.
  enum class whence_mode : int {
    set = MPI_SEEK_SET,
    current = MPI_SEEK_CUR,
    end = MPI_SEEK_END
  };

  /// Owning wrapper around an MPI_File; the only public file type.  The file surface is
  /// member-only: the former free wrappers and container adapters are gone, and each
  /// data-access family has one concept-constrained member per form.
  ///
  /// One owning handle with one shared internal state. The state records the access mode, the
  /// current view (etype identity, MPI 4.1 section 14.3), the split collective epoch
  /// (section 14.4.5), and the number of outstanding wrapper-owned nonblocking requests.
  /// Nonblocking requests hold a lease on that state, so a request may outlive this object
  /// without dangling; the file handle itself must outlive the request (MPI 4.1 section
  /// 14.2.2: all outstanding requests and split operations must complete before
  /// MPI_File_close).
  ///
  /// Data access is one member per (operation, form):
  ///   * value:          T& (read) / const T& (write) -- one native datatype application;
  ///   * borrowed buffer: any contiguous, sized range of elements (std::vector, std::array,
  ///                      std::basic_string, std::span, C array of non-character T; writes
  ///                      also accept std::string_view) -- count = size, datatype = native T;
  ///   * region<T>:       one application of a per-call layout -- count = 1, datatype = the
  ///                      layout (the buffer holds layout.size() elements; layout.size() is
  ///                      never the MPI count).
  /// A nonblocking/split begin write additionally requires borrowable<R&&> on a forwarding
  /// reference, so an rvalue container or scalar is rejected at compile time; the blocking
  /// write forms may borrow a temporary because they return only after the access completes.
  /// Every write/iwrite/split range form requires `element_buffer` (contiguous and sized), so
  /// a non-contiguous sequence such as std::list is rejected at overload resolution rather
  /// than failing inside <ranges>, matching the documented contiguous-buffer contract.
  /// A per-call region in a nonblocking/split request is duplicated at initiation and the
  /// duplicate is the datatype MPI holds, so the caller's layout may die after the call while
  /// the data buffer may not.
  ///
  /// Units (MPI's mixed-unit model, section 14.1.1):
  ///   - set_view displacement: bytes from the start of the file (passed unchanged, never
  ///     scaled by the etype extent);
  ///   - size/resize/preallocate: bytes;
  ///   - byte_offset: input is etype units relative to the view, output is absolute bytes;
  ///   - seek/seek_shared/position/get_position_shared: etype units relative to the view;
  ///   - read_at/write_at and all other data access: etype units (holes are not counted).
  /// The etype is the validated native datatype for a layout set_view and MPI_BYTE for the
  /// non-template set_view. Before the first set_view the view is the initial byte view
  /// (disp 0, MPI_BYTE etype/filetype, section 14.2.1).
  ///
  /// Collectivity and blocking (sections 14.2, 14.3, 14.4.1, 14.6.1):
  ///   - open, close, remove (not collective), resize, preallocate, sync, set_view, set_info,
  ///     set_atomicity, seek_shared and every *_at_all/*_all/*_ordered family are collective;
  ///   - the blocking collective calls and MPI_File_close may block until every rank enters;
  ///   - close is idempotent, performs the MPI close (and its implicit sync), and is
  ///     collective. All nonblocking requests and split operations must be complete first;
  ///   - read_all/write_all and their nonblocking/split forms use each rank's individual file
  ///     pointer (section 14.4.3). read_ordered/write_ordered, the read_shared/write_shared
  ///     (iread_shared/iwrite_shared) data-access family, and seek_shared/get_position_shared
  ///     use the shared file pointer (section 14.4.4).
  ///
  /// Close/destructor/release policy:
  ///   - close() refuses an active or unknown split epoch and outstanding wrapper-owned
  ///     requests, and throws std::runtime_error before any MPI call. It throws
  ///     std::runtime_error after an external MPI_Finalize without touching the handle.
  ///   - ~file() is noexcept and may block: it finishes an active split with the recorded
  ///     operation and buffer, then closes collectively when no request is outstanding. It
  ///     abandons (and counts) the handle when the split state is unknown, when the destructor
  ///     runs on a different thread than the begin (MPI 4.1 section 14.4.5 requires the same
  ///     thread), when requests are outstanding, or when MPI is no longer active. It never
  ///     throws and never calls MPI after finalize. The finish/close path is collective: if
  ///     the peers of this rank have already taken a different collective path, MPI itself
  ///     may deadlock; an explicit end followed by an explicit close is the supported
  ///     teardown.
  ///   - release() returns the raw handle; the caller then owns every raw-MPI obligation
  ///     including the collective close. It makes no MPI call except when an active region
  ///     split owns a duplicated layout: that duplicate is freed here (MPI_Type_free), and the
  ///     subsequent raw split end takes no datatype and still succeeds. Release during a split
  ///     is nevertheless not a supported spelling because the wrapper can no longer validate
  ///     or complete the epoch.
  ///   - The pending-request count covers requests created by this wrapper. Requests created
  ///     from the raw handle through native_handle() are invisible to close()/sync() and
  ///     remain the caller's raw-MPI obligation. set_view, resize and preallocate also reject
  ///     outstanding wrapper-owned requests before any MPI call (sections 14.2.4, 14.2.5 and
  ///     14.3); set_atomicity does not, because MPI permits it with outstanding nonblocking
  ///     operations (section 14.6.1).
  ///
  /// Split collectives (MPI 4.1 section 14.4.5): at most one active split per handle; begin
  /// and end are collective and must be called from the same thread; the end must match the
  /// begin operation and pass the same buffer. A wrong, double or missing end, a mismatched
  /// buffer, a collective I/O call during a split, and a close during a split all throw
  /// std::runtime_error locally and never reach MPI. sync, set_view, resize and preallocate
  /// also refuse during a split because their own clauses require split operations to be
  /// complete; set_info, set_atomicity and seek_shared carry no such requirement and are
  /// accepted (§14.2.8, §14.6.1, §14.4.4). A begin or end that fails inside MPI poisons the
  /// epoch to "unknown"; close and further begins/ends refuse, and the destructor abandons the
  /// handle. Noncollective data access, queries, seek/position, byte_offset and the three
  /// split-legal collective controls remain legal during a split.
  ///
  /// Sequential files (MPI 4.1 section 14.6.2): only shared-pointer data access is permitted;
  /// seek_shared/get_position_shared and the individual-pointer operations are rejected
  /// locally, with the single exception of get_byte_offset; resize/preallocate are rejected;
  /// set_view requires MPI_DISPLACEMENT_CURRENT.
  ///
  /// Deliberately outside 1.0: get_view (needs the owning raw MPI_Datatype bridge, deferred to
  /// 1.1), register_datarep/user data representations, file error-handler objects (the default
  /// MPI_ERRORS_RETURN makes mpi_error reachable), and a raw heterogeneous set_view. Use
  /// native_handle() for those raw MPI operations.
  ///
  /// Type matching (MPI 4.1 section 14.6.6): a typed access is locally rejected unless the
  /// recorded view is unknown (raw/adopted handle or a failed set_view), the view etype is
  /// MPI_BYTE, or the buffer element type has the same MPI datatype identity as the view
  /// etype. This is deliberately stricter than MPI's "integral number of contiguous etype
  /// copies" rule; the standard rule remains available through native_handle() and for
  /// unknown views. A datatype set through raw MPI after a wrapper set_view leaves the
  /// recorded token stale; mixing raw view changes with wrapper typed access is unsupported.
  ///
  /// Errors: state/epoch/mode misuse throws std::runtime_error; argument/value faults throw
  /// std::invalid_argument; an MPI failure throws mp1::mpi_error. Local checks never make an
  /// MPI call. A local throw before a collective call is a job-level failure: peers may
  /// already be inside that collective.
  ///
  /// A file request's drop policy is wait-only: dropping or destroying a request waits, and a
  /// failed wait/release leaves the pending lease counted, so close() keeps refusing
  /// conservatively. MPI 4.1's default file error handler is MPI_ERRORS_RETURN in section 14.7,
  /// so mpi_error is reachable without installing a handler. MPI_SOURCE/MPI_TAG/MPI_ERROR of
  /// a file status are undefined (section 14.4.1); count/elements are meaningful.
  class file {
  public:
    // ---- lifecycle: 5 ----
    /// Closed file (MPI_FILE_NULL); no MPI call.
    file() noexcept = default;

    /// Open a file; see open().
    file(const mpi_communicator& comm, const std::filesystem::path& name, access_mode mode,
         const info& i = {}) {
      open(comm, name, mode, i);
    }

    /// May block (the target's split end and collective close). Makes no MPI call after
    /// finalize. Copy construction/assignment are implicitly deleted by the move members.
    file(file&& other) noexcept = default;

    /// May block (the target's split end and collective close).
    file& operator=(file&& other) noexcept {
      if (this != &other) {
        detail::close_file_noexcept(state_);
        state_ = std::move(other.state_);
      }
      return *this;
    }

    /// Noexcept; may block in the matching split end and the collective close; abandons the
    /// handle when the state is unknown, requests are outstanding, or MPI is gone.
    ~file() noexcept {
      detail::close_file_noexcept(state_);
    }

    // ---- control: 7 ----
    /// Open a file collectively over `comm` (intracommunicator only), MPI 4.1 section 14.2.1.
    /// All ranks must reference the same file; amode must be identical on every rank and is
    /// validated locally before MPI (exactly one access bit; create/no_replace not with
    /// read_only; sequential not with read_write). A failed open throws mpi_error and leaves
    /// the object closed and reusable.
    void open(const mpi_communicator& comm, const std::filesystem::path& name, access_mode mode,
              const info& i = {}) {
      if (is_open())
        detail::fail_with<std::runtime_error>("mp1::file::open",
                                              std::source_location::current(), "already open");
      detail::require_mpi("mp1::file::open");
      detail::require_valid(comm, "mp1::file::open");
      if (comm.is_inter()) {
        throw_invalid("mp1::file::open", "MPI-IO requires an intracommunicator");
      }
      detail::validate_file_amode(static_cast<int>(mode), "mp1::file::open");

      // Allocate the state before the MPI call: a post-open allocation failure cannot leak the
      // handle, and the state carries the recorded amode and the initial byte view.
      auto state = std::make_shared<detail::file_state>();
      state->amode = static_cast<int>(mode);

      const std::string filename = name.string();
      MPI_File opened{MPI_FILE_NULL};
      const int rc = MPI_File_open(comm.native_handle(), filename.c_str(),
                                   static_cast<int>(mode), i.native_handle(), &opened);
      if (rc != MPI_SUCCESS)
        throw mpi_error{rc, "MPI_File_open"};

      state->handle = opened;
      state_ = std::move(state);
    }

    /// Take ownership of a raw MPI_File. Requires active MPI and a non-null handle; the
    /// access mode is queried and validated before ownership is transferred (if that or the
    /// state allocation throws, the caller still owns the handle). The current view is
    /// unknown, so typed access falls back to MPI's own type-signature rule.
    [[nodiscard]] static file adopt(MPI_File handle) {
      detail::require_mpi("mp1::file::adopt");
      if (handle == MPI_FILE_NULL)
        throw_invalid("mp1::file::adopt", "MPI_FILE_NULL");
      int amode{0};
      detail::check_file(MPI_File_get_amode(handle, &amode), "mp1::file::adopt");
      detail::validate_file_amode(amode, "mp1::file::adopt");

      auto state = std::make_shared<detail::file_state>();
      state->handle = handle;
      state->amode = amode;
      state->view.known = false;
      file adopted;
      adopted.state_ = std::move(state);
      return adopted;
    }

    /// Collective; idempotent. Throws std::runtime_error after finalize, during an active or
    /// unknown split, or with outstanding wrapper-owned requests, without touching the
    /// handle. On success or MPI failure the object is closed and cannot be closed again; an
    /// MPI failure throws mpi_error and counts the handle as abandoned (MPI may or may not
    /// have deallocated it).
    void close() {
      if (not state_ or state_->handle == MPI_FILE_NULL)
        return;
      const context ctx{"mp1::file::close"};
      detail::require_mpi_checked(ctx);
      if (state_->split.phase == detail::split_phase::active) {
        detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                              "close during a split collective");
      }
      if (state_->split.phase == detail::split_phase::unknown) {
        detail::fail_with<std::runtime_error>(
            ctx.operation, std::source_location::current(),
            "close while the split collective state is unknown after a failed begin/end");
      }
      detail::require_no_pending_requests(*state_, ctx);

      MPI_File handle = state_->handle;
      state_->handle = MPI_FILE_NULL;
      const int rc = MPI_File_close(&handle);
      state_.reset();
      if (rc != MPI_SUCCESS) {
        detail::note_abandoned_handle();
        detail::check_file(rc, ctx);
      }
    }

    /// Hand the raw handle and every raw-MPI obligation (outstanding requests, split end,
    /// collective close) to the caller. A second release, a closed file, or a moved-from file
    /// returns MPI_FILE_NULL. MPI-free except when an active region split owns a duplicated
    /// layout, which is freed here with MPI_Type_free.
    [[nodiscard]] MPI_File release() noexcept {
      if (not state_ or state_->handle == MPI_FILE_NULL)
        return MPI_FILE_NULL;
      MPI_File handle = state_->handle;
      state_->handle = MPI_FILE_NULL;
      state_->split.reset();
      state_.reset();
      return handle;
    }

    /// \return true if a native handle is held; no MPI call
    [[nodiscard]] bool is_open() const noexcept {
      return state_ and state_->handle != MPI_FILE_NULL;
    }

    /// \return the raw MPI_File, non-owning; no MPI call
    [[nodiscard]] MPI_File native_handle() const noexcept {
      return state_ ? state_->handle : MPI_FILE_NULL;
    }

    /// MPI_File_delete; not collective (MPI 4.1 section 14.2.3). Behavior with the file open
    /// elsewhere is implementation-dependent. Throws mpi_error on failure.
    static void remove(const std::filesystem::path& name, const info& i = {}) {
      detail::require_mpi("mp1::file::remove");
      const std::string filename = name.string();
      detail::check_file(MPI_File_delete(filename.c_str(), i.native_handle()),
                         "MPI_File_delete");
    }

    // ---- metadata/positioning: 15 ----
    /// Collective; set the file size in bytes (MPI 4.1 section 14.2.4). Erroneous on a
    /// sequential file or with an outstanding nonblocking request; both are rejected locally
    /// before MPI.
    void resize(offset_t new_size) {
      const context ctx{"mp1::file::resize"};
      MPI_File handle = detail::require_collective_file(state_, false, ctx);
      detail::require_no_pending_requests(*state_, ctx);
      detail::check_file(MPI_File_set_size(handle, new_size), ctx);
    }

    /// Collective; ensure the file is at least `new_size` bytes (section 14.2.5). Erroneous on
    /// a sequential file or with an outstanding nonblocking request; both are rejected locally
    /// before MPI.
    void preallocate(offset_t new_size) {
      const context ctx{"mp1::file::preallocate"};
      MPI_File handle = detail::require_collective_file(state_, false, ctx);
      detail::require_no_pending_requests(*state_, ctx);
      detail::check_file(MPI_File_preallocate(handle, new_size), ctx);
    }

    /// \return file size in bytes
    [[nodiscard]] offset_t size() const {
      MPI_File handle = detail::require_file(state_, "mp1::file::size");
      MPI_Offset result{0};
      detail::check_file(MPI_File_get_size(handle, &result), "MPI_File_get_size");
      return result;
    }

    /// \return the access mode recorded at open/adopt; no MPI call
    [[nodiscard]] access_mode mode() const {
      (void)detail::require_file(state_, "mp1::file::mode");
      return static_cast<access_mode>(state_->amode);
    }

    /// Collective; flush previous writes and make other ranks' updates visible (section
    /// 14.6.1). Erroneous with outstanding nonblocking requests or an active split; both are
    /// rejected locally.
    void sync() {
      const context ctx{"mp1::file::sync"};
      MPI_File handle = detail::require_collective_file(state_, true, ctx);
      detail::require_no_pending_requests(*state_, ctx);
      detail::check_file(MPI_File_sync(handle), ctx);
    }

    /// Collective; set the byte view (disp in BYTES, MPI_BYTE etype/filetype). The view resets
    /// both file pointers to zero (section 14.3). Erroneous with an outstanding nonblocking
    /// request; rejected locally before MPI. In sequential mode disp must be
    /// MPI_DISPLACEMENT_CURRENT and it is invalid otherwise.
    void set_view(offset_t disp, const char* datarep = "native", const info& i = {}) {
      const context ctx{"mp1::file::set_view"};
      detail::require_file_datarep(datarep, ctx);
      MPI_File handle = detail::require_collective_file(state_, true, ctx);
      detail::require_no_pending_requests(*state_, ctx);
      detail::require_file_displacement(*state_, disp, ctx);
      const int rc =
          MPI_File_set_view(handle, disp, MPI_BYTE, MPI_BYTE, datarep, i.native_handle());
      if (rc != MPI_SUCCESS) {
        state_->view.known = false;
        detail::check_file(rc, ctx);
      }
      state_->view = detail::file_view_state{true, true, MPI_BYTE};
    }

    /// Collective; etype is the validated native datatype, filetype is `filetype` (section 14.3).
    /// datarep and the file-representation extents of the etype must be identical on all
    /// ranks; disp (an absolute BYTE displacement, passed unchanged), filetype and info may
    /// vary. Resets both file pointers. Erroneous with an outstanding nonblocking request;
    /// rejected locally before MPI. A derived etype whose typemap order is proven non-monotone
    /// (descending or negative displacements) is rejected for every access mode. The filetype
    /// is checked the same way against the decoded exact type map (negative or descending
    /// displacements are rejected; §14.3), and a filetype with a proven overlapping typemap is
    /// rejected when the file was opened for writing (§14.3.1). The remaining §14.3 rule — a
    /// hole extent must be a multiple of the etype extent — is not decidable from the
    /// basic-entry intervals and remains the caller's obligation for an opaque/`from_raw`
    /// filetype; the public layout factories produce extent-aligned maps.
    template<element T>
    void set_view(offset_t disp, const layout<T>& filetype, const char* datarep = "native",
                  const info& i = {}) {
      const context ctx{"mp1::file::set_view"};
      detail::require_file_data_type<T>();
      if (not filetype.is_valid())
        throw_invalid("mp1::file::set_view", "invalid layout");
      detail::require_file_datarep(datarep, ctx);
      MPI_File handle = detail::require_collective_file(state_, true, ctx);
      detail::require_no_pending_requests(*state_, ctx);
      detail::require_file_displacement(*state_, disp, ctx);
      detail::require_file_view_etype<T>(ctx);
      detail::require_file_view_filetype(*state_, filetype, ctx);
      const MPI_Datatype etype = detail::datatype_info_of<T>().type;
      const int rc = MPI_File_set_view(handle, disp, etype, filetype.native_handle(), datarep,
                                       i.native_handle());
      if (rc != MPI_SUCCESS) {
        state_->view.known = false;
        detail::check_file(rc, ctx);
      }
      state_->view = detail::file_view_state{true, etype == MPI_BYTE, etype};
    }

    /// Noncollective; set the individual file pointer in etype units (section 14.4.3).
    /// Erroneous on a sequential file.
    void seek(offset_t offset, whence_mode whence) {
      detail::file_call(state_, false, false, "mp1::file::seek",
                        [offset, whence](MPI_File handle) {
                          return MPI_File_seek(handle, offset, static_cast<int>(whence));
                        });
    }

    /// Collective; all ranks pass identical offset/whence (section 14.4.4). Erroneous on a
    /// sequential file.
    void seek_shared(offset_t offset, whence_mode whence) {
      // §14.4.4 has no split-completion requirement: seek_shared is a collective control, not
      // a collective I/O operation, so it is legal between a split begin and its end.
      detail::file_call_during_split(
          state_, false, "mp1::file::seek_shared", [offset, whence](MPI_File handle) {
            return MPI_File_seek_shared(handle, offset, static_cast<int>(whence));
          });
    }

    /// \return the individual file pointer in etype units; erroneous on a sequential file
    [[nodiscard]] offset_t position() const {
      MPI_File handle =
          detail::require_noncollective_file(state_, false, "mp1::file::position");
      MPI_Offset result{0};
      detail::check_file(MPI_File_get_position(handle, &result), "MPI_File_get_position");
      return result;
    }

    /// \return the shared file pointer in etype units (section 14.4.4); erroneous on a
    /// sequential file
    [[nodiscard]] offset_t get_position_shared() const {
      MPI_File handle =
          detail::require_noncollective_file(state_, false, "mp1::file::get_position_shared");
      MPI_Offset result{0};
      detail::check_file(MPI_File_get_position_shared(handle, &result),
                         "MPI_File_get_position_shared");
      return result;
    }

    /// \return absolute byte offset of etype `offset` in the current view; the one
    /// individual-pointer operation permitted on a sequential file (MPI 4.1 section 14.4.3).
    /// Provider caveat: a provider may reject this call on a sequential file with
    /// MPI_ERR_UNSUPPORTED_OPERATION even though the standard permits it, and pinned
    /// MPICH 4.3.2 has been observed to return an incorrect offset (0) instead of an error
    /// after set_view(MPI_DISPLACEMENT_CURRENT, layout<char>::contiguous(1)), while Open MPI
    /// returns the standard-conforming value. Keep the standard-conforming behavior and
    /// surface a provider error as mp1::mpi_error; verify the call on the target provider and
    /// prefer a non-sequential mode when the byte offset is required. Do not narrow this
    /// contract for one provider.
    [[nodiscard]] offset_t byte_offset(offset_t offset) const {
      MPI_File handle =
          detail::require_noncollective_file(state_, true, "mp1::file::byte_offset");
      MPI_Offset result{0};
      detail::check_file(MPI_File_get_byte_offset(handle, offset, &result),
                         "MPI_File_get_byte_offset");
      return result;
    }

    /// Collective; replace the file hints (section 14.2.8). All ranks must pass a non-null
    /// info; a null info is rejected locally (Open MPI returns MPI_ERR_INFO for
    /// MPI_INFO_NULL, MPICH accepts it, so the portable contract is non-null).
    void set_info(const info& i) {
      // §14.2.8 has no split-completion requirement: set_info is a collective control, not a
      // collective I/O operation, so it is legal between a split begin and its end.
      MPI_File handle =
          detail::require_collective_file_during_split(state_, true, "mp1::file::set_info");
      if (i.native_handle() == MPI_INFO_NULL)
        throw_invalid("mp1::file::set_info", "null info");
      detail::check_file(MPI_File_set_info(handle, i.native_handle()), "MPI_File_set_info");
    }

    /// \return an owning copy of the MPI_File_get_info table
    [[nodiscard]] info get_info() const {
      MPI_File handle = detail::require_file(state_, "mp1::file::get_info");
      MPI_Info native{MPI_INFO_NULL};
      detail::check_file(MPI_File_get_info(handle, &native), "MPI_File_get_info");
      return detail::copy_native_info(native, "MPI_File_get_info");
    }

    /// \return owning group of processes that opened the file
    [[nodiscard]] group get_group() const {
      MPI_File handle = detail::require_file(state_, "mp1::file::get_group");
      MPI_Group native{MPI_GROUP_NULL};
      detail::check_file(MPI_File_get_group(handle, &native), "MPI_File_get_group");
      return detail::group_factory::make(native);
    }

    // ---- atomicity/extent/handler: 6 ----
    /// Collective; set atomic mode (section 14.6.1). The flag must be identical on all ranks;
    /// it is not validated locally beyond the bool type.
    void set_atomicity(bool flag) {
      // §14.6.1 permits set_atomicity with outstanding nonblocking operations and sets no
      // split-completion requirement: it is legal between a split begin and its end.
      detail::file_call_during_split(
          state_, true, "mp1::file::set_atomicity",
          [flag](MPI_File handle) { return MPI_File_set_atomicity(handle, flag ? 1 : 0); });
    }

    /// \return current atomic mode (section 14.6.1); noncollective
    [[nodiscard]] bool get_atomicity() const {
      MPI_File handle = detail::require_file(state_, "mp1::file::get_atomicity");
      int flag{0};
      detail::check_file(MPI_File_get_atomicity(handle, &flag), "MPI_File_get_atomicity");
      return flag != 0;
    }

    /// \return the extent of T in the file representation of this handle (section 14.5.1);
    /// MPI_Aint, or MPI_UNDEFINED when MPI cannot represent it
    template<element T>
    [[nodiscard]] MPI_Aint get_type_extent() const {
      detail::require_file_data_type<T>();
      MPI_File handle = detail::require_file(state_, "mp1::file::get_type_extent");
      MPI_Aint extent{0};
      detail::check_file(
          MPI_File_get_type_extent(handle, detail::datatype_info_of<T>().type, &extent),
          "MPI_File_get_type_extent");
      return extent;
    }

    /// \return the extent of `filetype` in the file representation of this handle
    template<element T>
    [[nodiscard]] MPI_Aint get_type_extent(const layout<T>& filetype) const {
      detail::require_file_data_type<T>();
      if (not filetype.is_valid())
        throw_invalid("mp1::file::get_type_extent", "invalid layout");
      MPI_File handle = detail::require_file(state_, "mp1::file::get_type_extent");
      MPI_Aint extent{0};
      detail::check_file(MPI_File_get_type_extent(handle, filetype.native_handle(), &extent),
                         "MPI_File_get_type_extent");
      return extent;
    }

    /// MPI_File_set_errhandler with a predefined handler (MPI 4.1 section 9.3.3).
    /// Non-collective, process-local; requires active MPI and an open file. Files do not
    /// inherit the initial handler and default to MPI_ERRORS_RETURN; see section 14.7.
    void set_error_handler(error_handler handler) {
      const context ctx{"mp1::file::set_error_handler"};
      detail::require_mpi(ctx);
      if (not is_open())
        detail::fail_with<std::runtime_error>(ctx, std::source_location::current(),
                                              "file is not open");
      detail::check_file(
          MPI_File_set_errhandler(native_handle(), detail::native_error_handler(ctx, handler)),
          ctx);
    }

    /// MPI_File_get_errhandler plus predefined classification; std::nullopt when the attached
    /// handler is not one of the three predefined handlers. The returned reference is freed
    /// with MPI_Errhandler_free before returning (MPI 4.1 section 9.3, Table 11.1).
    [[nodiscard]] std::optional<error_handler> get_error_handler() const {
      detail::require_mpi("mp1::file::get_error_handler");
      if (not is_open())
        detail::fail_with<std::runtime_error>("mp1::file::get_error_handler",
                                              std::source_location::current(),
                                              "file is not open");
      MPI_Errhandler handler{MPI_ERRHANDLER_NULL};
      detail::check_file(MPI_File_get_errhandler(native_handle(), &handler),
                         "MPI_File_get_errhandler");
      const std::optional<error_handler> result = detail::classify_error_handler(handler);
      if (handler != MPI_ERRHANDLER_NULL)
        detail::check_mpi(MPI_Errhandler_free(&handler), "MPI_Errhandler_free");
      return result;
    }

    // ---- data access: 96 = 14 read x 3 + 14 write x 3 + 6 split end x 2 ----
    /// read family `read`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read(T& value) const {
      const context ctx{"mp1::file::read"};

      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read(R&& buffer) const {
      const context ctx{"mp1::file::read"};

      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read(region<T> buffer) const {
      const context ctx{"mp1::file::read"};

      return detail::file_read_status<T>(
          state_, false, false, ctx, detail::file_read_region(buffer, false, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_c(handle, data, count, datatype, native);
          });
    }
    /// read family `read_at`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_at(offset_t off, T& value) const {
      const context ctx{"mp1::file::read_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_read_value(value),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `read_at`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_at(offset_t off, R&& buffer) const {
      const context ctx{"mp1::file::read_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_read_range(buffer, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `read_at`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read_at(offset_t off, region<T> buffer) const {
      const context ctx{"mp1::file::read_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<T>(
          state_, false, false, ctx, detail::file_read_region(buffer, false, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_c(handle, off, data, count, datatype, native);
          });
    }
    /// read family `read_shared`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_shared(T& value) const {
      const context ctx{"mp1::file::read_shared"};

      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, false, true, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_shared_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_shared`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_shared(R&& buffer) const {
      const context ctx{"mp1::file::read_shared"};

      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, true, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_shared_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_shared`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read_shared(region<T> buffer) const {
      const context ctx{"mp1::file::read_shared"};

      return detail::file_read_status<T>(
          state_, false, true, ctx, detail::file_read_region(buffer, false, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_shared_c(handle, data, count, datatype, native);
          });
    }
    /// read family `read_all`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_all(T& value) const {
      const context ctx{"mp1::file::read_all"};

      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_all_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_all`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_all(R&& buffer) const {
      const context ctx{"mp1::file::read_all"};

      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_all_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_all`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read_all(region<T> buffer) const {
      const context ctx{"mp1::file::read_all"};

      return detail::file_read_status<T>(
          state_, true, false, ctx, detail::file_read_region(buffer, false, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_all_c(handle, data, count, datatype, native);
          });
    }
    /// read family `read_at_all`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_at_all(offset_t off, T& value) const {
      const context ctx{"mp1::file::read_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_read_value(value),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `read_at_all`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_at_all(offset_t off, R&& buffer) const {
      const context ctx{"mp1::file::read_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_read_range(buffer, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `read_at_all`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read_at_all(offset_t off, region<T> buffer) const {
      const context ctx{"mp1::file::read_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_status<T>(
          state_, true, false, ctx, detail::file_read_region(buffer, false, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_read_at_all_c(handle, off, data, count, datatype, native);
          });
    }
    /// read family `read_ordered`: value form (one native datatype application).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_ordered(T& value) const {
      const context ctx{"mp1::file::read_ordered"};

      return detail::file_read_status<std::remove_cv_t<T>>(
          state_, true, true, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_ordered_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_ordered`: borrowed contiguous buffer (count = size, native T).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_ordered(R&& buffer) const {
      const context ctx{"mp1::file::read_ordered"};

      return detail::file_read_status<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, true, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_ordered_c(handle, data, count, datatype, native);
          });
    }

    /// read family `read_ordered`: one region occurrence (count = 1, datatype = layout).
    template<class T>
    requires(!std::is_const_v<T>)
    status read_ordered(region<T> buffer) const {
      const context ctx{"mp1::file::read_ordered"};

      return detail::file_read_status<T>(
          state_, true, true, ctx, detail::file_read_region(buffer, false, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_read_ordered_c(handle, data, count, datatype, native);
          });
    }
    /// read family `iread`: value form, nonblocking.
    template<element_value T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread(T& value) const {
      const context ctx{"mp1::file::iread"};

      return detail::file_read_request<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread`: borrowed contiguous buffer, nonblocking.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    [[nodiscard]] request iread(R&& buffer) const {
      const context ctx{"mp1::file::iread"};

      return detail::file_read_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread`: one region occurrence, nonblocking; the layout is
    /// duplicated into the request lease.
    template<class T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread(region<T> buffer) const {
      const context ctx{"mp1::file::iread"};

      return detail::file_read_request<T>(
          state_, false, false, ctx, detail::file_read_region(buffer, true, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_c(handle, data, count, datatype, native);
          });
    }
    /// read family `iread_at`: value form, nonblocking.
    template<element_value T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_at(offset_t off, T& value) const {
      const context ctx{"mp1::file::iread_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_read_value(value),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `iread_at`: borrowed contiguous buffer, nonblocking.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    [[nodiscard]] request iread_at(offset_t off, R&& buffer) const {
      const context ctx{"mp1::file::iread_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_read_range(buffer, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `iread_at`: one region occurrence, nonblocking; the layout is
    /// duplicated into the request lease.
    template<class T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_at(offset_t off, region<T> buffer) const {
      const context ctx{"mp1::file::iread_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<T>(
          state_, false, false, ctx, detail::file_read_region(buffer, true, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_c(handle, off, data, count, datatype, native);
          });
    }
    /// read family `iread_shared`: value form, nonblocking.
    template<element_value T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_shared(T& value) const {
      const context ctx{"mp1::file::iread_shared"};

      return detail::file_read_request<std::remove_cv_t<T>>(
          state_, false, true, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_shared_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread_shared`: borrowed contiguous buffer, nonblocking.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    [[nodiscard]] request iread_shared(R&& buffer) const {
      const context ctx{"mp1::file::iread_shared"};

      return detail::file_read_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, true, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_shared_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread_shared`: one region occurrence, nonblocking; the layout is
    /// duplicated into the request lease.
    template<class T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_shared(region<T> buffer) const {
      const context ctx{"mp1::file::iread_shared"};

      return detail::file_read_request<T>(
          state_, false, true, ctx, detail::file_read_region(buffer, true, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_shared_c(handle, data, count, datatype, native);
          });
    }
    /// read family `iread_all`: value form, nonblocking.
    template<element_value T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_all(T& value) const {
      const context ctx{"mp1::file::iread_all"};

      return detail::file_read_request<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_read_value(value),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_all_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread_all`: borrowed contiguous buffer, nonblocking.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    [[nodiscard]] request iread_all(R&& buffer) const {
      const context ctx{"mp1::file::iread_all"};

      return detail::file_read_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_read_range(buffer, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_all_c(handle, data, count, datatype, native);
          });
    }

    /// read family `iread_all`: one region occurrence, nonblocking; the layout is
    /// duplicated into the request lease.
    template<class T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_all(region<T> buffer) const {
      const context ctx{"mp1::file::iread_all"};

      return detail::file_read_request<T>(
          state_, true, false, ctx, detail::file_read_region(buffer, true, ctx),
          [](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iread_all_c(handle, data, count, datatype, native);
          });
    }
    /// read family `iread_at_all`: value form, nonblocking.
    template<element_value T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_at_all(offset_t off, T& value) const {
      const context ctx{"mp1::file::iread_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_read_value(value),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `iread_at_all`: borrowed contiguous buffer, nonblocking.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    [[nodiscard]] request iread_at_all(offset_t off, R&& buffer) const {
      const context ctx{"mp1::file::iread_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_read_range(buffer, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// read family `iread_at_all`: one region occurrence, nonblocking; the layout is
    /// duplicated into the request lease.
    template<class T>
    requires(!std::is_const_v<T>)
    [[nodiscard]] request iread_at_all(offset_t off, region<T> buffer) const {
      const context ctx{"mp1::file::iread_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_read_request<T>(
          state_, true, false, ctx, detail::file_read_region(buffer, true, ctx),
          [off](MPI_File handle, void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iread_at_all_c(handle, off, data, count, datatype, native);
          });
    }
    /// read split begin `read_at_all_begin`: value form (same-thread begin/end).
    template<element_value T>
    requires(!std::is_const_v<T>)
    void read_at_all_begin(offset_t off, T& value) {
      const context ctx{"mp1::file::read_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_read_value(value);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::read_at_all, transfer.data, false, false, ctx,
          std::move(transfer.layout_keepalive),
          [off, data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_at_all_begin_c(handle, off, data, count, datatype);
          });
    }

    /// read split begin `read_at_all_begin`: borrowed buffer; an rvalue container is
    /// rejected because the buffer must stay put until the matching end.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    void read_at_all_begin(offset_t off, R&& buffer) {
      const context ctx{"mp1::file::read_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_read_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::read_at_all, transfer.data, false, false, ctx,
          std::move(transfer.layout_keepalive),
          [off, data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_at_all_begin_c(handle, off, data, count, datatype);
          });
    }

    /// read split begin `read_at_all_begin`: region; the layout is duplicated into the
    /// split epoch; end it with a span over region.data()/region.map().size().
    template<class T>
    requires(!std::is_const_v<T>)
    void read_at_all_begin(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::read_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_read_region(buffer, true, ctx);
      detail::file_begin<T>(state_, detail::split_kind::read_at_all, transfer.data, false,
                            false, ctx, std::move(transfer.layout_keepalive),
                            [off, data = transfer.data, count = transfer.count,
                             datatype = transfer.datatype](MPI_File handle) {
                              return MPI_File_read_at_all_begin_c(handle, off, data, count,
                                                                  datatype);
                            });
    }
    /// read split begin `read_all_begin`: value form (same-thread begin/end).
    template<element_value T>
    requires(!std::is_const_v<T>)
    void read_all_begin(T& value) {
      const context ctx{"mp1::file::read_all_begin"};

      auto transfer = detail::file_read_value(value);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::read_all, transfer.data, false, false, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_all_begin_c(handle, data, count, datatype);
          });
    }

    /// read split begin `read_all_begin`: borrowed buffer; an rvalue container is
    /// rejected because the buffer must stay put until the matching end.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    void read_all_begin(R&& buffer) {
      const context ctx{"mp1::file::read_all_begin"};

      auto transfer = detail::file_read_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::read_all, transfer.data, false, false, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_all_begin_c(handle, data, count, datatype);
          });
    }

    /// read split begin `read_all_begin`: region; the layout is duplicated into the
    /// split epoch; end it with a span over region.data()/region.map().size().
    template<class T>
    requires(!std::is_const_v<T>)
    void read_all_begin(region<T> buffer) {
      const context ctx{"mp1::file::read_all_begin"};

      auto transfer = detail::file_read_region(buffer, true, ctx);
      detail::file_begin<T>(state_, detail::split_kind::read_all, transfer.data, false, false,
                            ctx, std::move(transfer.layout_keepalive),
                            [data = transfer.data, count = transfer.count,
                             datatype = transfer.datatype](MPI_File handle) {
                              return MPI_File_read_all_begin_c(handle, data, count, datatype);
                            });
    }
    /// read split begin `read_ordered_begin`: value form (same-thread begin/end).
    template<element_value T>
    requires(!std::is_const_v<T>)
    void read_ordered_begin(T& value) {
      const context ctx{"mp1::file::read_ordered_begin"};

      auto transfer = detail::file_read_value(value);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::read_ordered, transfer.data, false, true, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_ordered_begin_c(handle, data, count, datatype);
          });
    }

    /// read split begin `read_ordered_begin`: borrowed buffer; an rvalue container is
    /// rejected because the buffer must stay put until the matching end.
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    void read_ordered_begin(R&& buffer) {
      const context ctx{"mp1::file::read_ordered_begin"};

      auto transfer = detail::file_read_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::read_ordered, transfer.data, false, true, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_read_ordered_begin_c(handle, data, count, datatype);
          });
    }

    /// read split begin `read_ordered_begin`: region; the layout is duplicated into the
    /// split epoch; end it with a span over region.data()/region.map().size().
    template<class T>
    requires(!std::is_const_v<T>)
    void read_ordered_begin(region<T> buffer) {
      const context ctx{"mp1::file::read_ordered_begin"};

      auto transfer = detail::file_read_region(buffer, true, ctx);
      detail::file_begin<T>(state_, detail::split_kind::read_ordered, transfer.data, false,
                            true, ctx, std::move(transfer.layout_keepalive),
                            [data = transfer.data, count = transfer.count,
                             datatype = transfer.datatype](MPI_File handle) {
                              return MPI_File_read_ordered_begin_c(handle, data, count,
                                                                   datatype);
                            });
    }
    /// write family `write`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write(const T& value) {
      const context ctx{"mp1::file::write"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write(const R& buffer) {
      const context ctx{"mp1::file::write"};

      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, false, false, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write`: one region occurrence (read-only buffer use).
    template<class T>
    status write(region<T> buffer) {
      const context ctx{"mp1::file::write"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_region(buffer, false, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_c(handle, data, count, datatype, native);
          });
    }
    /// write family `write_at`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write_at(offset_t off, const T& value) {
      const context ctx{"mp1::file::write_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_value(value),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `write_at`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write_at(offset_t off, const R& buffer) {
      const context ctx{"mp1::file::write_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, false, false, ctx, detail::file_send_range(buffer, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `write_at`: one region occurrence (read-only buffer use).
    template<class T>
    status write_at(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::write_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_region(buffer, false, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_c(handle, off, data, count, datatype, native);
          });
    }
    /// write family `write_shared`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write_shared(const T& value) {
      const context ctx{"mp1::file::write_shared"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, true, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_shared_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_shared`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write_shared(const R& buffer) {
      const context ctx{"mp1::file::write_shared"};

      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, false, true, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_shared_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_shared`: one region occurrence (read-only buffer use).
    template<class T>
    status write_shared(region<T> buffer) {
      const context ctx{"mp1::file::write_shared"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, false, true, ctx, detail::file_send_region(buffer, false, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_shared_c(handle, data, count, datatype, native);
          });
    }
    /// write family `write_all`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write_all(const T& value) {
      const context ctx{"mp1::file::write_all"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_all_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_all`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write_all(const R& buffer) {
      const context ctx{"mp1::file::write_all"};

      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, true, false, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_all_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_all`: one region occurrence (read-only buffer use).
    template<class T>
    status write_all(region<T> buffer) {
      const context ctx{"mp1::file::write_all"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_region(buffer, false, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_all_c(handle, data, count, datatype, native);
          });
    }
    /// write family `write_at_all`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write_at_all(offset_t off, const T& value) {
      const context ctx{"mp1::file::write_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_value(value),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `write_at_all`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write_at_all(offset_t off, const R& buffer) {
      const context ctx{"mp1::file::write_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, true, false, ctx, detail::file_send_range(buffer, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `write_at_all`: one region occurrence (read-only buffer use).
    template<class T>
    status write_at_all(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::write_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_region(buffer, false, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Status* native) {
            return MPI_File_write_at_all_c(handle, off, data, count, datatype, native);
          });
    }
    /// write family `write_ordered`: value form (blocking sends may borrow a temporary).
    template<element_value T>
    status write_ordered(const T& value) {
      const context ctx{"mp1::file::write_ordered"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, true, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_ordered_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_ordered`: borrowed contiguous buffer (const range).
    template<class R>
    requires(element_buffer<R &&>)
    status write_ordered(const R& buffer) {
      const context ctx{"mp1::file::write_ordered"};

      return detail::file_write_status<std::remove_cv_t<std::ranges::range_value_t<R>>>(
          state_, true, true, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_ordered_c(handle, data, count, datatype, native);
          });
    }

    /// write family `write_ordered`: one region occurrence (read-only buffer use).
    template<class T>
    status write_ordered(region<T> buffer) {
      const context ctx{"mp1::file::write_ordered"};

      return detail::file_write_status<std::remove_cv_t<T>>(
          state_, true, true, ctx, detail::file_send_region(buffer, false, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Status* native) {
            return MPI_File_write_ordered_c(handle, data, count, datatype, native);
          });
    }
    /// write family `iwrite`: value form; borrowable<T&&> rejects an rvalue
    /// scalar (the request retains the address until completion).
    template<element_value T>
    requires(borrowable<T &&>)
    [[nodiscard]] request iwrite(T&& value) {
      const context ctx{"mp1::file::iwrite"};

      return detail::file_write_request<std::remove_cvref_t<T>>(
          state_, false, false, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite`: borrowed contiguous buffer; borrowable<R&&> rejects
    /// an rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    [[nodiscard]] request iwrite(R&& buffer) {
      const context ctx{"mp1::file::iwrite"};

      return detail::file_write_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite`: one region occurrence; the layout is duplicated
    /// into the request lease.
    template<class T>
    [[nodiscard]] request iwrite(region<T> buffer) {
      const context ctx{"mp1::file::iwrite"};

      return detail::file_write_request<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_region(buffer, true, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_c(handle, data, count, datatype, native);
          });
    }
    /// write family `iwrite_at`: value form; borrowable<T&&> rejects an rvalue
    /// scalar (the request retains the address until completion).
    template<element_value T>
    requires(borrowable<T &&>)
    [[nodiscard]] request iwrite_at(offset_t off, T&& value) {
      const context ctx{"mp1::file::iwrite_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<std::remove_cvref_t<T>>(
          state_, false, false, ctx, detail::file_send_value(value),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `iwrite_at`: borrowed contiguous buffer; borrowable<R&&> rejects
    /// an rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    [[nodiscard]] request iwrite_at(offset_t off, R&& buffer) {
      const context ctx{"mp1::file::iwrite_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, false, ctx, detail::file_send_range(buffer, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `iwrite_at`: one region occurrence; the layout is duplicated
    /// into the request lease.
    template<class T>
    [[nodiscard]] request iwrite_at(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::iwrite_at"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<std::remove_cv_t<T>>(
          state_, false, false, ctx, detail::file_send_region(buffer, true, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_c(handle, off, data, count, datatype, native);
          });
    }
    /// write family `iwrite_shared`: value form; borrowable<T&&> rejects an rvalue
    /// scalar (the request retains the address until completion).
    template<element_value T>
    requires(borrowable<T &&>)
    [[nodiscard]] request iwrite_shared(T&& value) {
      const context ctx{"mp1::file::iwrite_shared"};

      return detail::file_write_request<std::remove_cvref_t<T>>(
          state_, false, true, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_shared_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite_shared`: borrowed contiguous buffer; borrowable<R&&> rejects
    /// an rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    [[nodiscard]] request iwrite_shared(R&& buffer) {
      const context ctx{"mp1::file::iwrite_shared"};

      return detail::file_write_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, false, true, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_shared_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite_shared`: one region occurrence; the layout is duplicated
    /// into the request lease.
    template<class T>
    [[nodiscard]] request iwrite_shared(region<T> buffer) {
      const context ctx{"mp1::file::iwrite_shared"};

      return detail::file_write_request<std::remove_cv_t<T>>(
          state_, false, true, ctx, detail::file_send_region(buffer, true, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_shared_c(handle, data, count, datatype, native);
          });
    }
    /// write family `iwrite_all`: value form; borrowable<T&&> rejects an rvalue
    /// scalar (the request retains the address until completion).
    template<element_value T>
    requires(borrowable<T &&>)
    [[nodiscard]] request iwrite_all(T&& value) {
      const context ctx{"mp1::file::iwrite_all"};

      return detail::file_write_request<std::remove_cvref_t<T>>(
          state_, true, false, ctx, detail::file_send_value(value),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_all_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite_all`: borrowed contiguous buffer; borrowable<R&&> rejects
    /// an rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    [[nodiscard]] request iwrite_all(R&& buffer) {
      const context ctx{"mp1::file::iwrite_all"};

      return detail::file_write_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_send_range(buffer, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_all_c(handle, data, count, datatype, native);
          });
    }

    /// write family `iwrite_all`: one region occurrence; the layout is duplicated
    /// into the request lease.
    template<class T>
    [[nodiscard]] request iwrite_all(region<T> buffer) {
      const context ctx{"mp1::file::iwrite_all"};

      return detail::file_write_request<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_region(buffer, true, ctx),
          [](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
             MPI_Request* native) {
            return MPI_File_iwrite_all_c(handle, data, count, datatype, native);
          });
    }
    /// write family `iwrite_at_all`: value form; borrowable<T&&> rejects an rvalue
    /// scalar (the request retains the address until completion).
    template<element_value T>
    requires(borrowable<T &&>)
    [[nodiscard]] request iwrite_at_all(offset_t off, T&& value) {
      const context ctx{"mp1::file::iwrite_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<std::remove_cvref_t<T>>(
          state_, true, false, ctx, detail::file_send_value(value),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `iwrite_at_all`: borrowed contiguous buffer; borrowable<R&&> rejects
    /// an rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    [[nodiscard]] request iwrite_at_all(offset_t off, R&& buffer) {
      const context ctx{"mp1::file::iwrite_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<
          std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, true, false, ctx, detail::file_send_range(buffer, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_all_c(handle, off, data, count, datatype, native);
          });
    }

    /// write family `iwrite_at_all`: one region occurrence; the layout is duplicated
    /// into the request lease.
    template<class T>
    [[nodiscard]] request iwrite_at_all(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::iwrite_at_all"};
      detail::require_file_offset(off, ctx);
      return detail::file_write_request<std::remove_cv_t<T>>(
          state_, true, false, ctx, detail::file_send_region(buffer, true, ctx),
          [off](MPI_File handle, const void* data, MPI_Count count, MPI_Datatype datatype,
                MPI_Request* native) {
            return MPI_File_iwrite_at_all_c(handle, off, data, count, datatype, native);
          });
    }
    /// write split begin `write_at_all_begin`: value form; borrowable<T&&> rejects an
    /// rvalue scalar (the begin records its address until the matching end).
    template<element_value T>
    requires(borrowable<T &&>)
    void write_at_all_begin(offset_t off, T&& value) {
      const context ctx{"mp1::file::write_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_send_value(value);
      detail::file_begin<std::remove_cvref_t<T>>(
          state_, detail::split_kind::write_at_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [off, data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_at_all_begin_c(handle, off, data, count, datatype);
          });
    }

    /// write split begin `write_at_all_begin`: borrowed buffer; borrowable<R&&> rejects an
    /// rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    void write_at_all_begin(offset_t off, R&& buffer) {
      const context ctx{"mp1::file::write_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_send_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::write_at_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [off, data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_at_all_begin_c(handle, off, data, count, datatype);
          });
    }

    /// write split begin `write_at_all_begin`: region (read-only use); the layout is
    /// duplicated into the split epoch; end it with a span over the region.
    template<class T>
    void write_at_all_begin(offset_t off, region<T> buffer) {
      const context ctx{"mp1::file::write_at_all_begin"};
      detail::require_file_offset(off, ctx);
      auto transfer = detail::file_send_region(buffer, true, ctx);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::write_at_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [off, data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_at_all_begin_c(handle, off, data, count, datatype);
          });
    }
    /// write split begin `write_all_begin`: value form; borrowable<T&&> rejects an
    /// rvalue scalar (the begin records its address until the matching end).
    template<element_value T>
    requires(borrowable<T &&>)
    void write_all_begin(T&& value) {
      const context ctx{"mp1::file::write_all_begin"};

      auto transfer = detail::file_send_value(value);
      detail::file_begin<std::remove_cvref_t<T>>(
          state_, detail::split_kind::write_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_all_begin_c(handle, data, count, datatype);
          });
    }

    /// write split begin `write_all_begin`: borrowed buffer; borrowable<R&&> rejects an
    /// rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    void write_all_begin(R&& buffer) {
      const context ctx{"mp1::file::write_all_begin"};

      auto transfer = detail::file_send_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::write_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_all_begin_c(handle, data, count, datatype);
          });
    }

    /// write split begin `write_all_begin`: region (read-only use); the layout is
    /// duplicated into the split epoch; end it with a span over the region.
    template<class T>
    void write_all_begin(region<T> buffer) {
      const context ctx{"mp1::file::write_all_begin"};

      auto transfer = detail::file_send_region(buffer, true, ctx);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::write_all, transfer.data, true, false, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_all_begin_c(handle, data, count, datatype);
          });
    }
    /// write split begin `write_ordered_begin`: value form; borrowable<T&&> rejects an
    /// rvalue scalar (the begin records its address until the matching end).
    template<element_value T>
    requires(borrowable<T &&>)
    void write_ordered_begin(T&& value) {
      const context ctx{"mp1::file::write_ordered_begin"};

      auto transfer = detail::file_send_value(value);
      detail::file_begin<std::remove_cvref_t<T>>(
          state_, detail::split_kind::write_ordered, transfer.data, true, true, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_ordered_begin_c(handle, data, count, datatype);
          });
    }

    /// write split begin `write_ordered_begin`: borrowed buffer; borrowable<R&&> rejects an
    /// rvalue container but accepts an lvalue container or a borrowed span.
    template<class R>
    requires(element_buffer<R &&> && borrowable<R &&>)
    void write_ordered_begin(R&& buffer) {
      const context ctx{"mp1::file::write_ordered_begin"};

      auto transfer = detail::file_send_range(buffer, ctx);
      detail::file_begin<std::remove_cv_t<std::ranges::range_value_t<std::remove_cvref_t<R>>>>(
          state_, detail::split_kind::write_ordered, transfer.data, true, true, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_ordered_begin_c(handle, data, count, datatype);
          });
    }

    /// write split begin `write_ordered_begin`: region (read-only use); the layout is
    /// duplicated into the split epoch; end it with a span over the region.
    template<class T>
    void write_ordered_begin(region<T> buffer) {
      const context ctx{"mp1::file::write_ordered_begin"};

      auto transfer = detail::file_send_region(buffer, true, ctx);
      detail::file_begin<std::remove_cv_t<T>>(
          state_, detail::split_kind::write_ordered, transfer.data, true, true, ctx,
          std::move(transfer.layout_keepalive),
          [data = transfer.data, count = transfer.count,
           datatype = transfer.datatype](MPI_File handle) {
            return MPI_File_write_ordered_begin_c(handle, data, count, datatype);
          });
    }
    /// split end `read_at_all_end`: value form (the end must match the begin).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_at_all_end(T& value) {
      T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::read_at_all, buffer,
                              "mp1::file::read_at_all_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_at_all_end(handle, buffer, native);
                              });
    }

    /// split end `read_at_all_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_at_all_end(R&& buffer) {
      void* const data = detail::file_read_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::read_at_all, data,
                              "mp1::file::read_at_all_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_at_all_end(handle, data, native);
                              });
    }
    /// split end `read_all_end`: value form (the end must match the begin).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_all_end(T& value) {
      T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::read_all, buffer,
                              "mp1::file::read_all_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_all_end(handle, buffer, native);
                              });
    }

    /// split end `read_all_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_all_end(R&& buffer) {
      void* const data = detail::file_read_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::read_all, data,
                              "mp1::file::read_all_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_all_end(handle, data, native);
                              });
    }
    /// split end `read_ordered_end`: value form (the end must match the begin).
    template<element_value T>
    requires(!std::is_const_v<T>)
    status read_ordered_end(T& value) {
      T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::read_ordered, buffer,
                              "mp1::file::read_ordered_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_ordered_end(handle, buffer, native);
                              });
    }

    /// split end `read_ordered_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
             !is_region_v<std::remove_cvref_t<R>>)
    status read_ordered_end(R&& buffer) {
      void* const data = detail::file_read_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::read_ordered, data,
                              "mp1::file::read_ordered_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_read_ordered_end(handle, data, native);
                              });
    }
    /// split end `write_at_all_end`: value form (the end must match the begin).
    template<element_value T>
    status write_at_all_end(const T& value) {
      const T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::write_at_all, buffer,
                              "mp1::file::write_at_all_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_at_all_end(handle, buffer, native);
                              });
    }

    /// split end `write_at_all_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(element_buffer<R &&>)
    status write_at_all_end(const R& buffer) {
      const void* const data = detail::file_send_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::write_at_all, data,
                              "mp1::file::write_at_all_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_at_all_end(handle, data, native);
                              });
    }
    /// split end `write_all_end`: value form (the end must match the begin).
    template<element_value T>
    status write_all_end(const T& value) {
      const T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::write_all, buffer,
                              "mp1::file::write_all_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_all_end(handle, buffer, native);
                              });
    }

    /// split end `write_all_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(element_buffer<R &&>)
    status write_all_end(const R& buffer) {
      const void* const data = detail::file_send_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::write_all, data,
                              "mp1::file::write_all_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_all_end(handle, data, native);
                              });
    }
    /// split end `write_ordered_end`: value form (the end must match the begin).
    template<element_value T>
    status write_ordered_end(const T& value) {
      const T* const buffer = std::addressof(value);
      return detail::file_end(state_, detail::split_kind::write_ordered, buffer,
                              "mp1::file::write_ordered_end",
                              [buffer](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_ordered_end(handle, buffer, native);
                              });
    }

    /// split end `write_ordered_end`: borrowed buffer (same address as its begin).
    template<class R>
    requires(element_buffer<R &&>)
    status write_ordered_end(const R& buffer) {
      const void* const data = detail::file_send_end_buffer(buffer);
      return detail::file_end(state_, detail::split_kind::write_ordered, data,
                              "mp1::file::write_ordered_end",
                              [data](MPI_File handle, MPI_Status* native) {
                                return MPI_File_write_ordered_end(handle, data, native);
                              });
    }

  private:
    std::shared_ptr<detail::file_state> state_;
  };

}  // namespace mp1

#endif
