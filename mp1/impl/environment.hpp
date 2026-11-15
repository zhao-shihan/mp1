#if !(defined MP1_IMPL_ENVIRONMENT_HPP)

#define MP1_IMPL_ENVIRONMENT_HPP

#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/info.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>


namespace mp1 {

  /// Level of thread support provided by the underlying MPI implementation.
  enum class threading_mode : int {
    single = MPI_THREAD_SINGLE,
    funneled = MPI_THREAD_FUNNELED,
    serialized = MPI_THREAD_SERIALIZED,
    multiple = MPI_THREAD_MULTIPLE
  };

  static_assert(MPI_THREAD_SINGLE < MPI_THREAD_FUNNELED and
                    MPI_THREAD_FUNNELED < MPI_THREAD_SERIALIZED and
                    MPI_THREAD_SERIALIZED < MPI_THREAD_MULTIPLE,
                "MPI thread levels must be monotonically ordered");

  /// \return true if MPI_Init or MPI_Init_thread has been called (legal before init and after
  /// finalize). Observes an MPI_Finalize issued outside mp1.
  inline bool initialized() noexcept {
    if (detail::mpi_finalized())
      detail::publish_finalized();
    return detail::mpi_initialized();
  }

  /// \return true if MPI_Finalize has completed (legal before init and after finalize)
  inline bool finalized() noexcept {
    if (detail::mpi_finalized()) {
      detail::publish_finalized();
      return true;
    }
    return false;
  }

  /// \return initialized() && !finalized(); observes an external MPI_Finalize
  inline bool available() noexcept {
    return detail::mpi_active_checked();
  }

  /// \return pair of MPI major and minor version; legal before init and after finalize
  inline std::pair<int, int> get_version() {
    int major{0};
    int minor{0};
    MPI_Get_version(&major, &minor);
    return {major, minor};
  }

  /// \return implementation library version string; legal before init and after finalize
  inline std::string get_library_version() {
    char buffer[MPI_MAX_LIBRARY_VERSION_STRING + 1] = {};
    int length{0};
    (void)MPI_Get_library_version(buffer, &length);
    length = std::max(length, 0);
    if (length > MPI_MAX_LIBRARY_VERSION_STRING)
      length = MPI_MAX_LIBRARY_VERSION_STRING;
    buffer[length] = '\0';
    return std::string{buffer};
  }

  /// MPI_Get_hw_resource_info: hardware resource information as an owning info table.
  ///
  /// The key/value interpretation is implementation-defined (for example `hwloc://...` keys
  /// on Open MPI); the table may be empty or null on implementations that do not report
  /// resources (for example MPICH). Requires active MPI; this is not an always-available
  /// operation (use info::create_env for a table that works before init and after finalize).
  [[nodiscard]] inline info get_hw_resource_info() {
    detail::require_mpi("mp1::get_hw_resource_info");
    MPI_Info handle{MPI_INFO_NULL};
    detail::check_mpi(MPI_Get_hw_resource_info(&handle), {"MPI_Get_hw_resource_info"});
    return detail::info_access::adopt(handle);
  }

  namespace detail {

    inline threading_mode query_thread_level() {
      int level{MPI_THREAD_SINGLE};
      check_mpi(MPI_Query_thread(&level), {"MPI_Query_thread"});
      return static_cast<threading_mode>(level);
    }

    /// Required-level enforcement.  MPI guarantees MPI_THREAD_SINGLE < FUNNELED < SERIALIZED <
    /// MULTIPLE, so an integer comparison is valid.
    inline void enforce_thread_level(threading_mode required, threading_mode provided_level,
                                     bool owned) {
      if (static_cast<int>(provided_level) >= static_cast<int>(required))
        return;
      std::string message =
          std::format("required thread level {} but MPI provided {}",
                      static_cast<int>(required), static_cast<int>(provided_level));
      if (owned) {
        message +=
            "; MPI remains initialized and is owned by mp1: call mp1::finalize() on all ranks";
      }
      detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                            message);
    }

    /// Shared body of the four init overloads.  argc/argv are null for the no-argument forms;
    /// required is empty for the accept-any-level forms (which use MPI_Init and record the
    /// level MPI actually provides with MPI_Query_thread, not an assumed SINGLE).
    ///
    /// Returns true iff this call performed the initialization (and therefore owns MPI).  The
    /// lifecycle mutex is held across MPI_Init*/MPI_Query_thread so concurrent init callers
    /// block and then observe the completed transition; exactly one MPI_Init* call is issued.
    inline bool init_impl(int* argc, char*** argv, std::optional<threading_mode> required) {
      std::lock_guard<std::mutex> lock{lifecycle_mutex()};
      auto& phase = lifecycle_phase_word();
      switch (phase.load(std::memory_order_acquire)) {
        case lifecycle_phase::active_owned:
        case lifecycle_phase::active_adopted: {
          threading_mode level = static_cast<threading_mode>(
              provided_thread_level().load(std::memory_order_acquire));
          if (phase.load(std::memory_order_acquire) == lifecycle_phase::active_adopted) {
            // The hot-path adoption records the actual level with MPI_Query_thread; an
            // explicit init() re-queries it anyway so a level that could not be recorded
            // there (MPI_Query_thread failed) is repaired.  This preserves the explicit-init
            // behavior unchanged.
            try {
              level = query_thread_level();
            } catch (...) {
              level = threading_mode::single;
            }
            provided_thread_level().store(static_cast<int>(level), std::memory_order_release);
          }
          if (required)
            enforce_thread_level(*required, level, /*owned=*/false);
          return false;
        }
        case lifecycle_phase::finalizing:
        case lifecycle_phase::finalized:
          detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                                "init after finalize");
        case lifecycle_phase::init_failed:
          detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                                "previous MPI initialization failed");
        case lifecycle_phase::initializing:
          detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                                "MPI initialization is in progress");
        case lifecycle_phase::idle:
          break;
      }

      if (mpi_finalized()) {
        phase.store(lifecycle_phase::finalized, std::memory_order_release);
        detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                              "init after finalize");
      }

      if (mpi_initialized()) {
        // Adoption: the application owns MPI.  The query may fail only in exotic
        // implementations; record the conservative SINGLE level and continue.
        threading_mode level = threading_mode::single;
        try {
          level = query_thread_level();
        } catch (...) {
          level = threading_mode::single;
        }
        provided_thread_level().store(static_cast<int>(level), std::memory_order_release);
        phase.store(lifecycle_phase::active_adopted, std::memory_order_release);
        if (required)
          enforce_thread_level(*required, level, /*owned=*/false);
        return false;
      }

      phase.store(lifecycle_phase::initializing, std::memory_order_release);
      int provided_level = MPI_THREAD_SINGLE;
      int rc = MPI_SUCCESS;
      const char* operation = "MPI_Init";
      if (required) {
        operation = "MPI_Init_thread";
        rc = MPI_Init_thread(argc, argv, static_cast<int>(*required), &provided_level);
      } else {
        rc = MPI_Init(argc, argv);
        if (rc == MPI_SUCCESS) {
          // MPI 4.1 §11.2.1: MPI_Init has the same effect as MPI_Init_thread with
          // required = SINGLE, but a provider configured with only a higher level (for
          // example a MULTIPLE-only launch) initializes that level instead.  Record what
          // MPI actually provides; on a failed query keep the conservative SINGLE.
          int queried = MPI_THREAD_SINGLE;
          if (MPI_Query_thread(&queried) == MPI_SUCCESS)
            provided_level = queried;
        }
      }
      if (rc != MPI_SUCCESS) {
        phase.store(lifecycle_phase::init_failed, std::memory_order_release);
        throw mpi_error{rc, operation};
      }
      provided_thread_level().store(provided_level, std::memory_order_release);
      phase.store(lifecycle_phase::active_owned, std::memory_order_release);
      if (required) {
        enforce_thread_level(*required, static_cast<threading_mode>(provided_level),
                             /*owned=*/true);
      }
      return true;
    }

  }  // namespace detail

  /// Initialize MPI.
  ///
  /// MPI 4.1 §11.2.1: MPI_INIT has the same effect as MPI_INIT_THREAD with
  /// required = MPI_THREAD_SINGLE; the thread levels are monotonic.  The default forms
  /// therefore call MPI_Init and record the level MPI actually provides via MPI_Query_thread
  /// (§11.2.1 permits MPI_INIT to install a higher level when only that level is available;
  /// an implementation-defined query failure keeps the conservative SINGLE).  This is the
  /// thread-safe lifecycle default, not an MPI_THREAD_MULTIPLE promise: use
  /// init(threading_mode::multiple) when the program uses mp1 from several threads.  The call
  /// is serialized with every other init/finalize; concurrent init callers all return after the
  /// single MPI_Init.  A second call while MPI is active is a no-op (it adopts an MPI
  /// environment initialized by the application without taking ownership).  Throws
  /// std::runtime_error after finalize and mp1::mpi_error if MPI_Init fails.
  ///
  /// There is no implicit finalize at exit: call finalize(), hold an environment object, or
  /// opt in with mp1::finalize_at_exit().
  inline void init() {
    detail::init_impl(nullptr, nullptr, std::nullopt);
  }

  /// Initialize MPI, letting MPI remove its arguments (MPI_Init; see init()).
  inline void init(int& argc, char**& argv) {
    detail::init_impl(&argc, &argv, std::nullopt);
  }

  /// Initialize MPI with a required thread level (MPI_Init_thread).
  ///
  /// Throws std::runtime_error when the provided level is lower than required.  On the
  /// mp1-owned path MPI is left initialized and mp1 owns it: the exception message says so and
  /// all ranks must still call mp1::finalize().  mp1 never rolls back with MPI_Finalize, which
  /// is collective over connected processes.
  inline void init(threading_mode required) {
    detail::init_impl(nullptr, nullptr, required);
  }

  /// Initialize MPI with a required thread level, updating argc/argv (see init(threading_mode)).
  inline void init(int& argc, char**& argv, threading_mode required) {
    detail::init_impl(&argc, &argv, required);
  }

  /// Finalize MPI; only legal for MPI initialized by mp1.  May block (collective).
  ///
  /// MPI 4.1 §11.2.2: MPI_FINALIZE is collective over all connected processes, does not free
  /// objects created by MPI calls, and after it returns no World-Model procedure may be called
  /// except the §11.4.1/Table 11.1 functions.
  ///
  /// Live MPI objects at the call (windows, files, requests, spawned/connected
  /// intercommunicators) are the application's MPI precondition, not an mp1-managed resource:
  /// MPI_FINALIZE does not free them, and mp1 does not count, refuse, diagnose or release them.
  /// The caller must complete every request, close every RMA epoch, free every window, close
  /// every file and disconnect every spawned/connected intercommunicator before calling this
  /// function.
  ///
  /// Throws std::runtime_error when MPI is not active, was not initialized by mp1 (the external
  /// owner must finalize it), or is already finalized; throws mp1::mpi_error if MPI_Finalize
  /// fails (the process is then treated as finalized; a failed collective finalize is not
  /// retried).
  ///
  /// Finalization order (MPI 4.1 §11.2.2/§11.6.2): all blocking operations returned, every
  /// request complete/inactive/released, every RMA epoch closed, every window freed, every file
  /// closed (including split epochs), every spawned intercommunicator disconnected on both
  /// ends, and every thread that used mp1 joined before this call.
  inline void finalize() {
    const context ctx{"mp1::finalize"};
    std::lock_guard<std::mutex> lock{detail::lifecycle_mutex()};
    auto& phase = detail::lifecycle_phase_word();
    if (detail::mpi_finalized()) {
      phase.store(detail::lifecycle_phase::finalized, std::memory_order_release);
      detail::fail_with<std::runtime_error>(ctx, std::source_location::current(),
                                            "MPI is already finalized");
    }
    switch (phase.load(std::memory_order_acquire)) {
      case detail::lifecycle_phase::idle:
      case detail::lifecycle_phase::active_adopted:
        if (not detail::mpi_initialized())
          detail::fail_with<std::runtime_error>(ctx, std::source_location::current(),
                                                "MPI is not active");
        detail::fail_with<std::runtime_error>(ctx, std::source_location::current(),
                                              "MPI was not initialized by mp1");
      case detail::lifecycle_phase::init_failed:
        detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                              "previous MPI initialization failed");
      case detail::lifecycle_phase::finalizing:
      case detail::lifecycle_phase::finalized:
        detail::fail_with<std::runtime_error>(ctx, std::source_location::current(),
                                              "MPI is already finalized");
      case detail::lifecycle_phase::initializing:
        detail::fail_with<std::runtime_error>({"mp1::init"}, std::source_location::current(),
                                              "MPI initialization is in progress");
      case detail::lifecycle_phase::active_owned:
        break;
    }

    phase.store(detail::lifecycle_phase::finalizing, std::memory_order_release);
    const int rc = MPI_Finalize();
    phase.store(detail::lifecycle_phase::finalized, std::memory_order_release);
    detail::check_mpi(rc, {"MPI_Finalize"});
  }

  /// Opt-in implicit finalization at process exit (atomic, idempotent, safe before or after
  /// init; MPI 4.1 §11.2.2 requires MPI_FINALIZE before a normal exit).  Live MPI objects at
  /// exit are the same caller precondition as finalize(): the handler does not count, diagnose
  /// or release them.  Documented single-threaded only: the handler runs on the exiting thread while
  /// other threads may still be inside MPI, which crashes the installed implementations.  Prefer
  /// an explicit finalize() or an environment object.
  inline void finalize_at_exit() noexcept {
    static const bool registered = (std::atexit(&detail::finalize_at_exit) == 0);
    (void)registered;
  }

  /// \return the live level of thread support; requires active MPI
  inline threading_mode query_thread() {
    detail::require_mpi("mp1::query_thread");
    return detail::query_thread_level();
  }

  /// \return the thread level recorded by the last successful init or by the first adoption of
  /// an externally initialized MPI environment; single if none.  The accept-any-level init
  /// records the level MPI actually provides (MPI_Query_thread after MPI_Init) instead of
  /// assuming SINGLE, so this is truthful in a MULTIPLE-only configuration; adopting an MPI
  /// environment records its actual MPI_Query_thread level, so this is not stale after the
  /// first mp1 call.
  inline threading_mode provided() noexcept {
    return static_cast<threading_mode>(
        detail::provided_thread_level().load(std::memory_order_acquire));
  }

  /// \return true if the calling thread is the main thread; requires active MPI
  inline bool is_thread_main() {
    detail::require_mpi("mp1::is_thread_main");
    int flag{0};
    detail::check_mpi(MPI_Is_thread_main(&flag), {"MPI_Is_thread_main"});
    return flag != 0;
  }

  /// \return name of the processor; requires active MPI
  inline std::string processor_name() {
    detail::require_mpi("mp1::processor_name");
    char buffer[MPI_MAX_PROCESSOR_NAME + 1] = {};
    int length{0};
    detail::check_mpi(MPI_Get_processor_name(buffer, &length), {"MPI_Get_processor_name"});
    length = std::max(length, 0);
    if (length > MPI_MAX_PROCESSOR_NAME)
      length = MPI_MAX_PROCESSOR_NAME;
    buffer[length] = '\0';
    return std::string{buffer};
  }

  /// \return wall-clock time in seconds; requires active MPI
  inline double wtime() {
    detail::require_mpi("mp1::wtime");
    return MPI_Wtime();
  }

  /// \return resolution of wtime(); requires active MPI
  inline double wtick() {
    detail::require_mpi("mp1::wtick");
    return MPI_Wtick();
  }

  /// \return whether wtime() values are synchronized across MPI_COMM_WORLD; false if the
  /// attribute is absent. Requires active MPI.
  inline bool wtime_is_global() {
    detail::require_mpi("mp1::wtime_is_global");
    void* value{nullptr};
    int flag{0};
    detail::check_mpi(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &value, &flag),
                      {"MPI_Comm_get_attr"});
    return flag != 0 and value != nullptr and *static_cast<int*>(value) != 0;
  }

  /// MPI_BUFFER_AUTOMATIC: pass as the buffer argument to buffer_attach or
  /// attach_buffer(comm, buffer_automatic, 0) to let MPI manage the buffered-send buffer.
  /// Not a dereferenceable address. Not constexpr because MPI implementations define the
  /// sentinel as a cast from an integer pointer value.
  inline void* const buffer_automatic = MPI_BUFFER_AUTOMATIC;

  namespace detail {

    /// MPI 4.1 §3.6: a detached size that cannot be represented in `count_t` is reported as
    /// MPI_UNDEFINED, and the size of a detached MPI_BUFFER_AUTOMATIC buffer is undefined (any
    /// bit pattern is legal). The conversion therefore must never throw; it reports
    /// MPI_UNDEFINED instead.
    [[nodiscard]] inline count_t checked_count_or_undefined(MPI_Count size) noexcept {
      if (size < static_cast<MPI_Count>(std::numeric_limits<count_t>::min()) or
          size > static_cast<MPI_Count>(std::numeric_limits<count_t>::max()))
        return MPI_UNDEFINED;
      return static_cast<count_t>(size);
    }

  }  // namespace detail

  /// Attach a process-specific buffer for buffered sends (MPI_Buffer_attach_c, MPI 4.1 §3.6);
  /// requires active MPI.
  ///
  /// `size` is in **bytes**, not elements, and must be non-negative for an explicit buffer; a
  /// negative size throws std::invalid_argument before any MPI call. mp1 performs no
  /// MPI_BSEND_OVERHEAD arithmetic here (bsend_buffer is the convenience wrapper that does).
  /// Pass buffer_automatic as buffer to enable automatic buffering; MPI then ignores the size,
  /// so any value (including a negative one) is accepted. A negative size is normalized to 0
  /// before the MPI call because some providers validate the otherwise-ignored argument and
  /// would reject a legal automatic attach.
  ///
  /// Only one process-specific buffer may be attached at a time; attaching while one is
  /// attached is erroneous per MPI 4.1 §3.6 and the wrapper does not track it.
  ///
  /// The counted form removes the 32-bit `int` ceiling of MPI_Buffer_attach. Interaction with
  /// the flush family: buffer_flush()/buffer_iflush() flush the attached buffer and never
  /// detach it, so the buffer must stay attached until the flush completes. buffer_detach()
  /// is itself a blocking flush (MPI 4.1 §3.6): complete (or destroy, which waits for) every
  /// outstanding buffer_iflush() request before calling it. Detaching while an iflush request
  /// is still active is not supported; there is no process-buffer handle to correlate, so the
  /// wrapper cannot detect it.
  inline void buffer_attach(void* buffer, count_t size) {
    detail::require_mpi("mp1::buffer_attach");
    if (buffer == buffer_automatic) {
      // MPI ignores the size for automatic buffering (MPI 4.1 §3.6). Some providers still
      // validate it and reject a negative value, so pass 0 for a negative size: the observable
      // contract "any size is accepted with buffer_automatic" then holds on every provider.
      const MPI_Count mpi_size = size < 0 ? MPI_Count{0} : static_cast<MPI_Count>(size);
      detail::check_mpi(MPI_Buffer_attach_c(buffer, mpi_size), {"MPI_Buffer_attach_c"});
      return;
    }
    detail::check_mpi(
        MPI_Buffer_attach_c(buffer, detail::checked_mpi_count(size, "mp1::buffer_attach")),
        {"MPI_Buffer_attach_c"});
  }

  /// Detach the process-specific buffered-send buffer; may block until pending buffered sends
  /// complete (MPI_Buffer_detach_c, MPI 4.1 §3.6). Requires active MPI.
  ///
  /// \return `{address, size in bytes}`. The size is a count_t, so the old 32-bit ceiling is
  /// gone; MPI_UNDEFINED is returned when the detached size cannot be represented (or, for
  /// buffer_automatic, when MPI reports an undefined size). If buffer_automatic was attached,
  /// buffer_automatic is returned and the size is undefined: do not use it as an allocation
  /// size. As a blocking flush, this must not run while a buffer_iflush() request is still
  /// active.
  [[nodiscard]] inline std::pair<void*, count_t> buffer_detach() {
    detail::require_mpi("mp1::buffer_detach");
    void* buffer{nullptr};
    MPI_Count size{0};
    detail::check_mpi(MPI_Buffer_detach_c(&buffer, &size), {"MPI_Buffer_detach_c"});
    return {buffer, detail::checked_count_or_undefined(size)};
  }

  /// RAII guard for the one MPI buffered-send buffer.
  ///
  /// It always attaches an explicit allocation, so it cannot enable MPI_BUFFER_AUTOMATIC. Use
  /// buffer_attach(buffer_automatic, 0) and buffer_detach() for automatic buffering instead.
  class bsend_buffer {
  public:
    /// Allocate size + MPI_BSEND_OVERHEAD bytes and attach them. Throws std::invalid_argument
    /// for a negative size, std::bad_array_new_length when the attachment size overflows int,
    /// and mp1::mpi_error if MPI_Buffer_attach fails.
    explicit bsend_buffer(int size) {
      if (size < 0) {
        detail::fail_as<std::invalid_argument>({"mp1::bsend_buffer"},
                                               std::source_location::current(),
                                               "size {} is negative", size);
      }
      const long long total = static_cast<long long>(size) + MPI_BSEND_OVERHEAD;
      if (total > static_cast<long long>(std::numeric_limits<int>::max()))
        throw std::bad_array_new_length{};
      buffer_ = ::operator new(static_cast<std::size_t>(total));
      try {
        buffer_attach(buffer_, static_cast<int>(total));
      } catch (...) {
        ::operator delete(buffer_);
        buffer_ = nullptr;
        throw;
      }
      attached_ = true;
    }

    /// Detach when MPI is active, then free the allocation. Never throws.
    ~bsend_buffer() noexcept {
      if (buffer_ == nullptr)
        return;
      if (attached_ and detail::mpi_active()) {
        void* detached{nullptr};
        int size{0};
        (void)MPI_Buffer_detach(&detached, &size);
      }
      ::operator delete(buffer_);
      buffer_ = nullptr;
    }

    bsend_buffer(const bsend_buffer&) = delete;
    bsend_buffer& operator=(const bsend_buffer&) = delete;
    bsend_buffer(bsend_buffer&&) = delete;
    bsend_buffer& operator=(bsend_buffer&&) = delete;

  private:
    void* buffer_{nullptr};
    bool attached_{false};
  };

  /// Optional RAII lifecycle guard. Owns MPI only when it called init itself; an already
  /// initialized environment is adopted and never finalized. The destructor finalizes owned
  /// MPI (collective, may block), so all ranks must leave the owning scope together. Free
  /// init()/finalize() users may ignore this class.
  ///
  /// The destructor does not police the MPI finalization precondition: live MPI objects at that
  /// point (windows, files, requests, spawned/connected intercommunicators) are the caller's MPI
  /// obligation, not an mp1-managed resource, so the destructor neither counts them nor skips
  /// MPI_Finalize because of them.  It calls MPI_Finalize whenever it owns a still-active MPI
  /// environment.
  ///
  /// environment() calls MPI_Init and records the level MPI actually provides (it does not
  /// assume SINGLE); environment(threading_mode) calls MPI_Init_thread and enforces the
  /// requested level.  The destructor is noexcept and never blocks on the lifecycle mutex; it
  /// is a no-op when finalization already happened.
  class environment {
  public:
    environment() : owns_{detail::init_impl(nullptr, nullptr, std::nullopt)} {
    }

    environment(int& argc, char**& argv)
        : owns_{detail::init_impl(&argc, &argv, std::nullopt)} {
    }

    explicit environment(threading_mode required)
        : owns_{detail::init_impl(nullptr, nullptr, required)} {
    }

    environment(int& argc, char**& argv, threading_mode required)
        : owns_{detail::init_impl(&argc, &argv, required)} {
    }

    ~environment() noexcept {
      if (owns_)
        detail::finalize_noexcept();
    }

    environment(const environment&) = delete;
    environment& operator=(const environment&) = delete;
    environment(environment&&) = delete;
    environment& operator=(environment&&) = delete;

  private:
    bool owns_{false};
  };

}  // namespace mp1

#endif
