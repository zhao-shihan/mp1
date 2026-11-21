#if !(defined MP1_IMPL_ERROR_HPP)

#define MP1_IMPL_ERROR_HPP

#include "mp1/impl/fwd.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <mpi.h>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>


namespace mp1 {

  /// Operation name of one public entry, for example "mp1::gather".
  ///
  /// A local check uses the public-entry literal, passed down from the entry.  An MPI-return
  /// check may instead carry the MPI routine name (for example "MPI_File_open"); `where()` is
  /// the check site either way.  `source_location::function_name()` is never used for the
  /// operation: it is implementation-defined and carries the full signature and template
  /// arguments.
  struct context {
    std::string_view operation;

    /// A context is almost always written as `context{"mp1::gather"}` at the public entry;
    /// the converting constructor also lets the check helpers take a bare operation literal.
    constexpr context(std::string_view op) noexcept : operation{op} {
    }

    constexpr context(const char* op) noexcept : operation{op} {
    }
  };

  namespace detail {

    inline int error_class_of(int code) noexcept {
      int error_class = code;
      if (MPI_Error_class(code, &error_class) != MPI_SUCCESS) [[unlikely]]
        error_class = code;
      return error_class;
    }

    inline std::string error_string_of(int code) {
      char buffer[MPI_MAX_ERROR_STRING + 1] = {};
      int length{0};
      if (MPI_Error_string(code, buffer, &length) != MPI_SUCCESS) [[unlikely]]
        return std::string{"MPI error "} + std::to_string(code);
      length = std::max(length, 0);
      if (length > MPI_MAX_ERROR_STRING)
        length = MPI_MAX_ERROR_STRING;
      buffer[length] = '\0';
      return std::string{buffer};
    }

    /// "mp1: <operation> (<file>:<line>): " — the common prefix of every mp1 diagnostic.
    /// `<file>:<line>` is the *check* site (the line that made the decision), not the public
    /// entry and not the caller.
    [[nodiscard]] inline std::string format_prefix(std::string_view operation,
                                                   std::source_location where) {
      std::string text{"mp1: "};
      if (not operation.empty()) {
        text += operation;
        text += ' ';
      }
      std::format_to(std::back_inserter(text), "({}:{}): ", where.file_name(), where.line());
      return text;
    }

    /// Format one diagnostic message with the check-site prefix.
    template<class... Args>
    [[nodiscard]] std::string format_message(std::string_view operation,
                                             std::source_location where,
                                             std::format_string<Args...> format,
                                             Args&&... args) {
      std::string text = format_prefix(operation, where);
      std::format_to(std::back_inserter(text), format, std::forward<Args>(args)...);
      return text;
    }

    /// Preformatted variant: the message was already assembled by the caller.
    [[nodiscard]] inline std::string format_message(std::string_view operation,
                                                    std::source_location where,
                                                    std::string_view message) {
      std::string text = format_prefix(operation, where);
      text.append(message.data(), message.size());
      return text;
    }

    /// Throw `E` with one formatted diagnostic.  The exception types of the library are
    /// preserved: local contract violations use std::invalid_argument (through mp1::fail),
    /// lifecycle misuse std::runtime_error, an unrepresentable count std::overflow_error.
    template<class E, class... Args>
    [[noreturn]] void fail_as(const context& ctx, std::source_location where,
                              std::format_string<Args...> format, Args&&... args) {
      throw E{format_message(ctx.operation, where, format, std::forward<Args>(args)...)};
    }

    /// Preformatted std::runtime_error / std::overflow_error / std::logic_error variant.
    template<class E>
    [[noreturn]] void fail_with(const context& ctx, std::source_location where,
                                std::string_view message) {
      throw E{format_message(ctx.operation, where, message)};
    }

  }  // namespace detail

  /// Run-time error reported by the MPI implementation.
  ///
  /// Thrown only when an MPI procedure returns a code other than MPI_SUCCESS; local contract
  /// violations are reported with std::invalid_argument before any MPI call, lifecycle misuse
  /// with std::runtime_error, and an unrepresentable MPI count with std::overflow_error.  There
  /// is no mp1 error hierarchy beyond this type.
  ///
  /// Fatal-handler policy (MPI 4.1 §2.8: by default an error detected during execution aborts
  /// the parallel computation, except for file operations; §9.3 defines the error-handler
  /// object model): with MPI's default handler an MPI fault aborts the job inside MPI; no
  /// exception is thrown.  mp1 never installs a handler implicitly.  mpi_error is reachable
  /// only when the application selected a returning handler (for example
  /// mp1::install_recoverable_errors()).
  ///
  /// code(), error_class(), operation(), slot() and where() are MPI-free; class_name() and
  /// what() may call MPI_Error_string, which is always available (MPI 4.1 Table 11.1) and
  /// therefore safe before init and after finalize.
  ///
  /// `where` is the library check line that detected the failure, passed as
  /// std::source_location::current() by the check helper; it is not the caller's line.  The
  /// caller's location is in the exception's stack trace.
  class mpi_error : public std::runtime_error {
  public:
    /// slot names the completed slot for a multi-completion call (0-based); -1 otherwise.
    mpi_error(int code, std::string_view operation, int slot = -1,
              std::source_location where = std::source_location::current())
        : std::runtime_error(std::string{}),
          code_{code},
          error_class_{detail::error_class_of(code)},
          operation_{operation},
          slot_{slot},
          where_{where} {
      what_ = detail::format_prefix(operation_, where_);
      what_ += detail::error_string_of(code_);
      std::format_to(std::back_inserter(what_), " [code={}, class={}:{}", code_, error_class_,
                     detail::error_string_of(error_class_));
      if (slot_ >= 0)
        std::format_to(std::back_inserter(what_), ", slot={}", slot_);
      what_ += ']';
    }

    /// \return MPI return code that produced this exception
    [[nodiscard]] int code() const noexcept {
      return code_;
    }

    /// \return MPI error class of #code
    [[nodiscard]] int error_class() const noexcept {
      return error_class_;
    }

    /// \return implementation-defined name of #error_class; safe before init and after finalize
    [[nodiscard]] std::string class_name() const {
      return detail::error_string_of(error_class_);
    }

    /// \return operation name recorded for this error: the public entry for a local check,
    /// the MPI routine name for an MPI-return check; empty when none was available
    [[nodiscard]] std::string_view operation() const noexcept {
      return operation_;
    }

    /// \return completed slot for a multi-completion error, -1 otherwise
    [[nodiscard]] int slot() const noexcept {
      return slot_;
    }

    /// \return the library check site that reported the failure
    [[nodiscard]] std::source_location where() const noexcept {
      return where_;
    }

    /// \return true if #error_class is the given MPI error class constant
    /// (for example MPI_ERR_IN_STATUS)
    [[nodiscard]] bool is_class(int error_class) const noexcept {
      return error_class_ == error_class;
    }

    [[nodiscard]] const char* what() const noexcept override {
      return what_.c_str();
    }

  private:
    int code_;
    int error_class_;
    std::string operation_;
    int slot_;
    std::source_location where_;
    std::string what_;
  };

  // -----------------------------------------------------------------------------------------
  // The error layer helpers.
  //
  // `where` precedes the format string on purpose: a defaulted source_location placed after a
  // parameter pack is a non-deduced context.  Every check passes
  // std::source_location::current() explicitly, so `where` names the check line inside the
  // library.  No public entry takes a source_location parameter.
  // -----------------------------------------------------------------------------------------

  /// Throw std::invalid_argument with the check-site diagnostic:
  /// `mp1: <operation> (<file>:<line>): <message>`.
  template<class... Args>
  [[noreturn]] void fail(const context& ctx, std::source_location where,
                         std::format_string<Args...> format, Args&&... args) {
    throw std::invalid_argument{
        detail::format_message(ctx.operation, where, format, std::forward<Args>(args)...)};
  }

  /// Local precondition: throw std::invalid_argument unless `condition` holds.
  template<class... Args>
  void require(bool condition, const context& ctx, std::source_location where,
               std::format_string<Args...> format, Args&&... args) {
    if (not condition) [[unlikely]]
      throw std::invalid_argument{
          detail::format_message(ctx.operation, where, format, std::forward<Args>(args)...)};
  }

  /// Throw std::invalid_argument from an already-assembled message.  Static messages stay
  /// std::string_view; only the cold throw path allocates.
  [[noreturn]] inline void throw_invalid(
      std::string_view operation, std::string_view message,
      std::source_location where = std::source_location::current()) {
    throw std::invalid_argument{detail::format_message(operation, where, message)};
  }

  // -----------------------------------------------------------------------------------------
  // Predefined error-handler control (MPI 4.1 §9.3).
  //
  // Only the predefined handlers are modelled: there is no owning errhandler type and no user
  // callback API.  MPI 4.1 §9.3 states that the predefined handlers are
  // MPI_ERRORS_ARE_FATAL, MPI_ERRORS_ABORT and MPI_ERRORS_RETURN, that a get "behaves as if a
  // new error handler object is created" and that MPI_ERRHANDLER_FREE should be called on the
  // returned reference (Table 11.1 makes MPI_ERRHANDLER_FREE always available).  Windows and
  // files do not inherit the initial handler (§9.3; §12.6, §14.7), and the default file
  // handler is MPI_ERRORS_RETURN; see §14.7.
  // -----------------------------------------------------------------------------------------

  /// Predefined MPI error handlers (MPI 4.1 §9.3).
  enum class error_handler : unsigned char {
    are_fatal,  // MPI_ERRORS_ARE_FATAL
    abort,      // MPI_ERRORS_ABORT, added in MPI 4.0
    returns     // MPI_ERRORS_RETURN
  };

  namespace detail {

    /// Map an enumerator to the predefined MPI handle.  An out-of-range value can only come
    /// from an explicit cast; reject it locally (std::invalid_argument).
    [[nodiscard]] inline MPI_Errhandler native_error_handler(
        const context& ctx, error_handler handler,
        std::source_location where = std::source_location::current()) {
      switch (handler) {
        case error_handler::are_fatal:
          return MPI_ERRORS_ARE_FATAL;
        case error_handler::abort:
          return MPI_ERRORS_ABORT;
        case error_handler::returns:
          return MPI_ERRORS_RETURN;
      }
      detail::fail_as<std::invalid_argument>(ctx, where, "invalid error_handler value");
    }

    /// Best-effort classification of a returned handler reference.  MPI does not define
    /// handle equality, so a value that is not one of the three predefined handles (for
    /// example a user-created handler installed by raw MPI) yields std::nullopt.
    [[nodiscard]] inline std::optional<error_handler> classify_error_handler(
        MPI_Errhandler handler) noexcept {
      if (handler == MPI_ERRORS_ARE_FATAL)
        return error_handler::are_fatal;
      if (handler == MPI_ERRORS_ABORT)
        return error_handler::abort;
      if (handler == MPI_ERRORS_RETURN)
        return error_handler::returns;
      return std::nullopt;
    }

  }  // namespace detail

  /// Attach a predefined error handler to a communicator view (MPI_Comm_set_errhandler).
  /// Non-collective and process-local.  Requires active MPI and a valid communicator.
  void set_error_handler(const mpi_communicator& comm, error_handler handler);

  /// Attach a predefined error handler to a window view (MPI_Win_set_errhandler, §9.3.2).
  /// Windows do not inherit the initial handler and start with MPI_ERRORS_ARE_FATAL
  /// (§12.6.1); requires active MPI and a valid window.
  void set_error_handler(const mpi_window& window, error_handler handler);

  /// \return the predefined handler currently attached to the communicator, or nullopt when
  /// the attached handler is not predefined (best-effort classification).  The returned MPI
  /// reference is freed with MPI_Errhandler_free before returning (§9.3).
  [[nodiscard]] std::optional<error_handler> get_error_handler(const mpi_communicator& comm);

  /// \return the predefined handler currently attached to the window, or nullopt; the
  /// returned reference is freed (§9.3).
  [[nodiscard]] std::optional<error_handler> get_error_handler(const mpi_window& window);

  /// Set MPI_COMM_WORLD and MPI_COMM_SELF to MPI_ERRORS_RETURN.  This handler makes every
  /// handled error return its code, so mp1::mpi_error becomes reachable.  Both are set because
  /// the installed implementations disagree about which communicator's handler governs an
  /// error that is not associated with a communicator (§2.8: such errors are raised on
  /// MPI_COMM_SELF).  Non-collective, process-local, idempotent; requires active MPI.  New
  /// communicators inherit MPI_COMM_WORLD's handler; windows and files do not inherit (§9.3).
  /// This overwrites any handler the application installed on WORLD/SELF.
  void install_recoverable_errors();

}  // namespace mp1

#endif
