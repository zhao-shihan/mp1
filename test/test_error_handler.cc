// Scenario test for predefined-only error-handler control.
//
// MPI 4.1 basis: §9.3 defines MPI_ERRORS_ARE_FATAL / MPI_ERRORS_ABORT /
// MPI_ERRORS_RETURN, says a get behaves as if a new error-handler object is created and
// MPI_ERRHANDLER_FREE should be called on the returned reference (Table 11.1 makes
// MPI_ERRHANDLER_FREE always available), and states that windows and files do not inherit
// the initial handler; §9.3.1/§9.3.2/§9.3.3 give the communicator/window/file set/get
// procedures; §12.6.1 associates MPI_ERRORS_ARE_FATAL with a new window; §14.7 makes
// MPI_ERRORS_RETURN the default file handler.  §2.8 says errors not associated with an
// object are raised on MPI_COMM_SELF, which is why install_recoverable_errors() sets both
// WORLD and SELF.
//
// One process, one rank; it provides its own main so it can probe before init and after
// finalize.  MPI_Errhandler_free is wrapped by the test harness to count the free of the
// returned reference.

#include "mp1/mp1.hpp"

#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
#endif


namespace {

  int failure_count = 0;

  void report_failure(const char* file, int line, const char* expression) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    ++failure_count;
  }

#define MP1_CHECK(condition)                          \
  do {                                                \
    if (not(condition))                               \
      report_failure(__FILE__, __LINE__, #condition); \
  } while (false)

  template<class F>
  bool throws_runtime_error(F&& function) {
    try {
      function();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_invalid_argument(F&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // A user-created handler installed through raw MPI: mp1 must classify it as non-predefined
  // (nullopt) while still freeing the reference returned by the get.
  void custom_comm_errhandler(MPI_Comm* comm, int* error_code, ...) {
    (void)comm;
    (void)error_code;
  }

  std::filesystem::path scratch_path(const char* name) {
    return std::filesystem::temp_directory_path() / (std::string{"mp1_error_handler_"} + name);
  }

  bool matches(const std::optional<mp1::error_handler>& handler, mp1::error_handler expected) {
    return handler.has_value() and *handler == expected;
  }

  bool contains(const std::string& text, std::string_view needle) {
    return text.find(needle) != std::string::npos;
  }

  // The closed error taxonomy and the check-site `where` of the error model.
  //
  // Every helper is exercised at a known call site, so the diagnostic must name the check
  // line in this file and never the public entry of an operation.  The library-side checks
  // (`span_from`, p2p, file) additionally prove that `where` names a library header.
  void check_error_model_local() {
    const mp1::context ctx{"mp1::error_probe"};

    // fail: std::invalid_argument, `mp1: <op> (<file>:<line>): <formatted message>`.
    std::string message;
    try {
      mp1::fail(ctx, std::source_location::current(), "value {} is invalid", 7);
    } catch (const std::invalid_argument& error) {
      message = error.what();
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(message.starts_with("mp1: mp1::error_probe ("));
    MP1_CHECK(contains(message, "): value 7 is invalid"));
    MP1_CHECK(contains(message, "test_error_handler.cc"));

    // require(false): std::invalid_argument with the condition's message; require(true) is a
    // no-op that does not format.
    message.clear();
    try {
      mp1::require(1 == 2, ctx, std::source_location::current(), "condition {}", false);
    } catch (const std::invalid_argument& error) {
      message = error.what();
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(contains(message, "): condition false"));
    mp1::require(true, ctx, std::source_location::current(), "must not be formatted {}", 1);

    // throw_invalid: the static-message path, still std::invalid_argument.
    message.clear();
    try {
      mp1::throw_invalid("mp1::error_probe", "static message");
    } catch (const std::invalid_argument& error) {
      message = error.what();
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(message.starts_with("mp1: mp1::error_probe ("));
    MP1_CHECK(contains(message, "): static message"));

    // mpi_error: code, class, operation, slot, where and the code/class suffix.  MPI's
    // error-string queries are always available, so this works before MPI_Init.
    const std::source_location here = std::source_location::current();
    const mp1::mpi_error direct{MPI_ERR_ARG, "mp1::error_probe", 3, here};
    MP1_CHECK(direct.code() == MPI_ERR_ARG);
    MP1_CHECK(direct.error_class() == MPI_ERR_ARG);
    MP1_CHECK(direct.is_class(MPI_ERR_ARG));
    MP1_CHECK(direct.operation() == "mp1::error_probe");
    MP1_CHECK(direct.slot() == 3);
    MP1_CHECK(direct.where().line() == here.line());
    MP1_CHECK(std::string_view{direct.where().file_name()} == here.file_name());
    // The class name is provider-defined prose; only its presence is portable.
    MP1_CHECK(not direct.class_name().empty());
    const std::string direct_what = direct.what();
    MP1_CHECK(direct_what.starts_with("mp1: mp1::error_probe ("));
    MP1_CHECK(contains(direct_what, ", slot=3]"));
    MP1_CHECK(contains(direct_what, " [code="));

    // The single-completion form has slot -1 and no slot suffix.
    const mp1::mpi_error single{MPI_SUCCESS, "mp1::error_probe"};
    MP1_CHECK(single.slot() == -1);
    MP1_CHECK(not contains(single.what(), "slot="));

    // A library-side check reports the library header and line, not this caller.
    message.clear();
    try {
      (void)mp1::span_from<int>(nullptr, -1);
    } catch (const std::invalid_argument& error) {
      message = error.what();
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(message.starts_with("mp1: mp1::span_from ("));
    MP1_CHECK(contains(message, "vocabulary.hpp:"));
    MP1_CHECK(not contains(message, "test_error_handler.cc"));
    MP1_CHECK(contains(message, "): count -1 is negative"));

    // Checked conversions preserve their exception types.
    MP1_CHECK(throws_invalid_argument([&] { (void)mp1::detail::checked_mpi_count(-1, ctx); }));
    bool overflow = false;
    try {
      (void)mp1::detail::checked_count_add(std::numeric_limits<mp1::count_t>::max(), 1, ctx);
    } catch (const std::overflow_error&) {
      overflow = true;
    }
    MP1_CHECK(overflow);
  }

  // Library-side MPI failures report the check line of the library header.  Requires an
  // active MPI environment and a returning handler.
  void check_error_model_runtime() {
    // p2p envelope validation: the check site is p2p.hpp, not this test file.
    std::string message;
    try {
      mp1::send(mp1::comm_world(), std::string{"x"}, -7);
    } catch (const std::invalid_argument& error) {
      message = error.what();
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(message.starts_with("mp1: mp1::send ("));
    MP1_CHECK(contains(message, "p2p.hpp:"));
    MP1_CHECK(not contains(message, "test_error_handler.cc"));
    MP1_CHECK(contains(message, "is outside [0,"));

    // An MPI failure surfaces as mpi_error with the library check site.  Files default to
    // MPI_ERRORS_RETURN (MPI 4.1 section 14.7), so a missing file is a returned error code.
    const std::filesystem::path missing = scratch_path("missing_for_error_model");
    std::error_code ignored;
    std::filesystem::remove(missing, ignored);
    bool got_mpi_error = false;
    try {
      mp1::file failing;
      failing.open(mp1::comm_self(), missing, mp1::access_mode::read_only);
    } catch (const mp1::mpi_error& error) {
      got_mpi_error = true;
      MP1_CHECK(error.code() != MPI_SUCCESS);
      MP1_CHECK(error.operation() == "MPI_File_open");
      const std::string_view file = error.where().file_name();
      MP1_CHECK(not file.empty());
      MP1_CHECK(contains(std::string{file}, "file.hpp"));
      MP1_CHECK(not contains(std::string{file}, "test_error_handler.cc"));
      MP1_CHECK(std::string{error.what()}.starts_with("mp1: MPI_File_open ("));
      MP1_CHECK(error.slot() == -1);
    } catch (...) {
      MP1_CHECK(false);
    }
    MP1_CHECK(got_mpi_error);
  }

}  // namespace


int main() {
  // ---- before init ---------------------------------------------------------------------
  // There is no deferred view: wrapping a valid raw handle while MPI is inactive throws
  // std::runtime_error at construction.  An invalid view is local misuse
  // (std::invalid_argument) and stays MPI-free.
  MP1_CHECK(throws_runtime_error([] {
    const mp1::mpi_communicator active_handle{MPI_COMM_WORLD};
    (void)active_handle;
  }));
  MP1_CHECK(throws_runtime_error([] { mp1::install_recoverable_errors(); }));
  const mp1::mpi_communicator invalid_comm;
  const mp1::mpi_window invalid_window;
  MP1_CHECK(throws_invalid_argument(
      [&] { mp1::set_error_handler(invalid_comm, mp1::error_handler::returns); }));
  MP1_CHECK(throws_invalid_argument([&] { (void)mp1::get_error_handler(invalid_comm); }));
  MP1_CHECK(throws_invalid_argument(
      [&] { mp1::set_error_handler(invalid_window, mp1::error_handler::returns); }));
  MP1_CHECK(throws_invalid_argument([&] { (void)mp1::get_error_handler(invalid_window); }));

  check_error_model_local();

  mp1::init();

  // ---- communicator set/get and the free of the returned reference ----------------------
  MP1_CHECK(matches(mp1::get_error_handler(mp1::comm_world()), mp1::error_handler::are_fatal));

  mp1::communicator copy = mp1::communicator::dup(mp1::comm_world());
  const mp1::error_handler handlers[] = {
      mp1::error_handler::are_fatal, mp1::error_handler::abort, mp1::error_handler::returns};
  for (const mp1::error_handler handler : handlers) {
    mp1::set_error_handler(copy, handler);
    MP1_CHECK(matches(mp1::get_error_handler(copy), handler));
  }
  mp1::set_error_handler(copy, mp1::error_handler::returns);

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    (void)mp1::get_error_handler(copy);
    MP1_CHECK(mpi_wrap::count("MPI_Errhandler_free") == 1);
  }
#endif

  // A handler created by raw MPI is not one of the predefined handles: nullopt, still freed.
  MPI_Errhandler custom{MPI_ERRHANDLER_NULL};
  MP1_CHECK(MPI_Comm_create_errhandler(&custom_comm_errhandler, &custom) == MPI_SUCCESS);
  MP1_CHECK(custom != MPI_ERRHANDLER_NULL);
  MP1_CHECK(MPI_Comm_set_errhandler(copy.native_handle(), custom) == MPI_SUCCESS);
#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    MP1_CHECK(not mp1::get_error_handler(copy).has_value());
    MP1_CHECK(mpi_wrap::count("MPI_Errhandler_free") == 1);
  } else
#endif
  {
    MP1_CHECK(not mp1::get_error_handler(copy).has_value());
  }
  // The reference handed to MPI_Comm_set_errhandler stays caller-owned; the one returned by
  // the get was already freed by mp1.
  MP1_CHECK(MPI_Errhandler_free(&custom) == MPI_SUCCESS);
  copy.free();

  // ---- window set/get (default MPI_ERRORS_ARE_FATAL, MPI 4.1 §12.6.1) ------------------
  mp1::window window = mp1::window::allocate(mp1::comm_self(), {.size = 0, .disp_unit = 1});
  MP1_CHECK(matches(mp1::get_error_handler(window), mp1::error_handler::are_fatal));
  mp1::set_error_handler(window, mp1::error_handler::returns);
  MP1_CHECK(matches(mp1::get_error_handler(window), mp1::error_handler::returns));

  // ---- file set/get (default MPI_ERRORS_RETURN, MPI 4.1 §14.7) -------------------------
  const std::filesystem::path path = scratch_path("recoverable.bin");
  std::error_code remove_error;
  std::filesystem::remove(path, remove_error);
  mp1::file file;
  file.open(mp1::comm_self(), path, mp1::access_mode::read_write | mp1::access_mode::create);
  MP1_CHECK(matches(file.get_error_handler(), mp1::error_handler::returns));
  file.set_error_handler(mp1::error_handler::are_fatal);
  MP1_CHECK(matches(file.get_error_handler(), mp1::error_handler::are_fatal));
  file.set_error_handler(mp1::error_handler::returns);

  // One free per get across all three object classes.
#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    (void)mp1::get_error_handler(mp1::comm_world());
    (void)mp1::get_error_handler(window);
    (void)file.get_error_handler();
    MP1_CHECK(mpi_wrap::count("MPI_Errhandler_free") == 3);
  }
#endif

  // With a returning handler attached, a raw invalid file view raises an error code instead
  // of aborting (the file default is already MPI_ERRORS_RETURN, §14.7).
  {
    const int rc = MPI_File_set_view(file.native_handle(), 0, MPI_BYTE, MPI_BYTE,
                                     "not-a-real-representation", MPI_INFO_NULL);
    MP1_CHECK(rc != MPI_SUCCESS);
  }

  file.close();
  std::filesystem::remove(path, remove_error);
  window.free();

  // ---- install_recoverable_errors: both WORLD and SELF, idempotent, returning ------------
  mp1::install_recoverable_errors();
  MP1_CHECK(matches(mp1::get_error_handler(mp1::comm_world()), mp1::error_handler::returns));
  MP1_CHECK(matches(mp1::get_error_handler(mp1::comm_self()), mp1::error_handler::returns));
  mp1::install_recoverable_errors();  // idempotent
  check_error_model_runtime();
  // An invalid communicator is an MPI error; with returning handlers this must return a code
  // (the same sequence aborts under the default MPI_ERRORS_ARE_FATAL handler).
  {
    int rank{0};
    MP1_CHECK(MPI_Comm_rank(MPI_COMM_NULL, &rank) != MPI_SUCCESS);
  }

  // ---- after finalize ------------------------------------------------------------------
  const mp1::mpi_communicator stale_comm = mp1::comm_world();
  mp1::mpi_window stale_window;
  {
    mp1::window temporary =
        mp1::window::allocate(mp1::comm_self(), {.size = 0, .disp_unit = 1});
    stale_window = temporary;
    temporary.free();
  }
  mp1::file closed_file;

  mp1::finalize();

  MP1_CHECK(throws_runtime_error([&] { (void)mp1::get_error_handler(stale_comm); }));
  MP1_CHECK(throws_runtime_error(
      [&] { mp1::set_error_handler(stale_comm, mp1::error_handler::returns); }));
  MP1_CHECK(throws_runtime_error([&] { (void)mp1::get_error_handler(stale_window); }));
  MP1_CHECK(throws_runtime_error(
      [&] { mp1::set_error_handler(stale_window, mp1::error_handler::returns); }));
  MP1_CHECK(throws_runtime_error([&] { (void)closed_file.get_error_handler(); }));
  MP1_CHECK(throws_runtime_error(
      [&] { closed_file.set_error_handler(mp1::error_handler::returns); }));
  MP1_CHECK(throws_runtime_error([] { mp1::install_recoverable_errors(); }));

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported())
    MP1_CHECK(mpi_wrap::after_finalize_calls() == 0);
#endif

  if (failure_count != 0) {
    std::fprintf(stderr, "test_error_handler: %d failure(s)\n", failure_count);
    return 1;
  }
  return 0;
}
