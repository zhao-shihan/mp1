// Standalone external-finalize scenario: an application calls MPI_Finalize() directly (the
// mp1 external-lifecycle case) while file handles are open.
//
// MPI 4.1 section 11.2.2: once MPI_Finalize returns, no MPI procedure may be called. The
// wrapper's close() must therefore throw without touching the handle, and the destructor must
// abandon the handle without an MPI call. MPI 4.1 section 14.2.1 requires files to be closed
// before finalize; this scenario documents the fallback behavior for the application error.
//
// Return codes: 0 success, 10+ a specific failed assertion. The executable is a CTest
// scenario (its own main); it never calls mp1::finalize().

#include "mp1/mp1.hpp"
#include "mpi_wrap.hpp"

#include <cstdio>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>


namespace {

  int fail(const char* message) {
    std::fprintf(stderr, "file_external_finalize: %s\n", message);
    return 10;
  }

}  // namespace

int main(int argc, char** argv) {
  mp1::init(argc, argv);

  const std::filesystem::path first =
      std::filesystem::temp_directory_path() / "mp1_external_finalize_release.bin";
  const std::filesystem::path second =
      std::filesystem::temp_directory_path() / "mp1_external_finalize_abandon.bin";
  std::error_code error;
  std::filesystem::remove(first, error);
  std::filesystem::remove(second, error);

  std::optional<mp1::file> released_file{std::in_place};
  std::optional<mp1::file> abandoned_file{std::in_place};
  released_file->open(mp1::comm_world(), first,
                      mp1::access_mode::read_write | mp1::access_mode::create);
  abandoned_file->open(mp1::comm_world(), second,
                       mp1::access_mode::read_write | mp1::access_mode::create);
  (void)released_file->size();  // prime the cached lifecycle fast path

  // External finalize, deliberately not through mp1.
  if (MPI_Finalize() != MPI_SUCCESS)
    return fail("MPI_Finalize failed");
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();

  bool released_close_threw{false};
  try {
    released_file->close();
  } catch (const std::runtime_error&) {
    released_close_threw = true;
  } catch (...) {
    return fail("close threw the wrong exception type after external finalize");
  }
  if (not released_close_threw)
    return fail("close did not throw after external finalize");
  if (not released_file->is_open())
    return fail("a failed close changed the open state");
  // The settled-phase guard is one MPI_Finalized query (Table 11.1), never the redundant
  // MPI_Initialized pair; publishing the observed finalize makes the later paths free.
  if (mpi_wrap::supported()) {
    if (mpi_wrap::count("MPI_Initialized") != 0)
      return fail("a release path issued MPI_Initialized after the phase settled");
    if (mpi_wrap::count("MPI_Finalized") != 1)
      return fail("the first close did not use exactly one MPI_Finalized guard");
  }

  // release() is the no-MPI escape: it hands the raw handle out and counts no abandonment.
  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  MPI_File handed_back = released_file->release();
  if (handed_back == MPI_FILE_NULL)
    return fail("release returned MPI_FILE_NULL for an open handle");
  if (released_file->is_open())
    return fail("release left the wrapper open");
  if (released_file->release() != MPI_FILE_NULL)
    return fail("a second release returned a handle");

  bool abandoned_close_threw{false};
  try {
    abandoned_file->close();
  } catch (const std::runtime_error&) {
    abandoned_close_threw = true;
  } catch (...) {
    return fail("close threw the wrong exception type on the abandoned handle");
  }
  if (not abandoned_close_threw)
    return fail("close did not throw for the abandoned handle");

  // The destructor must make no MPI call and count exactly one abandoned handle.
  abandoned_file.reset();
  if (mp1::detail::abandoned_handle_count() != abandoned_before + 1)
    return fail("the after-finalize destructor did not abandon exactly one handle");
  if (mpi_wrap::supported() and mpi_wrap::count("MPI_Finalized") != 1)
    return fail("a later release path repeated the lifecycle finalize query");

  // The external raw handle cannot be closed after finalize; the OS reclaims it at exit.
  (void)handed_back;
  std::filesystem::remove(first, error);
  std::filesystem::remove(second, error);
  return 0;
}
