// Owned accept-any init on a MULTIPLE-only provider (MPI 4.1 section 11.2.1).
//
// MPI_INIT has the same effect as MPI_INIT_THREAD with required = MPI_THREAD_SINGLE, but a
// conforming implementation started with only a higher level available installs that level
// instead ("a call to MPI_INIT will also initialize the MPI thread support level to
// MPI_THREAD_MULTIPLE").  This executable replaces MPI_Init with one that forwards to
// MPI_Init_thread(MULTIPLE) so mp1's accept-any init path runs against a process that
// really is initialized at MPI_THREAD_MULTIPLE.  provided() must
// record the level MPI actually provides, and the explicit init(multiple) must not throw.
//
// This needs a process-wide MPI_Init override, so it is its own executable (registered as a
// rank-1 scenario) rather than a test_environment scenario.

#include "mp1/impl/environment.hpp"

#include <cstdio>
#include <cstdlib>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
#endif


// Process-wide lifecycle shim: MPI_Init installs MPI_THREAD_MULTIPLE.  The main executable's
// strong definition interposes the provider's MPI_Init on both supported providers.
extern "C" int MPI_Init(int* argc, char*** argv) {
  int provided = MPI_THREAD_SINGLE;
  return MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
}

namespace {

  int failure_count = 0;

  void check(bool condition, const char* expression, int line) {
    if (not condition) {
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, line, expression);
      ++failure_count;
    }
  }

#define MP1_CHECK(condition) check((condition), #condition, __LINE__)

  bool throws_runtime_error(void (*function)()) {
    try {
      function();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  void require_multiple() {
    mp1::init(mp1::threading_mode::multiple);
  }

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  MP1_CHECK(not mp1::initialized());
  MP1_CHECK(mp1::provided() == mp1::threading_mode::single);

  {
    // The accept-any form calls MPI_Init; the shim makes the provider initialize MULTIPLE.
    mp1::environment environment;
    MP1_CHECK(mp1::initialized());
    MP1_CHECK(mp1::available());

    const mp1::threading_mode live_level = mp1::query_thread();
    // The defect: provided() used to record an assumed SINGLE instead of the live level.
    MP1_CHECK(mp1::provided() == live_level);
    // A thread-compliant provider grants the shim's MULTIPLE requirement; if it cannot, the
    // requirement must fail correctly.  The equality above is provider-independent.
    if (live_level >= mp1::threading_mode::multiple) {
      MP1_CHECK(mp1::provided() == mp1::threading_mode::multiple);
      MP1_CHECK(not throws_runtime_error(&require_multiple));
      MP1_CHECK(mp1::provided() == mp1::query_thread());
    } else {
      MP1_CHECK(throws_runtime_error(&require_multiple));
    }

    // The --wrap counter cannot observe the accept-any entry point here: the executable
    // defines MPI_Init itself (the shim), so mp1's inline call binds to that definition and
    // the link-time wrapper for MPI_Init is bypassed.  The call-entry assertion for the
    // accept-any form lives in test_environment_owned, which has no override.

    mp1::finalize();
    MP1_CHECK(mp1::finalized());
    MP1_CHECK(not mp1::available());
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      MP1_CHECK(mpi_wrap::after_finalize_calls() == 0);
#endif
  }

  // The environment destructor ran after finalize: it must be a no-op (owned but inactive).
  MP1_CHECK(mp1::finalized());

  if (failure_count != 0) {
    std::fprintf(stderr, "test_environment_owned_multiple: %d failure(s)\n", failure_count);
    return 1;
  }
  return 0;
}
