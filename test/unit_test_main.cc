// Shared Boost.Test main for every `mp1_add_test(... KIND unit)` target.
//
// This is the only place the Boost.Test main is defined.  It is compiled once
// into the static library `mp1_test_main`; unit-test sources include
// <boost/test/unit_test.hpp>, do not define BOOST_TEST_MODULE, and never use
// <boost/test/included/unit_test.hpp>.
//
// BOOST_TEST_ALTERNATIVE_INIT_API selects the `bool init_unit_test()` entry
// point used by the prebuilt Boost.Test libraries in the supported toolchains.
// `BOOST_TEST_NO_MAIN` suppresses the main that Boost.Test would otherwise
// provide, leaving exactly one main in mp1_test_main.

#define BOOST_TEST_MODULE mp1
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API

#include "mp1/impl/environment.hpp"

#include <boost/test/unit_test.hpp>

int main(int argc, char* argv[]) {
  // MPI 4.1 §11.2.2: each process must call MPI_Finalize before it exits.  The library has no
  // implicit finalizer; a unit-test process uses the documented opt-in escape so a test that
  // only calls mp1::init() still exits cleanly (a launcher may report an unfinalized exit as
  // abnormal termination).  Register after the framework returns so this handler runs before
  // any framework/static cleanup handler; a process manager that watches the exit path is
  // sensitive to a finalize that runs after its cleanup.
  // Tests that finalize themselves observe a no-op handler.
  const int result = boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
  mp1::finalize_at_exit();
  return result;
}
