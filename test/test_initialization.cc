// Initialization test: the process starts with MPI inactive, mp1::init() initializes MPI and
// the predefined communicator views are usable, and mp1::finalize() releases MPI.
//
// The process-wide lifecycle scenarios (owned/external/none) live in test_environment.cc; this
// file keeps the original initialization intent of the test suite.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(initialization) {
  BOOST_TEST(not mp1::initialized());
  BOOST_TEST(not mp1::finalized());

  mp1::init();
  BOOST_TEST(mp1::initialized());
  BOOST_TEST(mp1::available());
  BOOST_TEST(not mp1::finalized());

  const auto comm_world = mp1::comm_world();
  BOOST_TEST(comm_world.is_valid());
  BOOST_TEST((comm_world.rank() >= 0));
  BOOST_TEST((comm_world.rank() < comm_world.size()));

  const auto comm_self = mp1::comm_self();
  BOOST_TEST(comm_self.is_valid());
  BOOST_TEST((comm_self.size() == 1));
  BOOST_TEST((comm_self.rank() == 0));

  mp1::finalize();
  BOOST_TEST(mp1::finalized());
  BOOST_TEST(not mp1::available());
}
