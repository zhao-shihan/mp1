// Barrier tests: blocking and nonblocking collective synchronization.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>


bool barrier_test() {
  const auto comm_world = mp1::comm_world();
  mp1::barrier(comm_world);
  return true;
}


bool ibarrier_test() {
  const auto comm_world = mp1::comm_world();
  auto r{mp1::ibarrier(comm_world)};
  r.wait();
  return true;
}


BOOST_AUTO_TEST_CASE(barrier) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(barrier_test());
  BOOST_TEST(ibarrier_test());
}
