// API consumer skeleton.  This is not part of the in-tree build; the
// package CI job configures test/consumer out of tree against an installed mp1
// prefix, with CMAKE_CXX_STANDARD=14 in the consumer project.  The point is to
// prove that mp1::mp1 raises the consumer to C++20 and exposes the
// lifecycle/communicator API and the version macros.

#include <cstdio>
#include <functional>
#include <mp1/mp1.hpp>

static_assert(__cplusplus >= 202002L, "mp1::mp1 must require C++20 for consumers");

int main() {
  mp1::init();

  auto world = mp1::comm_world();
  if (!world.is_valid()) {
    std::fprintf(stderr, "consumer: comm_world() is invalid\n");
    mp1::finalize();
    return 1;
  }
  if (world.size() < 1) {
    std::fprintf(stderr, "consumer: comm_world().size() < 1\n");
    mp1::finalize();
    return 2;
  }

  // The installed headers must expose both the native-MPI_Op and the functor reduction
  // forms; std::plus<int> maps to MPI_SUM through the operator adapter.
  const int local = world.rank();
  const int expected = world.size() * (world.size() - 1) / 2;
  int native_total = 0;
  int functor_total = 0;
  mp1::allreduce(world, MPI_SUM, local, native_total);
  mp1::allreduce(world, std::plus<int>{}, local, functor_total);
  if (native_total != expected or functor_total != expected) {
    std::fprintf(stderr, "consumer: reduction mismatch (native=%d functor=%d expected=%d)\n",
                 native_total, functor_total, expected);
    mp1::finalize();
    return 3;
  }

  std::printf("mp1 %s: rank %d of %d\n", MP1_VERSION_STRING, world.rank(), world.size());
  mp1::finalize();
  return 0;
}
