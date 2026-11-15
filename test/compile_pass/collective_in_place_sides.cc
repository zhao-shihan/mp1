// Positive control for the collective wrong-side probes: the sentinel is accepted in the
// MPI-legal slot of each family, and a dummy span in the non-root ignored position compiles.
#include "mp1/mp1.hpp"

#include <functional>
#include <span>

void control(const mp1::mpi_communicator& c) {
  int scalar = 0;
  int array[4] = {0, 0, 0, 0};
  int recv[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  mp1::gather(c, 0, mp1::in_place, std::span<int>{recv});
  mp1::gather(c, 0, scalar, std::span<int>{});
  mp1::scatter(c, 0, std::span<const int>{array}, mp1::in_place);
  mp1::scatter(c, 0, std::span<const int>{}, std::span<int>{array, 1});
  mp1::allgather(c, mp1::in_place, std::span<int>{recv});
  mp1::alltoall(c, mp1::in_place, std::span<int>{recv});
  mp1::reduce(c, std::plus<int>{}, 0, mp1::in_place, std::span<int>{array});
  mp1::reduce(c, std::plus<int>{}, 0, scalar, std::span<int>{});
  mp1::allreduce(c, std::plus<int>{}, mp1::in_place, std::span<int>{array});
  mp1::scan(c, std::plus<int>{}, mp1::in_place, std::span<int>{array});
  mp1::exscan(c, std::plus<int>{}, mp1::in_place, std::span<int>{array});
  mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place, std::span<int>{recv},
                      std::span<const mp1::count_t>{});
  mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place, std::span<int>{recv}, 0);
  mp1::reduce_local(std::plus<int>{}, std::span<const int>{array}, std::span<int>{array});
}
