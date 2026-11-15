// Positive control for the rvalue-send and persistent-container probes: a named container is
// borrowable, and the persistent form accepts a fixed std::array.
#include "mp1/mp1.hpp"

#include <array>
#include <functional>
#include <span>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::vector<int> send(2, 0);
  std::array<int, 2> recv{0, 0};
  (void)mp1::igather(c, 0, send, std::span<int>{recv});
  (void)mp1::iallgather(c, send, std::span<int>{recv});
  (void)mp1::gather_init(c, 0, std::span<int>{send}, std::span<int>{recv});
  std::array<int, 2> fixed_send{0, 0};
  (void)mp1::gather_init(c, 0, fixed_send, std::span<int>{recv});
  (void)mp1::reduce_scatter_init(c, std::plus<int>{}, std::span<int>{send},
                                 std::span<int>{recv}, std::span<const mp1::count_t>{});
}
