#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <span>


int main() {
  mp1::environment env;
  const auto world{mp1::comm_world()};
  const int rank{world.rank()};

  // disp_unit is in bytes here: target_disp is scaled by the target process's disp_unit.
  constexpr mp1::count_t n{8};
  auto win = mp1::window::allocate(
      world, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* local{static_cast<int*>(win.base())};
  for (mp1::count_t i{0}; i < n; ++i)
    local[i] = rank * 100 + static_cast<int>(i);

  // Active target: one collective fence opens the epoch, the next one completes it.
  mp1::fence(win);
  const int send[2]{1000, 1001};
  if (rank == 0)
    mp1::put(win, std::span<const int>{send, 2}, 1, 3, 2);  // 2 ints to rank 1, element 3
  mp1::fence(win);

  int received[2]{-1, -1};
  if (rank == 0)
    mp1::get(win, std::span<int>{received, 2}, 1, 3, 2);
  mp1::fence(win);  // data is valid only after this fence
  if (rank == 0)
    std::cout << "fence get: " << received[0] << ' ' << received[1] << '\n';

  // Passive target: every rank accumulates one into rank 0's element 0.
  if (rank == 0)
    local[0] = 0;
  mp1::fence(win);
  const int one{1};
  mp1::lock(win, mp1::lock_type::exclusive, 0);
  mp1::accumulate(win, std::span<const int>{&one, 1}, 0, 0, 1, MPI_SUM);
  mp1::unlock(win, 0);  // completes origin and target for rank 0
  mp1::fence(win);      // target-side synchronization before a local load
  if (rank == 0)
    std::cout << "lock accumulate total: " << local[0] << '\n';

  win.free();  // collective; may block until passive-target accesses complete
  return EXIT_SUCCESS;
}
