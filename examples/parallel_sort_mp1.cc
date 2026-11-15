#include <algorithm>
#include <cstdlib>
#include <mp1/mp1.hpp>
#include <random>
#include <span>
#include <vector>


static std::random_device rd;
static std::mt19937_64 mt(rd());


double get_random() {
  std::uniform_real_distribution<double> dist(0, 1);
  return dist(mt);
}


void fill_random(std::vector<double>& v) {
  std::generate(std::begin(v), std::end(v), get_random);
}


// parallel sort algorithm for distributed memory computers
//
// algorithm works as follows:
//   1) each process draws (size-1) random samples from its local data
//   2) all processes gather local random samples => size*(size-1) samples
//   3) size*(size-1) samples are sorted locally
//   4) pick (size-1) pivot elements from the globally sorted sample
//   5) partition local data with respect to the pivot elements into size bins
//   6) redistribute data such that data in bin i goes to process with rank i
//   7) sort redistributed data locally
//
// Note that the amount of data at each process changes during the algorithm.
// In worst case, a single process may hold finally all data.
//
template<typename T>
void parallel_sort(std::vector<T>& v) {
  const auto comm_world{mp1::comm_world()};
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  std::vector<T> local_pivots, pivots(size * (size - 1));
  std::sample(begin(v), end(v), std::back_inserter(local_pivots), size - 1, mt);
  const mp1::layout<T> pivot_block = mp1::layout<T>::contiguous(size - 1);
  const mp1::layout<T> pivot_aggregate = mp1::layout<T>::contiguous(size * (size - 1));
  mp1::allgather(comm_world, mp1::region<T>{local_pivots.data(), pivot_block},
                 mp1::region<T>{pivots.data(), pivot_aggregate});
  std::sort(begin(pivots), end(pivots));
  local_pivots.clear();
  for (std::size_t i{1}; i < static_cast<std::size_t>(size); ++i)
    local_pivots.push_back(pivots[i * (size - 1)]);
  swap(local_pivots, pivots);
  std::vector<typename std::vector<T>::iterator> pivot_pos;
  pivot_pos.push_back(begin(v));
  for (T p : pivots)
    pivot_pos.push_back(std::partition(pivot_pos.back(), end(v), [p](T x) { return x < p; }));
  pivot_pos.push_back(end(v));
  std::vector<int> local_block_sizes, block_sizes(size * size);
  for (std::size_t i{0}; i < pivot_pos.size() - 1; ++i)
    local_block_sizes.push_back(
        static_cast<int>(std::distance(pivot_pos[i], pivot_pos[i + 1])));
  const mp1::layout<int> size_block = mp1::layout<int>::contiguous(size);
  const mp1::layout<int> size_aggregate = mp1::layout<int>::contiguous(size * size);
  mp1::allgather(comm_world, mp1::region<int>{local_block_sizes.data(), size_block},
                 mp1::region<int>{block_sizes.data(), size_aggregate});
  std::vector<mp1::layout<T>> send_layouts, recv_layouts;
  int send_pos{0}, recv_pos{0};
  for (int i{0}; i < size; ++i) {
    mp1::indexed_parameter send_parameter;
    send_parameter.add(block_sizes[rank * size + i], send_pos);
    send_layouts.push_back(mp1::layout<T>::indexed(send_parameter));
    send_pos += block_sizes[rank * size + i];
    mp1::indexed_parameter recv_parameter;
    recv_parameter.add(block_sizes[rank + size * i], recv_pos);
    recv_layouts.push_back(mp1::layout<T>::indexed(recv_parameter));
    recv_pos += block_sizes[rank + size * i];
  }
  std::vector<T> v_2(recv_pos);
  mp1::alltoallv(comm_world, v, send_layouts, std::span<const mp1::count_t>{}, v_2,
                 recv_layouts);
  std::sort(begin(v_2), end(v_2));
  swap(v, v_2);
}


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  const int size{comm_world.size()};

  const std::size_t N{100000000 / static_cast<std::size_t>(size)};
  std::vector<double> v(N);
  fill_random(v);
  parallel_sort(v);
  return EXIT_SUCCESS;
}
