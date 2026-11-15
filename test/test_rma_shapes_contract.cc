// RMA 42/42 origin/result/target shape contract.
//
// The six simple data-movement families (put, get, accumulate, rput, rget, raccumulate) have
// four shapes each = 24: origin {span, region} x target {count_t, layout<T>}.  The two
// fetching families (get_accumulate, rget_accumulate) have eight shapes each = 16:
// origin {span, region} x result {span, region} x target {count_t, layout<T>}.  The two
// single-element atomics add 1 + 1.  Total 24 + 16 + 2 = 42, expressed by the 18 public
// declarations.
//
// Every call targets the calling rank's own window segment inside an exclusive passive-target
// epoch, so the expected values are exact without inter-rank races.  After the epoch the
// documented synchronization sequence makes the target-side update visible for the local
// checks (MPI 4.1 §12.7 rules 4 and 6; 12.7.4 volatile loads).
//
// MPI 4.1 basis: §12.3.1 (put), §12.3.2 (get), §12.3.4 (accumulate/get_accumulate/FOP/CAS),
// §12.3.5 (request-based RMA), §12.7.1/§12.7.2 (atomicity and ordering), §12.5.3/§12.5.4
// (passive-target epochs).  Runs at 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_RMA_SHAPES_WRAP 1
#else
#define MP1_RMA_SHAPES_WRAP 0
#endif


namespace {

  template<class Function>
  bool throws_invalid_argument(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // Window memory can be updated by another process; a volatile load prevents the compiler
  // from caching it across synchronization calls (MPI 4.1 12.7.4).
  [[nodiscard]] int window_load(const int* p) {
    return *static_cast<const volatile int*>(p);
  }

  // Make all origin-side completion and target updates visible before the target rank reads
  // its own window memory.
  void sync_after_passive(const mp1::mpi_communicator& c, const mp1::mpi_window& w) {
    mp1::barrier(c);
    mp1::lock_all(w);
    mp1::sync(w);
    mp1::unlock_all(w);
    mp1::barrier(c);
  }

  // One span over the first `count` elements of `data` (the documented raw-buffer spelling).
  template<class T>
  [[nodiscard]] std::span<T> span_of(T* data, mp1::count_t count) {
    return std::span<T>{data, static_cast<std::size_t>(count)};
  }

  struct rma_shapes_fixture {
    rma_shapes_fixture() {
      mp1::init();
    }

    ~rma_shapes_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(rma_shapes_fixture);

BOOST_AUTO_TEST_SUITE(rma_shapes_contract)

BOOST_AUTO_TEST_CASE(put_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = -1;

  mp1::fence(w);
  mp1::barrier(c);

  // 1. span origin, flat target count.
  const int flat[2] = {1, 2};
  // 2. region origin, flat target count.
  const int region_flat[2] = {3, 4};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  // 3. span origin, layout target (elements 4 and 6).
  const int flat_to_layout[2] = {5, 6};
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  // 4. region origin with a hole, layout target (elements 8 and 11).
  const int region_origin[3] = {7, -99, 8};
  const mp1::layout<int> origin_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> target_stride_three = mp1::layout<int>::vector(2, 1, 3);

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  mp1::put(w, span_of(flat, 2), rank, 0, 2);
  mp1::put(w, mp1::region{region_flat, two}, rank, 2, 2);
  mp1::put(w, span_of(flat_to_layout, 2), rank, 4, target_stride_two);
  mp1::put(w, mp1::region{region_origin, origin_stride_two}, rank, 8, target_stride_three);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 2);
  BOOST_TEST(window_load(target + 2) == 3);
  BOOST_TEST(window_load(target + 3) == 4);
  BOOST_TEST(window_load(target + 4) == 5);
  BOOST_TEST(window_load(target + 5) == -1);
  BOOST_TEST(window_load(target + 6) == 6);
  BOOST_TEST(window_load(target + 7) == -1);
  BOOST_TEST(window_load(target + 8) == 7);
  BOOST_TEST(window_load(target + 9) == -1);
  BOOST_TEST(window_load(target + 10) == -1);
  BOOST_TEST(window_load(target + 11) == 8);
  w.free();
}

BOOST_AUTO_TEST_CASE(get_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 200 + i;

  mp1::fence(w);
  mp1::barrier(c);

  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> result_stride_two = mp1::layout<int>::vector(2, 1, 2);

  int flat[2] = {-1, -1};
  int region_flat[2] = {-1, -1};
  int flat_to_layout[2] = {-1, -1};
  int region_result[3] = {-1, -99, -1};

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // 1. span result, flat target count.
  mp1::get(w, span_of(flat, 2), rank, 32, 2);
  // 2. region result, flat target count.
  mp1::get(w, mp1::region{region_flat, two}, rank, 34, 2);
  // 3. span result, layout target.
  mp1::get(w, span_of(flat_to_layout, 2), rank, 36, target_stride_two);
  // 4. region result with a hole, layout target.
  mp1::get(w, mp1::region{region_result, result_stride_two}, rank, 40, two);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  BOOST_TEST(flat[0] == 232);
  BOOST_TEST(flat[1] == 233);
  BOOST_TEST(region_flat[0] == 234);
  BOOST_TEST(region_flat[1] == 235);
  BOOST_TEST(flat_to_layout[0] == 236);
  BOOST_TEST(flat_to_layout[1] == 238);
  BOOST_TEST(region_result[0] == 240);
  BOOST_TEST(region_result[2] == 241);
  w.free();
}

BOOST_AUTO_TEST_CASE(accumulate_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;

  mp1::fence(w);
  mp1::barrier(c);

  const int one[2] = {1, 1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const int region_one[2] = {1, 1};
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const int region_origin[3] = {1, -99, 1};
  const mp1::layout<int> origin_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> target_stride_three = mp1::layout<int>::vector(2, 1, 3);

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // 1. span origin, flat target count.
  mp1::accumulate(w, span_of(one, 2), rank, 0, 2, MPI_SUM);
  // 2. region origin, flat target count.
  mp1::accumulate(w, mp1::region{region_one, two}, rank, 2, 2, MPI_SUM);
  // 3. span origin, layout target.
  mp1::accumulate(w, span_of(one, 2), rank, 4, target_stride_two, MPI_SUM);
  // 4. region origin, layout target.
  mp1::accumulate(w, mp1::region{region_origin, origin_stride_two}, rank, 8,
                  target_stride_three, MPI_SUM);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 1);
  BOOST_TEST(window_load(target + 2) == 1);
  BOOST_TEST(window_load(target + 3) == 1);
  BOOST_TEST(window_load(target + 4) == 1);
  BOOST_TEST(window_load(target + 5) == 0);
  BOOST_TEST(window_load(target + 6) == 1);
  BOOST_TEST(window_load(target + 8) == 1);
  BOOST_TEST(window_load(target + 9) == 0);
  BOOST_TEST(window_load(target + 10) == 0);
  BOOST_TEST(window_load(target + 11) == 1);
  w.free();
}

BOOST_AUTO_TEST_CASE(rput_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const int one[2] = {1, 1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const int region_origin[3] = {1, -99, 1};
  const mp1::layout<int> origin_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> target_stride_three = mp1::layout<int>::vector(2, 1, 3);

  std::vector<mp1::request> requests;
  mp1::lock_all(w);
  // 1. span origin, flat target count.
  requests.push_back(mp1::rput(w, span_of(one, 2), rank, 0, 2));
  // 2. region origin, flat target count.
  requests.push_back(mp1::rput(w, mp1::region{one, two}, rank, 2, 2));
  // 3. span origin, layout target.
  requests.push_back(mp1::rput(w, span_of(one, 2), rank, 4, target_stride_two));
  // 4. region origin, layout target.
  requests.push_back(mp1::rput(w, mp1::region{region_origin, origin_stride_two}, rank, 8,
                               target_stride_three));
  (void)mp1::wait_all(requests);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 1);
  BOOST_TEST(window_load(target + 2) == 1);
  BOOST_TEST(window_load(target + 3) == 1);
  BOOST_TEST(window_load(target + 4) == 1);
  BOOST_TEST(window_load(target + 5) == 0);
  BOOST_TEST(window_load(target + 6) == 1);
  BOOST_TEST(window_load(target + 8) == 1);
  BOOST_TEST(window_load(target + 9) == 0);
  BOOST_TEST(window_load(target + 10) == 0);
  BOOST_TEST(window_load(target + 11) == 1);
  w.free();
}

BOOST_AUTO_TEST_CASE(rget_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 200 + i;
  mp1::fence(w);
  mp1::barrier(c);

  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> result_stride_two = mp1::layout<int>::vector(2, 1, 2);

  int flat[2] = {-1, -1};
  int region_flat[2] = {-1, -1};
  int flat_to_layout[2] = {-1, -1};
  int region_result[3] = {-1, -99, -1};
  std::vector<mp1::request> requests;

  mp1::lock_all(w);
  // 1. span result, flat target count.
  requests.push_back(mp1::rget(w, span_of(flat, 2), rank, 32, 2));
  // 2. region result, flat target count.
  requests.push_back(mp1::rget(w, mp1::region{region_flat, two}, rank, 34, 2));
  // 3. span result, layout target.
  requests.push_back(mp1::rget(w, span_of(flat_to_layout, 2), rank, 36, target_stride_two));
  // 4. region result with a hole, layout target.
  requests.push_back(
      mp1::rget(w, mp1::region{region_result, result_stride_two}, rank, 40, two));
  (void)mp1::wait_all(requests);
  mp1::flush_all(w);
  mp1::unlock_all(w);

  BOOST_TEST(flat[0] == 232);
  BOOST_TEST(flat[1] == 233);
  BOOST_TEST(region_flat[0] == 234);
  BOOST_TEST(region_flat[1] == 235);
  BOOST_TEST(flat_to_layout[0] == 236);
  BOOST_TEST(flat_to_layout[1] == 238);
  BOOST_TEST(region_result[0] == 240);
  BOOST_TEST(region_result[2] == 241);
  w.free();
}

BOOST_AUTO_TEST_CASE(raccumulate_four_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const int one[2] = {1, 1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> target_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const int region_origin[3] = {1, -99, 1};
  const mp1::layout<int> origin_stride_two = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> target_stride_three = mp1::layout<int>::vector(2, 1, 3);

  std::vector<mp1::request> requests;
  mp1::lock_all(w);
  // 1. span origin, flat target count.
  requests.push_back(mp1::raccumulate(w, span_of(one, 2), rank, 0, 2, MPI_SUM));
  // 2. region origin, flat target count.
  requests.push_back(mp1::raccumulate(w, mp1::region{one, two}, rank, 2, 2, MPI_SUM));
  // 3. span origin, layout target.
  requests.push_back(mp1::raccumulate(w, span_of(one, 2), rank, 4, target_stride_two, MPI_SUM));
  // 4. region origin, layout target.
  requests.push_back(mp1::raccumulate(w, mp1::region{region_origin, origin_stride_two}, rank, 8,
                                      target_stride_three, MPI_SUM));
  (void)mp1::wait_all(requests);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 1);
  BOOST_TEST(window_load(target + 2) == 1);
  BOOST_TEST(window_load(target + 3) == 1);
  BOOST_TEST(window_load(target + 4) == 1);
  BOOST_TEST(window_load(target + 5) == 0);
  BOOST_TEST(window_load(target + 6) == 1);
  BOOST_TEST(window_load(target + 8) == 1);
  BOOST_TEST(window_load(target + 9) == 0);
  BOOST_TEST(window_load(target + 10) == 0);
  BOOST_TEST(window_load(target + 11) == 1);
  w.free();
}

BOOST_AUTO_TEST_CASE(get_accumulate_eight_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 32;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 10;
  mp1::fence(w);
  mp1::barrier(c);

  const int one[2] = {1, 1};
  const int region_one[2] = {1, 1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> result_hole_layout = mp1::layout<int>::vector(2, 1, 2);
  int result[8][2] = {};
  int result_hole[8][3] = {};

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // 1. span origin, span result, flat target count.
  mp1::get_accumulate(w, span_of(one, 2), span_of(result[0], 2), rank, 0, 2, MPI_SUM);
  // 2. span origin, span result, layout target.
  mp1::get_accumulate(w, span_of(one, 2), span_of(result[1], 2), rank, 0, two, MPI_SUM);
  // 3. span origin, region result, flat target count.
  mp1::get_accumulate(w, span_of(one, 2), mp1::region{result[2], two}, rank, 0, 2, MPI_SUM);
  // 4. span origin, region result, layout target.
  mp1::get_accumulate(w, span_of(one, 2), mp1::region{result[3], two}, rank, 0, two, MPI_SUM);
  // 5. region origin, span result, flat target count.
  mp1::get_accumulate(w, mp1::region{region_one, two}, span_of(result[4], 2), rank, 0, 2,
                      MPI_SUM);
  // 6. region origin, span result, layout target.
  mp1::get_accumulate(w, mp1::region{region_one, two}, span_of(result[5], 2), rank, 0, two,
                      MPI_SUM);
  // 7. region origin, region result (hole), flat target count.
  mp1::get_accumulate(w, mp1::region{region_one, two},
                      mp1::region{result_hole[6], result_hole_layout}, rank, 0, 2, MPI_SUM);
  // 8. region origin, region result (hole), layout target.
  mp1::get_accumulate(w, mp1::region{region_one, two},
                      mp1::region{result_hole[7], result_hole_layout}, rank, 0, two, MPI_SUM);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  for (int shape = 0; shape < 6; ++shape) {
    BOOST_TEST(result[shape][0] == 10 + shape);
    BOOST_TEST(result[shape][1] == 10 + shape);
  }
  BOOST_TEST(result_hole[6][0] == 16);
  BOOST_TEST(result_hole[6][2] == 16);
  BOOST_TEST(result_hole[7][0] == 17);
  BOOST_TEST(result_hole[7][2] == 17);
  BOOST_TEST(window_load(target + 0) == 18);
  BOOST_TEST(window_load(target + 1) == 18);
  w.free();
}

BOOST_AUTO_TEST_CASE(rget_accumulate_eight_shapes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 32;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 10;
  mp1::fence(w);
  mp1::barrier(c);

  const int one[2] = {1, 1};
  const int region_one[2] = {1, 1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> result_hole_layout = mp1::layout<int>::vector(2, 1, 2);
  int result[8][2] = {};
  int result_hole[8][3] = {};
  std::vector<mp1::request> requests;

  mp1::lock_all(w);
  // 1. span origin, span result, flat target count.
  requests.push_back(
      mp1::rget_accumulate(w, span_of(one, 2), span_of(result[0], 2), rank, 0, 2, MPI_SUM));
  // 2. span origin, span result, layout target.
  requests.push_back(
      mp1::rget_accumulate(w, span_of(one, 2), span_of(result[1], 2), rank, 0, two, MPI_SUM));
  // 3. span origin, region result, flat target count.
  requests.push_back(mp1::rget_accumulate(w, span_of(one, 2), mp1::region{result[2], two}, rank,
                                          0, 2, MPI_SUM));
  // 4. span origin, region result, layout target.
  requests.push_back(mp1::rget_accumulate(w, span_of(one, 2), mp1::region{result[3], two}, rank,
                                          0, two, MPI_SUM));
  // 5. region origin, span result, flat target count.
  requests.push_back(mp1::rget_accumulate(w, mp1::region{region_one, two},
                                          span_of(result[4], 2), rank, 0, 2, MPI_SUM));
  // 6. region origin, span result, layout target.
  requests.push_back(mp1::rget_accumulate(w, mp1::region{region_one, two},
                                          span_of(result[5], 2), rank, 0, two, MPI_SUM));
  // 7. region origin, region result (hole), flat target count.
  requests.push_back(mp1::rget_accumulate(w, mp1::region{region_one, two},
                                          mp1::region{result_hole[6], result_hole_layout}, rank,
                                          0, 2, MPI_SUM));
  // 8. region origin, region result (hole), layout target.
  requests.push_back(mp1::rget_accumulate(w, mp1::region{region_one, two},
                                          mp1::region{result_hole[7], result_hole_layout}, rank,
                                          0, two, MPI_SUM));
  (void)mp1::wait_all(requests);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  for (int shape = 0; shape < 6; ++shape) {
    BOOST_TEST(result[shape][0] == 10 + shape);
    BOOST_TEST(result[shape][1] == 10 + shape);
  }
  BOOST_TEST(result_hole[6][0] == 16);
  BOOST_TEST(result_hole[6][2] == 16);
  BOOST_TEST(result_hole[7][0] == 17);
  BOOST_TEST(result_hole[7][2] == 17);
  BOOST_TEST(window_load(target + 0) == 18);
  BOOST_TEST(window_load(target + 1) == 18);
  w.free();
}

BOOST_AUTO_TEST_CASE(single_element_atomics) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  target[0] = 5;
  target[1] = 7;
  mp1::fence(w);
  mp1::barrier(c);

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // fetch_and_op is one shape: pointer origin, pointer result, predefined op.
  const int one = 1;
  int fetched = -1;
  mp1::fetch_and_op(w, &one, &fetched, rank, 0, MPI_SUM);
  // compare_and_swap is one shape: pointer origin, compare and result.
  const int replacement = 9;
  const int expected = 6;  // 5 + 1
  int cas_old = -1;
  mp1::compare_and_swap(w, &replacement, &expected, &cas_old, rank, 0);
  // A non-matching compare leaves the target unchanged.
  const int wrong = 100;
  const int unchanged = 100;
  int second_old = -1;
  mp1::compare_and_swap(w, &wrong, &unchanged, &second_old, rank, 1);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  BOOST_TEST(fetched == 5);
  BOOST_TEST(cas_old == 6);
  BOOST_TEST(second_old == 7);
  BOOST_TEST(window_load(target + 0) == 9);
  BOOST_TEST(window_load(target + 1) == 7);
  w.free();
}

BOOST_AUTO_TEST_CASE(overlap_and_no_op_controls) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 10;
  mp1::fence(w);
  mp1::barrier(c);

  int buffer[elements] = {5, 5, 5, 5, 5, 5, 5, 5};
  int result[2] = {-1, -1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // The both-flat form rejects identical and partially overlapping byte ranges before MPI
  // (MPI 4.1 §12.3.4: the origin and result buffers must be disjoint).
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 2), span_of(buffer, 2), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 2), span_of(buffer + 1, 2), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::rget_accumulate(w, span_of(buffer, 2), span_of(buffer, 2), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::fetch_and_op(w, &buffer[0], &buffer[0], rank, 0, MPI_SUM); }));

  // A mixed flat/layout origin/result pair is the caller's obligation: the plan has no
  // layout-object address, so only the both-flat pair is byte-checked.  A mixed pair over
  // distinct buffers is therefore accepted and executes the normal call.
  int mixed_origin[2] = {1, 1};
  int mixed_result[2] = {-1, -1};
  mp1::get_accumulate(w, mp1::region{mixed_origin, two}, span_of(mixed_result, 2), rank, 2, 2,
                      MPI_SUM);
  mp1::get_accumulate(w, span_of(mixed_origin, 2), mp1::region{mixed_result, two}, rank, 2, 2,
                      MPI_SUM);

  // MPI_NO_OP ignores the origin arguments: a null/empty origin is the atomic-read spelling
  // and its size is not checked (MPI 4.1 §12.3.4).  The result is written.
  mp1::get_accumulate(w, std::span<const int>{}, span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  // The single-element control kept from the base test: a null origin with MPI_NO_OP is the
  // atomic read and returns the old target value, while a null origin with an active op is
  // rejected before MPI (the origin arguments are ignored only under NO_OP).
  int fop_old = -1;
  mp1::fetch_and_op(w, static_cast<const int*>(nullptr), &fop_old, rank, 0, MPI_NO_OP);
  BOOST_TEST(fop_old == 10);
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::fetch_and_op(w, static_cast<const int*>(nullptr), &fop_old, rank, 0, MPI_SUM);
  }));
  mp1::unlock(w, rank);
  BOOST_TEST(result[0] == 10);
  BOOST_TEST(result[1] == 10);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 10);
  BOOST_TEST(window_load(target + 1) == 10);

  // MPI_PROC_NULL is a normalized no-op for every data-movement family; the count checks are
  // skipped as part of the whole no-op and the epoch must still be closed (MPI 4.1 §12.3).
  mp1::lock_all(w);
  const int one = 1;
  int value = -1;
  mp1::put(w, std::span<const int>{}, MPI_PROC_NULL, 0, 123);
  mp1::get(w, std::span<int>{}, MPI_PROC_NULL, 0, 123);
  mp1::accumulate(w, std::span<const int>{}, MPI_PROC_NULL, 0, 123, MPI_SUM);
  mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, MPI_PROC_NULL, 0, 123,
                      MPI_NO_OP);
  mp1::fetch_and_op(w, &one, &value, MPI_PROC_NULL, 0, MPI_SUM);
  mp1::compare_and_swap(w, &one, &one, &value, MPI_PROC_NULL, 0);
  mp1::request r1 = mp1::rput(w, std::span<const int>{}, MPI_PROC_NULL, 0, 123);
  mp1::request r2 = mp1::rget(w, std::span<int>{}, MPI_PROC_NULL, 0, 123);
  mp1::request r3 = mp1::raccumulate(w, std::span<const int>{}, MPI_PROC_NULL, 0, 123, MPI_SUM);
  mp1::request r4 = mp1::rget_accumulate(w, std::span<const int>{}, std::span<int>{},
                                         MPI_PROC_NULL, 0, 123, MPI_NO_OP);
  r1.wait();
  r2.wait();
  r3.wait();
  r4.wait();
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 10);
  BOOST_TEST(window_load(target + 1) == 10);
  w.free();
}

BOOST_AUTO_TEST_SUITE_END()
