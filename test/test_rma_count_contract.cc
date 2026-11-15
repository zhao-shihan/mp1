// RMA count/disjointness contract.
//
// The span/region API replaced the pointer+count call shapes, so the equalities the old shape
// encoded are explicit, local checks before the MPI call:
//
//   origin.size()/result.size() == target_count          (flat target)
//   origin.size()/result.size() == target.size()         (layout target)
//   a region contributes map().size()
//   the both-flat origin/result byte ranges are disjoint (get_accumulate/rget_accumulate/
//                                                        fetch_and_op/compare_and_swap)
//
// A mismatch throws std::invalid_argument before any MPI call, target_count is never inferred
// from a buffer size, MPI_PROC_NULL and a null origin under MPI_NO_OP are exempt, and a zero
// element count is legal.  Mixed flat/layout or layout/layout origin/result overlap is the
// caller's obligation because the plan has no layout-object address (documented limitation).
//
// MPI 4.1 basis: §12.3.2 (get), §12.3.4 (accumulate/get_accumulate, "the data ... must fit,
// without truncation" and "the origin and result buffers must be disjoint"; Get Accumulate:
// MPI_NO_OP ignores the origin arguments), §12.3.5 (request-based RMA).  Runs at 2 and 4
// ranks; every call targets the calling rank's own window segment or is rejected locally.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_RMA_COUNT_WRAP 1
#else
#define MP1_RMA_COUNT_WRAP 0
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

  [[nodiscard]] int window_load(const int* p) {
    return *static_cast<const volatile int*>(p);
  }

  void sync_after_passive(const mp1::mpi_communicator& c, const mp1::mpi_window& w) {
    mp1::barrier(c);
    mp1::lock_all(w);
    mp1::sync(w);
    mp1::unlock_all(w);
    mp1::barrier(c);
  }

  template<class T>
  [[nodiscard]] std::span<T> span_of(T* data, mp1::count_t count) {
    return std::span<T>{data, static_cast<std::size_t>(count)};
  }

  // The wrapped RMA entry points whose local-rejection zero-call assertion is available.  The
  // put/get/request entry points are not wrapped, so their rejection is proven by the
  // exception type and a valid window/rank (there is no other throw path).
  void expect_no_accumulate_call() {
#if MP1_RMA_COUNT_WRAP
    BOOST_TEST(mpi_wrap::count("MPI_Accumulate_c") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Get_accumulate_c") == 0);
#endif
  }

  struct rma_count_fixture {
    rma_count_fixture() {
      mp1::init();
    }

    ~rma_count_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(rma_count_fixture);

BOOST_AUTO_TEST_SUITE(rma_count_contract)

BOOST_AUTO_TEST_CASE(flat_target_size_equality) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  int buffer[elements] = {1, 2, 3, 4, 5, 6, 7, 8};
  int result[elements] = {};
  const int one = 1;

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // The origin/result size is the fact checked; a longer or shorter buffer is rejected before
  // MPI rather than truncated or accepted.
  BOOST_TEST(throws_invalid_argument([&] { mp1::put(w, span_of(buffer, 2), rank, 0, 3); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::put(w, span_of(buffer, 3), rank, 0, 2); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::get(w, span_of(result, 2), rank, 0, 3); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::get(w, span_of(result, 3), rank, 0, 2); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, span_of(buffer, 2), rank, 0, 3, MPI_SUM); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::raccumulate(w, span_of(buffer, 2), rank, 0, 3, MPI_SUM); }));

  // get_accumulate and rget_accumulate check both sides.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 2), span_of(result, 3), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 2), span_of(result, 2), rank, 0, 3, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::rget_accumulate(w, span_of(buffer, 2), span_of(result, 3), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::rget(w, span_of(result, 2), rank, 0, 3); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::rput(w, span_of(buffer, 2), rank, 0, 3); }));

  // The equal sizes are accepted and move exactly that many elements.
  mp1::put(w, span_of(buffer, 2), rank, 0, 2);
  mp1::get(w, span_of(result, 2), rank, 0, 2);
  mp1::accumulate(w, span_of(&one, 1), rank, 2, 1, MPI_SUM);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 2);
  BOOST_TEST(window_load(target + 2) == 1);
  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  w.free();
}

BOOST_AUTO_TEST_CASE(layout_target_size_equality) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 16;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> three = mp1::layout<int>::contiguous(3);
  int buffer[3] = {1, 2, 3};
  int result[3] = {};

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // A span/region size must equal target.size(), not target_count and not the byte size.
  BOOST_TEST(throws_invalid_argument([&] { mp1::put(w, span_of(buffer, 2), rank, 0, three); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::put(w, span_of(buffer, 3), rank, 0, two); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::get(w, span_of(result, 2), rank, 0, three); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, span_of(buffer, 2), rank, 0, three, MPI_SUM); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 2), span_of(result, 3), rank, 0, two, MPI_SUM);
  }));

  // A region contributes map().size(), including a region with a hole.
  const int origin_data[3] = {4, -1, 5};
  const mp1::layout<int> origin_stride = mp1::layout<int>::vector(2, 1, 2);
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::put(w, mp1::region{origin_data, origin_stride}, rank, 0, three); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::get(w, mp1::region{result, origin_stride}, rank, 0, three); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, mp1::region{origin_data, origin_stride}, span_of(result, 3), rank, 0,
                        two, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(buffer, 3), mp1::region{result, origin_stride}, rank, 0, two,
                        MPI_SUM);
  }));

  // Matching sizes are accepted for every shape.
  mp1::put(w, span_of(buffer, 2), rank, 0, two);
  mp1::put(w, mp1::region{origin_data, origin_stride}, rank, 2, two);
  mp1::get(w, span_of(result, 2), rank, 0, two);
  mp1::accumulate(w, span_of(buffer, 2), rank, 4, two, MPI_SUM);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 1);
  BOOST_TEST(window_load(target + 1) == 2);
  BOOST_TEST(window_load(target + 2) == 4);
  BOOST_TEST(window_load(target + 3) == 5);
  BOOST_TEST(window_load(target + 4) == 1);
  BOOST_TEST(window_load(target + 5) == 2);
  BOOST_TEST(result[0] == 1);
  BOOST_TEST(result[1] == 2);
  w.free();
}

BOOST_AUTO_TEST_CASE(target_count_is_not_ignored) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const int two_elements[2] = {7, 8};

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // The same two-element span is accepted with target_count == 2 and rejected with any other
  // target_count: the count is the caller's value, never inferred from the buffer and never
  // ignored.
  mp1::put(w, span_of(two_elements, 2), rank, 0, 2);
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::put(w, span_of(two_elements, 2), rank, 0, 1); }));
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::put(w, span_of(two_elements, 2), rank, 0, 3); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, span_of(two_elements, 2), rank, 0, 3, MPI_SUM); }));
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 7);
  BOOST_TEST(window_load(target + 1) == 8);
  BOOST_TEST(window_load(target + 2) == 0);
  w.free();
}

BOOST_AUTO_TEST_CASE(zero_count_and_empty_layout) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 3;
  mp1::fence(w);
  mp1::barrier(c);

  int result = -1;
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
  const mp1::layout<int> empty_region = mp1::layout<int>::contiguous(0);

  mp1::lock_all(w);
  // A zero element count is legal for every family; an empty span is null-data + size 0, and
  // the region form has an empty map.
  mp1::put(w, std::span<const int>{}, rank, 0, 0);
  mp1::get(w, std::span<int>{}, rank, 0, 0);
  mp1::accumulate(w, std::span<const int>{}, rank, 0, 0, MPI_SUM);
  mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, rank, 0, 0, MPI_SUM);
  mp1::put(w, mp1::region<const int>{nullptr, empty_region}, rank, 0, 0);
  mp1::put(w, std::span<const int>{}, rank, 0, empty);
  mp1::get(w, std::span<int>{}, rank, 0, empty);
  mp1::accumulate(w, mp1::region<const int>{nullptr, empty_region}, rank, 0, empty, MPI_SUM);
  mp1::request rput = mp1::rput(w, std::span<const int>{}, rank, 0, 0);
  mp1::request rget = mp1::rget(w, std::span<int>{}, rank, 0, 0);
  mp1::request racc = mp1::raccumulate(w, std::span<const int>{}, rank, 0, 0, MPI_SUM);
  mp1::request rga =
      mp1::rget_accumulate(w, std::span<const int>{}, std::span<int>{}, rank, 0, 0, MPI_SUM);
  rput.wait();
  rget.wait();
  racc.wait();
  rga.wait();
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 3);
  BOOST_TEST(window_load(target + 1) == 3);
  BOOST_TEST(result == -1);
  w.free();
}

BOOST_AUTO_TEST_CASE(proc_null_and_no_op_exemptions) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 11;
  mp1::fence(w);
  mp1::barrier(c);

  const int one = 1;
  int result[2] = {-1, -1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);

  // MPI_PROC_NULL: the whole data-movement call is a normalized no-op, so even an empty
  // buffer with a bogus positive target count, a negative displacement and a negative target
  // count are accepted (MPI ignores all of it; §12.3 Communication Calls).  The epoch still
  // has to be closed.
  mp1::lock_all(w);
  mp1::put(w, std::span<const int>{}, MPI_PROC_NULL, -1, 123);
  mp1::get(w, std::span<int>{}, MPI_PROC_NULL, -1, 123);
  mp1::accumulate(w, std::span<const int>{}, MPI_PROC_NULL, -1, 123, MPI_SUM);
  mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, MPI_PROC_NULL, -1, 123,
                      MPI_NO_OP);
  mp1::fetch_and_op(w, static_cast<const int*>(nullptr), &result[0], MPI_PROC_NULL, -1,
                    MPI_NO_OP);
  mp1::compare_and_swap(w, &one, &one, &result[0], MPI_PROC_NULL, -1);
  mp1::request r1 = mp1::rput(w, std::span<const int>{}, MPI_PROC_NULL, -1, 123);
  mp1::request r2 = mp1::rget(w, std::span<int>{}, MPI_PROC_NULL, -1, 123);
  mp1::request r3 =
      mp1::raccumulate(w, std::span<const int>{}, MPI_PROC_NULL, -1, 123, MPI_SUM);
  mp1::request r4 = mp1::rget_accumulate(w, std::span<const int>{}, std::span<int>{},
                                         MPI_PROC_NULL, -1, 123, MPI_NO_OP);
  r1.wait();
  r2.wait();
  r3.wait();
  r4.wait();

  // MPI_NO_OP: the origin arguments are ignored (§12.3.4), so the null-origin spelling skips
  // the origin size check.  Both the empty span and a null-data region are accepted; the
  // result is still written and still size-checked.
  mp1::get_accumulate(w, std::span<const int>{}, span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  mp1::get_accumulate(w, mp1::region<const int>{nullptr, two}, span_of(result, 2), rank, 0, 2,
                      MPI_NO_OP);
  mp1::request rga = mp1::rget_accumulate(w, std::span<const int>{}, span_of(result, 2), rank,
                                          0, 2, MPI_NO_OP);
  rga.wait();

  // MPI_NO_OP ignores every origin argument, so a real non-empty origin is legal even when it
  // aliases the result (the standard atomic-read idiom).  The wrapper passes the dummy origin
  // for the call and skips both the size equality and the both-flat overlap check.
  mp1::get_accumulate(w, span_of(result, 0), span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  mp1::get_accumulate(w, mp1::region<const int>{result, two}, span_of(result, 2), rank, 0, 2,
                      MPI_NO_OP);
  mp1::get_accumulate(w, span_of(result, 2), span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  mp1::request rga_alias =
      mp1::rget_accumulate(w, span_of(result, 2), span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  rga_alias.wait();
  // The active operator is unchanged: an empty flat origin still fails the size equality and a
  // non-empty aliasing origin still fails the disjointness check.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(result, 0), span_of(result, 2), rank, 0, 2, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, span_of(result, 2), span_of(result, 2), rank, 0, 2, MPI_SUM);
  }));

  // Every MPI_NO_OP origin is ignored, so even a wrongly sized origin is accepted (and never
  // read); the result is still size-checked and written.
  const int wrong_size[1] = {1};
  mp1::get_accumulate(w, span_of(wrong_size, 1), span_of(result, 2), rank, 0, 2, MPI_NO_OP);
  // A null result is never exempt: the result is written even for MPI_NO_OP.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, rank, 0, 2, MPI_NO_OP);
  }));
  // fetch_and_op: the same exemption covers the single-element atomic read, including an
  // origin that aliases the result buffer.
  mp1::fetch_and_op(w, &result[0], &result[0], rank, 0, MPI_NO_OP);
  // The NO_OP zero-count shape is legal.
  mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, rank, 0, empty, MPI_NO_OP);

  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  // NO_OP returns the old target value and leaves the target unchanged.
  BOOST_TEST(result[0] == 11);
  BOOST_TEST(result[1] == 11);
  BOOST_TEST(window_load(target + 0) == 11);
  BOOST_TEST(window_load(target + 1) == 11);
  w.free();
}

BOOST_AUTO_TEST_CASE(both_flat_origin_result_disjointness) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  target[2] = 1;  // the compare-and-swap expected value
  mp1::fence(w);
  mp1::barrier(c);

  int buffer[elements] = {1, 1, 1, 1, 1, 1, 1, 1};
  int adjacent[2] = {-1, -1};

#if MP1_RMA_COUNT_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // Identical and partially overlapping both-flat ranges are rejected before MPI
  // (MPI 4.1 §12.3.4: "The origin and result buffers must be disjoint").
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
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::compare_and_swap(w, &buffer[0], &buffer[2], &buffer[0], rank, 0); }));
#if MP1_RMA_COUNT_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Get_accumulate_c") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Accumulate_c") == 0);
#endif

  // Adjacent, non-overlapping ranges are accepted.
  mp1::get_accumulate(w, span_of(buffer, 2), span_of(adjacent, 2), rank, 0, 2, MPI_SUM);
  // CAS's compare input is a separate read-only buffer and may alias origin or result
  // (§12.3.4 Compare and Swap makes only origin and result disjoint).
  const int expected = 1;
  const int replacement = 5;
  int cas_old = -1;
  mp1::compare_and_swap(w, &buffer[0], &buffer[0], &cas_old, rank, 2);
  mp1::compare_and_swap(w, &replacement, &expected, &cas_old, rank, 2);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);

  BOOST_TEST(adjacent[0] == 0);
  BOOST_TEST(adjacent[1] == 0);
  BOOST_TEST(cas_old == 1);
  BOOST_TEST(window_load(target + 2) == 5);

  // Mixed flat/layout and layout/layout pairs have no locally knowable cross-object byte
  // range, so no local overlap check exists (documented limitation).  A mixed
  // pair over distinct buffers executes the normal call.
  int mixed_origin[2] = {2, 2};
  int mixed_result_span[2] = {-1, -1};
  int mixed_result_region[2] = {-1, -1};
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  mp1::get_accumulate(w, mp1::region{mixed_origin, two}, span_of(mixed_result_span, 2), rank, 4,
                      2, MPI_SUM);
  mp1::get_accumulate(w, span_of(mixed_origin, 2), mp1::region{mixed_result_region, two}, rank,
                      4, 2, MPI_SUM);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(mixed_result_span[0] == 0);
  BOOST_TEST(mixed_result_span[1] == 0);
  BOOST_TEST(mixed_result_region[0] == 2);
  BOOST_TEST(mixed_result_region[1] == 2);
  w.free();
}

BOOST_AUTO_TEST_CASE(null_buffers_and_negative_count) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  int result[2] = {};

#if MP1_RMA_COUNT_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // A negative target count is rejected, even when the buffer size would otherwise match.
  BOOST_TEST(throws_invalid_argument([&] { mp1::put(w, span_of(result, 1), rank, 0, -1); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::get(w, span_of(result, 1), rank, 0, -1); }));
  // A null-data region with a non-empty map and an active (non-NO_OP) call is rejected.
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::put(w, mp1::region<const int>{nullptr, two}, rank, 0, 2); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::get(w, mp1::region<int>{nullptr, two}, rank, 0, 2); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, mp1::region<const int>{nullptr, two}, span_of(result, 2), rank, 0, 2,
                        MPI_SUM);
  }));
  // A null result with a nonzero target count is rejected (the result is always written).
  BOOST_TEST(throws_invalid_argument([&] { mp1::get(w, std::span<int>{}, rank, 0, 2); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, std::span<const int>{}, std::span<int>{}, rank, 0, 2, MPI_NO_OP);
  }));
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 0) == 0);
  expect_no_accumulate_call();
  w.free();
}

BOOST_AUTO_TEST_CASE(counts_are_element_counts_not_bytes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  constexpr int elements = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  mp1::fence(w);
  mp1::barrier(c);

  const int buffer[2] = {31, 32};

  mp1::lock(w, mp1::lock_type::exclusive, rank);
  // target_count is an element count: two ints move two elements; passing the byte size (8)
  // is a size mismatch and is rejected.
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::put(w, span_of(buffer, 2), rank, 0, elements); }));
  // target_disp is scaled by the target's disp_unit (sizeof(int) here), so element 3 is at
  // displacement 3, never byte 3.
  mp1::put(w, span_of(buffer, 2), rank, 3, 2);
  mp1::unlock(w, rank);
  sync_after_passive(c, w);
  BOOST_TEST(window_load(target + 3) == 31);
  BOOST_TEST(window_load(target + 4) == 32);
  w.free();
}

BOOST_AUTO_TEST_SUITE_END()
