// Exact-byte null positions for every reduction family and the per-slot dummy
// normalization of an empty position.
//
// The documented contract (README "A data position is empty when its total byte size is
// zero"): an empty position may be null and is presented to MPI as the per-slot dummy; a
// non-empty position with a null pointer throws std::invalid_argument before any MPI data
// call, and before a managed functor's MPI_Op (and its callback datatype) is created.  The
// exact byte size is used, not the element count: a zero-size layout or a zero-size native
// datatype with a positive count is empty (MPI 4.1 5.1.11).
//
// Every rejection is a local throw on every rank (or only on the root for a root-only
// receive), so no rank enters MPI while another skips the call.  The wrap counters are
// rank-local; a build without -Wl,--wrap= still runs the behavioral half of each case.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_REDUCTION_NULL_WRAP 1
#else
#define MP1_REDUCTION_NULL_WRAP 0
#endif


namespace {

  struct mpi_fixture {
    mpi_fixture() {
      mp1::init();
    }

    ~mpi_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  template<class Call>
  [[nodiscard]] bool throws_invalid_argument(Call&& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // Zero the wrapped-call counters immediately before the call under test.  A build without
  // the wrap shim has no counters; every assertion below is then a no-op.
  void reset_calls() {
#if MP1_REDUCTION_NULL_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
  }

  void require_calls(const char* symbol, std::size_t expected) {
#if MP1_REDUCTION_NULL_WRAP
    if (mpi_wrap::supported()) {
      BOOST_TEST_CONTEXT(symbol) {
        BOOST_TEST(mpi_wrap::count(symbol) == expected);
      }
    }
#else
    (void)symbol;
    (void)expected;
#endif
  }

  // A non-flat (non base-repetition) layout over a 4-int buffer: it selects elements 0 and 2.
  // A reduction with this receive layout stages through the local pack path, which is
  // exactly the shape that used to reach MPI: a flat null send with a non-flat receive.
  [[nodiscard]] mp1::layout<int> holey_int_layout() {
    return mp1::layout<int>::vector(2, 1, 2);
  }

  // A managed (non-predefined) functor; it forces MPI_Op_create plus the callback datatype
  // commit on first use, which a rejected call must not reach.
  struct managed_sum {
    int operator()(const int& left, const int& right) const noexcept {
      return left + right;
    }
  };

  void require_sequence(const std::vector<int>& values, std::initializer_list<int> expected) {
    BOOST_TEST(values.size() == expected.size());
    std::size_t index = 0;
    for (const int value : expected) {
      if (index < values.size())
        BOOST_TEST_CONTEXT("index " << index) {
          BOOST_TEST(values[index] == value);
        }
      ++index;
    }
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(reduction_null_positions)

// ---------------------------------------------------------------------------------------
// reduce_local: no communicator; both positions are significant.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduce_local_null_send_positive_is_rejected) {
  const std::plus<int> plus{};
  std::vector<int> recv{10, 20, 30, 40};

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_local(plus, std::span<const int>{static_cast<const int*>(nullptr), 4},
                      std::span<int>{recv});
  }));
  // The exception is thrown before the native call and before the receive buffer is touched.
  require_calls("MPI_Reduce_local_c", 0);
  require_sequence(recv, {10, 20, 30, 40});
}

BOOST_AUTO_TEST_CASE(reduce_local_null_recv_positive_is_rejected) {
  const std::plus<int> plus{};
  std::vector<int> send{1, 2, 3, 4};

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_local(plus, std::span<const int>{send},
                      std::span<int>{static_cast<int*>(nullptr), 4});
  }));
  require_calls("MPI_Reduce_local_c", 0);
}

BOOST_AUTO_TEST_CASE(reduce_local_null_staged_position_positive_is_rejected) {
  // Both mixed shapes: a flat null on one side and a non-flat position on the other.  The
  // staged plan builder used to be reached before any validation.
  const std::plus<int> plus{};
  const mp1::layout<int> holey = holey_int_layout();
  std::vector<int> recv{10, -1, 30, -1};

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_local(plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
                      mp1::region<int>{recv.data(), holey});
  }));
  require_calls("MPI_Reduce_local_c", 0);
  require_calls("MPI_Pack_c", 0);
  require_calls("MPI_Pack_size_c", 0);
  require_sequence(recv, {10, -1, 30, -1});

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    const std::vector<int> send{1, 2, 3, 4};
    mp1::reduce_local(plus, mp1::region<const int>{send.data(), holey},
                      mp1::region<int>{static_cast<int*>(nullptr), holey});
  }));
  require_calls("MPI_Reduce_local_c", 0);
  require_calls("MPI_Pack_c", 0);
}

// ---------------------------------------------------------------------------------------
// allreduce: every rank's position is significant.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(allreduce_null_send_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allreduce(c, plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
                   std::span<int>{recv});
  }));
  require_calls("MPI_Allreduce_c", 0);
  require_sequence(recv, {-1, -1});
}

BOOST_AUTO_TEST_CASE(allreduce_null_send_with_nonflat_recv_is_rejected) {
  // Before the exact-byte preflight, the code staged the receive side and handed the null
  // flat send pointer to MPI_Allreduce_c.  The preflight must reject before any pack or
  // native call.
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  const mp1::layout<int> holey = holey_int_layout();
  std::vector<int> recv{10, -1, 30, -1};

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allreduce(c, plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
                   mp1::region<int>{recv.data(), holey});
  }));
  require_calls("MPI_Allreduce_c", 0);
  require_calls("MPI_Pack_c", 0);
  require_calls("MPI_Pack_size_c", 0);
  require_sequence(recv, {10, -1, 30, -1});
}

BOOST_AUTO_TEST_CASE(allreduce_null_recv_with_nonflat_send_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  const mp1::layout<int> holey = holey_int_layout();
  std::vector<int> send{1, 2, 3, 4};

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allreduce(c, plus, mp1::region<const int>{send.data(), holey},
                   mp1::region<int>{static_cast<int*>(nullptr), holey});
  }));
  require_calls("MPI_Allreduce_c", 0);
}

BOOST_AUTO_TEST_CASE(nonblocking_reduction_null_position_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::iallreduce(c, plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
                          std::span<int>{recv});
  }));
  require_calls("MPI_Iallreduce_c", 0);
  require_sequence(recv, {-1, -1});
}

BOOST_AUTO_TEST_CASE(persistent_reduction_null_position_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::allreduce_init(c, plus,
                              std::span<const int>{static_cast<const int*>(nullptr), 2},
                              std::span<int>{recv});
  }));
  // No initiation (and no staging) may happen; the callback datatype commit is the observable
  // part of managed-op creation that a predefined op does not perform.
  require_calls("MPI_Type_commit", 0);
  require_sequence(recv, {-1, -1});
}

// ---------------------------------------------------------------------------------------
// reduce / scan / exscan.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduce_null_send_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce(c, plus, 0, std::span<const int>{static_cast<const int*>(nullptr), 2},
                std::span<int>{recv});
  }));
  require_calls("MPI_Reduce_c", 0);
  require_sequence(recv, {-1, -1});
}

BOOST_AUTO_TEST_CASE(reduce_root_null_recv_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const std::plus<int> plus{};
  const std::vector<int> send(2, 1);

  // Only the root's receive position is significant, so only the root rejects and the other
  // ranks must not enter the collective: the root catches its local throw, everyone rendezvous
  // at the barrier afterwards.
  if (rank == 0) {
    reset_calls();
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::reduce(c, plus, 0, std::span<const int>{send},
                  std::span<int>{static_cast<int*>(nullptr), 2});
    }));
    require_calls("MPI_Reduce_c", 0);
  }
  mp1::barrier(c);
}

BOOST_AUTO_TEST_CASE(reduce_nonroot_null_recv_is_ignored) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const std::vector<int> send{rank, 10 * rank};
  std::vector<int> recv(2, -1);

  // The non-root receive buffer is ignored by MPI 4.1 6.6, so an empty-looking null span is
  // deliberately not inspected on a non-root rank; the root's real receive keeps the result.
  if (rank == 0)
    mp1::reduce(c, plus, 0, std::span<const int>{send}, std::span<int>{recv});
  else
    mp1::reduce(c, plus, 0, std::span<const int>{send},
                std::span<int>{static_cast<int*>(nullptr), 2});

  if (rank == 0) {
    const int expected = size * (size - 1) / 2;
    require_sequence(recv, {expected, 10 * expected});
  }
}

BOOST_AUTO_TEST_CASE(scan_and_exscan_null_positions_are_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::scan(c, plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
              std::span<int>{recv});
  }));
  require_calls("MPI_Scan_c", 0);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::exscan(c, plus, std::span<const int>{static_cast<const int*>(nullptr), 2},
                std::span<int>{recv});
  }));
  require_calls("MPI_Exscan_c", 0);
  require_sequence(recv, {-1, -1});
}

BOOST_AUTO_TEST_CASE(managed_functor_null_position_is_rejected_before_mpi_op_create) {
  // A managed functor creates its callback datatype (MPI_Type_contiguous_c + MPI_Type_commit)
  // and then MPI_Op_create on first use.  The position preflight must run first, so a rejected
  // call creates neither.  The functor is unique to this case, so no earlier test can have
  // warmed the per-(T,F) caches.
  const mp1::mpi_communicator c = mp1::comm_world();
  std::vector<int> recv(2, -1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allreduce(c, managed_sum{}, std::span<const int>{static_cast<const int*>(nullptr), 2},
                   std::span<int>{recv});
  }));
  require_calls("MPI_Type_commit", 0);
  require_calls("MPI_Allreduce_c", 0);
  require_sequence(recv, {-1, -1});
}

// ---------------------------------------------------------------------------------------
// reduce_scatter / reduce_scatter_block.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reduce_scatter_null_send_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> recv(1, -1);
  std::vector<mp1::count_t> counts(static_cast<std::size_t>(c.size()), 1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_scatter(c, plus,
                        std::span<const int>{static_cast<const int*>(nullptr),
                                             static_cast<std::size_t>(c.size())},
                        std::span<int>{recv}, counts);
  }));
  require_calls("MPI_Reduce_scatter_c", 0);
  require_sequence(recv, {-1});
}

BOOST_AUTO_TEST_CASE(reduce_scatter_null_recv_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> send(static_cast<std::size_t>(c.size()), 1);
  std::vector<mp1::count_t> counts(static_cast<std::size_t>(c.size()), 1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_scatter(c, plus, std::span<const int>{send},
                        std::span<int>{static_cast<int*>(nullptr), 1}, counts);
  }));
  require_calls("MPI_Reduce_scatter_c", 0);
}

BOOST_AUTO_TEST_CASE(reduce_scatter_block_null_recv_positive_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  std::vector<int> send(static_cast<std::size_t>(c.size()), 1);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_scatter_block(c, plus, std::span<const int>{send},
                              std::span<int>{static_cast<int*>(nullptr), 1}, 1);
  }));
  require_calls("MPI_Reduce_scatter_block_c", 0);
}

BOOST_AUTO_TEST_CASE(reduce_scatter_null_staged_recv_is_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const std::plus<int> plus{};
  // Two elements per rank, so the element counts agree and only the null non-empty receive
  // can reject the call (a size mismatch would otherwise mask the null check).
  std::vector<int> send(static_cast<std::size_t>(2 * c.size()), 1);
  const mp1::layout<int> holey = holey_int_layout();
  std::vector<int> recv{10, -1, 30, -1};
  std::vector<mp1::count_t> counts(static_cast<std::size_t>(c.size()), 2);

  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_scatter(c, plus, std::span<const int>{send},
                        mp1::region<int>{static_cast<int*>(nullptr), holey}, counts);
  }));
  require_calls("MPI_Reduce_scatter_c", 0);
  require_calls("MPI_Pack_c", 0);
  require_sequence(recv, {10, -1, 30, -1});
}

// ---------------------------------------------------------------------------------------
// Empty positions: null with zero exact bytes is legal, one native call gets the dummy.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(zero_count_null_positions_are_accepted) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int size = c.size();
  const std::plus<int> plus{};
  constexpr const int* null_send = nullptr;
  int* null_recv = nullptr;
  std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);

  // reduce_local: one local call, no throw.
  reset_calls();
  mp1::reduce_local(plus, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0});
  require_calls("MPI_Reduce_local_c", 1);

  // allreduce / scan / exscan: every rank passes the same empty positions.
  reset_calls();
  mp1::allreduce(c, plus, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0});
  require_calls("MPI_Allreduce_c", 1);

  reset_calls();
  mp1::scan(c, plus, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0});
  require_calls("MPI_Scan_c", 1);

  reset_calls();
  mp1::exscan(c, plus, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0});
  require_calls("MPI_Exscan_c", 1);

  // reduce: the root's receive is significant and empty; the non-root's is ignored.
  reset_calls();
  mp1::reduce(c, plus, 0, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0});
  require_calls("MPI_Reduce_c", 1);

  // reduce_scatter / reduce_scatter_block with all-zero counts.
  reset_calls();
  mp1::reduce_scatter(c, plus, std::span<const int>{null_send, 0}, std::span<int>{null_recv, 0},
                      zeros);
  require_calls("MPI_Reduce_scatter_c", 1);

  reset_calls();
  mp1::reduce_scatter_block(c, plus, std::span<const int>{null_send, 0},
                            std::span<int>{null_recv, 0}, 0);
  require_calls("MPI_Reduce_scatter_block_c", 1);

  // The same rule for a typed empty layout: byte_size() == 0 is empty, so null is legal and
  // no staging call happens.
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
  reset_calls();
  mp1::allreduce(c, plus, mp1::region<const int>{nullptr, empty},
                 mp1::region<int>{nullptr, empty});
  require_calls("MPI_Allreduce_c", 1);
  require_calls("MPI_Pack_c", 0);
  require_calls("MPI_Pack_size_c", 0);
}

// ---------------------------------------------------------------------------------------
// Behavior preservation: valid flat and staged reductions keep computing the same result.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(valid_flat_and_staged_reductions_are_unchanged) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int expected = size * (size - 1) / 2;

  // reduce_local.
  {
    const std::vector<int> send{rank, 10 * rank};
    std::vector<int> recv{100, 100};
    mp1::reduce_local(plus, std::span<const int>{send}, std::span<int>{recv});
    require_sequence(recv, {100 + rank, 100 + 10 * rank});
  }

  // allreduce, flat.
  {
    const std::vector<int> send{rank, 10 * rank};
    std::vector<int> recv(2, -1);
    mp1::allreduce(c, plus, std::span<const int>{send}, std::span<int>{recv});
    require_sequence(recv, {expected, 10 * expected});
  }

  // allreduce, staged non-flat receive (the mixed flat-null/non-flat control): elements 0 and
  // 2 are reduced, the holes keep their values.
  {
    const mp1::layout<int> holey = holey_int_layout();
    const std::vector<int> send{rank, 10 * rank};
    std::vector<int> recv{100, -1, 200, -1};
    mp1::allreduce(c, plus, std::span<const int>{send}, mp1::region<int>{recv.data(), holey});
    // The mapped positions are overwritten by the reduction result; the holes keep their
    // values (the staging unpack writes only the layout's type map).
    require_sequence(recv, {expected, -1, 10 * expected, -1});
  }

  // reduce_local, staged non-flat send and receive.
  {
    const mp1::layout<int> holey = holey_int_layout();
    const std::vector<int> send{rank, -1, 10 * rank, -1};
    std::vector<int> recv{100, -1, 200, -1};
    mp1::reduce_local(plus, mp1::region<const int>{send.data(), holey},
                      mp1::region<int>{recv.data(), holey});
    require_sequence(recv, {100 + rank, -1, 200 + 10 * rank, -1});
  }

  // reduce_scatter, flat and in-place, keeps the per-rank block.
  {
    std::vector<int> send(static_cast<std::size_t>(size), 1);
    for (int peer = 0; peer < size; ++peer)
      send[static_cast<std::size_t>(peer)] = peer + 1;
    std::vector<int> recv(1, -1);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    mp1::reduce_scatter(c, plus, std::span<const int>{send}, std::span<int>{recv}, counts);
    require_sequence(recv, {size * (rank + 1)});
  }
}

// ---------------------------------------------------------------------------------------
// An absent plan keeps its direction, so an ignored send and an ignored receive use
// different dummies (some implementations reject aliased buffers even at count 0).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(absent_plans_use_the_per_slot_dummy) {
  using mp1::detail::buffer_slot;
  using mp1::detail::dummy_buffer;
  using mp1::detail::transfer_direction;
  using mp1::detail::xfer_plan;

  const xfer_plan<int> absent_send = xfer_plan<int>::absent(transfer_direction::send);
  const xfer_plan<int> absent_recv = xfer_plan<int>::absent(transfer_direction::receive);
  BOOST_TEST(absent_send.ptr() == dummy_buffer(buffer_slot::send));
  BOOST_TEST(absent_recv.ptr() == dummy_buffer(buffer_slot::recv));
  BOOST_TEST(absent_send.ptr() != absent_recv.ptr());
  BOOST_TEST(absent_send.count() == 0);
  BOOST_TEST(absent_recv.count() == 0);
}

BOOST_AUTO_TEST_SUITE_END()
