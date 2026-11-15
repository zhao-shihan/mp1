// The five blocking receive policies and the four
// nonblocking/persistent policies.
//
//   1 value       T&                     one element; a longer message is a truncation error
//   2 fixed       span/array/C array     the extent is the capacity; no resize
//   3 region      region<T>              count 1 with the layout; overlap map rejected
//   4 auto-size   vector/string          MPI_Mprobe + MPI_Mrecv; resized to the count
//   5 bounded     vector/string+capacity upper bound; resized down; longer is truncation
//
// Nonblocking/persistent: 1, 2, 3, 5 (no auto-size).  MPI 4.1 §3.2.4 (truncation),
// §3.8.2 (matched probe), §5.1.11 (receive layout role), §3.10 (MPI_PROC_NULL).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#endif


namespace {

  // Temporarily install MPI_ERRORS_RETURN so a truncation error becomes an mp1::mpi_error
  // instead of aborting the job; restores the previous handler on scope exit.
  // A 12-byte reflected datatype: a message whose byte size is not a multiple of this extent
  // makes the provider report MPI_UNDEFINED for the probed count.
  struct triple3 {
    int a = 0;
    int b = 0;
    int c = 0;
  };

  class world_return_errors_scope {
  public:
    world_return_errors_scope() {
      (void)MPI_Comm_get_errhandler(MPI_COMM_WORLD, &previous_);
      (void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    }

    world_return_errors_scope(const world_return_errors_scope&) = delete;
    world_return_errors_scope& operator=(const world_return_errors_scope&) = delete;

    ~world_return_errors_scope() {
      (void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, previous_);
      (void)MPI_Errhandler_free(&previous_);
    }

  private:
    MPI_Errhandler previous_{MPI_ERRORS_ARE_FATAL};
  };

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

}  // namespace

MP1_REFLECTION(triple3, a, b, c)


struct mp1_receive_policy_fixture {
  mp1_receive_policy_fixture() {
    mp1::init();
  }

  ~mp1_receive_policy_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_receive_policy_fixture);

BOOST_AUTO_TEST_SUITE(receive_policy_contract)

BOOST_AUTO_TEST_CASE(policy_1_value) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  if (rank == 0) {
    mp1::send(world, 41, 1, mp1::tag{7000});
  } else if (rank == 1) {
    int value{0};
    const mp1::status status = mp1::recv(world, value, 0, mp1::tag{7000});
    BOOST_TEST(value == 41);
    BOOST_TEST(status.get_count<int>() == 1);
    BOOST_TEST(status.source() == 0);
  }
  world_barrier();

  // A two-element message into a one-element value is a truncation error (§3.2.4).
  if (rank == 0) {
    const std::array<int, 2> sent{{1, 2}};
    mp1::ssend(world, sent, 1, mp1::tag{7001});
  } else if (rank == 1) {
    world_return_errors_scope return_errors;
    int value{0};
    bool threw{false};
    try {
      (void)mp1::recv(world, value, 0, mp1::tag{7001});
    } catch (const mp1::mpi_error&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(policy_2_fixed) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // A larger fixed extent is legal; only the received prefix is written.
  if (rank == 0) {
    const std::array<int, 2> sent{{7, 8}};
    mp1::send(world, sent, 1, mp1::tag{7010});
  } else if (rank == 1) {
    std::array<int, 3> received{{-1, -1, -1}};
    const mp1::status status = mp1::recv(world, received, 0, mp1::tag{7010});
    BOOST_TEST(received[0] == 7);
    BOOST_TEST(received[1] == 8);
    BOOST_TEST(received[2] == -1);
    BOOST_TEST(status.get_count<int>() == 2);
  }
  world_barrier();

  // A smaller fixed extent is a truncation error, not a silent short read.
  if (rank == 0) {
    const std::array<int, 3> sent{{1, 2, 3}};
    mp1::ssend(world, sent, 1, mp1::tag{7011});
  } else if (rank == 1) {
    world_return_errors_scope return_errors;
    std::array<int, 2> received{{-1, -1}};
    bool threw{false};
    try {
      (void)mp1::recv(world, received, 0, mp1::tag{7011});
    } catch (const mp1::mpi_error&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();

  // C array and std::span are the same fixed policy.
  if (rank == 0) {
    const std::array<int, 2> sent{{9, 10}};
    mp1::send(world, sent, 1, mp1::tag{7012});
    mp1::send(world, sent, 1, mp1::tag{7013});
  } else if (rank == 1) {
    int received[2] = {-1, -1};
    (void)mp1::recv(world, received, 0, mp1::tag{7012});
    BOOST_TEST(received[1] == 10);
    std::array<int, 2> span_received{};
    (void)mp1::recv(world, std::span(span_received), 0, mp1::tag{7013});
    BOOST_TEST(span_received[0] == 9);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(policy_3_region) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const mp1::layout<int> map = mp1::layout<int>::contiguous(2);

  if (rank == 0) {
    int sent[3] = {11, 12, 99};
    mp1::send(world, mp1::region{&sent[0], map}, 1, mp1::tag{7020});
  } else if (rank == 1) {
    int received[2] = {-1, -1};
    const mp1::status status = mp1::recv(world, mp1::region{&received[0], map}, 0, mp1::tag{7020});
    BOOST_TEST(received[0] == 11);
    BOOST_TEST(received[1] == 12);
    BOOST_TEST(status.get_count(map) == 1);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(policy_4_auto_size) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  if (rank == 0) {
    const std::vector<int> sent{21, 22, 23};
    mp1::send(world, sent, 1, mp1::tag{7030});
  } else if (rank == 1) {
    std::vector<int> received{99, 99, 99, 99, 99};
    const mp1::status status = mp1::recv(world, received, 0, mp1::tag{7030});
    BOOST_TEST(received.size() == 3u);
    BOOST_TEST(received[2] == 23);
    BOOST_TEST(status.get_count<int>() == 3);
  }
  world_barrier();

  // A zero-element message resizes the target to zero.
  if (rank == 0) {
    const std::vector<int> sent;
    mp1::send(world, sent, 1, mp1::tag{7031});
  } else if (rank == 1) {
    std::vector<int> received{1, 2, 3};
    const mp1::status status = mp1::recv(world, received, 0, mp1::tag{7031});
    BOOST_TEST(received.empty());
    BOOST_TEST(status.get_count<int>() == 0);
  }
  world_barrier();

  // A message whose probed count is MPI_UNDEFINED (a type signature that is not an integral
  // number of the target element type) throws before the message is consumed.  The provider
  // computes the count, so this case is exercised only when the provider reports MPI_UNDEFINED;
  // the portable contract is the local `count < 0` rejection documented in recv_container_impl.
  if (rank == 0) {
    const double sent = 1.5;  // 8 bytes, not a multiple of the 12-byte triple3 extent
    mp1::send(world, sent, 1, mp1::tag{7032});
  } else if (rank == 1) {
    std::vector<triple3> received;
    const bool undefined = mp1::probe(world, 0, mp1::tag{7032}).get_count<triple3>() < 0;
    bool threw{false};
    try {
      (void)mp1::recv(world, received, 0, mp1::tag{7032});
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    // The local `count < 0` rejection happens after MPI_Mprobe; the scope guard consumes the
    // matched message, so no unmatched message is left behind either way.
    if (undefined)
      BOOST_TEST(threw);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(policy_5_bounded) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  if (rank == 0) {
    const std::vector<int> sent{31, 32};
    mp1::send(world, sent, 1, mp1::tag{7040});
  } else if (rank == 1) {
    std::vector<int> received(8, -1);
    const mp1::status status = mp1::recv(world, received, mp1::capacity{8}, 0, mp1::tag{7040});
    BOOST_TEST(received.size() == 2u);
    BOOST_TEST(received[1] == 32);
    BOOST_TEST(status.get_count<int>() == 2);
  }
  world_barrier();

  if (rank == 0) {
    const std::vector<int> sent{41, 42, 43};
    mp1::ssend(world, sent, 1, mp1::tag{7041});
  } else if (rank == 1) {
    world_return_errors_scope return_errors;
    std::vector<int> received(4, -1);
    bool threw{false};
    try {
      (void)mp1::recv(world, received, mp1::capacity{1}, 0, mp1::tag{7041});
    } catch (const mp1::mpi_error&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(nonblocking_and_persistent_policies) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const mp1::layout<int> map = mp1::layout<int>::contiguous(1);

  // irecv 1/2/3/5: all posted before the matching send, all complete.
  if (rank == 0) {
    mp1::send(world, 51, 1, mp1::tag{7050});
    mp1::send(world, 52, 1, mp1::tag{7051});
    int one = 53;
    mp1::send(world, mp1::region{&one, map}, 1, mp1::tag{7052});
    mp1::send(world, 54, 1, mp1::tag{7053});
  } else if (rank == 1) {
    int value{0};
    mp1::request r1 = mp1::irecv(world, value, 0, mp1::tag{7050});
    std::array<int, 1> fixed{};
    mp1::request r2 = mp1::irecv(world, std::span(fixed), 0, mp1::tag{7051});
    int typed{0};
    mp1::request r3 = mp1::irecv(world, mp1::region{&typed, map}, 0, mp1::tag{7052});
    std::vector<int> bounded;
    mp1::request r4 = mp1::irecv(world, bounded, mp1::capacity{4}, 0, mp1::tag{7053});
    (void)r1.wait();
    (void)r2.wait();
    (void)r3.wait();
    (void)r4.wait();
    BOOST_TEST(value == 51);
    BOOST_TEST(fixed[0] == 52);
    BOOST_TEST(typed == 53);
    BOOST_TEST(bounded.size() == 1u);
    BOOST_TEST(bounded[0] == 54);
  }
  world_barrier();

  // recv_init 1/2/3/5: distinct tags, one matched message per persistent request.
  if (rank == 0) {
    mp1::send(world, 60, 1, mp1::tag{7060});
    mp1::send(world, 61, 1, mp1::tag{7060});
    mp1::send(world, 62, 1, mp1::tag{7061});
    mp1::send(world, 63, 1, mp1::tag{7062});
    mp1::send(world, 64, 1, mp1::tag{7063});
  } else if (rank == 1) {
    int value{0};
    mp1::request p1 = mp1::recv_init(world, value, 0, mp1::tag{7060});
    p1.start();
    (void)p1.wait();
    BOOST_TEST(value == 60);
    p1.start();
    (void)p1.wait();
    BOOST_TEST(value == 61);

    std::array<int, 1> fixed{};
    mp1::request p2 = mp1::recv_init(world, std::span(fixed), 0, mp1::tag{7061});
    p2.start();
    (void)p2.wait();
    BOOST_TEST(fixed[0] == 62);

    int typed{0};
    mp1::request p3 = mp1::recv_init(world, mp1::region{&typed, map}, 0, mp1::tag{7062});
    p3.start();
    (void)p3.wait();
    BOOST_TEST(typed == 63);

    std::vector<int> bounded;
    mp1::request p4 = mp1::recv_init(world, bounded, mp1::capacity{2}, 0, mp1::tag{7063});
    p4.start();
    (void)p4.wait();
    BOOST_TEST(bounded.size() == 1u);
    BOOST_TEST(bounded[0] == 64);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(proc_null_leaves_policy_target_untouched) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();

  int value = 7;
  const mp1::status value_status = mp1::recv(world, value, MPI_PROC_NULL);
  BOOST_TEST(value_status.source() == MPI_PROC_NULL);
  BOOST_TEST(value == 7);

  std::array<int, 3> fixed{{1, 2, 3}};
  (void)mp1::recv(world, fixed, MPI_PROC_NULL);

  std::vector<int> auto_sized{1, 2, 3};
  (void)mp1::recv(world, auto_sized, MPI_PROC_NULL);
  BOOST_TEST(auto_sized.size() == 3u);

  std::vector<int> bounded{1, 2, 3};
  (void)mp1::recv(world, bounded, mp1::capacity{8}, MPI_PROC_NULL);
  BOOST_TEST(bounded.size() == 3u);

  const mp1::layout<int> map = mp1::layout<int>::contiguous(3);
  std::array<int, 3> typed{{1, 2, 3}};
  (void)mp1::recv(world, mp1::region{typed.data(), map}, MPI_PROC_NULL);
  BOOST_TEST(typed[2] == 3);
}

BOOST_AUTO_TEST_SUITE_END()
