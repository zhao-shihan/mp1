// Variable-collective fast-path regressions: the direct path is selected by the recorded
// `packed` fact, not by byte_size == true_extent. A descending receive map and a
// duplicate+hole send map must stage and agree with native MPI byte-for-byte.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <vector>


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

  class raw_datatype {
  public:
    explicit raw_datatype(MPI_Datatype handle) : handle_{handle} {
    }
    raw_datatype(const raw_datatype&) = delete;
    raw_datatype& operator=(const raw_datatype&) = delete;
    ~raw_datatype() {
      if (handle_ != MPI_DATATYPE_NULL)
        (void)MPI_Type_free(&handle_);
    }
    [[nodiscard]] MPI_Datatype get() const noexcept {
      return handle_;
    }

  private:
    MPI_Datatype handle_;
  };

  [[nodiscard]] raw_datatype make_raw_indexed(MPI_Datatype base) {
    const MPI_Count blocklengths[2] = {1, 1};
    const MPI_Count displacements[2] = {1, 0};
    MPI_Datatype type = MPI_DATATYPE_NULL;
    BOOST_REQUIRE(MPI_Type_indexed_c(2, blocklengths, displacements, base, &type) ==
                  MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_commit(&type) == MPI_SUCCESS);
    return raw_datatype{type};
  }

  [[nodiscard]] raw_datatype make_raw_duplicate_hole() {
    const int blocklengths[3] = {1, 1, 1};
    const MPI_Aint displacements[3] = {0, 0, 8};
    MPI_Datatype type = MPI_DATATYPE_NULL;
    BOOST_REQUIRE(MPI_Type_create_hindexed(3, blocklengths, displacements, MPI_INT, &type) ==
                  MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_commit(&type) == MPI_SUCCESS);
    return raw_datatype{type};
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(variable_layout_fastpath_test)

BOOST_AUTO_TEST_CASE(descending_receive_map_is_staged) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = 0;

  const std::array<int, 2> send{rank * 10, rank * 10 + 1};
  const mp1::layout<int> send_block = mp1::layout<int>::contiguous(2);
  mp1::indexed_parameter descending_parameter;
  descending_parameter.add(1, 1);
  descending_parameter.add(1, 0);
  const mp1::layout<int> descending = mp1::layout<int>::indexed(descending_parameter);
  BOOST_TEST(not descending.dense());

  std::vector<mp1::layout<int>> recv_blocks(static_cast<std::size_t>(size), descending);
  std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(2 * i);

  std::vector<int> recv(static_cast<std::size_t>(2 * size), -1);
  mp1::gatherv(comm, root, send, recv, recv_blocks, displs);

  // Native reference: one count of the same descending datatype per rank; native
  // displacements are in units of the datatype extent (8 bytes), so rank i is at i.
  const raw_datatype native_type = make_raw_indexed(MPI_INT);
  std::vector<int> native_recv(static_cast<std::size_t>(2 * size), -1);
  std::vector<int> recvcounts(static_cast<std::size_t>(size), 1);
  std::vector<int> native_displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    native_displs[static_cast<std::size_t>(i)] = i;
  const int rc =
      MPI_Gatherv(send.data(), 2, MPI_INT, native_recv.data(), recvcounts.data(),
                  native_displs.data(), native_type.get(), root, comm.native_handle());
  BOOST_REQUIRE(rc == MPI_SUCCESS);

  if (rank == root) {
    BOOST_TEST(recv == native_recv, boost::test_tools::per_element());
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(2 * i)] == 10 * i + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(2 * i + 1)] == 10 * i);
    }
  }
}

BOOST_AUTO_TEST_CASE(duplicate_hole_send_map_is_staged) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = 0;

  // The send type map is {0, 0, 8}: a duplicate and a hole whose sizes cancel, so
  // byte_size == true_extent == 12 but the map is not packed.
  mp1::hindexed_parameter duplicate_parameter;
  duplicate_parameter.add(1, 0);
  duplicate_parameter.add(1, 0);
  duplicate_parameter.add(1, 8);
  const mp1::layout<int> duplicate_hole = mp1::layout<int>::hindexed(duplicate_parameter);
  BOOST_TEST(duplicate_hole.byte_size() == duplicate_hole.true_byte_extent());
  BOOST_TEST(not duplicate_hole.dense());

  const std::array<int, 4> send{rank * 10, rank * 10 + 1, rank * 10 + 2, rank * 10 + 3};
  std::vector<mp1::layout<int>> recv_blocks(static_cast<std::size_t>(size),
                                            mp1::layout<int>::contiguous(3));
  std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(3 * i);
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
  mp1::gatherv(comm, root, mp1::region{send.data(), duplicate_hole}, recv, recv_blocks, displs);

  const raw_datatype native_type = make_raw_duplicate_hole();
  std::vector<int> native_recv(static_cast<std::size_t>(3 * size), -1);
  std::vector<int> recvcounts(static_cast<std::size_t>(size), 3);
  std::vector<int> native_displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    native_displs[static_cast<std::size_t>(i)] = 3 * i;
  const int rc =
      MPI_Gatherv(send.data(), 1, native_type.get(), native_recv.data(), recvcounts.data(),
                  native_displs.data(), MPI_INT, root, comm.native_handle());
  BOOST_REQUIRE(rc == MPI_SUCCESS);

  if (rank == root) {
    BOOST_TEST(recv == native_recv, boost::test_tools::per_element());
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == 10 * i);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == 10 * i);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == 10 * i + 2);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
