// Contract test for status content provenance.
//
// The accessors must decide from the operation family that produced the status, never from the
// MPI_SOURCE/MPI_TAG/MPI_ERROR field values: a conforming provider may leave the undefined
// fields of a file or send status at any values, including the ones that look like the MPI
// empty status. The central regression is a status whose fields are the empty sentinels while
// MPI internally holds a count of 4: an `empty()`-field heuristic returns 0, provenance returns
// 4. MPI 4.1: 3.2.5/3.7.3 (point-to-point status content), 3.7.3 (empty status definition),
// 14.4.1 (file status content).
//
// It provides no main: the shared test harness links the Boost test main.

#include "mp1/impl/environment.hpp"
#include "mp1/impl/file.hpp"
#include "mp1/impl/p2p.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
#endif

namespace {

  int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  // MPI empty-looking fields with an internal element count of 4 (MPI 4.1 3.7.3: a status is
  // empty only when it is *also* configured so the count query returns 0).
  MPI_Status empty_looking_status_with_elements(int elements) {
    MPI_Status native{};
    native.MPI_SOURCE = MPI_ANY_SOURCE;
    native.MPI_TAG = MPI_ANY_TAG;
    native.MPI_ERROR = MPI_SUCCESS;
    BOOST_TEST(MPI_Status_set_elements_c(&native, MPI_INT, elements) == MPI_SUCCESS);
    return native;
  }

  void expect_any_envelope(const mp1::status& s) {
    BOOST_TEST(s.source() == MPI_ANY_SOURCE);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
  }

}  // namespace

struct mp1_status_provenance_fixture {
  mp1_status_provenance_fixture() {
    mp1::init();
  }

  ~mp1_status_provenance_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized by another path; never throw from a fixture dtor.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_status_provenance_fixture);

BOOST_AUTO_TEST_SUITE(status_provenance)

BOOST_AUTO_TEST_CASE(receive_status_with_empty_fields_queries_the_count) {
  const MPI_Status native = empty_looking_status_with_elements(4);
  // Every field is the empty-status sentinel, so a field-value heuristic would return 0.
  const mp1::status received = mp1::detail::status_access::make_receive(native);
  BOOST_TEST(received.get_count<int>() == 4);
  BOOST_TEST(received.get_elements<int>() == 4);
  // The same status through the layout overload (one contiguous int per repetition).
  const mp1::layout<int> one = mp1::layout<int>::contiguous(1);
  BOOST_TEST(received.get_count(one) == 4);
  BOOST_TEST(received.get_elements(one) == 4);
  expect_any_envelope(received);

  // The legacy constructor name keeps the receive provenance.
  const mp1::status legacy = mp1::detail::status_access::make(native);
  BOOST_TEST(legacy.get_count<int>() == 4);
}

BOOST_AUTO_TEST_CASE(send_status_has_no_count_and_is_normalized) {
  MPI_Status raw{};
  raw.MPI_SOURCE = 3;
  raw.MPI_TAG = 77;
  raw.MPI_ERROR = MPI_SUCCESS;
  const mp1::status sent = mp1::detail::status_access::make_send(raw, false);
  expect_any_envelope(sent);
  BOOST_TEST(sent.error() == MPI_SUCCESS);
  BOOST_TEST(sent.get_count<int>() == 0);
  BOOST_TEST(sent.get_elements<int>() == 0);
  // MPI_Test_cancelled is defined for a send status (MPI 4.1 3.7.3).
  BOOST_TEST(not sent.is_cancelled());

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    BOOST_TEST(sent.get_count<int>() == 0);
    BOOST_TEST(sent.get_elements<int>() == 0);
    BOOST_TEST(mpi_wrap::total() == 0);
  }
#endif
}

BOOST_AUTO_TEST_CASE(pending_status_carries_only_the_error) {
  MPI_Status raw{};
  raw.MPI_SOURCE = 11;
  raw.MPI_TAG = 22;
  raw.MPI_ERROR = MPI_ERR_PENDING;
  const mp1::status pending = mp1::detail::status_access::make_pending(raw);
  BOOST_TEST(pending.error() == MPI_ERR_PENDING);
  expect_any_envelope(pending);
  BOOST_TEST(pending.get_count<int>() == 0);
  BOOST_TEST(pending.get_elements<int>() == 0);
  BOOST_TEST(not pending.is_cancelled());

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    BOOST_TEST(pending.get_count<int>() == 0);
    BOOST_TEST(pending.get_elements<int>() == 0);
    BOOST_TEST(not pending.is_cancelled());
    BOOST_TEST(mpi_wrap::total() == 0);
  }
#endif
}

BOOST_AUTO_TEST_CASE(normalized_and_default_status_are_mpi_free) {
  const MPI_Status raw = empty_looking_status_with_elements(4);
  const mp1::status normalized = mp1::detail::status_access::make_normalized(raw);
  expect_any_envelope(normalized);
  BOOST_TEST(normalized.error() == MPI_SUCCESS);
  BOOST_TEST(normalized.get_count<int>() == 0);
  BOOST_TEST(normalized.get_elements<int>() == 0);
  BOOST_TEST(not normalized.is_cancelled());

  const mp1::status empty;
  expect_any_envelope(empty);
  BOOST_TEST(empty.get_count<int>() == 0);
  BOOST_TEST(empty.get_elements<int>() == 0);
  BOOST_TEST(not empty.is_cancelled());

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    (void)normalized.get_count<int>();
    (void)normalized.get_elements<int>();
    (void)normalized.is_cancelled();
    (void)empty.get_count<int>();
    (void)empty.get_elements<int>();
    (void)empty.is_cancelled();
    BOOST_TEST(mpi_wrap::total() == 0);
  }
#endif
}

BOOST_AUTO_TEST_CASE(send_request_status_has_no_count) {
  const auto self = mp1::comm_self();
  int value{7};
  int received{0};
  MPI_Request recv_raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Irecv(&received, 1, MPI_INT, 0, 70, MPI_COMM_SELF, &recv_raw) == MPI_SUCCESS);
  mp1::request send_request = mp1::isend(self, value, 0, mp1::tag{70});
  const mp1::status sent = send_request.wait();
  // A standard send completes when the sender's buffer is reusable; the matching receive is
  // not necessarily complete yet (Intel MPI can return before the receive lands). Wait for
  // the posted receive before inspecting the payload.
  MPI_Status recv_status{};
  BOOST_TEST(MPI_Wait(&recv_raw, &recv_status) == MPI_SUCCESS);
  BOOST_TEST(received == 7);
  expect_any_envelope(sent);
  BOOST_TEST(sent.get_count<int>() == 0);
  BOOST_TEST(sent.get_elements<int>() == 0);

#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    BOOST_TEST(sent.get_count<int>() == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Get_count_c") == 0);
  }
#endif

  BOOST_TEST(recv_status.MPI_TAG == 70);
}

BOOST_AUTO_TEST_CASE(receive_request_status_has_a_count) {
  const auto self = mp1::comm_self();
  int value{9};
  int received{0};
  mp1::request recv_request = mp1::irecv(self, received, 0, mp1::tag{71});
  mp1::request send_request = mp1::isend(self, value, 0, mp1::tag{71});
  const mp1::status sent = recv_request.wait();
  BOOST_TEST(received == 9);
  BOOST_TEST(sent.get_count<int>() == 1);
  BOOST_TEST(sent.get_elements<int>() == 1);
  BOOST_TEST(sent.source() == 0);
  BOOST_TEST(sent.tag().value == 71);
  (void)send_request.wait();
}

BOOST_AUTO_TEST_CASE(constructed_file_status_has_count_and_normalized_envelope) {
  // A conforming provider may leave SOURCE/TAG at the empty sentinels for a file status
  // (MPI 4.1 14.4.1). The constructed status makes that explicit while the internal count is 4.
  MPI_Status raw = empty_looking_status_with_elements(4);
  raw.MPI_ERROR = MPI_SUCCESS;
  const mp1::status file = mp1::detail::status_access::make_file(raw, false);
  expect_any_envelope(file);
  BOOST_TEST(file.get_count<int>() == 4);
  BOOST_TEST(file.get_elements<int>() == 4);
  BOOST_TEST(file.error() == MPI_SUCCESS);  // the undefined field is forced to SUCCESS
  BOOST_TEST(not file.is_cancelled());
}

BOOST_AUTO_TEST_CASE(file_status_count_is_defined_and_envelope_normalized) {
  const int rank = world_rank();
  // Each rank opens its own file (MPI_COMM_SELF). The timestamp keeps concurrent rank-count
  // runs (ctest may run the 2- and 4-rank entries together) from sharing a path.
  const auto run_id = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("mp1_status_provenance_" + std::to_string(rank) + "_" + std::to_string(run_id) + ".bin");
  const mp1::layout<int> four_ints = mp1::layout<int>::contiguous(4);
  const mp1::layout<int> two_ints = mp1::layout<int>::contiguous(2);

  {
    mp1::file file;
    file.open(mp1::comm_self(), path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, four_ints);
    file.resize(static_cast<mp1::offset_t>(4 * sizeof(int)));
    const std::array<int, 4> written{{11, 22, 33, 44}};
    (void)file.write_at(0, written);
    file.sync();

    std::array<int, 2> readback{{0, 0}};
    const mp1::status read = file.read_at(0, mp1::region{readback.data(), two_ints});
    BOOST_TEST(readback[0] == 11);
    BOOST_TEST(readback[1] == 22);
    // MPI 4.1 14.4.1: count/elements are defined for a file data access; source/tag are not,
    // so the wrapper normalizes them to ANY instead of forwarding provider garbage.
    BOOST_TEST(read.get_count<int>() == 2);
    BOOST_TEST(read.get_elements<int>() == 2);
    expect_any_envelope(read);
    BOOST_TEST(read.error() == MPI_SUCCESS);
    // MPI_Test_cancelled is allowed for a file data-access status (MPI 4.1 14.4.1); the flag
    // itself is provider-controlled, so only the query being available is exercised here. The
    // constructed-status case below asserts the deterministic value.
    (void)read.is_cancelled();

#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
      BOOST_TEST(read.get_count<int>() == 2);
      BOOST_TEST(mpi_wrap::count("MPI_Get_count_c") == 1);
    }
#endif

    // The nonblocking file request takes the same make_file provenance.
    std::array<int, 2> async_readback{{0, 0}};
    mp1::request read_request = file.iread_at(0, mp1::region{async_readback.data(), two_ints});
    const mp1::status async_status = read_request.wait();
    BOOST_TEST(async_readback[0] == 11);
    BOOST_TEST(async_readback[1] == 22);
    BOOST_TEST(async_status.get_count<int>() == 2);
    expect_any_envelope(async_status);
    // Open MPI 6.0.x leaves the cancellation flag uninitialized for a nonblocking file
    // request, so no value is asserted for the real path; the query must merely be available.
    (void)async_status.is_cancelled();

    file.close();
  }
  mp1::file::remove(path);
}

BOOST_AUTO_TEST_SUITE_END()
