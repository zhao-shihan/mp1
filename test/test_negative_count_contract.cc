// Checked negative counts and displacements.
//
// The public count type is checked before any MPI call: `span_from` rejects a negative count,
// a `blocks` descriptor rejects negative count/displacement entries, and file access rejects
// a negative view displacement and a negative etype offset.  The local throw is
// std::invalid_argument and, when the wrap shim is available, makes no MPI call.
//
// Runs at 2 and 4 ranks.  The negative collective cases are entered by every rank so the
// local rejection cannot deadlock a peer that already entered the same collective.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>


#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_NEGATIVE_HAVE_WRAP 1
#else
#define MP1_NEGATIVE_HAVE_WRAP 0
#endif

namespace {

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] bool throws_invalid_argument(auto&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  [[nodiscard]] bool wrap_supported() {
#if MP1_NEGATIVE_HAVE_WRAP
    return mpi_wrap::supported();
#else
    return false;
#endif
  }

  void reset_calls() {
#if MP1_NEGATIVE_HAVE_WRAP
    if (wrap_supported())
      mpi_wrap::reset_counts();
#endif
  }

  [[nodiscard]] std::size_t total_calls() {
#if MP1_NEGATIVE_HAVE_WRAP
    return wrap_supported() ? mpi_wrap::total() : 0;
#else
    return 0;
#endif
  }

  struct negative_count_fixture {
    negative_count_fixture() {
      mp1::init();
    }

    ~negative_count_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a fixture destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(negative_count_fixture);

BOOST_AUTO_TEST_SUITE(negative_count_contract)

// The pointer+count conversion is checked; a negative count never becomes a huge
// std::size_t.
BOOST_AUTO_TEST_CASE(span_from_rejects_negative_count) {
  int values[2] = {0, 0};
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::span_from(values, -1); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::span_from<int>(nullptr, -1); }));
  BOOST_TEST(mp1::span_from(values, 2).size() == 2);
}

// A negative count or displacement is a local contract violation and is rejected before the
// collective (no MPI call under the wrap shim).
BOOST_AUTO_TEST_CASE(blocks_reject_negative_entries_before_mpi) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int size = c.size();
  const std::vector<int> send(1, 1);
  std::vector<int> recv(static_cast<std::size_t>(size), -1);

  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
  counts[0] = -1;
  std::vector<mp1::count_t> displacements(static_cast<std::size_t>(size), 0);

  reset_calls();
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::allgatherv(c, send, recv, mp1::blocks{counts, displacements}); }));
  BOOST_TEST(total_calls() == 0);

  counts[0] = 1;
  displacements[0] = -1;
  reset_calls();
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::allgatherv(c, send, recv, mp1::blocks{counts, displacements}); }));
  BOOST_TEST(total_calls() == 0);
  world_barrier();
}

// A negative view displacement and a negative etype offset are rejected locally; the file
// handle stays usable and can still be closed collectively.
BOOST_AUTO_TEST_CASE(file_units_reject_negative_values) {
  const mp1::mpi_communicator world = mp1::comm_world();
  // The same path on every rank: MPI_File_open is collective.  A per-run seed keeps the .2
  // and .4 CTest entries (which run concurrently) from sharing a file.
  std::uint64_t seed{0};
  if (world_rank() == 0) {
    seed = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    seed ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&seed));
  }
  MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("mp1_negative_count_" + std::to_string(seed) + ".bin");
  std::error_code ignored;
  std::filesystem::remove(path, ignored);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.resize(static_cast<mp1::offset_t>(4 * sizeof(int)));

  const mp1::layout<int> int_etype = mp1::layout<int>::contiguous(1);
  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] { file.set_view(-1, int_etype); }));
  BOOST_TEST(total_calls() == 0);
  BOOST_TEST(file.is_open());

  // The failed set_view recorded no view; a valid byte view restores a known baseline.
  file.set_view(0, "native");

  int value = 0;
  reset_calls();
  BOOST_TEST(throws_invalid_argument([&] { (void)file.read_at(-1, value); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)file.write_at(-1, value); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)file.iread_at(-1, value); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)file.iwrite_at(-1, value); }));
  BOOST_TEST(total_calls() == 0);

  file.close();
  world_barrier();
  if (world_rank() == 0)
    std::filesystem::remove(path, ignored);
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
