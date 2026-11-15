// File etype-unit and view-displacement contract.
//
// MPI-IO uses a mixed-unit model (MPI 4.1 §14.1.1/§14.4.1): the view displacement is an
// absolute byte count, while seek/position/data-access offsets are etype counts relative to
// the view and byte_offset converts between them.  The byte-unit contract is that the wrapper
// must pass `set_view(disp)` through unchanged: with an MPI_INT etype and disp = 1 the first
// int is written at absolute byte 1, never at byte 0 or byte 1 * extent.  The test also pins
// size() as bytes, a `region<T>` as exactly one occurrence (MPI count 1) of its layout
// (`layout.size()` elements), and read-at-EOF as a short read with a zero count.
//
// Each rank writes and reads its own disjoint file region: the test checks unit conversions,
// not the provider's cross-rank visibility/cache behavior (MPICH 4.3.2 returns a stale first
// read for another rank's write even after a collective MPI_File_sync).  The non-monotone
// etype and overlapping-filetype rows are covered by test_file_typed_view_contract;
// this file owns the units/displacement rows.  Runs at 2 and 4 ranks: open/resize/set_view/
// sync/close are collective and every rank enters them in the same order.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <system_error>
#include <vector>


namespace {

  int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_etype_units_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  struct file_etype_fixture {
    file_etype_fixture() {
      mp1::init();
      const int rank = world_rank();
      std::uint64_t seed{0};
      if (rank == 0) {
        seed = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        seed ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&seed));
      }
      MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
      run_id = seed;
      base_dir = std::filesystem::temp_directory_path() /
                 ("mp1_etype_units_" + std::to_string(run_id));
      std::error_code error;
      if (rank == 0) {
        std::filesystem::remove_all(base_dir, error);
        error.clear();
        std::filesystem::create_directories(base_dir, error);
      }
      world_barrier();
    }

    ~file_etype_fixture() {
      world_barrier();
      if (world_rank() == 0) {
        std::error_code error;
        std::filesystem::remove_all(base_dir, error);
      }
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(file_etype_fixture);

BOOST_AUTO_TEST_SUITE(file_etype_units_contract)

// The byte-unit contract: set_view's `disp` is a byte displacement, not an etype count.  With an int etype and
// disp = 1, etype offset 0 lands at absolute byte 1; byte_offset() reports the same absolute
// byte, seek/position count etypes, and a raw MPI_BYTE view finds the int bytes at byte 1.
BOOST_AUTO_TEST_CASE(view_displacement_is_bytes) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world.size();
  const std::filesystem::path path = test_path("disp_bytes");
  constexpr mp1::offset_t displacement = 1;
  const int per_rank = 3;
  const std::size_t payload =
      static_cast<std::size_t>(1 + size * per_rank * static_cast<int>(sizeof(int)));
  const std::vector<int> values{0x11223344, 0x55667788, static_cast<int>(0x7f7f7f7f)};
  const mp1::offset_t first = static_cast<mp1::offset_t>(rank) * per_rank;

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  // MPI_File_open is collective but does not order the operations that follow
  // it across ranks: a fast rank can resize the file before a slow rank returns
  // from open.  Synchronize before observing shared metadata so the assertion
  // holds when sibling CTest entries keep the machine busy.
  world_barrier();
  BOOST_TEST(file.size() == 0);  // the file was just created
  // A barrier synchronizes entry, not exit: without this second barrier the
  // fastest rank can leave the barrier, pass the check above and enter the
  // collective resize while a slow rank is still sampling size(), which would
  // then observe the new length instead of zero.
  world_barrier();
  file.resize(static_cast<mp1::offset_t>(payload));
  world_barrier();
  BOOST_TEST(file.size() == static_cast<mp1::offset_t>(payload));  // size() is bytes

  const mp1::layout<int> int_etype = mp1::layout<int>::contiguous(1);
  file.set_view(displacement, int_etype);

  // The displacement is an absolute byte count: etype offset n must report byte
  // displacement + n * sizeof(int), independent of any MPI call argument conversion.
  for (int i = 0; i < per_rank; ++i) {
    const mp1::offset_t offset = first + i;
    BOOST_TEST(file.byte_offset(offset) ==
               displacement + offset * static_cast<mp1::offset_t>(sizeof(int)));
  }
  BOOST_TEST(file.byte_offset(0) == displacement);

  for (int i = 0; i < per_rank; ++i)
    (void)file.write_at(first + i, values[static_cast<std::size_t>(i)]);
  file.sync();

  // seek/position are etype units relative to the view; an individual-pointer read at etype
  // offset first + 2 reads the byte-displacement-shifted int of this rank's region.
  file.seek(first + 2, mp1::whence_mode::set);
  BOOST_TEST(file.position() == first + 2);
  int pointer_read = -1;
  (void)file.read(pointer_read);
  BOOST_TEST(pointer_read == values[2]);
  BOOST_TEST(file.position() == first + 3);

  // Read the raw bytes through the MPI_BYTE view: this rank's ints start at byte
  // displacement + first * sizeof(int), so the byte before that is not part of any int.
  file.set_view(0, "native");
  std::vector<unsigned char> bytes(payload, 0);
  (void)file.read_at(0, bytes);
  // Compare only this rank's int slices: the rest of the file (holes and other ranks'
  // regions) is not initialized by this rank, so its raw bytes are not portable.
  bool own_slices_match = true;
  for (int i = 0; i < per_rank; ++i) {
    const auto offset = static_cast<std::size_t>(
        displacement + (first + i) * static_cast<mp1::offset_t>(sizeof(int)));
    own_slices_match = own_slices_match and
                       std::memcmp(bytes.data() + offset, &values[static_cast<std::size_t>(i)],
                                   sizeof(int)) == 0;
  }
  BOOST_TEST(own_slices_match);

  file.close();
  remove_path(path);
}

// A region is one occurrence of its layout: MPI count = 1 and layout.size() elements, so a
// status reports one repetition and layout.size() basic elements.
BOOST_AUTO_TEST_CASE(region_is_one_layout_occurrence) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world.size();
  const std::filesystem::path path = test_path("region_count");
  const mp1::layout<int> int_etype = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> pair = mp1::layout<int>::contiguous(2);
  const mp1::offset_t first = static_cast<mp1::offset_t>(rank) * 2;

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.resize(static_cast<mp1::offset_t>(1 + size * 2 * static_cast<int>(sizeof(int))));
  file.set_view(1, int_etype);

  std::array<int, 2> written{rank + 100, rank + 200};
  (void)file.write_at(first, mp1::region{written.data(), pair});
  file.sync();

  std::array<int, 2> read{};
  const mp1::status status = file.read_at(first, mp1::region{read.data(), pair});
  BOOST_TEST(status.get_count(pair) == 1);     // occurrences of the layout
  BOOST_TEST(status.get_elements(pair) == 2);  // layout.size() basic elements
  BOOST_TEST(read == written);

  file.close();
  remove_path(path);
}

// Reading at or past EOF is a short read: MPI returns MPI_SUCCESS and the status count is the
// number of elements actually read (0 when the offset is at EOF).
BOOST_AUTO_TEST_CASE(read_at_eof_is_a_short_read) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world.size();
  const std::filesystem::path path = test_path("eof");
  const mp1::layout<int> int_etype = mp1::layout<int>::contiguous(1);
  const int per_rank = 3;
  const mp1::offset_t first = static_cast<mp1::offset_t>(rank) * per_rank;

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.resize(static_cast<mp1::offset_t>(1 + size * per_rank * static_cast<int>(sizeof(int))));
  file.set_view(1, int_etype);
  for (int i = 0; i < per_rank; ++i)
    (void)file.write_at(first + i, rank * 1000 + i);
  file.sync();

  // The file's last etype ends immediately before etype offset size * per_rank; a read of two
  // ints at that offset is completely past EOF and reads nothing.
  std::array<int, 2> buffer{-1, -1};
  const mp1::status at_eof = file.read_at(static_cast<mp1::offset_t>(size * per_rank),
                                          std::span<int>{buffer.data(), buffer.size()});
  BOOST_TEST(at_eof.get_count<int>() == 0);
  BOOST_TEST(buffer[0] == -1);
  BOOST_TEST(buffer[1] == -1);

  // The last int of this rank's region is a normal one-element read.
  std::array<int, 1> last{-1};
  const mp1::status last_status = file.read_at(first + per_rank - 1, std::span<int>{last});
  BOOST_TEST(last_status.get_count<int>() == 1);
  BOOST_TEST(last[0] == rank * 1000 + per_rank - 1);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
