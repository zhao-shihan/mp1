// Collapsed file surface probe: the 15-call container/region probe, the byte-unit
// set_view displacement probe, a short read at EOF, and close() with an outstanding request.
//
// The 15 data-access calls are exactly:
//   vector write/read, array write/read, string write/read, C-array write/read,
//   span write/read, region write/read, nonblocking span write/read, nonblocking region write.
// Every call selects the concept-constrained member; no free wrapper exists.
//
// MPI 4.1 basis: §14.3 (set_view displacement is an absolute byte count), §14.4.1 (offsets
// are etype units; a read may be short at EOF), §14.4.5 (no close during an outstanding split),
// §14.2.2 (all nonblocking requests complete before MPI_File_close).

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


namespace {

  int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template<class Function>
  bool throws_runtime_error(Function&& function) {
    try {
      function();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_p3_surface_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  struct file_collapsed_surface_fixture {
    file_collapsed_surface_fixture() {
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
      base_dir =
          std::filesystem::temp_directory_path() / ("mp1_p3_surface_" + std::to_string(run_id));
      std::error_code error;
      if (rank == 0) {
        std::filesystem::remove_all(base_dir, error);
        error.clear();
        std::filesystem::create_directories(base_dir, error);
      }
      world_barrier();
    }

    ~file_collapsed_surface_fixture() {
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


BOOST_TEST_GLOBAL_FIXTURE(file_collapsed_surface_fixture);

BOOST_AUTO_TEST_SUITE(file_collapsed_surface)

BOOST_AUTO_TEST_CASE(fifteen_call_container_region_probe) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("fifteen");
  const mp1::layout<int> int_view = mp1::layout<int>::contiguous(1);
  const mp1::layout<char> char_view = mp1::layout<char>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_view);
  file.resize(static_cast<mp1::offset_t>(size * 64 * sizeof(int)));
  const mp1::offset_t base = static_cast<mp1::offset_t>(rank * 64);

  std::vector<int> vector{rank + 1, rank + 2, rank + 3};
  std::vector<int> vector_readback(3, -1);
  std::array<int, 3> array{{rank + 4, rank + 5, rank + 6}};
  std::array<int, 3> array_readback{{-1, -1, -1}};
  int c_array[3] = {rank + 7, rank + 8, rank + 9};
  int c_array_readback[3] = {-1, -1, -1};
  std::vector<int> span_storage(3, -1);
  const std::span<int> span{span_storage};
  int region_storage[3] = {-1, -1, -1};
  const mp1::layout<int> region_map = mp1::layout<int>::contiguous(3);

  // 1-2: vector write/read.
  (void)file.write_at(base, vector);
  (void)file.read_at(base, vector_readback);
  BOOST_TEST(vector_readback == vector);

  // 3-4: std::array write/read.
  (void)file.write_at(base + 3, array);
  (void)file.read_at(base + 3, array_readback);
  BOOST_TEST(array_readback == array);

  // 5-6: C array write/read (the built-in array is the range, not a span).
  (void)file.write_at(base + 6, c_array);
  (void)file.read_at(base + 6, c_array_readback);
  for (int i = 0; i < 3; ++i)
    BOOST_TEST(c_array_readback[i] == c_array[i]);

  // 7-8: std::span write/read.
  (void)file.write_at(base + 9, std::span<const int>{array});
  (void)file.read_at(base + 9, span);
  BOOST_TEST(span_storage == std::vector<int>(array.begin(), array.end()));

  // 9-10: region write/read (count = 1 application of the layout).
  (void)file.write_at(base + 12, mp1::region{array.data(), region_map});
  (void)file.read_at(base + 12, mp1::region{region_storage, region_map});
  for (int i = 0; i < 3; ++i)
    BOOST_TEST(region_storage[i] == array[static_cast<std::size_t>(i)]);

  // 11-12: string write/read through the char view.  set_view is collective.
  file.set_view(0, char_view);
  const std::string text = std::string{"r"} + std::to_string(rank);
  std::string text_readback(text.size(), '\0');
  (void)file.write_at(static_cast<mp1::offset_t>(rank * 8), text);
  (void)file.read_at(static_cast<mp1::offset_t>(rank * 8), text_readback);
  BOOST_TEST(text_readback == text);

  // 13-15: nonblocking span write/read and nonblocking region write.  A named lvalue (or a
  // borrowed span) is required; the region layout is duplicated into the request lease.
  file.set_view(0, int_view);
  mp1::request span_write = file.iwrite_at(base + 15, std::span<const int>{array});
  span_write.wait();
  mp1::request span_read = file.iread_at(base + 15, span);
  span_read.wait();
  BOOST_TEST(span_storage == std::vector<int>(array.begin(), array.end()));
  mp1::request region_write = file.iwrite_at(base + 18, mp1::region{array.data(), region_map});
  region_write.wait();

  file.sync();
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(set_view_displacement_is_bytes) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("disp_bytes");
  const int size = world_size();

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.resize(static_cast<mp1::offset_t>(size) * 64);
  // The initial view is MPI_BYTE at disp 0: fill the file with a sentinel byte pattern.
  std::array<unsigned char, 16> sentinel{};
  sentinel.fill(0x7f);
  (void)file.write_at(0, sentinel);
  file.sync();

  // disp = 1 BYTE, not one etype.  With an int etype (extent 4) a wrongly scaled wrapper
  // would place the view at byte 4.
  file.set_view(1, mp1::layout<int>::contiguous(1));
  BOOST_TEST(file.byte_offset(0) == 1);
  const int magic = 0x01020304;
  (void)file.write_at(0, magic);
  int readback = 0;
  (void)file.read_at(0, readback);
  BOOST_TEST(readback == magic);
  file.sync();
  file.close();

  // Raw byte inspection: byte 0 is untouched, the int starts at byte 1, and byte 5 is
  // untouched (bytes 1..4 hold the int).  The sentinel above covered bytes 0..15.
  MPI_File native{MPI_FILE_NULL};
  BOOST_REQUIRE(MPI_File_open(MPI_COMM_WORLD, path.string().c_str(), MPI_MODE_RDONLY,
                              MPI_INFO_NULL, &native) == MPI_SUCCESS);
  std::array<unsigned char, 16> bytes{};
  BOOST_REQUIRE(MPI_File_read_at(native, 0, bytes.data(), 16, MPI_BYTE, MPI_STATUS_IGNORE) ==
                MPI_SUCCESS);
  BOOST_REQUIRE(MPI_File_close(&native) == MPI_SUCCESS);
  BOOST_TEST(bytes[0] == 0x7f);
  BOOST_TEST(bytes[5] == 0x7f);
  bool magic_found = true;
  const unsigned char* const magic_bytes = reinterpret_cast<const unsigned char*>(&magic);
  for (std::size_t i = 0; i < sizeof(int); ++i)
    magic_found = magic_found and bytes[1 + i] == magic_bytes[i];
  BOOST_TEST(magic_found);

  remove_path(path);
}

BOOST_AUTO_TEST_CASE(short_read_at_eof_is_reported_not_thrown) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("short_read");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, mp1::layout<int>::contiguous(1));
  file.resize(static_cast<mp1::offset_t>(2 * sizeof(int)));
  const std::array<int, 2> data{{11, 22}};
  (void)file.write_at(0, data);
  file.sync();

  std::array<int, 4> buffer{{-1, -1, -1, -1}};
  const mp1::status short_status = file.read_at(0, buffer);
  BOOST_TEST(short_status.get_count<int>() == 2);
  BOOST_TEST(buffer[0] == 11);
  BOOST_TEST(buffer[1] == 22);
  BOOST_TEST(buffer[2] == -1);
  BOOST_TEST(buffer[3] == -1);

  std::array<int, 2> beyond{{-1, -1}};
  const mp1::status eof_status = file.read_at(static_cast<mp1::offset_t>(4), beyond);
  BOOST_TEST(eof_status.get_count<int>() == 0);
  BOOST_TEST(beyond[0] == -1);
  BOOST_TEST(beyond[1] == -1);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(close_refuses_outstanding_request) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("close_outstanding");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, mp1::layout<int>::contiguous(1));
  file.resize(static_cast<mp1::offset_t>(sizeof(int)));

  int value = 0;
  mp1::request outstanding = file.iread_at(0, value);
  BOOST_TEST(throws_runtime_error([&file] { file.close(); }));
  BOOST_TEST(file.is_open());
  outstanding.wait();
  file.close();  // succeeds once the wrapper-owned request is complete
  BOOST_TEST(not file.is_open());
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
