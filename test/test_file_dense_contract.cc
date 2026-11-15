// File dense data-access contract: the pointer+count and contiguous-container spellings of the
// blocking and split-collective families, the MPI_File_*_c lowering, split begin/end buffer
// identity, pending-request rejection, and the empty/null/negative count rules.
//
// The typed-view etype rules (MPI 4.1 §14.3.1) are covered by
// test_file_typed_view_contract.cc; this file keeps the dense-access contract.
//
// MPI 4.1 basis: §14.4 (data access), §14.4.5 (split-collective begin/end), §14.2.4/§14.2.5
// (set_view/resize with outstanding nonblocking requests), §14.9 (dense access is the
// (buffer, count, datatype) spelling).  Runs at 2 and 4 ranks; every collective file call is
// entered by all ranks in the same order.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_FILE_DENSE_WRAP 1
#else
#define MP1_FILE_DENSE_WRAP 0
#endif


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
    return base_dir / (std::string{"mp1_dense_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  // The dense spelling uses native_datatype<T>() as the filetype; one contiguous int etype.
  [[nodiscard]] mp1::layout<int> int_etype() {
    return mp1::layout<int>::contiguous(1);
  }

  struct file_dense_fixture {
    file_dense_fixture() {
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
          std::filesystem::temp_directory_path() / ("mp1_dense_" + std::to_string(run_id));
      std::error_code error;
      if (rank == 0) {
        std::filesystem::remove_all(base_dir, error);
        error.clear();
        std::filesystem::create_directories(base_dir, error);
      }
      world_barrier();
    }

    ~file_dense_fixture() {
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


BOOST_TEST_GLOBAL_FIXTURE(file_dense_fixture);

BOOST_AUTO_TEST_SUITE(file_dense_contract)

BOOST_AUTO_TEST_CASE(blocking_pointer_count_and_containers) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("blocking");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size * 64 * sizeof(int)));
  const mp1::offset_t base = static_cast<mp1::offset_t>(rank * 48);

  int data[4] = {rank, rank + 1, rank + 2, rank + 3};
  int readback[4] = {-1, -1, -1, -1};
  (void)file.write_at(base, std::span<int>{data});
  file.sync();
#if MP1_FILE_DENSE_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    (void)file.read_at(base, std::span<int>{readback});
    BOOST_TEST(mpi_wrap::count("MPI_File_read_at_c") == 1u);
  } else
#endif
  {
    (void)file.read_at(base, std::span<int>{readback});
  }
  for (int i = 0; i < 4; ++i)
    BOOST_TEST(readback[i] == data[i]);

  // Contiguous-container forms use the container's current size.
  std::vector<int> vector_data{rank + 10, rank + 11, rank + 12};
  std::vector<int> vector_readback(3, -1);
  (void)file.write_at(base + 4, vector_data);
  file.sync();
#if MP1_FILE_DENSE_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    (void)file.read_at(base + 4, vector_readback);
    BOOST_TEST(mpi_wrap::count("MPI_File_read_at_c") == 1u);
  } else
#endif
  {
    (void)file.read_at(base + 4, vector_readback);
  }
  BOOST_TEST(vector_readback == vector_data);

  std::array<int, 2> array_data{{rank + 20, rank + 21}};
  std::array<int, 2> array_readback{{-1, -1}};
  (void)file.write_at(base + 8, array_data);
  (void)file.read_at(base + 8, array_readback);
  BOOST_TEST(array_readback == array_data);

  // Sequential-pointer dense forms.
  file.seek(base + 12, mp1::whence_mode::set);
  (void)file.write(vector_data);
  file.seek(base + 12, mp1::whence_mode::set);
  std::vector<int> sequential_readback(3, -1);
  (void)file.read(sequential_readback);
  BOOST_TEST(sequential_readback == vector_data);

  // Collective individual-pointer dense forms.
  (void)file.write_at_all(base + 16, std::span<int>{data});
  (void)file.read_at_all(base + 16, std::span<int>{readback});
  for (int i = 0; i < 4; ++i)
    BOOST_TEST(readback[i] == data[i]);

  file.seek(base + 24, mp1::whence_mode::set);
  (void)file.write_all(std::span<int>{data});
  file.seek(base + 24, mp1::whence_mode::set);
  std::array<int, 4> all_readback{{-1, -1, -1, -1}};
  (void)file.read_all(all_readback);
  for (int i = 0; i < 4; ++i)
    BOOST_TEST(all_readback[static_cast<std::size_t>(i)] == data[i]);

  // Zero-count no-ops accept null buffers; a positive count requires a buffer, and a negative
  // count is rejected before MPI.
  (void)file.read_at(base, std::span<int>{});
  (void)file.write_at(base, std::span<const int>{});
  std::vector<int> empty_container;
  (void)file.read_at(base, empty_container);
  (void)file.write_at(base, empty_container);
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.read_at(base, mp1::span_from(static_cast<int*>(nullptr), 2)); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.read_at(base, mp1::span_from(data, -1)); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.write_at(base, mp1::span_from(data, -1)); }));

  file.sync();
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(shared_and_ordered_dense_forms) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path shared_path = test_path("shared");
  const std::filesystem::path ordered_path = test_path("ordered");

  // Shared file pointer: only rank 0 performs the access, and the file is reopened to reset
  // the shared pointer for the read pass.
  {
    mp1::file file;
    file.open(world, shared_path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, int_etype());
    file.resize(static_cast<mp1::offset_t>(size * 8 * sizeof(int)));
    if (rank == 0) {
      const int shared_data[3] = {51, 52, 53};
      (void)file.write_shared(std::span<const int>{shared_data});
    }
    file.sync();
    file.close();
  }
  {
    mp1::file reader;
    reader.open(world, shared_path, mp1::access_mode::read_only);
    reader.set_view(0, int_etype());
    if (rank == 0) {
      int shared_readback[3] = {-1, -1, -1};
      (void)reader.read_shared(std::span<int>{shared_readback});
      BOOST_TEST(shared_readback[0] == 51);
      BOOST_TEST(shared_readback[1] == 52);
      BOOST_TEST(shared_readback[2] == 53);
    }
    reader.close();
  }

  // Ordered collective dense write appends each rank's block in rank order; the read pass
  // starts from a fresh shared pointer after reopen.
  {
    mp1::file file;
    file.open(world, ordered_path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, int_etype());
    file.resize(static_cast<mp1::offset_t>(size * 4 * sizeof(int)));
    std::vector<int> ordered_data{rank * 10, rank * 10 + 1};
    (void)file.write_ordered(ordered_data);
    file.sync();
    file.close();
  }
  {
    mp1::file reader;
    reader.open(world, ordered_path, mp1::access_mode::read_only);
    reader.set_view(0, int_etype());
    std::vector<int> ordered_readback(2, -1);
    (void)reader.read_ordered(ordered_readback);
    BOOST_TEST(ordered_readback[0] == rank * 10);
    BOOST_TEST(ordered_readback[1] == rank * 10 + 1);
    reader.close();
  }

  remove_path(shared_path);
  remove_path(ordered_path);
}

BOOST_AUTO_TEST_CASE(split_collectives_pointer_and_container) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("split");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size * 96 * sizeof(int)));
  const mp1::offset_t base = static_cast<mp1::offset_t>(rank * 64);

  const int data[3] = {rank, rank + 1, rank + 2};
  int readback[3] = {-1, -1, -1};
  file.write_at_all_begin(base, std::span<const int>{data});
  (void)file.write_at_all_end(data);
  file.read_at_all_begin(base, std::span<int>{readback});
  (void)file.read_at_all_end(readback);
  for (int i = 0; i < 3; ++i)
    BOOST_TEST(readback[i] == data[i]);

  std::vector<int> vector_data{rank + 10, rank + 11, rank + 12};
  std::vector<int> vector_readback(3, -1);
  file.write_at_all_begin(base + 4, vector_data);
  (void)file.write_at_all_end(vector_data);
  file.read_at_all_begin(base + 4, vector_readback);
  (void)file.read_at_all_end(vector_readback);
  BOOST_TEST(vector_readback == vector_data);

  // Sequential-pointer split forms.
  file.seek(base + 8, mp1::whence_mode::set);
  file.write_all_begin(std::span<const int>{data});
  (void)file.write_all_end(data);
  file.seek(base + 8, mp1::whence_mode::set);
  std::array<int, 3> all_readback{{-1, -1, -1}};
  file.read_all_begin(all_readback);
  (void)file.read_all_end(all_readback);
  for (std::size_t i = 0; i < all_readback.size(); ++i)
    BOOST_TEST(all_readback[i] == data[i]);

  // A split end must receive the same buffer as its begin.  A mismatch is a local
  // std::runtime_error before MPI; the epoch stays active, so end with the recorded buffer.
  std::array<int, 2> first{{rank, rank + 1}};
  std::array<int, 2> second{{-1, -1}};
  file.write_at_all_begin(base + 12, first);
  BOOST_TEST(throws_runtime_error([&] { (void)file.write_at_all_end(second); }));
  (void)file.write_at_all_end(first);

  std::array<int, 2> begin_buffer{{0, 0}};
  file.read_at_all_begin(base + 12, begin_buffer);
  BOOST_TEST(throws_runtime_error([&] {
    std::array<int, 2> different{{0, 0}};
    (void)file.read_at_all_end(different);
  }));
  (void)file.read_at_all_end(begin_buffer);
  BOOST_TEST(begin_buffer[0] == rank);
  BOOST_TEST(begin_buffer[1] == rank + 1);

#if MP1_FILE_DENSE_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    file.read_at_all_begin(base, std::span<int>{readback});
    (void)file.read_at_all_end(readback);
    BOOST_TEST(mpi_wrap::count("MPI_File_read_at_all_begin_c") == 1u);
  }
#endif

  file.sync();
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(split_ordered_collectives) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("ordered_split");

  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, int_etype());
    file.resize(static_cast<mp1::offset_t>(size * 8 * sizeof(int)));
    std::array<int, 2> data{{rank * 5, rank * 5 + 1}};
    file.write_ordered_begin(data);
    (void)file.write_ordered_end(data);
    file.sync();
    file.close();
  }
  {
    mp1::file reader;
    reader.open(world, path, mp1::access_mode::read_only);
    reader.set_view(0, int_etype());
    std::array<int, 2> readback{{-1, -1}};
    reader.read_ordered_begin(readback);
    (void)reader.read_ordered_end(readback);
    BOOST_TEST(readback[0] == rank * 5);
    BOOST_TEST(readback[1] == rank * 5 + 1);
    reader.close();
  }

  remove_path(path);
}

BOOST_AUTO_TEST_CASE(pending_request_rejection) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("pending");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(world_size() * 4 * sizeof(int)));

  int value = 0;
  mp1::request request = file.iread_at(0, value);
  BOOST_TEST(throws_runtime_error([&] { file.set_view(0, int_etype()); }));
  BOOST_TEST(throws_runtime_error(
      [&] { file.resize(static_cast<mp1::offset_t>(world_size() * 4 * sizeof(int))); }));
  request.wait();

  // Recovery: once the request is complete, the same operations succeed.
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(world_size() * 4 * sizeof(int)));
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
