// File dense nonblocking data-access contract: the pointer+count and contiguous-container
// spellings of the ten nonblocking families, their MPI_File_i*_c lowering, the dense-path
// datatype facts (native_datatype<T>() is predefined, so no MPI_Type_commit/free is issued),
// zero/null and count rules, nonzero offsets, shared-pointer ordering and collective
// completion.
//
// Families: iread_at/iwrite_at, iread/iwrite, iread_shared/iwrite_shared,
// iread_at_all/iwrite_at_all and iread_all/iwrite_all (MPI 4.1 §14.4, §14.4.3, §14.4.4,
// §14.4.5, §14.9).  The test runs at 2 and 4 ranks; every collective call is entered by all
// ranks in the same order, and each rank owns a private byte range of every test file.
//
// MPI 4.1 basis: §14.4 (data access), §14.4.1 (nonblocking buffer lifetime), §14.4.3
// (individual file pointer), §14.4.4 (shared file pointer), §14.4.5 (collective and split
// collectives), §14.9 (dense access is the (buffer, count, datatype) spelling).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_FILE_NB_WRAP 1
#else
#define MP1_FILE_NB_WRAP 0
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

  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_nb_dense_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  // One contiguous native element is the view etype; the dense spelling uses the same
  // predefined type as the access filetype.
  [[nodiscard]] mp1::layout<int> int_etype() {
    return mp1::layout<int>::contiguous(1);
  }

  [[nodiscard]] mp1::layout<char> char_etype() {
    return mp1::layout<char>::contiguous(1);
  }

  void reset_wrap_counts() {
#if MP1_FILE_NB_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
  }

  void expect_wrap_count(const char* symbol, std::size_t expected) {
#if MP1_FILE_NB_WRAP
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count(symbol) == expected);
#else
    (void)symbol;
    (void)expected;
#endif
  }

  // The dense path passes native_datatype<T>() straight to MPI_File_*_c; it must not construct,
  // commit or free a derived datatype.  MPI_Type_contiguous is not in the canonical wrap table,
  // so the commit/free counters are the observable proxy for "no derived datatype was built".
  // Counters are rank-local and were reset immediately before the initiation under test.
  void expect_no_derived_datatype() {
#if MP1_FILE_NB_WRAP
    if (mpi_wrap::supported()) {
      BOOST_TEST(mpi_wrap::count("MPI_Type_commit") == 0u);
      BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 0u);
    }
#endif
  }

  void expect_dense_initiation(const char* symbol) {
    expect_wrap_count(symbol, 1);
    expect_no_derived_datatype();
  }

  struct file_nonblocking_dense_fixture {
    file_nonblocking_dense_fixture() {
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
          std::filesystem::temp_directory_path() / ("mp1_nb_dense_" + std::to_string(run_id));
      std::error_code error;
      if (rank == 0) {
        std::filesystem::remove_all(base_dir, error);
        error.clear();
        std::filesystem::create_directories(base_dir, error);
      }
      world_barrier();
    }

    ~file_nonblocking_dense_fixture() {
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

  // A shared-file-pointer round trip.  The shared pointer is deterministic only when the
  // processes issue their operations in a fixed order, so rank `turn` performs the operation
  // while every other rank waits at the barrier.  The file is reopened before the read pass to
  // reset the shared pointer to zero.  The callables take the file by reference because the
  // helper owns it.
  template<class T, class WriteCall, class ReadCall, class VerifyCall>
  void shared_roundtrip(const mp1::mpi_communicator& world, const std::filesystem::path& path,
                        const mp1::layout<T>& view, mp1::offset_t bytes_per_rank,
                        WriteCall&& write_call, ReadCall&& read_call,
                        VerifyCall&& verify_call) {
    const int rank = world_rank();
    const int size = world_size();
    const mp1::offset_t file_bytes =
        static_cast<mp1::offset_t>(size) * bytes_per_rank + static_cast<mp1::offset_t>(64);
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
      file.set_view(0, view);
      file.resize(file_bytes);
      for (int turn = 0; turn < size; ++turn) {
        if (rank == turn) {
          reset_wrap_counts();
          mp1::request request = write_call(file);
          expect_dense_initiation("MPI_File_iwrite_shared_c");
          (void)request.wait();
        }
        world_barrier();
      }
      file.sync();
      file.close();
    }
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_only);
      file.set_view(0, view);
      for (int turn = 0; turn < size; ++turn) {
        if (rank == turn) {
          reset_wrap_counts();
          mp1::request request = read_call(file);
          expect_dense_initiation("MPI_File_iread_shared_c");
          (void)request.wait();
          verify_call();
        }
        world_barrier();
      }
      file.close();
    }
    remove_path(path);
  }

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(file_nonblocking_dense_fixture);

BOOST_AUTO_TEST_SUITE(file_nonblocking_dense_contract)

BOOST_AUTO_TEST_CASE(explicit_offset_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("at");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size) * 64 * sizeof(int));

  // Nonzero, rank-private explicit offsets in etype units.
  const mp1::offset_t base = static_cast<mp1::offset_t>(rank) * 64 + 1;
  std::array<int, 4> data{{rank + 1, rank + 2, rank + 3, rank + 4}};
  std::array<int, 4> readback{{-1, -1, -1, -1}};

  reset_wrap_counts();
  mp1::request write_pointer = file.iwrite_at(base, std::span<int>{data});
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)write_pointer.wait();
  file.sync();

  reset_wrap_counts();
  mp1::request read_pointer = file.iread_at(base, std::span<int>{readback});
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)read_pointer.wait();
  BOOST_TEST(readback == data);

  // Contiguous vector container: the current size is the count.
  std::vector<int> vector_data{rank * 10 + 1, rank * 10 + 2, rank * 10 + 3};
  std::vector<int> vector_readback(3, -1);
  reset_wrap_counts();
  mp1::request write_vector = file.iwrite_at(base + 8, vector_data);
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)write_vector.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_vector = file.iread_at(base + 8, vector_readback);
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)read_vector.wait();
  BOOST_TEST(vector_readback == vector_data);

  // Fixed-extent array container.
  std::array<int, 2> array_data{{rank * 20 + 1, rank * 20 + 2}};
  std::array<int, 2> array_readback{{-1, -1}};
  reset_wrap_counts();
  mp1::request write_array = file.iwrite_at(base + 16, array_data);
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)write_array.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_array = file.iread_at(base + 16, array_readback);
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)read_array.wait();
  BOOST_TEST(array_readback == array_data);

  // The buffer element datatype must match the view etype, so the string spelling uses a
  // separate collective view switch to char.
  file.sync();
  file.set_view(0, char_etype());
  const mp1::offset_t char_base = static_cast<mp1::offset_t>(rank) * 64 + 2;
  const std::string string_data = std::string{"rank"} + static_cast<char>('0' + rank);
  std::string string_readback(string_data.size(), '\0');
  reset_wrap_counts();
  mp1::request write_string = file.iwrite_at(char_base, string_data);
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)write_string.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_string = file.iread_at(char_base, string_readback);
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)read_string.wait();
  BOOST_TEST(string_readback == string_data);

  // Restore the int view so the int zero/null checks exercise the same etype.
  file.set_view(0, int_etype());

  // Zero-count/no-buffer forms are no-ops that still reach the _c entry point.
  reset_wrap_counts();
  mp1::request zero_read = file.iread_at(0, std::span<int>{static_cast<int*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)zero_read.wait();
  reset_wrap_counts();
  mp1::request zero_write =
      file.iwrite_at(0, std::span<const int>{static_cast<const int*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)zero_write.wait();
  std::vector<int> empty;
  reset_wrap_counts();
  mp1::request empty_read = file.iread_at(0, empty);
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)empty_read.wait();
  reset_wrap_counts();
  mp1::request empty_write = file.iwrite_at(0, empty);
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)empty_write.wait();

  // A null buffer with a positive count and a negative count are rejected before MPI.
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iread_at(0, mp1::span_from(static_cast<int*>(nullptr), 2)); }));
  expect_wrap_count("MPI_File_iread_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iwrite_at(0, mp1::span_from(static_cast<const int*>(nullptr), 2)); }));
  expect_wrap_count("MPI_File_iwrite_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iread_at(0, mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iread_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iwrite_at(0, mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iwrite_at_c", 0);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(individual_pointer_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("individual");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size) * 64 * sizeof(int));

  const mp1::offset_t base = static_cast<mp1::offset_t>(rank) * 64 + 1;
  std::array<int, 4> data{{rank + 11, rank + 12, rank + 13, rank + 14}};
  std::array<int, 4> readback{{-1, -1, -1, -1}};

  file.seek(base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_pointer = file.iwrite(std::span<int>{data});
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)write_pointer.wait();
  file.sync();
  file.seek(base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_pointer = file.iread(std::span<int>{readback});
  expect_dense_initiation("MPI_File_iread_c");
  (void)read_pointer.wait();
  BOOST_TEST(readback == data);

  std::vector<int> vector_data{rank * 10 + 21, rank * 10 + 22};
  std::vector<int> vector_readback(2, -1);
  file.seek(base + 8, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_vector = file.iwrite(vector_data);
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)write_vector.wait();
  file.sync();
  file.seek(base + 8, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_vector = file.iread(vector_readback);
  expect_dense_initiation("MPI_File_iread_c");
  (void)read_vector.wait();
  BOOST_TEST(vector_readback == vector_data);

  std::array<int, 3> array_data{{rank * 30 + 1, rank * 30 + 2, rank * 30 + 3}};
  std::array<int, 3> array_readback{{-1, -1, -1}};
  file.seek(base + 16, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_array = file.iwrite(array_data);
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)write_array.wait();
  file.sync();
  file.seek(base + 16, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_array = file.iread(array_readback);
  expect_dense_initiation("MPI_File_iread_c");
  (void)read_array.wait();
  BOOST_TEST(array_readback == array_data);

  file.sync();
  file.set_view(0, char_etype());
  const mp1::offset_t char_base = static_cast<mp1::offset_t>(rank) * 64 + 2;
  const std::string string_data = std::string{"ptr"} + static_cast<char>('0' + rank);
  std::string string_readback(string_data.size(), '\0');
  file.seek(char_base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_string = file.iwrite(string_data);
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)write_string.wait();
  file.sync();
  file.seek(char_base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_string = file.iread(string_readback);
  expect_dense_initiation("MPI_File_iread_c");
  (void)read_string.wait();
  BOOST_TEST(string_readback == string_data);

  reset_wrap_counts();
  mp1::request zero_read = file.iread(std::span<char>{static_cast<char*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iread_c");
  (void)zero_read.wait();
  reset_wrap_counts();
  mp1::request zero_write =
      file.iwrite(std::span<const char>{static_cast<const char*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)zero_write.wait();

  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iread(mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iread_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iwrite(mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iwrite_c", 0);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(shared_pointer_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const mp1::layout<int> int_view = int_etype();

  // pointer + count
  {
    const std::filesystem::path path = test_path("shared_pointer");
    const int write_values[2] = {rank * 10 + 1, rank * 10 + 2};
    int read_values[2] = {-1, -1};
    shared_roundtrip(
        world, path, int_view, 2 * static_cast<mp1::offset_t>(sizeof(int)),
        [&](mp1::file& file) { return file.iwrite_shared(std::span<const int>{write_values}); },
        [&](mp1::file& file) { return file.iread_shared(std::span<int>{read_values}); },
        [&] {
          BOOST_TEST(read_values[0] == write_values[0]);
          BOOST_TEST(read_values[1] == write_values[1]);
        });
  }

  // vector container
  {
    const std::filesystem::path path = test_path("shared_vector");
    std::vector<int> write_values{rank * 10 + 3, rank * 10 + 4};
    std::vector<int> read_values(2, -1);
    shared_roundtrip(
        world, path, int_view, 2 * static_cast<mp1::offset_t>(sizeof(int)),
        [&](mp1::file& file) { return file.iwrite_shared(write_values); },
        [&](mp1::file& file) { return file.iread_shared(read_values); },
        [&] { BOOST_TEST(read_values == write_values); });
  }

  // array container
  {
    const std::filesystem::path path = test_path("shared_array");
    const std::array<int, 2> write_values{{rank * 10 + 5, rank * 10 + 6}};
    std::array<int, 2> read_values{{-1, -1}};
    shared_roundtrip(
        world, path, int_view, 2 * static_cast<mp1::offset_t>(sizeof(int)),
        [&](mp1::file& file) { return file.iwrite_shared(write_values); },
        [&](mp1::file& file) { return file.iread_shared(read_values); },
        [&] { BOOST_TEST(read_values == write_values); });
  }

  // string container with the matching char view
  {
    const std::filesystem::path path = test_path("shared_string");
    const mp1::layout<char> char_view = char_etype();
    const std::string write_values = std::string{"sh"} + static_cast<char>('0' + rank);
    std::string read_values(write_values.size(), '\0');
    shared_roundtrip(
        world, path, char_view, static_cast<mp1::offset_t>(write_values.size()),
        [&](mp1::file& file) { return file.iwrite_shared(write_values); },
        [&](mp1::file& file) { return file.iread_shared(read_values); },
        [&] { BOOST_TEST(read_values == write_values); });
  }

  // Zero-count shared forms are still lowered to the _c entry point; they do not advance the
  // shared pointer.
  {
    const std::filesystem::path path = test_path("shared_zero");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, int_view);
    file.resize(static_cast<mp1::offset_t>(64));
    reset_wrap_counts();
    mp1::request zero_write =
        file.iwrite_shared(std::span<const int>{static_cast<const int*>(nullptr), 0});
    expect_dense_initiation("MPI_File_iwrite_shared_c");
    (void)zero_write.wait();
    reset_wrap_counts();
    mp1::request zero_read = file.iread_shared(std::span<int>{static_cast<int*>(nullptr), 0});
    expect_dense_initiation("MPI_File_iread_shared_c");
    (void)zero_read.wait();
    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument([&] {
      (void)file.iwrite_shared(mp1::span_from(static_cast<const int*>(nullptr), 2));
    }));
    expect_wrap_count("MPI_File_iwrite_shared_c", 0);
    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)file.iread_shared(mp1::span_from(static_cast<int*>(nullptr), 2)); }));
    expect_wrap_count("MPI_File_iread_shared_c", 0);
    file.sync();
    file.close();
    remove_path(path);
  }
}

BOOST_AUTO_TEST_CASE(explicit_offset_collective_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("at_all");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size) * 64 * sizeof(int));

  const mp1::offset_t base = static_cast<mp1::offset_t>(rank) * 64 + 1;
  std::array<int, 4> data{{rank + 21, rank + 22, rank + 23, rank + 24}};
  std::array<int, 4> readback{{-1, -1, -1, -1}};

  reset_wrap_counts();
  mp1::request write_pointer = file.iwrite_at_all(base, std::span<int>{data});
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)write_pointer.wait();
  file.sync();

  // Collective visibility: every rank reads its successor's private block.
  const int next = (rank + 1) % size;
  const mp1::offset_t next_base = static_cast<mp1::offset_t>(next) * 64 + 1;
  std::array<int, 4> cross_readback{{-1, -1, -1, -1}};
  reset_wrap_counts();
  mp1::request read_cross = file.iread_at_all(next_base, std::span<int>{cross_readback});
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)read_cross.wait();
  BOOST_TEST(cross_readback[0] == next + 21);
  BOOST_TEST(cross_readback[1] == next + 22);
  BOOST_TEST(cross_readback[2] == next + 23);
  BOOST_TEST(cross_readback[3] == next + 24);

  reset_wrap_counts();
  mp1::request read_own = file.iread_at_all(base, std::span<int>{readback});
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)read_own.wait();
  BOOST_TEST(readback == data);

  // vector container
  std::vector<int> vector_data{rank * 10 + 31, rank * 10 + 32};
  std::vector<int> vector_readback(2, -1);
  reset_wrap_counts();
  mp1::request write_vector = file.iwrite_at_all(base + 8, vector_data);
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)write_vector.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_vector = file.iread_at_all(base + 8, vector_readback);
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)read_vector.wait();
  BOOST_TEST(vector_readback == vector_data);

  // array container
  std::array<int, 2> array_data{{rank * 20 + 31, rank * 20 + 32}};
  std::array<int, 2> array_readback{{-1, -1}};
  reset_wrap_counts();
  mp1::request write_array = file.iwrite_at_all(base + 16, array_data);
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)write_array.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_array = file.iread_at_all(base + 16, array_readback);
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)read_array.wait();
  BOOST_TEST(array_readback == array_data);

  // string container under the matching char view
  file.sync();
  file.set_view(0, char_etype());
  const mp1::offset_t char_base = static_cast<mp1::offset_t>(rank) * 64 + 2;
  const std::string string_data = std::string{"all"} + static_cast<char>('0' + rank);
  std::string string_readback(string_data.size(), '\0');
  reset_wrap_counts();
  mp1::request write_string = file.iwrite_at_all(char_base, string_data);
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)write_string.wait();
  file.sync();
  reset_wrap_counts();
  mp1::request read_string = file.iread_at_all(char_base, string_readback);
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)read_string.wait();
  BOOST_TEST(string_readback == string_data);

  // Restore the int view for the int zero-count collective forms.
  file.set_view(0, int_etype());

  // Zero-count collective forms: all ranks enter the same collective in the same order.
  reset_wrap_counts();
  mp1::request zero_write =
      file.iwrite_at_all(0, std::span<const int>{static_cast<const int*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)zero_write.wait();
  reset_wrap_counts();
  mp1::request zero_read = file.iread_at_all(0, std::span<int>{static_cast<int*>(nullptr), 0});
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)zero_read.wait();

  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iread_at_all(0, mp1::span_from(static_cast<int*>(nullptr), 1)); }));
  expect_wrap_count("MPI_File_iread_at_all_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument([&] {
    (void)file.iwrite_at_all(0, mp1::span_from(static_cast<const int*>(nullptr), 1));
  }));
  expect_wrap_count("MPI_File_iwrite_at_all_c", 0);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(collective_pointer_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("all");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size) * 64 * sizeof(int));

  const mp1::offset_t base = static_cast<mp1::offset_t>(rank) * 64 + 1;
  std::array<int, 4> data{{rank + 41, rank + 42, rank + 43, rank + 44}};
  std::array<int, 4> readback{{-1, -1, -1, -1}};

  file.seek(base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_pointer = file.iwrite_all(std::span<int>{data});
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)write_pointer.wait();
  file.sync();

  file.seek(base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_pointer = file.iread_all(std::span<int>{readback});
  expect_dense_initiation("MPI_File_iread_all_c");
  (void)read_pointer.wait();
  BOOST_TEST(readback == data);

  std::vector<int> vector_data{rank * 10 + 51, rank * 10 + 52};
  std::vector<int> vector_readback(2, -1);
  file.seek(base + 8, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_vector = file.iwrite_all(vector_data);
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)write_vector.wait();
  file.sync();
  file.seek(base + 8, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_vector = file.iread_all(vector_readback);
  expect_dense_initiation("MPI_File_iread_all_c");
  (void)read_vector.wait();
  BOOST_TEST(vector_readback == vector_data);

  std::array<int, 3> array_data{{rank * 30 + 1, rank * 30 + 2, rank * 30 + 3}};
  std::array<int, 3> array_readback{{-1, -1, -1}};
  file.seek(base + 16, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_array = file.iwrite_all(array_data);
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)write_array.wait();
  file.sync();
  file.seek(base + 16, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_array = file.iread_all(array_readback);
  expect_dense_initiation("MPI_File_iread_all_c");
  (void)read_array.wait();
  BOOST_TEST(array_readback == array_data);

  file.sync();
  file.set_view(0, char_etype());
  const mp1::offset_t char_base = static_cast<mp1::offset_t>(rank) * 64 + 2;
  const std::string string_data = std::string{"col"} + static_cast<char>('0' + rank);
  std::string string_readback(string_data.size(), '\0');
  file.seek(char_base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request write_string = file.iwrite_all(string_data);
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)write_string.wait();
  file.sync();
  file.seek(char_base, mp1::whence_mode::set);
  reset_wrap_counts();
  mp1::request read_string = file.iread_all(string_readback);
  expect_dense_initiation("MPI_File_iread_all_c");
  (void)read_string.wait();
  BOOST_TEST(string_readback == string_data);

  // Restore the int view for the int empty-container form.
  file.set_view(0, int_etype());

  std::vector<int> empty;
  reset_wrap_counts();
  mp1::request empty_write = file.iwrite_all(empty);
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)empty_write.wait();

  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iread_all(mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iread_all_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iwrite_all(mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iwrite_all_c", 0);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(zero_null_and_local_rejections_across_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int size = world_size();
  const std::filesystem::path path = test_path("zero");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, int_etype());
  file.resize(static_cast<mp1::offset_t>(size) * 32 * sizeof(int));

  int* const null_read = nullptr;
  const int* const null_write = nullptr;
  std::array<int, 4> data{{1, 2, 3, 4}};
  std::vector<int> empty;

  // iread_at / iwrite_at
  reset_wrap_counts();
  mp1::request at_read_zero = file.iread_at(0, std::span<int>{null_read, 0});
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)at_read_zero.wait();
  reset_wrap_counts();
  mp1::request at_write_zero = file.iwrite_at(0, std::span<const int>{null_write, 0});
  expect_dense_initiation("MPI_File_iwrite_at_c");
  (void)at_write_zero.wait();
  reset_wrap_counts();
  mp1::request at_read_empty = file.iread_at(0, empty);
  expect_dense_initiation("MPI_File_iread_at_c");
  (void)at_read_empty.wait();

  // iread / iwrite
  reset_wrap_counts();
  mp1::request read_zero = file.iread(std::span<int>{null_read, 0});
  expect_dense_initiation("MPI_File_iread_c");
  (void)read_zero.wait();
  reset_wrap_counts();
  mp1::request write_zero = file.iwrite(std::span<const int>{null_write, 0});
  expect_dense_initiation("MPI_File_iwrite_c");
  (void)write_zero.wait();

  // iread_shared / iwrite_shared
  reset_wrap_counts();
  mp1::request shared_read_zero = file.iread_shared(std::span<int>{null_read, 0});
  expect_dense_initiation("MPI_File_iread_shared_c");
  (void)shared_read_zero.wait();
  reset_wrap_counts();
  mp1::request shared_write_zero = file.iwrite_shared(std::span<const int>{null_write, 0});
  expect_dense_initiation("MPI_File_iwrite_shared_c");
  (void)shared_write_zero.wait();

  // iread_at_all / iwrite_at_all
  reset_wrap_counts();
  mp1::request at_all_read_zero = file.iread_at_all(0, std::span<int>{null_read, 0});
  expect_dense_initiation("MPI_File_iread_at_all_c");
  (void)at_all_read_zero.wait();
  reset_wrap_counts();
  mp1::request at_all_write_zero = file.iwrite_at_all(0, std::span<const int>{null_write, 0});
  expect_dense_initiation("MPI_File_iwrite_at_all_c");
  (void)at_all_write_zero.wait();

  // iread_all / iwrite_all
  reset_wrap_counts();
  mp1::request all_read_zero = file.iread_all(std::span<int>{null_read, 0});
  expect_dense_initiation("MPI_File_iread_all_c");
  (void)all_read_zero.wait();
  reset_wrap_counts();
  mp1::request all_write_zero = file.iwrite_all(std::span<const int>{null_write, 0});
  expect_dense_initiation("MPI_File_iwrite_all_c");
  (void)all_write_zero.wait();

  // A null buffer with a positive count is rejected before every MPI entry point.
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iread_at(0, mp1::span_from(null_read, 1)); }));
  expect_wrap_count("MPI_File_iread_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iwrite_at(0, mp1::span_from(null_write, 1)); }));
  expect_wrap_count("MPI_File_iwrite_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument([&] { (void)file.iread(mp1::span_from(null_read, 1)); }));
  expect_wrap_count("MPI_File_iread_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iwrite(mp1::span_from(null_write, 1)); }));
  expect_wrap_count("MPI_File_iwrite_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iread_shared(mp1::span_from(null_read, 1)); }));
  expect_wrap_count("MPI_File_iread_shared_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iwrite_shared(mp1::span_from(null_write, 1)); }));
  expect_wrap_count("MPI_File_iwrite_shared_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iread_at_all(0, mp1::span_from(null_read, 1)); }));
  expect_wrap_count("MPI_File_iread_at_all_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iwrite_at_all(0, mp1::span_from(null_write, 1)); }));
  expect_wrap_count("MPI_File_iwrite_at_all_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iread_all(mp1::span_from(null_read, 1)); }));
  expect_wrap_count("MPI_File_iread_all_c", 0);
  reset_wrap_counts();
  BOOST_TEST(
      throws_invalid_argument([&] { (void)file.iwrite_all(mp1::span_from(null_write, 1)); }));
  expect_wrap_count("MPI_File_iwrite_all_c", 0);

  // A negative count is rejected by the shared count validator.
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iread_at(0, mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iread_at_c", 0);
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)file.iwrite_at(0, mp1::span_from(data.data(), -1)); }));
  expect_wrap_count("MPI_File_iwrite_at_c", 0);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
