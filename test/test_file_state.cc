// File state-machine, lifetime, view/etype and 1.0-addition contract tests.
//
// MPI rules exercised here (official MPI 4.1 sections):
//   - 14.2.1 open/amode, 14.2.2 close collective and request completion,
//   - 14.3 views and the initial byte view,
//   - 14.4.4 shared file pointer seek/position,
//   - 14.4.5 split collectives: one active begin, matching end, same buffer/thread,
//   - 14.5.1 MPI_File_get_type_extent,
//   - 14.6.1 atomicity and sync, 14.6.2 sequential rules, 14.6.6 type matching.
//
// Every collective call is made by all ranks with identical control flow, so the same source
// runs at 2 and 4 ranks. Non-collective per-rank files use MPI_COMM_SELF.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>


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

  template<class F>
  bool throws_invalid_argument(F&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_runtime_error(F&& function) {
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

  std::filesystem::path test_path(const std::string& name) {
    return base_dir / (std::string{"mp1_state_"} + name + ".bin");
  }

  std::filesystem::path rank_path(const std::string& name, int rank) {
    return base_dir / (std::string{"mp1_state_"} + name + "_" + std::to_string(rank) + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

}  // namespace


struct mp1_file_state_fixture {
  mp1_file_state_fixture() {
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
        std::filesystem::temp_directory_path() / ("mp1_state_file_" + std::to_string(run_id));
    std::error_code error;
    if (rank == 0) {
      std::filesystem::remove_all(base_dir, error);
      error.clear();
      std::filesystem::create_directories(base_dir, error);
    }
    world_barrier();
  }

  ~mp1_file_state_fixture() {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove_all(base_dir, error);
    }
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Another owner may finalize; a fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_file_state_fixture);

BOOST_AUTO_TEST_SUITE(file_state_test)

BOOST_AUTO_TEST_CASE(split_state_machine_rejects_local_mismatches) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("split_state");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(size * sizeof(int)));

  int value = rank + 100;
  file.seek(rank, mp1::whence_mode::set);

  // End without begin never reaches MPI (Open MPI otherwise segfaults).
  BOOST_TEST(throws_runtime_error([&file, &value] { (void)file.read_at_all_end(value); }));
  BOOST_TEST(throws_runtime_error([&file, &value] { (void)file.write_all_end(value); }));

  file.write_all_begin(value);

  // A second begin while one is active, a mismatched end and a different end buffer are all
  // local errors; the correct end must still work afterwards.
  BOOST_TEST(throws_runtime_error([&file, &value] { file.write_all_begin(value); }));
  BOOST_TEST(throws_runtime_error([&file, &value] { (void)file.read_all_end(value); }));
  BOOST_TEST(throws_runtime_error([&file, &value] { (void)file.write_at_all_end(value); }));
  int other = value;
  BOOST_TEST(throws_runtime_error([&file, &other] { (void)file.write_all_end(other); }));
  BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
  BOOST_TEST(throws_runtime_error([&file] { file.close(); }));
  BOOST_TEST(file.is_open());

  // Non-collective access remains legal during the split.
  int during_split{0};
  (void)file.read_at(rank, during_split);

  (void)file.write_all_end(value);

  // Double end: the epoch is gone, so the second end is a local error too.
  BOOST_TEST(throws_runtime_error([&file, &value] { (void)file.write_all_end(value); }));

  file.sync();
  int readback{0};
  (void)file.read_at(rank, readback);
  BOOST_TEST(readback == value);
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(destructor_finishes_an_active_split) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("destructor_split");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  {
    // The buffer must outlive the file object: the destructor passes the recorded pointer to
    // the matching collective end.
    int value = rank + 200;
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, layout_one);
    file.resize(static_cast<mp1::offset_t>(size * sizeof(int)));
    file.seek(rank, mp1::whence_mode::set);
    file.write_all_begin(value);
  }  // ~file(): MPI_File_write_all_end + MPI_File_close, collectively on every rank

  mp1::file reader;
  reader.open(world, path, mp1::access_mode::read_only);
  reader.set_view(0, layout_one);
  int readback{0};
  (void)reader.read_at(rank, readback);
  BOOST_TEST(readback == rank + 200);
  reader.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(destructor_finishes_split_for_a_peer_rank) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("destructor_peer_split");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  // Reproducer: rank 0 destroys the file mid-split, rank 1 ends explicitly and
  // closes. The destructor must issue the matching collective end, otherwise rank 1 hangs.
  int value = rank + 250;
  std::optional<mp1::file> file{std::in_place};
  file->open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file->set_view(0, layout_one);
  file->resize(static_cast<mp1::offset_t>(size * sizeof(int)));
  file->seek(rank, mp1::whence_mode::set);
  file->write_all_begin(value);
  world_barrier();

  if (rank == 0) {
    file.reset();  // ~file(): MPI_File_write_all_end + MPI_File_close
  } else {
    (void)file->write_all_end(value);
    file->close();
  }
  world_barrier();

  mp1::file reader;
  reader.open(world, path, mp1::access_mode::read_only);
  reader.set_view(0, layout_one);
  int readback{0};
  (void)reader.read_at(rank, readback);
  BOOST_TEST(readback == rank + 250);
  reader.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(nonblocking_request_lease_outlives_the_file_object) {
  const mp1::mpi_communicator self = mp1::comm_self();
  const int rank = world_rank();
  const std::filesystem::path path = rank_path("request_lease", rank);
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);
  const int value = rank + 300;

  mp1::request request;
  {
    mp1::file file;
    file.open(self, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, layout_one);
    file.resize(static_cast<mp1::offset_t>(sizeof(int)));
    request = file.iwrite_at(0, value);

    // A pending wrapper-owned request makes close() and sync() refuse before MPI.
    BOOST_TEST(throws_runtime_error([&file] { file.close(); }));
    BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
    BOOST_TEST(file.is_open());
  }  // ~file(): the handle is abandoned because one request is still outstanding

  // The lease kept the shared state alive; completion works after the owner is gone.
  request.wait();

  {
    mp1::file reader;
    reader.open(self, path, mp1::access_mode::read_only);
    reader.set_view(0, layout_one);
    int readback{0};
    (void)reader.read_at(0, readback);
    BOOST_TEST(readback == value);
    reader.close();
  }
  mp1::file::remove(path);
}

BOOST_AUTO_TEST_CASE(view_etype_matching_and_byte_wildcard) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // Typed double view: an int access is rejected locally (no MPI call, no wrong data).
  {
    const std::filesystem::path path = test_path("etype_double");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, mp1::layout<double>::contiguous(1));
    int wrong{0};
    BOOST_TEST(throws_invalid_argument([&file, &wrong] { (void)file.read_at(0, wrong); }));
    BOOST_TEST(throws_invalid_argument([&file, &wrong] { (void)file.write_at(0, wrong); }));

    double value = 3.25;
    (void)file.write_at(0, value);
    double readback{0.0};
    (void)file.read_at(0, readback);
    BOOST_TEST(readback == value);

    // The byte view is MPI's wildcard (MPI 4.1 section 14.6.6).
    file.set_view(0);
    int byte_read{0};
    (void)file.read_at(0, byte_read);
    file.close();
    remove_path(path);
  }

  // Before the first set_view the view is the initial byte view: typed access is allowed.
  {
    const std::filesystem::path path = test_path("initial_byte_view");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    int value{0};
    (void)file.read_at(0, value);
    file.close();
    remove_path(path);
  }

  // std::pair is a valid file element type (datatype_traits gate, not trivial copyability).
  {
    const std::filesystem::path path = test_path("pair");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    const mp1::layout<std::pair<int, int>> pair_layout =
        mp1::layout<std::pair<int, int>>::contiguous(1);
    file.set_view(0, pair_layout);
    const std::pair<int, int> value{rank, rank + 1};
    (void)file.write_at(rank, value);
    std::pair<int, int> readback{0, 0};
    (void)file.read_at(rank, readback);
    BOOST_TEST(readback.first == value.first);
    BOOST_TEST(readback.second == value.second);
    file.close();
    remove_path(path);
  }

  // Zero-size layout with a null buffer: no element is accessed and the call is accepted.
  {
    const std::filesystem::path path = test_path("zero_size");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
    file.set_view(0, empty);
    int* null_buffer = nullptr;
    const mp1::status result = file.read_at(0, mp1::region{null_buffer, empty});
    BOOST_TEST(result.get_count<int>() == 0);
    file.read_at_all_begin(0, mp1::region{null_buffer, empty});
    const mp1::status split_result = file.read_at_all_end(std::span<int>{null_buffer, 0});
    BOOST_TEST(split_result.get_count<int>() == 0);
    file.close();
    remove_path(path);
  }
}

BOOST_AUTO_TEST_CASE(shared_pointer_atomicity_and_type_extent) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path path = test_path("shared_atomicity_extent");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(4 * sizeof(int)));

  file.set_atomicity(true);
  BOOST_TEST(file.get_atomicity());
  BOOST_TEST(file.get_atomicity());
  file.set_atomicity(false);
  BOOST_TEST(not file.get_atomicity());

  // Native representation: MPI_File_get_type_extent agrees with the local layout extent.
  const MPI_Aint scalar_extent = file.get_type_extent<int>();
  const MPI_Aint layout_extent = file.get_type_extent(layout_one);
  BOOST_TEST(scalar_extent == static_cast<MPI_Aint>(layout_one.byte_extent()));
  BOOST_TEST(layout_extent == scalar_extent);
  BOOST_TEST(file.get_type_extent<int>() == scalar_extent);

  // seek_shared is collective with identical arguments; get_position_shared is not.
  file.seek_shared(0, mp1::whence_mode::set);
  world_barrier();
  BOOST_TEST(file.get_position_shared() == 0);
  BOOST_TEST(file.get_position_shared() == 0);
  world_barrier();
  if (rank == 0)
    (void)file.write_shared(0x5A5A);
  world_barrier();
  file.sync();
  BOOST_TEST(file.get_position_shared() == 1);
  BOOST_TEST(file.size() == static_cast<mp1::offset_t>(4 * sizeof(int)));

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(amode_validation_and_direction) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("amode");

  // Errors that make the MPI call invalid are diagnosed before MPI; amode is uniform.
  mp1::file invalid;
  BOOST_TEST(throws_invalid_argument(
      [&invalid, &world, &path] { invalid.open(world, path, mp1::access_mode::create); }));
  BOOST_TEST(throws_invalid_argument([&invalid, &world, &path] {
    invalid.open(world, path, mp1::access_mode::read_only | mp1::access_mode::create);
  }));
  BOOST_TEST(throws_invalid_argument([&invalid, &world, &path] {
    invalid.open(world, path, mp1::access_mode::read_write | mp1::access_mode::sequential);
  }));
  BOOST_TEST(not invalid.is_open());

  mp1::file write_only;
  write_only.open(world, path, mp1::access_mode::write_only | mp1::access_mode::create);
  int value{0};
  BOOST_TEST(
      throws_runtime_error([&write_only, &value] { (void)write_only.read_at(0, value); }));
  BOOST_TEST(throws_runtime_error([&write_only, &value] { (void)write_only.read(value); }));
  write_only.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(adopt_and_release) {
  const mp1::mpi_communicator self = mp1::comm_self();
  const int rank = world_rank();
  const std::filesystem::path path = rank_path("adopt_release", rank);
  const std::string filename = path.string();

  MPI_File raw{MPI_FILE_NULL};
  BOOST_REQUIRE(MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDWR | MPI_MODE_CREATE,
                              MPI_INFO_NULL, &raw) == MPI_SUCCESS);

  mp1::file adopted = mp1::file::adopt(raw);
  BOOST_TEST(adopted.is_open());
  BOOST_TEST(adopted.native_handle() == raw);
  BOOST_TEST((static_cast<int>(adopted.mode()) & MPI_MODE_RDWR) != 0);

  // The adopted view is unknown, so typed access is delegated to MPI's type rule.
  int value = rank + 400;
  (void)adopted.write_at(0, value);

  MPI_File handed_back = adopted.release();
  BOOST_TEST(handed_back == raw);
  BOOST_TEST(not adopted.is_open());
  BOOST_TEST(adopted.release() == MPI_FILE_NULL);

  BOOST_REQUIRE(MPI_File_close(&handed_back) == MPI_SUCCESS);
  mp1::file::remove(path);
}

BOOST_AUTO_TEST_SUITE_END()
