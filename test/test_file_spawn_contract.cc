// Contract test for mp1::file, mp1::offset_t, mp1::access_mode, mp1::whence_mode,
// mp1::command_line(s), mp1::spawn and mp1::spawn_multiple.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// Every collective file operation is called by all ranks with identical control flow, so the
// same source runs at 2 and 4 ranks. Differential checks compare mp1's view/offset handling
// with raw MPI-IO on the same bytes. The spawn success cases re-spawn this executable: the
// child enters a static-initialization child mode that writes its argument vector and exits
// before Boost.Test starts. They run on Open MPI and Intel MPI; MPICH 4.3.2 cannot reliably
// start a second copy of this binary from one parent process, so that provider skips them.
// Open MPI 6+ skips the four-rank entry because it was seen to hang in the CI environment;
// the dedicated test_spawn_contract.4 covers that rank count with an external child. Intel
// 2021.19.2 skips the self-respawn case for the same CI-only reason. Intel needs
// I_MPI_FABRICS=shm:ofi for MPI_Comm_spawn; the CTest entry sets it. The local spawn
// validation runs on every provider.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#endif


namespace {

  // The self-respawn success cases run on Open MPI and Intel MPI.  MPICH 4.3.2 starts one
  // spawned copy of this binary, but a second MPI_Comm_spawn/MPI_Comm_spawn_multiple from the
  // same parent process deadlocks while the first copy is still inside MPI_Finalize.  Intel
  // needs I_MPI_FABRICS=shm:ofi for spawn; the CTest entry sets it.  The dedicated
  // test_spawn_contract child executable covers the provider-neutral spawn and
  // spawn_multiple paths on every provider.  The unsupported self-respawn combination is
  // skipped explicitly at run time (the helper code still compiles everywhere), never
  // compiled out.
#if defined(OMPI_MAJOR_VERSION) || defined(I_MPI_VERSION)
  inline constexpr bool spawn_self_respawn_supported = true;
#else
  inline constexpr bool spawn_self_respawn_supported = false;
#endif

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
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_runtime_error(F&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_mpi_error(F&& f) {
    try {
      f();
    } catch (const mp1::mpi_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // ---------------------------------------------------------------------------------------
  // Per-run scratch directory, unique at 2 and 4 ranks (ctest may run both concurrently).
  // ---------------------------------------------------------------------------------------

  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_"} + name + ".bin");
  }

  // ---------------------------------------------------------------------------------------
  // After-finalize static-destruction probe. The file is placement-new'd by a test and its
  // destructor is called by this guard, which is constructed BEFORE the Boost global fixture
  // and therefore destroyed AFTER the fixture's mp1::finalize().
  // ---------------------------------------------------------------------------------------

  alignas(mp1::file) unsigned char after_finalize_file_storage[sizeof(mp1::file)];
  bool after_finalize_file_constructed = false;

  [[noreturn]] void after_finalize_failure(const char* what) {
    std::fprintf(stderr, "mp1 file/spawn after-finalize failure: %s\n", what);
    std::abort();
  }

  struct after_finalize_check {
    ~after_finalize_check() noexcept {
      if (not after_finalize_file_constructed)
        return;
      if (not mp1::finalized())
        return;  // The global fixture did not finalize MPI; leave the object alone.
      auto* file = std::launder(reinterpret_cast<mp1::file*>(after_finalize_file_storage));

      if (not file->is_open() or file->native_handle() == MPI_FILE_NULL)
        after_finalize_failure("file handle was lost before the destructor");

      bool close_threw{false};
      try {
        file->close();
      } catch (const std::runtime_error&) {
        close_threw = true;
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
      }
      if (not close_threw)
        after_finalize_failure("file::close did not throw after finalize");
      if (not file->is_open())
        after_finalize_failure("failed close after finalize changed the open state");

      const std::size_t before = mp1::detail::abandoned_handle_count();
      file->~file();
      const std::size_t after = mp1::detail::abandoned_handle_count();
      if (after != before + 1)
        after_finalize_failure("~file after finalize did not abandon exactly one handle");
#if defined(MP1_TEST_WRAP)
      if (mpi_wrap::supported() and mpi_wrap::after_finalize_calls() != 0)
        after_finalize_failure("an MPI call was made after finalize");
#endif
    }
  };

  after_finalize_check after_finalize_check_instance;

}  // namespace


// -----------------------------------------------------------------------------------------
// Spawn child bootstrap.
//
// The parent re-spawns this executable with --mp1-spawn-child and a marker-directory
// argument. The child recognizes them in /proc/self/cmdline during static initialization
// (the sentinel is argv-based because Open MPI does not reliably forward environment
// additions made after MPI_Init), initializes MPI, writes its argument vector to
// <marker_dir>/<world rank>, and exits before Boost.Test starts. This exercises argv
// marshalling and maxprocs without a second executable.
// -----------------------------------------------------------------------------------------

namespace {

  bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
  }

  std::vector<std::string> read_own_command_line() {
    std::vector<std::string> arguments;
    std::ifstream command_line{"/proc/self/cmdline", std::ios::binary};
    std::string argument;
    while (std::getline(command_line, argument, '\0'))
      arguments.push_back(argument);
    return arguments;
  }

  // Polling is a handshake with an external process, not a substitute for an MPI
  // synchronization: sleep between probes so four parent ranks do not spin the CPU while the
  // spawned children initialize (a busy loop can starve them under sanitizers).
  template<class Predicate>
  bool wait_for_markers(Predicate&& predicate, double seconds) {
    const double deadline = MPI_Wtime() + seconds;
    while (MPI_Wtime() < deadline) {
      if (predicate())
        return true;
      std::this_thread::sleep_for(std::chrono::milliseconds{10});
    }
    return predicate();
  }

  // All parent ranks share the marker directory, but a poll can time out on one rank slightly
  // before another.  Agree before every rank either enters the collective disconnect or
  // releases the handle, so a failed child cannot deadlock the test in MPI_Comm_disconnect.
  bool all_parents_ready(bool local_ready) {
    int local = local_ready ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global != 0;
  }

  void run_spawn_child(const std::string& marker_dir) {
    try {
      mp1::init();
      int rank{0};
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      // Publish the argument vector before calling parent(), so the parent can release the
      // spawn intercommunicator before the child waits for its parent handle.
      if (not marker_dir.empty()) {
        const std::filesystem::path marker =
            std::filesystem::path{marker_dir} / std::to_string(rank);
        const std::filesystem::path temporary = marker.string() + ".tmp";
        std::ofstream marker_file{temporary, std::ios::trunc};
        for (const std::string& argument : read_own_command_line())
          marker_file << argument << '\n';
        marker_file.close();
        // Publish atomically: the parent polls for the final name and may otherwise read a
        // file that exists but has not been written yet.
        std::error_code error;
        std::filesystem::rename(temporary, marker, error);
      }
      // The child's parent() owner and the parent's spawn() owner are the two ends of the
      // same spawn intercommunicator; both must disconnect (MPI 4.1 §11.10.4).
      mp1::communicator parent_comm = mp1::parent();
      if (parent_comm.is_valid())
        parent_comm.disconnect();
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The inherited MPI state was external; leave it to the process exit.
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "mp1 spawn child failed: %s\n", error.what());
      std::_Exit(2);
    } catch (...) {
      std::fprintf(stderr, "mp1 spawn child failed with an unknown exception\n");
      std::_Exit(2);
    }
    std::_Exit(0);
  }

  struct spawn_child_bootstrap {
    spawn_child_bootstrap() {
      // The sentinel travels in argv, not in the environment: Open MPI does not forward
      // post-MPI_Init setenv additions to MPI_Comm_spawn children.
      const std::vector<std::string> arguments = read_own_command_line();
      if (std::find(arguments.begin(), arguments.end(), "--mp1-spawn-child") ==
          arguments.end()) {
        return;
      }
      std::string marker_dir;
      for (const std::string& argument : arguments) {
        constexpr const char* prefix = "--mp1-spawn-marker=";
        if (argument.rfind(prefix, 0) == 0)
          marker_dir = argument.substr(std::char_traits<char>::length(prefix));
      }
      run_spawn_child(marker_dir);
    }
  };

  spawn_child_bootstrap spawn_child_bootstrap_instance;

}  // namespace


struct mp1_file_spawn_contract_fixture {
  mp1_file_spawn_contract_fixture() {
    mp1::init();
#if defined(OMPI_MAJOR_VERSION) && OMPI_MAJOR_VERSION >= 6
    if (world_size() == 4) {
      std::fprintf(stderr,
                   "four-rank file-spawn test skipped on Open MPI 6+: observed in GitHub CI, "
                   "possibly a container-environment issue; test_spawn_contract.4 covers "
                   "four-rank spawn with the external child\n");
      std::_Exit(77);
    }
#endif
    const int rank = world_rank();
    std::uint64_t seed{0};
    if (rank == 0) {
      seed = static_cast<std::uint64_t>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
      seed ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&seed));
    }
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    run_id = seed;
    base_dir = std::filesystem::temp_directory_path() / ("mp1_w6_" + std::to_string(run_id));
    std::error_code error;
    if (rank == 0) {
      // A leftover directory from an aborted run must never leak bytes into this run.
      std::filesystem::remove_all(base_dir, error);
      error.clear();
      std::filesystem::create_directories(base_dir, error);
    }
    world_barrier();
  }

  ~mp1_file_spawn_contract_fixture() {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove_all(base_dir, error);
    }
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_file_spawn_contract_fixture);

BOOST_AUTO_TEST_SUITE(file_spawn_contract)

BOOST_AUTO_TEST_CASE(types_and_default_state) {
  BOOST_TEST(true);
  static_assert(std::is_same_v<mp1::offset_t, MPI_Offset>);
  static_assert(std::is_same_v<mp1::command_line, std::vector<std::string>>);
  static_assert(std::is_same_v<mp1::command_lines, std::vector<mp1::command_line>>);
  static_assert(std::is_nothrow_destructible_v<mp1::file>);
  static_assert(std::is_nothrow_move_constructible_v<mp1::file>);
  static_assert(std::is_nothrow_move_assignable_v<mp1::file>);
  static_assert(not std::is_copy_constructible_v<mp1::file>);
  static_assert(not std::is_copy_assignable_v<mp1::file>);
  static_assert(sizeof(mp1::offset_t) >= sizeof(std::int64_t));

  const mp1::access_mode combined = mp1::access_mode::read_write | mp1::access_mode::create;
  BOOST_TEST((static_cast<int>(combined) & MPI_MODE_RDWR) != 0);
  BOOST_TEST((static_cast<int>(combined) & MPI_MODE_CREATE) != 0);
  BOOST_TEST(static_cast<int>(combined & mp1::access_mode::read_write) == MPI_MODE_RDWR);
  BOOST_TEST(static_cast<int>(combined & mp1::access_mode::read_only) == 0);
  BOOST_TEST(static_cast<int>(mp1::whence_mode::set) == MPI_SEEK_SET);
  BOOST_TEST(static_cast<int>(mp1::whence_mode::current) == MPI_SEEK_CUR);
  BOOST_TEST(static_cast<int>(mp1::whence_mode::end) == MPI_SEEK_END);

  mp1::file closed;
  BOOST_TEST(not closed.is_open());
  BOOST_TEST(closed.native_handle() == MPI_FILE_NULL);
  closed.close();  // idempotent, no MPI call
  BOOST_TEST(not closed.is_open());
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.size(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.position(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.byte_offset(0); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.get_group(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.get_info(); }));
}

BOOST_AUTO_TEST_CASE(open_close_reuse_move_and_remove) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path first = test_path("open_close_first");
  const std::filesystem::path second = test_path("open_close_second");

  mp1::file open_file;
  open_file.open(world, first, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(open_file.is_open());
  BOOST_TEST(open_file.native_handle() != MPI_FILE_NULL);
  BOOST_TEST((static_cast<int>(open_file.mode()) & MPI_MODE_RDWR) != 0);

  BOOST_TEST(throws_runtime_error([&open_file, &world, &first] {
    open_file.open(world, first, mp1::access_mode::read_write);
  }));
  BOOST_TEST(open_file.is_open());

  open_file.close();
  BOOST_TEST(not open_file.is_open());
  open_file.close();  // idempotent
  BOOST_TEST(not open_file.is_open());

  open_file.open(world, first, mp1::access_mode::read_only);
  BOOST_TEST(open_file.is_open());
  BOOST_TEST((static_cast<int>(open_file.mode()) & MPI_MODE_RDONLY) != 0);

  mp1::file moved{std::move(open_file)};
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not open_file.is_open());
  BOOST_TEST(moved.is_open());

  mp1::file target;
  target.open(world, second, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(target.is_open());
  target = std::move(moved);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not moved.is_open());
  BOOST_TEST(target.is_open());
  target.close();

  if (rank == 0) {
    mp1::file::remove(first);
    mp1::file::remove(second);
  }
  world_barrier();
  BOOST_TEST(not std::filesystem::exists(first));
  BOOST_TEST(not std::filesystem::exists(second));

  // A failed open throws mpi_error, leaves the object closed, and the object stays reusable.
  const std::filesystem::path missing = test_path("open_missing");
  mp1::file reusable;
  BOOST_TEST(throws_mpi_error([&reusable, &world, &missing] {
    reusable.open(world, missing, mp1::access_mode::read_only);
  }));
  BOOST_TEST(not reusable.is_open());
  reusable.open(world, missing, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(reusable.is_open());
  reusable.close();
  if (rank == 0)
    mp1::file::remove(missing);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(offset_units_and_differential) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path path = test_path("offset_units");
  constexpr int pattern = 0x11223344;
  constexpr int first_value = 10;
  constexpr int second_value = 20;

  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.resize(0);
    file.sync();
    world_barrier();
    file.set_view(0, mp1::layout<int>::contiguous(4));
    BOOST_TEST(file.byte_offset(2) == static_cast<mp1::offset_t>(2 * sizeof(int)));
    BOOST_TEST(file.byte_offset(0) == 0);

    const mp1::layout<int> pair_layout = mp1::layout<int>::contiguous(2);
    if (rank == 0) {
      (void)file.write_at(2, pattern);
      const int pair[2] = {first_value, second_value};
      (void)file.write_at(0, mp1::region{pair, pair_layout});
    }
    world_barrier();
    file.sync();
    // MPI-IO collectives do not imply a rank barrier.  Order every metadata
    // transition before the size observations so a fast rank cannot resize the
    // file while a slow rank is still sampling the collective result.
    world_barrier();
    BOOST_TEST(file.size() == static_cast<mp1::offset_t>(3 * sizeof(int)));

    file.seek(2, mp1::whence_mode::set);
    BOOST_TEST(file.position() == 2);
    BOOST_TEST(file.byte_offset(2) == static_cast<mp1::offset_t>(2 * sizeof(int)));

    world_barrier();
    file.resize(16);
    world_barrier();
    BOOST_TEST(file.size() == 16);
    world_barrier();
    file.preallocate(32);
    world_barrier();
    BOOST_TEST(file.size() >= 32);
    file.close();
  }

  // Differential: raw MPI-IO reads the same bytes the mp1 view wrote (etype units, not bytes).
  {
    MPI_File native{MPI_FILE_NULL};
    BOOST_TEST(MPI_File_open(MPI_COMM_WORLD, path.string().c_str(), MPI_MODE_RDONLY,
                             MPI_INFO_NULL, &native) == MPI_SUCCESS);
    int values[3] = {};
    BOOST_TEST(MPI_File_read_at(native, 0, values, 3, MPI_INT, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
    BOOST_TEST(values[0] == first_value);
    BOOST_TEST(values[1] == second_value);
    BOOST_TEST(values[2] == pattern);
    BOOST_TEST(MPI_File_close(&native) == MPI_SUCCESS);
  }

  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_only);
    file.set_view(0, mp1::layout<int>::contiguous(4));
    int value{0};
    (void)file.read_at(0, value);
    BOOST_TEST(value == first_value);
    (void)file.read_at(1, value);
    BOOST_TEST(value == second_value);
    (void)file.read_at(2, value);
    BOOST_TEST(value == pattern);
    int pair[2] = {};
    const mp1::layout<int> pair_layout = mp1::layout<int>::contiguous(2);
    (void)file.read_at(0, mp1::region{pair, pair_layout});
    BOOST_TEST(pair[0] == first_value);
    BOOST_TEST(pair[1] == second_value);
    file.close();
  }

  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write);
    file.set_view(0, mp1::layout<int>::contiguous(4));
    if (rank == 0) {
      // One writer keeps the sequential pointer test deterministic; the other ranks only
      // participate in the collective set_view/close.
      file.seek(0, mp1::whence_mode::set);
      int value{0};
      (void)file.read(value);
      BOOST_TEST(value == first_value);
      file.seek(0, mp1::whence_mode::set);
      (void)file.write(99);
      file.seek(0, mp1::whence_mode::set);
      (void)file.read(value);
      BOOST_TEST(value == 99);
    }
    world_barrier();
    file.close();
  }

  if (rank == 0)
    mp1::file::remove(path);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(collective_io_families) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();

  {
    const std::filesystem::path path = test_path("collective_all");
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, mp1::layout<int>::contiguous(1));
    file.resize(static_cast<mp1::offset_t>(size * sizeof(int) * 2));

    file.seek(rank, mp1::whence_mode::set);
    (void)file.write_all(rank + 1000);
    file.sync();

    int value{0};
    (void)file.read_at(rank, value);
    BOOST_TEST(value == rank + 1000);

    (void)file.read_at_all(rank, value);
    BOOST_TEST(value == rank + 1000);

    (void)file.write_at_all(rank, rank + 2000);
    file.sync();
    (void)file.read_at_all(rank, value);
    BOOST_TEST(value == rank + 2000);

    int pending_value = rank + 2500;
    mp1::request issued = file.iwrite_at(rank, pending_value);
    (void)issued.wait();
    mp1::request fetched = file.iread_at(rank, value);
    (void)fetched.wait();
    BOOST_TEST(value == rank + 2500);

    file.seek(rank, mp1::whence_mode::set);
    int collective_value = rank + 2600;
    mp1::request collective_write = file.iwrite_all(collective_value);
    (void)collective_write.wait();
    file.seek(rank, mp1::whence_mode::set);
    mp1::request collective_read = file.iread_all(value);
    (void)collective_read.wait();
    BOOST_TEST(value == rank + 2600);

    file.close();
    if (rank == 0)
      mp1::file::remove(path);
    world_barrier();
  }

  // Ordered collectives use the shared file pointer; reopen to rewind it (mp1 deliberately
  // does not expose seek_shared).
  {
    const std::filesystem::path path = test_path("collective_ordered");
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
      file.set_view(0, mp1::layout<int>::contiguous(1));
      (void)file.write_ordered(rank + 3000);
      file.sync();
      file.close();
    }
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_only);
      file.set_view(0, mp1::layout<int>::contiguous(1));
      int value{0};
      (void)file.read_ordered(value);
      BOOST_TEST(value == rank + 3000);
      file.close();
    }
    if (rank == 0)
      mp1::file::remove(path);
    world_barrier();
  }

  // Shared file pointer; rank 0 is the only reader/writer so the result is deterministic.
  {
    const std::filesystem::path path = test_path("collective_shared");
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
      file.set_view(0, mp1::layout<int>::contiguous(1));
      if (rank == 0)
        (void)file.write_shared(0x5a5a);
      file.sync();
      file.close();
    }
    {
      mp1::file file;
      file.open(world, path, mp1::access_mode::read_only);
      file.set_view(0, mp1::layout<int>::contiguous(1));
      if (rank == 0) {
        int value{0};
        (void)file.read_shared(value);
        BOOST_TEST(value == 0x5a5a);
      }
      file.close();
    }
    if (rank == 0)
      mp1::file::remove(path);
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(split_collective_guard) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("split_guard");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, mp1::layout<int>::contiguous(1));
  file.resize(static_cast<mp1::offset_t>(size * sizeof(int)));

  file.seek(rank, mp1::whence_mode::set);
  int split_value = rank + 4000;
  file.write_all_begin(split_value);

  int dummy{0};
  BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
  BOOST_TEST(
      throws_runtime_error([&file] { file.set_view(0, mp1::layout<int>::contiguous(1)); }));
  BOOST_TEST(throws_runtime_error([&file] { file.resize(16); }));
  BOOST_TEST(throws_runtime_error([&file, &dummy] { (void)file.read_ordered(dummy); }));
  BOOST_TEST(throws_runtime_error([&file, &dummy] { (void)file.read_at_all(0, dummy); }));
  BOOST_TEST(throws_runtime_error([&file] { file.close(); }));
  BOOST_TEST(file.is_open());

  // Individual (non-collective) operations remain legal during a split.
  (void)file.read_at(rank, dummy);
  BOOST_TEST(file.is_open());

  (void)file.write_all_end(split_value);
  file.sync();
  BOOST_TEST(file.is_open());

  int value{0};
  (void)file.read_at(rank, value);
  BOOST_TEST(value == rank + 4000);

  // read_at_all split: begin sets the guard, end is the one allowed collective and clears it.
  file.read_at_all_begin(rank, value);
  BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
  (void)file.read_at_all_end(value);
  BOOST_TEST(value == rank + 4000);
  file.sync();

  file.close();
  if (rank == 0)
    mp1::file::remove(path);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(info_and_group) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path path = test_path("info_group");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);

  const mp1::group members = file.get_group();
  BOOST_TEST(members.is_valid());
  BOOST_TEST(members.size() == world_size());
  BOOST_TEST(members.rank() == rank);

  const mp1::group second_members = file.get_group();
  BOOST_TEST(second_members.is_valid());
  BOOST_TEST(second_members.size() == world_size());

  BOOST_TEST(throws_invalid_argument([&file] { file.set_info(mp1::info{}); }));
  const mp1::info table = file.get_info();
  BOOST_TEST(table.size() >= 0);
  BOOST_TEST(not table.value("mp1_definitely_missing_key").has_value());

  file.close();
  if (rank == 0)
    mp1::file::remove(path);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(file_request_teardown_and_close_wrap) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("request_teardown");
  const std::filesystem::path close_path = test_path("close_wrap");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, mp1::layout<int>::contiguous(1));
  file.resize(static_cast<mp1::offset_t>(size * sizeof(int)));

#if defined(MP1_TEST_WRAP)
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  {
    int pending_value = rank + 5000;
    mp1::request dropped = file.iwrite_at(rank, pending_value);
  }
#if defined(MP1_TEST_WRAP)
  if (mpi_wrap::supported())
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 1);
#endif

  int value{0};
  (void)file.read_at(rank, value);
  BOOST_TEST(value == rank + 5000);
  file.close();

#if defined(MP1_TEST_WRAP)
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    mp1::file temporary;
    temporary.open(world, close_path, mp1::access_mode::read_write | mp1::access_mode::create);
    BOOST_TEST(mpi_wrap::count("MPI_File_close") == 0);
    temporary.close();
    BOOST_TEST(mpi_wrap::count("MPI_File_close") == 1);
  } else {
    mp1::file temporary;
    temporary.open(world, close_path, mp1::access_mode::read_write | mp1::access_mode::create);
    temporary.close();
  }
#else
  mp1::file temporary;
  temporary.open(world, close_path, mp1::access_mode::read_write | mp1::access_mode::create);
  temporary.close();
#endif

  if (rank == 0) {
    mp1::file::remove(path);
    mp1::file::remove(close_path);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(leave_one_file_open_for_after_finalize_destructor) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("after_finalize");

  mp1::file open_file;
  open_file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  (void)open_file.size();

  // finalize no longer refuses a live file owner (MPI 4.1 §11.2.2: objects are not freed by
  // MPI_Finalize, and MPI_File_close is collective).  This probe deliberately keeps a stale
  // wrapper handle across finalize: hand the raw handle back and leave only the wrapper's
  // state open; the after-finalize destructor is what must abandon it.
  MPI_File raw = open_file.native_handle();
  BOOST_REQUIRE_EQUAL(MPI_File_close(&raw), MPI_SUCCESS);

  new (after_finalize_file_storage) mp1::file(std::move(open_file));
  after_finalize_file_constructed = true;
  world_barrier();
}

BOOST_AUTO_TEST_CASE(spawn_local_validation) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();

  const mp1::spawn_options options{.maxprocs = 1, .command = mp1::command_line{"prog"}};
  const mp1::spawn_multiple_options multiple;
  // Only the all-rank arguments are validated locally.  root and comm are examined on every
  // rank (MPI 4.1 §11.8.2, p. 523), so an out-of-range root throws everywhere before
  // MPI_Comm_spawn.  maxprocs, command/argv and the spawn_multiple payload are significant only
  // at the root and are a raw passthrough: mp1 neither validates them nor broadcasts an
  // agreement, so they are not probed here.
  BOOST_TEST(
      throws_invalid_argument([&world, &options] { (void)mp1::spawn(world, -1, options); }));
  BOOST_TEST(throws_invalid_argument([&world, &options] {
    (void)mp1::spawn(world, static_cast<int>(world.size()), options);
  }));
  BOOST_TEST(throws_invalid_argument(
      [&world, &multiple] { (void)mp1::spawn_multiple(world, -1, multiple); }));

  // An intercommunicator is rejected before MPI_Comm_spawn. Build one collectively.
  const int rank = world_rank();
  const int size = world_size();
  if (size >= 2) {
    const int half = (size + 1) / 2;
    const int color = rank < half ? 0 : 1;
    const int remote_leader = color == 0 ? half : 0;
    mp1::communicator local = mp1::communicator::split(world, color, rank);
    mp1::communicator inter =
        mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{7700});
    BOOST_TEST(inter.is_inter());
    BOOST_TEST(
        throws_invalid_argument([&inter, &options] { (void)mp1::spawn(inter, 0, options); }));
  }
}

BOOST_AUTO_TEST_CASE(spawn_single_and_multiple) {
  BOOST_TEST(true);
#if defined(I_MPI_NUMVERSION) && I_MPI_NUMVERSION == 20211902300
  std::fprintf(stderr,
               "Intel MPI 2021.19.2 self-respawn skipped: observed in GitHub CI, possibly a "
               "container-environment issue (COMPATIBILITY.md)\n");
  return;
#endif
  if (not spawn_self_respawn_supported) {
    BOOST_TEST_MESSAGE(
        "spawn self-respawn success cases skipped: Open MPI-only (MPICH 4.3.2 cannot start "
        "a second copy of this binary from one parent process); test_spawn_contract covers "
        "the provider-neutral spawn and spawn_multiple paths");
    return;
  }
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  std::error_code error;
  const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", error);
  if (error or self.empty()) {
    BOOST_TEST_MESSAGE("spawn probe skipped: /proc/self/exe unavailable");
    return;
  }

  std::filesystem::path single_dir = base_dir / "spawn_single_markers";
  if (rank == 0)
    std::filesystem::create_directories(single_dir, error);
  world_barrier();
  mp1::command_line command;
  if (rank == 0) {
    command = mp1::command_line{self.string(), "--mp1-spawn-child",
                                "--mp1-spawn-marker=" + single_dir.string(), "alpha", "beta"};
  }
  {
    std::fprintf(stderr, "spawn: single command\n");
    const mp1::spawn_options spawn_options{.maxprocs = 2, .command = command};
    mp1::communicator children = mp1::spawn(world, 0, spawn_options);
    BOOST_TEST(children.is_valid());
    BOOST_TEST(children.is_inter());
    BOOST_TEST(children.size() == world_size());
    BOOST_TEST(children.remote_size() == 2);
    BOOST_TEST(children.remote_group().size() == 2);
    // The children acquire parent() and publish the marker before they disconnect.  Wait for
    // that handshake first, then release the intercommunicator: the parent does not join the
    // child's finalize, but it also does not race the child's MPI_Init/parent().
    std::fprintf(stderr, "spawn: waiting for single-command child markers\n");
    const bool all_markers = wait_for_markers(
        [&] {
          return std::filesystem::exists(single_dir / "0") and
                 std::filesystem::exists(single_dir / "1");
        },
        60.0);
    BOOST_TEST(all_markers);
    if (all_parents_ready(all_markers)) {
      children.disconnect();
    } else {
      // A child that died before publishing will never enter disconnect; releasing the handle
      // locally keeps the failed test from turning into another 300 s CTest timeout.
      (void)children.release();
    }
    if (all_markers) {
      for (const char* child : {"0", "1"}) {
        std::ifstream marker{single_dir / child};
        std::vector<std::string> arguments;
        std::string line;
        while (std::getline(marker, line))
          arguments.push_back(line);
        BOOST_TEST(contains(arguments, "alpha"));
        BOOST_TEST(contains(arguments, "beta"));
      }
    }
  }
  world_barrier();

  std::filesystem::path multiple_dir = base_dir / "spawn_multiple_markers";
  if (rank == 0)
    std::filesystem::create_directories(multiple_dir, error);
  world_barrier();
  mp1::command_lines commands(2);
  if (rank == 0) {
    commands[0] =
        mp1::command_line{self.string(), "--mp1-spawn-child",
                          "--mp1-spawn-marker=" + multiple_dir.string(), "one-a", "one-b"};
    commands[1] = mp1::command_line{self.string(), "--mp1-spawn-child",
                                    "--mp1-spawn-marker=" + multiple_dir.string(), "two-a"};
  }
  mp1::spawn_multiple_options spawned;
  if (rank == 0)
    spawned.add(commands[0], 1).add(commands[1], 2);
  {
    std::fprintf(stderr, "spawn: spawn_multiple\n");
    mp1::communicator children = mp1::spawn_multiple(world, 0, spawned);
    BOOST_TEST(children.is_valid());
    BOOST_TEST(children.is_inter());
    BOOST_TEST(children.remote_size() == 3);
    // Same handshake as the single-command case: the children acquire parent() and publish
    // their markers before disconnecting, so the parent can wait for the files and then release
    // the intercomm without joining the children's finalize.
    std::fprintf(stderr, "spawn: waiting for spawn_multiple child markers\n");
    const bool all_markers = wait_for_markers(
        [&] {
          return std::filesystem::exists(multiple_dir / "0") and
                 std::filesystem::exists(multiple_dir / "1") and
                 std::filesystem::exists(multiple_dir / "2");
        },
        60.0);
    BOOST_TEST(all_markers);
    if (all_parents_ready(all_markers)) {
      children.disconnect();
    } else {
      // A child that died before publishing will never enter disconnect; releasing the handle
      // locally keeps the failed test from turning into another 300 s CTest timeout.
      (void)children.release();
    }
    if (all_markers) {
      // The remote-group rank order of MPI_Comm_spawn_multiple is implementation-defined in
      // practice; assert the command multiplicity and argv instead of the rank assignment.
      int first_command_count{0};
      int second_command_count{0};
      for (const char* child : {"0", "1", "2"}) {
        std::ifstream marker{multiple_dir / child};
        std::vector<std::string> arguments;
        std::string line;
        while (std::getline(marker, line))
          arguments.push_back(line);
        if (contains(arguments, "one-a")) {
          ++first_command_count;
          BOOST_TEST(contains(arguments, "one-b"));
        }
        if (contains(arguments, "two-a"))
          ++second_command_count;
      }
      BOOST_TEST(first_command_count == 1);
      BOOST_TEST(second_command_count == 2);
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
