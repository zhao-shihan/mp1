// File test: offset_t/MPI_Offset units, byte and etype views, all blocking and
// nonblocking families with layouts, split-collective guards, info/group accessors and the
// after-finalize close/destructor contract.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// the Boost.Test module macro. MPI is initialized by the global fixture below. Every collective
// file operation is called by all ranks with identical control flow, so the same source runs at
// 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
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

  // Reads are const and writes are non-const, for the blocking and nonblocking families
  // alike; the file surface is member-only.
  template<class F, class = void>
  struct const_blocking_writable : std::false_type {};

  template<class F>
  struct const_blocking_writable<
      F, std::void_t<decltype(std::declval<const F&>().write(std::declval<int>()))>>
      : std::true_type {};

  template<class F, class = void>
  struct const_blocking_readable : std::false_type {};

  template<class F>
  struct const_blocking_readable<
      F, std::void_t<decltype(std::declval<const F&>().read(std::declval<int&>()))>>
      : std::true_type {};

  template<class F, class = void>
  struct const_nonblocking_writable : std::false_type {};

  template<class F>
  struct const_nonblocking_writable<
      F, std::void_t<decltype(std::declval<const F&>().iwrite(std::declval<int>()))>>
      : std::true_type {};

  template<class F, class = void>
  struct const_nonblocking_readable : std::false_type {};

  template<class F>
  struct const_nonblocking_readable<
      F, std::void_t<decltype(std::declval<const F&>().iread(std::declval<int&>()))>>
      : std::true_type {};

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

  template<class F>
  bool throws_mpi_error(F&& function) {
    try {
      function();
    } catch (const mp1::mpi_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // Per-run scratch directory, unique at 2 and 4 ranks (ctest may run both concurrently).
  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  // One file is left open until after mp1::finalize(), so the throwing close() and the
  // noexcept destructor take their after-finalize paths. The check object is declared before
  // the Boost global fixture, so it is destroyed after the fixture finalized MPI.
  alignas(mp1::file) unsigned char after_finalize_storage[sizeof(mp1::file)];
  bool after_finalize_constructed = false;

  [[noreturn]] void after_finalize_failure(const char* what) {
    std::fprintf(stderr, "mp1 file after-finalize check: %s\n", what);
    std::abort();
  }

  struct file_after_finalize_check {
    ~file_after_finalize_check() noexcept {
      if (not after_finalize_constructed)
        return;
      if (not mp1::finalized())
        return;  // The fixture did not finalize MPI; leave the object alone.
      auto* file = std::launder(reinterpret_cast<mp1::file*>(after_finalize_storage));

      if (not file->is_open() or file->native_handle() == MPI_FILE_NULL)
        after_finalize_failure("the handle was lost before the destructor");

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
        after_finalize_failure("a failed close after finalize changed the open state");

      const std::size_t before = mp1::detail::abandoned_handle_count();
      file->~file();
      if (mp1::detail::abandoned_handle_count() != before + 1)
        after_finalize_failure(
            "the after-finalize destructor did not abandon exactly one handle");
    }
  };

  file_after_finalize_check after_finalize_check_instance;

}  // namespace


struct mp1_file_fixture {
  mp1_file_fixture() {
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
        std::filesystem::temp_directory_path() / ("mp1_w7a_file_" + std::to_string(run_id));
    std::error_code error;
    if (rank == 0) {
      std::filesystem::remove_all(base_dir, error);
      error.clear();
      std::filesystem::create_directories(base_dir, error);
    }
    world_barrier();
  }

  ~mp1_file_fixture() {
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

BOOST_TEST_GLOBAL_FIXTURE(mp1_file_fixture);

BOOST_AUTO_TEST_SUITE(file_test)

BOOST_AUTO_TEST_CASE(types_and_closed_state) {
  static_assert(std::is_same_v<mp1::offset_t, MPI_Offset>);
  static_assert(sizeof(mp1::offset_t) >= sizeof(std::int64_t));
  static_assert(std::is_nothrow_destructible_v<mp1::file>);
  static_assert(std::is_nothrow_move_constructible_v<mp1::file>);
  static_assert(std::is_nothrow_move_assignable_v<mp1::file>);
  static_assert(not std::is_copy_constructible_v<mp1::file>);
  static_assert(not std::is_copy_assignable_v<mp1::file>);

  // The documented read/write constness split holds for every family and member form.
  static_assert(const_blocking_readable<mp1::file>::value);
  static_assert(const_nonblocking_readable<mp1::file>::value);
  static_assert(not const_blocking_writable<mp1::file>::value);
  static_assert(not const_nonblocking_writable<mp1::file>::value);

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
  closed.close();  // idempotent no-op on a closed file
  BOOST_TEST(not closed.is_open());

  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.size(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.mode(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.position(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.byte_offset(0); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.get_info(); }));
  BOOST_TEST(throws_runtime_error([&closed] { (void)closed.get_group(); }));
  BOOST_TEST(throws_runtime_error([&closed] { closed.sync(); }));
  int value{0};
  BOOST_TEST(throws_runtime_error([&closed, &value] { (void)closed.read_at(0, value); }));
}

BOOST_AUTO_TEST_CASE(open_close_move_and_destructor) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  const std::filesystem::path first = test_path("open_close_first");
  mp1::file file;
  file.open(world, first, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(file.is_open());
  BOOST_TEST(file.native_handle() != MPI_FILE_NULL);
  BOOST_TEST((static_cast<int>(file.mode()) & MPI_MODE_RDWR) != 0);
  BOOST_TEST(throws_runtime_error(
      [&file, &world, &first] { file.open(world, first, mp1::access_mode::read_write); }));
  BOOST_TEST(file.is_open());
  file.close();
  BOOST_TEST(not file.is_open());
  file.close();  // idempotent
  BOOST_TEST(not file.is_open());

  // Move construction/assignment transfer ownership; the target's old handle is closed by the
  // move assignment.
  const std::filesystem::path second = test_path("open_close_second");
  mp1::file source;
  source.open(world, first, mp1::access_mode::read_only);
  mp1::file moved{std::move(source)};
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not source.is_open());
  BOOST_TEST(moved.is_open());
  mp1::file target;
  target.open(world, second, mp1::access_mode::read_write | mp1::access_mode::create);
  target = std::move(moved);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not moved.is_open());
  BOOST_TEST(target.is_open());
  target.close();

  // The destructor performs the close: the value written by rank 0 is visible after the scope.
  const std::filesystem::path destructor_path = test_path("destructor_close");
  {
    mp1::file scoped;
    scoped.open(world, destructor_path,
                mp1::access_mode::read_write | mp1::access_mode::create);
    scoped.set_view(0, layout_one);
    scoped.resize(static_cast<mp1::offset_t>(sizeof(int)));
    if (rank == 0)
      (void)scoped.write_at(0, 4242);
    scoped.sync();
  }
  {
    mp1::file reader;
    reader.open(world, destructor_path, mp1::access_mode::read_only);
    reader.set_view(0, layout_one);
    if (rank == 0) {
      int value{0};
      (void)reader.read_at(0, value);
      BOOST_TEST(value == 4242);
    }
    reader.close();
  }

  // A failed open throws mpi_error, leaves the object closed and the object stays reusable.
  const std::filesystem::path missing = test_path("open_missing");
  mp1::file reusable;
  BOOST_TEST(throws_mpi_error([&reusable, &world, &missing] {
    reusable.open(world, missing, mp1::access_mode::read_only);
  }));
  BOOST_TEST(not reusable.is_open());
  reusable.open(world, missing, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(reusable.is_open());
  reusable.close();

  // file::remove accepts the null default info and is not collective.
  if (rank == 0) {
    mp1::file::remove(first, mp1::info{});
    mp1::file::remove(second);
    mp1::file::remove(destructor_path);
    mp1::file::remove(missing);
    BOOST_TEST(
        throws_mpi_error([] { mp1::file::remove("mp1_w7a_definitely_missing_file.bin"); }));
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(byte_view_and_offsets) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("byte_view");

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.resize(static_cast<mp1::offset_t>(size));
  BOOST_TEST(file.size() == static_cast<mp1::offset_t>(size));

  // Before the first set_view the etype and filetype are MPI_BYTE: offsets are bytes.
  const char written = static_cast<char>('A' + rank);
  (void)file.write_at(rank, written);
  file.sync();
  BOOST_TEST(file.size() == static_cast<mp1::offset_t>(size));
  BOOST_TEST(file.byte_offset(rank) == static_cast<mp1::offset_t>(rank));

  file.seek(rank, mp1::whence_mode::set);
  BOOST_TEST(file.position() == rank);
  char sequential{0};
  (void)file.read(sequential);
  BOOST_TEST(sequential == written);
  BOOST_TEST(file.position() == rank + 1);

  file.seek(0, mp1::whence_mode::end);
  BOOST_TEST(file.position() == size);

  // The non-template set_view uses MPI_BYTE etype/filetype. Per-process displacements are
  // legal MPI-IO views; each rank reads back only the byte it wrote.
  file.set_view(rank, "native");
  BOOST_TEST(file.byte_offset(0) == rank);
  char byte_view{0};
  (void)file.read_at(0, byte_view);
  BOOST_TEST(byte_view == written);
  BOOST_TEST(file.size() == static_cast<mp1::offset_t>(size));

  // Collective set_view with a null datarep / invalid layout throws before MPI on every rank.
  BOOST_TEST(throws_invalid_argument([&file] { file.set_view(0, nullptr); }));
  BOOST_TEST(throws_invalid_argument([&file] {
    mp1::layout<int> invalid;
    file.set_view(0, invalid);
  }));
  BOOST_TEST(file.is_open());

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(explicit_offset_and_sequential_layout_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("layout_families");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> layout_two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> layout_four = mp1::layout<int>::contiguous(4);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_four);
  file.resize(static_cast<mp1::offset_t>(size * 8 * sizeof(int)));

  const int base = rank * 4;
  const std::array<int, 2> data{{rank * 100 + 1, rank * 100 + 2}};
  const std::array<int, 2> second_data{{rank * 100 + 3, rank * 100 + 4}};

  // Blocking explicit-offset layout form.
  (void)file.write_at(base, mp1::region{data.data(), layout_two});
  file.sync();
  std::array<int, 2> readback{{0, 0}};
  (void)file.read_at(base, mp1::region{readback.data(), layout_two});
  BOOST_TEST(readback == data);
  BOOST_TEST(file.byte_offset(base) == static_cast<mp1::offset_t>(base * sizeof(int)));

  // Scalar access at etype offsets; holes are not counted.
  int scalar{0};
  (void)file.read_at(base, scalar);
  BOOST_TEST(scalar == data[0]);
  (void)file.read_at(base + 1, scalar);
  BOOST_TEST(scalar == data[1]);

  // Sequential-pointer layout form.
  file.seek(base + 2, mp1::whence_mode::set);
  (void)file.write(mp1::region{second_data.data(), layout_two});
  file.seek(base + 2, mp1::whence_mode::set);
  std::array<int, 2> sequential_readback{{0, 0}};
  (void)file.read(mp1::region{sequential_readback.data(), layout_two});
  BOOST_TEST(sequential_readback == second_data);

  // Nonblocking explicit-offset layout form.
  const std::array<int, 2> third_data{{rank * 100 + 5, rank * 100 + 6}};
  auto write_request = file.iwrite_at(base, mp1::region{third_data.data(), layout_two});
  write_request.wait();
  std::array<int, 2> third_readback{{0, 0}};
  auto read_request = file.iread_at(base, mp1::region{third_readback.data(), layout_two});
  read_request.wait();
  BOOST_TEST(third_readback == third_data);

  // Nonblocking sequential-pointer layout form.
  file.seek(base, mp1::whence_mode::set);
  const std::array<int, 2> fourth_data{{rank * 100 + 7, rank * 100 + 8}};
  auto sequential_write = file.iwrite(mp1::region{fourth_data.data(), layout_two});
  sequential_write.wait();
  file.seek(base, mp1::whence_mode::set);
  std::array<int, 2> fourth_readback{{0, 0}};
  auto sequential_read = file.iread(mp1::region{fourth_readback.data(), layout_two});
  sequential_read.wait();
  BOOST_TEST(fourth_readback == fourth_data);

  // Shared file pointer: rank 0 alone keeps the result deterministic. set_view/resize are
  // collective, so every rank executes them; only rank 0 touches the shared pointer.
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(4 * sizeof(int)));
  if (rank == 0) {
    const std::array<int, 2> shared_data{{0x51, 0x52}};
    (void)file.write_shared(mp1::region{shared_data.data(), layout_two});
  }
  file.sync();
  file.close();

  // Reopen to rewind the shared pointer (mp1 does not expose seek_shared).
  mp1::file shared_reader;
  shared_reader.open(world, path, mp1::access_mode::read_only);
  shared_reader.set_view(0, layout_one);
  if (rank == 0) {
    std::array<int, 2> shared_readback{{0, 0}};
    (void)shared_reader.read_shared(mp1::region{shared_readback.data(), layout_two});
    BOOST_TEST(shared_readback == (std::array<int, 2>{0x51, 0x52}));
  }
  shared_reader.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(collective_and_ordered_layout_families) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("collective_families");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> layout_two = mp1::layout<int>::contiguous(2);
  const std::array<int, 2> data{{rank * 10 + 1, rank * 10 + 2}};

  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, layout_one);
    file.resize(static_cast<mp1::offset_t>(size * 4 * sizeof(int)));

    // Explicit-offset collective, blocking and nonblocking, with a layout.
    (void)file.write_at_all(rank * 2, mp1::region{data.data(), layout_two});
    file.sync();
    std::array<int, 2> at_all_readback{{0, 0}};
    (void)file.read_at_all(rank * 2, mp1::region{at_all_readback.data(), layout_two});
    BOOST_TEST(at_all_readback == data);

    const std::array<int, 2> async_data{{rank * 10 + 3, rank * 10 + 4}};
    auto async_write =
        file.iwrite_at_all(size * 2 + rank * 2, mp1::region{async_data.data(), layout_two});
    async_write.wait();
    std::array<int, 2> async_readback{{0, 0}};
    auto async_read =
        file.iread_at_all(size * 2 + rank * 2, mp1::region{async_readback.data(), layout_two});
    async_read.wait();
    BOOST_TEST(async_readback == async_data);
    file.close();
  }

  // Collective read_all/write_all use each rank's individual file pointer; seek first so
  // every rank owns a distinct block.
  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, layout_one);
    file.seek(rank * 2, mp1::whence_mode::set);
    (void)file.write_all(mp1::region{data.data(), layout_two});
    file.sync();
    file.close();
  }
  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_only);
    file.set_view(0, layout_one);
    file.seek(rank * 2, mp1::whence_mode::set);
    std::array<int, 2> readback{{0, 0}};
    (void)file.read_all(mp1::region{readback.data(), layout_two});
    BOOST_TEST(readback == data);
    file.close();
  }

  // Ordered collectives in rank order.
  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.set_view(0, layout_one);
    (void)file.write_ordered(mp1::region{data.data(), layout_two});
    file.sync();
    file.close();
  }
  {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_only);
    file.set_view(0, layout_one);
    std::array<int, 2> readback{{0, 0}};
    (void)file.read_ordered(mp1::region{readback.data(), layout_two});
    BOOST_TEST(readback == data);
    file.close();
  }

  remove_path(path);
}

BOOST_AUTO_TEST_CASE(split_collective_guard_and_end) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path path = test_path("split_guard");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> layout_two = mp1::layout<int>::contiguous(2);
  const std::array<int, 2> data{{rank * 10 + 5, rank * 10 + 6}};

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(world_size() * 2 * sizeof(int)));

  file.write_at_all_begin(rank * 2, mp1::region{data.data(), layout_two});
  int dummy{0};
  BOOST_TEST(not throws_runtime_error([&file] { (void)file.size(); }));
  BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
  BOOST_TEST(throws_runtime_error([&file] { file.close(); }));
  BOOST_TEST(throws_runtime_error([&file, &dummy] { (void)file.read_at_all(0, dummy); }));
  BOOST_TEST(file.is_open());
  // MPI 4.1 §14.4.5 forbids collective *I/O* during a split, not the collective controls whose
  // own clauses carry no completion requirement: set_info (§14.2.8), set_atomicity (§14.6.1)
  // and seek_shared (§14.4.4) must remain callable between begin and end.
  {
    const mp1::info hints = mp1::info::create();
    BOOST_TEST(not throws_runtime_error([&file, &hints] { file.set_info(hints); }));
    BOOST_TEST(not throws_runtime_error([&file] { file.set_atomicity(true); }));
    BOOST_TEST(not throws_runtime_error([&file] { file.set_atomicity(false); }));
    BOOST_TEST(
        not throws_runtime_error([&file] { file.seek_shared(0, mp1::whence_mode::set); }));
    BOOST_TEST(file.is_open());
  }
  // Non-collective operations stay legal during a split.
  (void)file.read_at(rank * 2, dummy);
  (void)file.write_at_all_end(data);
  file.sync();

  std::array<int, 2> readback{{0, 0}};
  file.read_at_all_begin(rank * 2, mp1::region{readback.data(), layout_two});
  BOOST_TEST(throws_runtime_error([&file] { file.sync(); }));
  (void)file.read_at_all_end(readback);
  BOOST_TEST(readback == data);

  // read_all split (shared pointer) after reopening to rewind it.
  file.close();
  {
    mp1::file reader;
    reader.open(world, path, mp1::access_mode::read_only);
    reader.set_view(0, layout_one);
    reader.seek(rank * 2, mp1::whence_mode::set);
    std::array<int, 2> all_readback{{0, 0}};
    reader.read_all_begin(mp1::region{all_readback.data(), layout_two});
    BOOST_TEST(throws_runtime_error([&reader] { reader.sync(); }));
    (void)reader.read_all_end(all_readback);
    BOOST_TEST(all_readback == data);
    reader.close();
  }

  // Ordered split collective: the shared pointer advances in rank order.
  {
    mp1::file writer;
    writer.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    writer.set_view(0, layout_one);
    writer.write_ordered_begin(mp1::region{data.data(), layout_two});
    BOOST_TEST(throws_runtime_error([&writer] { writer.close(); }));
    (void)writer.write_ordered_end(data);
    writer.sync();
    writer.close();
  }
  {
    mp1::file reader;
    reader.open(world, path, mp1::access_mode::read_only);
    reader.set_view(0, layout_one);
    std::array<int, 2> ordered_readback{{0, 0}};
    reader.read_ordered_begin(mp1::region{ordered_readback.data(), layout_two});
    BOOST_TEST(throws_runtime_error([&reader] { reader.sync(); }));
    (void)reader.read_ordered_end(ordered_readback);
    BOOST_TEST(ordered_readback == data);
    reader.close();
  }
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(info_group_and_set_info) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const std::filesystem::path path = test_path("info_group");

  mp1::info hints;
  hints.set("cb_buffer_size", "65536");
  mp1::file file{world, path, mp1::access_mode::read_write | mp1::access_mode::create, hints};
  BOOST_REQUIRE(file.is_open());

  const mp1::info table = file.get_info();
  BOOST_TEST(table.size() >= 0);
  BOOST_TEST(not table.value("mp1_definitely_missing_key").has_value());
  const auto buffer_size = table.value("cb_buffer_size");
  if (buffer_size)
    BOOST_TEST(*buffer_size == "65536");

  const mp1::group members = file.get_group();
  BOOST_TEST(members.is_valid());
  BOOST_TEST(members.size() == world.size());
  BOOST_TEST(members.rank() == world.rank());
  const mp1::group second_members = file.get_group();
  BOOST_TEST(second_members.is_valid());
  BOOST_TEST(second_members.size() == world.size());

  // set_info takes the info's native handle; a null info is rejected locally.
  mp1::info updated;
  updated.set("cb_buffer_size", "32768");
  file.set_info(updated);
  BOOST_TEST(throws_invalid_argument([&file] { file.set_info(mp1::info{}); }));

  file.close();
  if (rank == 0)
    mp1::file::remove(path);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(close_and_destructor_after_finalize) {
  // Leave one open file for the post-finalize check above. This case must stay last: the file
  // stays open until the global fixture finalized MPI.
  auto* file = new (after_finalize_storage) mp1::file{};
  const std::filesystem::path path = test_path("after_finalize");
  file->open(mp1::comm_world(), path, mp1::access_mode::read_write | mp1::access_mode::create);
  BOOST_TEST(file->is_open());
  BOOST_TEST(file->native_handle() != MPI_FILE_NULL);
  world_barrier();

  // Unlink the directory entry while the handle is open; the open handles keep the data alive
  // and the scratch directory stays clean after the after-finalize destructor.
  if (world_rank() == 0) {
    std::error_code error;
    std::filesystem::remove(path, error);
  }
  world_barrier();

  // finalize no longer refuses a live file owner (MPI 4.1 §11.2.2: objects are not freed by
  // MPI_Finalize, and MPI_File_close is collective).  This probe deliberately keeps a stale
  // wrapper handle across finalize, so hand the raw handle back and leave only the wrapper's
  // state open; the after-finalize destructor is what must abandon it.
  MPI_File raw = file->native_handle();
  BOOST_REQUIRE_EQUAL(MPI_File_close(&raw), MPI_SUCCESS);
  world_barrier();
  after_finalize_constructed = true;
}

BOOST_AUTO_TEST_SUITE_END()
