// File region lifetime tests: the documented "the caller's layout may die after initiation"
// guarantee for nonblocking and split-begin region accesses (MPI 4.1 §14.4.5, §14.6.7).
//
// The wrapper duplicates the per-call layout for a nonblocking/split-begin region.  The
// duplicate must be the datatype MPI holds (otherwise the keepalive guards a handle MPI never
// sees), and the duplicate must stay alive until the request completes / the matching split
// end returns.  The tests cover both halves:
//   * behavioral: destroy the caller's layout after initiation, complete, read the data back;
//   * mechanism (wrap builds only): MPI_File_*_c receives the MPI_Type_dup handle, not the
//     caller's layout handle; a blocking region call does not duplicate.
//
// The fixture owns MPI and a per-run temporary directory, so the file runs at the registered
// 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <span>
#include <system_error>
#include <type_traits>
#include <utility>

#if defined(MP1_TEST_WRAP)
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

  int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::filesystem::path base_dir;

  std::filesystem::path test_path(const std::string& name) {
    return base_dir / (std::string{"mp1_region_keepalive_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

#if MP1_TEST_HAVE_MPI_WRAP
  // The wrap shim reports a datatype as an integer token for equality comparison only (a
  // pointer on Open MPI, an integer on MPICH); this repeats the shim's own conversion for a
  // caller-held layout handle.  The Handle parameter keeps the discarded branch dependent.
  template<class Handle>
  std::uintptr_t datatype_token(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>)
      return reinterpret_cast<std::uintptr_t>(handle);
    else
      return static_cast<std::uintptr_t>(handle);
  }
#endif

}  // namespace


struct mp1_file_region_keepalive_fixture {
  mp1_file_region_keepalive_fixture() {
    mp1::init();
    const int rank = world_rank();
    std::uint64_t seed{0};
    if (rank == 0) {
      seed = static_cast<std::uint64_t>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
      seed ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&seed));
    }
    MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    base_dir = std::filesystem::temp_directory_path() /
               ("mp1_region_keepalive_" + std::to_string(seed));
    std::error_code error;
    if (rank == 0) {
      std::filesystem::remove_all(base_dir, error);
      error.clear();
      std::filesystem::create_directories(base_dir, error);
    }
    world_barrier();
  }

  ~mp1_file_region_keepalive_fixture() {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove_all(base_dir, error);
    }
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Another owner may have finalized; never throw from a fixture destructor.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_file_region_keepalive_fixture);

BOOST_AUTO_TEST_SUITE(file_region_keepalive_test)

BOOST_AUTO_TEST_CASE(caller_layout_may_die_after_nonblocking_region) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("nonblocking");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(size) * static_cast<mp1::offset_t>(sizeof(int)));

  int value = rank + 400;
  mp1::request write_request;
  {
    // The initiating layout dies after the call; the request owns the duplicate that MPI
    // holds, so wait() completes correctly.
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(1);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
    }
#endif
    write_request = file.iwrite_at(rank, mp1::region{&value, scoped});
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      // Exactly one duplicate was created, and the file initiation received the duplicate's
      // handle, not the caller's layout handle.
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 1);
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      const std::uintptr_t used = mpi_wrap::last_datatype("MPI_File_iwrite_at_c");
      BOOST_TEST(mpi_wrap::count("MPI_File_iwrite_at_c") == 1);
      BOOST_TEST(used == duplicate);
      BOOST_TEST(used != datatype_token(scoped.native_handle()));
    }
#endif
  }
  write_request.wait();

  int readback = 0;
  (void)file.read_at(rank, readback);
  BOOST_TEST(readback == value);

  // The same guarantee on the read side: the receive layout dies after initiation.
  int receive_target = 0;
  mp1::request read_request;
  {
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(1);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
    }
#endif
    read_request = file.iread_at(rank, mp1::region{&receive_target, scoped});
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 1);
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      const std::uintptr_t used = mpi_wrap::last_datatype("MPI_File_iread_at_c");
      BOOST_TEST(mpi_wrap::count("MPI_File_iread_at_c") == 1);
      BOOST_TEST(used == duplicate);
      BOOST_TEST(used != datatype_token(scoped.native_handle()));
    }
#endif
  }
  read_request.wait();
  BOOST_TEST(receive_target == value);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(caller_layout_may_die_after_split_begin) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("split");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(size) * static_cast<mp1::offset_t>(sizeof(int)));

  int value = rank + 500;
  {
    // The layout dies between begin and end; the epoch owns the duplicate.
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(1);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
    }
#endif
    file.write_at_all_begin(rank, mp1::region{&value, scoped});
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 1);
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      const std::uintptr_t used = mpi_wrap::last_datatype("MPI_File_write_at_all_begin_c");
      BOOST_TEST(mpi_wrap::count("MPI_File_write_at_all_begin_c") == 1);
      BOOST_TEST(used == duplicate);
      BOOST_TEST(used != datatype_token(scoped.native_handle()));
    }
#endif
  }  // the initiating layout is destroyed here; the split epoch is still active

  // The end takes no datatype; it must match the begin's buffer address.
  const mp1::status end_status = file.write_at_all_end(value);
  BOOST_TEST(end_status.error() == MPI_SUCCESS);

  int readback = 0;
  (void)file.read_at(rank, readback);
  BOOST_TEST(readback == value);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(blocking_region_borrows_without_duplicating) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("blocking");
  const mp1::layout<int> layout_one = mp1::layout<int>::contiguous(1);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, layout_one);
  file.resize(static_cast<mp1::offset_t>(size) * static_cast<mp1::offset_t>(sizeof(int)));

  int value = rank + 600;
  {
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(1);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
    }
#endif
    (void)file.write_at(rank, mp1::region{&value, scoped});
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      // A blocking region call borrows the caller's layout for the call; no duplicate.
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
      BOOST_TEST(mpi_wrap::count("MPI_File_write_at_c") == 1);
    }
#endif
  }

  int readback = 0;
  (void)file.read_at(rank, readback);
  BOOST_TEST(readback == value);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
