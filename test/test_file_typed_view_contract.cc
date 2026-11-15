// File typed-view etype rules (MPI 4.1 §14.3.1): a derived file-view etype may only have
// nonnegative, monotonically nondecreasing typemap displacements, and an overlapping etype or
// filetype is forbidden on a file opened for writing.
//
// Covered:
//   * a reflected etype whose member list is declared out of offset order is proven
//     non-monotone and rejected with std::invalid_argument for every access mode before
//     MPI_File_set_view;
//   * a padded (holey) reflected etype whose displacements are monotone is accepted and can be
//     written and read back through a typed view;
//   * an overlapping filetype is rejected locally for read_write and write_only, and left to
//     the provider for read_only.  A provider may reject an overlapping filetype at
//     MPI_File_set_view even for a read-only file, so the portable assertion is wrapper-local:
//     no std::invalid_argument from the wrapper, and a provider rejection is accepted as
//     mpi_error.
//
// The test runs at 2 and 4 ranks; set_view/resize are collective and every rank enters them in
// the same order.

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

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_TYPED_VIEW_WRAP 1
#else
#define MP1_TYPED_VIEW_WRAP 0
#endif


// Members are listed in an order that does not follow their offsets: the first registered
// member is at offset 4 and the second at offset 0, so the recorded typemap order is proven
// non-monotone.
struct non_monotone_view_type {
  int first{0};
  int second{0};
};

MP1_REFLECTION(non_monotone_view_type, second, first)

// A padded struct: the alignment hole between the members is legal (MPI 4.1 §14.3.1 only
// forbids descending/negative displacements), and the registration order follows the offsets.
struct padded_view_type {
  int first{0};
  alignas(16) int second{0};
};

MP1_REFLECTION(padded_view_type, first, second)


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

  // Returns the library-owned diagnostic of an expected std::invalid_argument, or an empty
  // string when the call did not throw that type.
  template<class Function>
  std::string invalid_argument_message(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument& error) {
      return std::string{error.what()};
    } catch (...) {
      return std::string{};
    }
    return std::string{};
  }

  std::uint64_t run_id{0};
  std::filesystem::path base_dir;

  std::filesystem::path test_path(const char* name) {
    return base_dir / (std::string{"mp1_typed_view_"} + name + ".bin");
  }

  void remove_path(const std::filesystem::path& path) {
    world_barrier();
    if (world_rank() == 0) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
    world_barrier();
  }

  [[nodiscard]] mp1::layout<non_monotone_view_type> non_monotone_filetype() {
    return mp1::layout<non_monotone_view_type>::contiguous(1);
  }

  [[nodiscard]] mp1::layout<padded_view_type> padded_filetype() {
    return mp1::layout<padded_view_type>::contiguous(1);
  }

  // The proven-overlapping filetype: blocklength 2 at element 0 and blocklength 2 at element 1.
  [[nodiscard]] mp1::layout<int> overlapping_filetype() {
    mp1::indexed_parameter parameter;
    parameter.add(2, 0);
    parameter.add(2, 1);
    return mp1::layout<int>::indexed(parameter);
  }

  void expect_no_wrapped_calls() {
#if MP1_TYPED_VIEW_WRAP
    if (mpi_wrap::supported()) {
      BOOST_TEST(mpi_wrap::count("MPI_Type_commit") == 0u);
      BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 0u);
    }
#endif
  }

  void reset_wrap_counts() {
#if MP1_TYPED_VIEW_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
  }

  // Create a file with enough bytes for every typed-view case, collectively.
  void create_sized_file(const mp1::mpi_communicator& world, const std::filesystem::path& path,
                         mp1::offset_t bytes) {
    mp1::file file;
    file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
    file.resize(bytes);
    file.close();
  }

  struct file_typed_view_fixture {
    file_typed_view_fixture() {
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
          std::filesystem::temp_directory_path() / ("mp1_typed_view_" + std::to_string(run_id));
      std::error_code error;
      if (rank == 0) {
        std::filesystem::remove_all(base_dir, error);
        error.clear();
        std::filesystem::create_directories(base_dir, error);
      }
      world_barrier();
    }

    ~file_typed_view_fixture() {
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


BOOST_TEST_GLOBAL_FIXTURE(file_typed_view_fixture);

BOOST_AUTO_TEST_SUITE(file_typed_view_contract)

BOOST_AUTO_TEST_CASE(non_monotone_etype_rejected_for_every_access_mode) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("non_monotone");
  create_sized_file(world, path, 64);
  const mp1::layout<non_monotone_view_type> filetype = non_monotone_filetype();

  // Build and cache the reflected etype before the counters are observed, so the assertion only
  // covers what set_view itself does.  The etype construction is allowed to commit its
  // datatype.
  (void)mp1::native_datatype<non_monotone_view_type>();

  const std::array<mp1::access_mode, 3> modes{
      mp1::access_mode::read_write, mp1::access_mode::write_only, mp1::access_mode::read_only};
  for (const mp1::access_mode mode : modes) {
    mp1::file file;
    file.open(world, path, mode);
    reset_wrap_counts();
    const std::string message = invalid_argument_message([&] { file.set_view(0, filetype); });
    BOOST_TEST(not message.empty());
    BOOST_TEST(message.find("monoton") != std::string::npos);
    expect_no_wrapped_calls();
    BOOST_TEST(file.is_open());
    // The rejected view must not corrupt the file object: resetting the byte view and closing
    // still works, and no view was recorded for the failed call.
    file.set_view(0, "native");
    file.close();
    world_barrier();
  }

  remove_path(path);
}

// MPI 4.1 §14.3 applies the same nonnegative/monotonically-nondecreasing typemap rule to the
// filetype as to the etype.  The etype here is plain int, so the rejection is the filetype's
// own descending map (element 1 at extent 1, then element 0 at extent 0), not the etype check.
BOOST_AUTO_TEST_CASE(non_monotone_filetype_rejected_for_every_access_mode) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("non_monotone_filetype");
  create_sized_file(world, path, 64);
  mp1::indexed_parameter parameter;
  parameter.add(1, 1);
  parameter.add(1, 0);
  const mp1::layout<int> filetype = mp1::layout<int>::indexed(parameter);
  BOOST_TEST(filetype.is_valid());
  BOOST_TEST(not filetype.has_overlapping_entries());
  (void)mp1::native_datatype<int>();  // cache the etype before observing wrapper calls

  const std::array<mp1::access_mode, 3> modes{
      mp1::access_mode::read_write, mp1::access_mode::write_only, mp1::access_mode::read_only};
  for (const mp1::access_mode mode : modes) {
    mp1::file file;
    file.open(world, path, mode);
    const std::string message = invalid_argument_message([&] { file.set_view(0, filetype); });
    BOOST_TEST(not message.empty());
    BOOST_TEST(message.find("monoton") != std::string::npos);
    BOOST_TEST(file.is_open());
    file.set_view(0, "native");
    file.close();
    world_barrier();
  }
  remove_path(path);
}

// A negative typemap displacement is rejected even for the raw-adoption escape, where the
// wrapper cannot rely on the factory's own non-negativity validation.
BOOST_AUTO_TEST_CASE(negative_filetype_displacement_rejected) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("negative_filetype");
  create_sized_file(world, path, 64);

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  const int blocklength = 1;
  const MPI_Aint negative_displacement = -static_cast<MPI_Aint>(sizeof(int));
  BOOST_REQUIRE(MPI_Type_create_hindexed(1, &blocklength, &negative_displacement, MPI_INT,
                                         &raw) == MPI_SUCCESS);
  const mp1::layout<int> filetype = mp1::layout<int>::from_raw(raw, 1);
  BOOST_TEST(filetype.true_byte_lower_bound() < 0);

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write);
  const std::string message = invalid_argument_message([&] { file.set_view(0, filetype); });
  BOOST_TEST(not message.empty());
  BOOST_TEST(message.find("negative") != std::string::npos);
  BOOST_TEST(file.is_open());
  file.set_view(0, "native");
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(monotone_padded_etype_accepted_and_usable) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const std::filesystem::path path = test_path("padded");

  // The alignment hole is a real padding range, not a packed image.
  static_assert(offsetof(padded_view_type, second) >= static_cast<std::size_t>(sizeof(int)),
                "the padded type must start its second member after a hole");
  static_assert(sizeof(padded_view_type) > offsetof(padded_view_type, second) + sizeof(int),
                "the padded type must end with tail padding");

  const mp1::layout<padded_view_type> filetype = padded_filetype();
  const mp1::offset_t etype_extent = static_cast<mp1::offset_t>(sizeof(padded_view_type));

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_write | mp1::access_mode::create);
  file.set_view(0, filetype);
  file.resize(static_cast<mp1::offset_t>(size) * 2 * etype_extent);

  // Offsets are etype units; every rank owns two etypes.
  const mp1::offset_t base = static_cast<mp1::offset_t>(rank) * 2;
  padded_view_type value;
  value.first = rank + 1;
  value.second = rank + 101;
  (void)file.write_at(base, value);
  file.sync();

  padded_view_type readback;
  (void)file.read_at(base, readback);
  BOOST_TEST(readback.first == value.first);
  BOOST_TEST(readback.second == value.second);

  // A one-etype view still accepts a multi-element contiguous filetype.
  file.set_view(0, mp1::layout<padded_view_type>::contiguous(2));
  padded_view_type pair[2];
  pair[0].first = rank + 201;
  pair[0].second = rank + 301;
  pair[1].first = rank + 202;
  pair[1].second = rank + 302;
  const mp1::offset_t pair_base = static_cast<mp1::offset_t>(rank) * 2;
  const mp1::layout<padded_view_type> pair_map = mp1::layout<padded_view_type>::contiguous(2);
  (void)file.write_at(pair_base, mp1::region{pair, pair_map});
  file.sync();
  std::array<padded_view_type, 2> pair_readback{};
  (void)file.read_at(pair_base, mp1::region{pair_readback.data(), pair_map});
  BOOST_TEST(pair_readback[0].first == pair[0].first);
  BOOST_TEST(pair_readback[0].second == pair[0].second);
  BOOST_TEST(pair_readback[1].first == pair[1].first);
  BOOST_TEST(pair_readback[1].second == pair[1].second);

  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(overlapping_filetype_rejected_for_write_modes) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("overlap_write");
  create_sized_file(world, path, 64);
  const mp1::layout<int> filetype = overlapping_filetype();
  BOOST_TEST(filetype.has_overlapping_entries());

  const std::array<mp1::access_mode, 2> modes{mp1::access_mode::read_write,
                                              mp1::access_mode::write_only};
  for (const mp1::access_mode mode : modes) {
    mp1::file file;
    file.open(world, path, mode);
    reset_wrap_counts();
    const std::string message = invalid_argument_message([&] { file.set_view(0, filetype); });
    BOOST_TEST(not message.empty());
    BOOST_TEST(message.find("overlapping") != std::string::npos);
    expect_no_wrapped_calls();
    BOOST_TEST(file.is_open());
    file.set_view(0, "native");
    file.close();
    world_barrier();
  }

  remove_path(path);
}

BOOST_AUTO_TEST_CASE(overlapping_filetype_read_only_is_provider_guarded) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("overlap_read");
  create_sized_file(world, path, 64);
  const mp1::layout<int> filetype = overlapping_filetype();
  BOOST_TEST(filetype.has_overlapping_entries());

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_only);

  // Wrapper-local contract: read_only is not locally rejected (the §14.3.1 overlap restriction
  // does not apply to a read-only file).  The provider may still refuse the view at
  // MPI_File_set_view, which the wrapper reports as mpi_error.
  bool accepted{false};
  bool provider_rejected{false};
  try {
    file.set_view(0, filetype);
    accepted = true;
  } catch (const std::invalid_argument& error) {
    BOOST_TEST(std::string{error.what()}.find("overlapping") == std::string::npos);
  } catch (const mp1::mpi_error& error) {
    provider_rejected = true;
    BOOST_TEST(std::string{error.what()}.find("set_view") != std::string::npos);
  }
  BOOST_TEST(static_cast<int>(accepted) + static_cast<int>(provider_rejected) > 0);
  BOOST_TEST(file.is_open());

  // The file object stays usable after either outcome; the byte view accepts every file.
  file.set_view(0, "native");
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_CASE(overlapping_per_call_read_layout_rejected) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const std::filesystem::path path = test_path("overlap_read_layout");
  create_sized_file(world, path, 64);
  const mp1::layout<int> layout = overlapping_filetype();
  BOOST_TEST(layout.has_overlapping_entries());

  mp1::file file;
  file.open(world, path, mp1::access_mode::read_only);
  file.set_view(0, mp1::layout<int>::contiguous(1));

  int buffer[4] = {0, 0, 0, 0};
  const std::string message =
      invalid_argument_message([&] { (void)file.read_at(0, mp1::region{buffer, layout}); });
  BOOST_TEST(not message.empty());
  BOOST_TEST(message.find("overlapping") != std::string::npos);

  // The same layout is not inspected on the write side: MPI only reads from that buffer, so
  // an overlap-free memory datatype is not required.  The read-only access mode is the reason
  // this call fails, not the overlap: a std::runtime_error, not std::invalid_argument.
  bool mode_rejected{false};
  try {
    (void)file.write_at(0, mp1::region{buffer, layout});
  } catch (const std::invalid_argument&) {
    BOOST_TEST(false);
  } catch (const std::runtime_error&) {
    mode_rejected = true;
  }
  BOOST_TEST(mode_rejected);
  BOOST_TEST(file.is_open());
  file.close();
  remove_path(path);
}

BOOST_AUTO_TEST_SUITE_END()
