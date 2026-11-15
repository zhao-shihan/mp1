// The error model's location contract.
//
// Every library diagnostic has the shape `mp1: <operation> (<file>:<line>): <message>`, where
// <file>:<line> is the *check site* inside the library (never the public entry, never the
// caller), `operation()` is the public entry (or the MPI routine for an MPI-return check), and
// `what()` keeps the provider error string plus the `[code=..., class=...]` suffix.  The layer
// takes no `what`/message parameters: the message is a std::format string at the check site,
// and every helper passes `std::source_location::current()` explicitly.  The C++20 <format>
// library is the stdlib floor (libstdc++ 13), checked below at compile time.
//
// This test complements test_error_handler.cc (the predefined error-handler control scenario,
// rank 1, before-init/after-finalize probes): here the location/message contract runs at 2
// and 4 ranks and includes the regression for the former double-prefixed root-send overlap
// diagnostic (`mp1: mp1: root send layout`).
//
// MPI 4.1 basis: §2.8/§9.3 (error reporting), §14.3.1 (overlapping filetype), §6.6 (scatterv
// root send).  `mpi_error` is reachable for MPI-IO because files default to
// MPI_ERRORS_RETURN (§14.7).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <filesystem>
#include <format>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>


#if !defined(__cpp_lib_format) || __cpp_lib_format < 201907L
#error "mp1 requires a C++20 <format> implementation (libstdc++ 13 floor)"
#endif

namespace {

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  struct error_location_fixture {
    error_location_fixture() {
      mp1::init();
    }

    ~error_location_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

  template<class Function>
  [[nodiscard]] std::string invalid_message(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument& error) {
      return std::string{error.what()};
    } catch (...) {
      return std::string{};
    }
    return std::string{};
  }

  [[nodiscard]] bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
  }

  [[nodiscard]] std::string source_site(const std::source_location& where) {
    return std::string{where.file_name()} + ":" + std::to_string(where.line());
  }

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(error_location_fixture);

BOOST_AUTO_TEST_SUITE(error_location_contract)

// The public helper shapes: a context literal, an explicit source_location::current() and a
// format string.  There is no `what` parameter overload (the negative is checked through a
// dependent concept so overload resolution is SFINAE, not a hard diagnostic).
template<class... Args>
concept fail_invocable = requires(Args&&... args) { mp1::fail(static_cast<Args&&>(args)...); };
template<class... Args>
concept require_invocable =
    requires(Args&&... args) { mp1::require(static_cast<Args&&>(args)...); };
static_assert(!fail_invocable<mp1::context, const char*>,
              "fail takes an explicit source_location, not a what-style message argument");
static_assert(!require_invocable<bool, mp1::context, const char*>);

BOOST_AUTO_TEST_CASE(helpers_take_the_check_site_explicitly) {
  const mp1::context ctx{"mp1::error_probe"};
  const std::source_location here = std::source_location::current();

  std::string message;
  try {
    mp1::fail(ctx, here, "value {} is invalid", 7);
  } catch (const std::invalid_argument& error) {
    message = error.what();
  }
  BOOST_TEST(message.starts_with("mp1: mp1::error_probe ("));
  BOOST_TEST(contains(message, "): value 7 is invalid"));
  BOOST_TEST(contains(message, here.file_name()));
  BOOST_TEST(contains(message, ":" + std::to_string(here.line()) + ")"));

  // require(true) is a no-op and does not format its arguments.
  mp1::require(true, ctx, std::source_location::current(), "must not be formatted {}", 1);

  std::string static_message;
  try {
    mp1::throw_invalid("mp1::error_probe", "static message");
  } catch (const std::invalid_argument& error) {
    static_message = error.what();
  }
  BOOST_TEST(static_message.starts_with("mp1: mp1::error_probe ("));
  BOOST_TEST(contains(static_message, "): static message"));
}

// mpi_error carries the code/class/slot and the check site; what() keeps the provider string
// and the bracketed suffix.
BOOST_AUTO_TEST_CASE(mpi_error_shape) {
  const std::source_location here = std::source_location::current();
  const mp1::mpi_error direct{MPI_ERR_ARG, "mp1::error_probe", 3, here};
  BOOST_TEST(direct.code() == MPI_ERR_ARG);
  BOOST_TEST(direct.error_class() == MPI_ERR_ARG);
  BOOST_TEST(direct.is_class(MPI_ERR_ARG));
  BOOST_TEST(direct.operation() == "mp1::error_probe");
  BOOST_TEST(direct.slot() == 3);
  BOOST_TEST(source_site(direct.where()) == source_site(here));
  BOOST_TEST(not direct.class_name().empty());
  const std::string text = direct.what();
  BOOST_TEST(text.starts_with("mp1: mp1::error_probe ("));
  BOOST_TEST(contains(text, source_site(here)));
  BOOST_TEST(contains(text, ", slot=3]"));
  BOOST_TEST(contains(text, " [code="));

  const mp1::mpi_error single{MPI_SUCCESS, "mp1::error_probe"};
  BOOST_TEST(single.slot() == -1);
  BOOST_TEST(not contains(single.what(), "slot="));
}

// Library-side checks report the library header line: span_from and the p2p envelope are
// checked before MPI, and the file-open failure reports file.hpp through mpi_error.
BOOST_AUTO_TEST_CASE(library_check_sites_are_inside_the_library) {
  const std::string span_message =
      invalid_message([] { (void)mp1::span_from<int>(nullptr, -1); });
  BOOST_TEST(span_message.starts_with("mp1: mp1::span_from ("));
  BOOST_TEST(contains(span_message, "vocabulary.hpp:"));
  BOOST_TEST(not contains(span_message, "test_error_location_contract.cc"));
  BOOST_TEST(contains(span_message, "): count -1 is negative"));

  const mp1::mpi_communicator world = mp1::comm_world();
  const std::string send_message =
      invalid_message([&] { mp1::send(world, std::string{"x"}, -7); });
  BOOST_TEST(send_message.starts_with("mp1: mp1::send ("));
  BOOST_TEST(contains(send_message, "p2p.hpp:"));
  BOOST_TEST(not contains(send_message, "test_error_location_contract.cc"));

  const std::filesystem::path missing =
      std::filesystem::temp_directory_path() / "mp1_error_location_missing_file.bin";
  std::error_code ignored;
  std::filesystem::remove(missing, ignored);
  bool got_mpi_error{false};
  try {
    mp1::file failing;
    failing.open(mp1::comm_self(), missing, mp1::access_mode::read_only);
  } catch (const mp1::mpi_error& error) {
    got_mpi_error = true;
    BOOST_TEST(error.code() != MPI_SUCCESS);
    BOOST_TEST(error.operation() == "MPI_File_open");
    BOOST_TEST(contains(std::string{error.where().file_name()}, "file.hpp"));
    BOOST_TEST(not contains(std::string{error.where().file_name()},
                            "test_error_location_contract.cc"));
    BOOST_TEST(std::string{error.what()}.starts_with("mp1: MPI_File_open ("));
    BOOST_TEST(error.slot() == -1);
  } catch (...) {
    BOOST_TEST(false);
  }
  BOOST_TEST(got_mpi_error);
}

// Regression: the root-send overlap diagnostic used a message fragment as
// the operation, producing `mp1: mp1: root send layout (...)` and losing `mp1::scatterv`.  The
// operation must be the public entry and the side label must appear exactly once after the
// single prefix.
BOOST_AUTO_TEST_CASE(root_send_overlap_diagnostic_keeps_one_operation_prefix) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const int root = size - 1;

  BOOST_TEST(size >= 2);
  if (rank == root) {
    // A proven-overlapping per-peer send layout: two entries at element 0.  The 8-element
    // aggregate keeps the sparse-fit check happy so the disjointness check is the one that
    // fires (it used to pass a fragment as the operation).
    const std::array entries{std::pair{mp1::count_t{1}, mp1::count_t{0}},
                             std::pair{mp1::count_t{1}, mp1::count_t{0}}};
    const mp1::indexed_parameter parameter{entries};
    const mp1::layout<int> overlapping = mp1::layout<int>::indexed(parameter);
    BOOST_TEST(overlapping.has_overlapping_entries());

    std::vector<int> send(8, 1);
    std::vector<int> recv(1, -1);
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), overlapping);
    std::vector<mp1::count_t> displacements(static_cast<std::size_t>(size), 0);

    const std::string message = invalid_message([&] {
      mp1::scatterv(c, root, send, std::span<const mp1::layout<int>>{layouts},
                    std::span<const mp1::count_t>{displacements}, std::span<int>{recv});
    });
    BOOST_TEST(message.starts_with("mp1: mp1::scatterv ("));
    BOOST_TEST(contains(message, "root send layout:"));
    BOOST_TEST(contains(message, "type-map entries overlap"));
    // A single `mp1: <operation>` prefix: a second prefix would start after the first one
    // (the legitimate `mp1: mp1::scatterv` must not be mistaken for the double prefix).
    BOOST_TEST(message.find("mp1: ", 5) == std::string::npos);
    BOOST_TEST(contains(message, "layout_lowering.hpp:"));
  }
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
