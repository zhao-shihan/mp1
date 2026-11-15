// MPI address geometry: MPI 4.1 §5.1.5 arithmetic at the reflection site and the checked
// conversion helpers.
//
// MPI_Aint is not promised to be a flat integer address, so mp1 routes every address
// difference through MPI_Aint_diff and every displacement add through MPI_Aint_add.  This
// file:
//
//  * checks the checked_address_diff/checked_address_add round trips on real object
//    addresses;
//  * checks the reflected struct displacements against the member addresses computed with
//    MPI_Aint_diff (the same values the old intrinsic subtraction produced on a flat
//    provider);
//  * pins is_undefined_count as an equality test against MPI_UNDEFINED (never a sign test);
//  * pins the std::in_range-based Aint/Count conversion boundaries.
//
// A non-flat address provider is not available locally; the portability claim is the
// normative rule plus the static check, not a runtime assertion.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace {

  struct aint_fixture {
    aint_fixture() {
      mp1::init();
    }

    ~aint_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  struct offset_probe {
    int first;
    double second;
    char third;
  };

  struct nested_probe {
    offset_probe inner;
    int tail;
  };

}  // namespace

MP1_REFLECTION(offset_probe, first, second, third)
MP1_REFLECTION(nested_probe, inner, tail)

BOOST_TEST_GLOBAL_FIXTURE(aint_fixture);


namespace {

  // Free a derived datatype returned by MPI_Type_get_contents_c.  The helper never frees a
  // predefined handle (the reflection result is derived, so its contents are too).
  class contents_handle {
  public:
    contents_handle() = default;
    contents_handle(const contents_handle&) = delete;
    contents_handle& operator=(const contents_handle&) = delete;
    ~contents_handle() {
      if (type_ != MPI_DATATYPE_NULL)
        (void)MPI_Type_free(&type_);
    }

    void reset(MPI_Datatype type) noexcept {
      type_ = type;
    }

    [[nodiscard]] MPI_Datatype get() const noexcept {
      return type_;
    }

  private:
    MPI_Datatype type_{MPI_DATATYPE_NULL};
  };

  // The displacement array of one struct combiner, read from MPI_Type_get_contents_c.
  std::vector<MPI_Count> struct_displacements(MPI_Datatype type) {
    MPI_Count num_integers = 0;
    MPI_Count num_addresses = 0;
    MPI_Count num_large_counts = 0;
    MPI_Count num_datatypes = 0;
    int combiner = -1;
    BOOST_REQUIRE(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses,
                                          &num_large_counts, &num_datatypes,
                                          &combiner) == MPI_SUCCESS);
    BOOST_REQUIRE(combiner == MPI_COMBINER_STRUCT);

    std::vector<int> integers(static_cast<std::size_t>(num_integers));
    std::vector<MPI_Aint> addresses(static_cast<std::size_t>(num_addresses));
    std::vector<MPI_Count> large(static_cast<std::size_t>(num_large_counts));
    std::vector<MPI_Datatype> types(static_cast<std::size_t>(num_datatypes), MPI_DATATYPE_NULL);
    int dummy_integer = 0;
    MPI_Aint dummy_address = 0;
    MPI_Count dummy_large = 0;
    MPI_Datatype dummy_type = MPI_DATATYPE_NULL;
    BOOST_REQUIRE(MPI_Type_get_contents_c(
                      type, num_integers, num_addresses, num_large_counts, num_datatypes,
                      integers.empty() ? &dummy_integer : integers.data(),
                      addresses.empty() ? &dummy_address : addresses.data(),
                      large.empty() ? &dummy_large : large.data(),
                      types.empty() ? &dummy_type : types.data()) == MPI_SUCCESS);

    // The *_c struct constructor returns {count, blocklengths, displacements} in the
    // large-count array (MPI 4.1 §5.1.13).
    BOOST_REQUIRE(num_large_counts >= 1);
    const MPI_Count count = large[0];
    BOOST_REQUIRE(count >= 0);
    BOOST_REQUIRE(num_large_counts >= 2 * count + 1);
    std::vector<MPI_Count> displacements;
    for (MPI_Count i = 0; i < count; ++i)
      displacements.push_back(large[static_cast<std::size_t>(1 + count + i)]);
    return displacements;
  }

}  // namespace


BOOST_AUTO_TEST_CASE(checked_address_diff_round_trip) {
  // Same sequential storage: the difference is defined and the checked helper must agree with
  // the MPI procedure, not with intrinsic subtraction.
  offset_probe objects[4] = {{1, 2.0, 'a'}, {3, 4.0, 'b'}, {5, 6.0, 'c'}, {7, 8.0, 'd'}};
  MPI_Aint high = 0;
  MPI_Aint low = 0;
  BOOST_REQUIRE(MPI_Get_address(&objects[3], &high) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Get_address(&objects[0], &low) == MPI_SUCCESS);

  const MPI_Aint direct = MPI_Aint_diff(high, low);
  const MPI_Aint checked =
      mp1::detail::checked_address_diff(high, low, "test::checked_address_diff");
  BOOST_TEST(checked == direct);
  // The documented round trip is the helper's overflow policy.
  BOOST_TEST(MPI_Aint_add(low, checked) == high);
  BOOST_TEST(checked > 0);

  // The member difference is the same quantity the reflection builder records.
  MPI_Aint member = 0;
  BOOST_REQUIRE(MPI_Get_address(&objects[0].third, &member) == MPI_SUCCESS);
  BOOST_TEST(mp1::detail::checked_address_diff(member, low, "test::checked_address_diff") ==
             static_cast<MPI_Aint>(offsetof(offset_probe, third)));

  // A zero difference is representable and round-trips.
  BOOST_TEST(mp1::detail::checked_address_diff(low, low, "test::checked_address_diff") == 0);
}

BOOST_AUTO_TEST_CASE(checked_address_add_round_trip) {
  offset_probe objects[2] = {{1, 2.0, 'a'}, {3, 4.0, 'b'}};
  MPI_Aint low = 0;
  BOOST_REQUIRE(MPI_Get_address(&objects[0], &low) == MPI_SUCCESS);

  const MPI_Aint displacement = static_cast<MPI_Aint>(sizeof(offset_probe));
  BOOST_TEST(mp1::detail::checked_address_add(low, static_cast<mp1::count_t>(displacement),
                                              "test::checked_address_add") ==
             MPI_Aint_add(low, displacement));
  BOOST_TEST(mp1::detail::checked_address_add(low, 0, "test::checked_address_add") == low);
  BOOST_CHECK_THROW(
      (void)mp1::detail::checked_address_add(low, -1, "test::checked_address_add"),
      std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(reflected_displacements_match_address_diff) {
  // The committed reflection is resized (MPI_COMBINER_RESIZED) so the struct displacements
  // are read from the wrapped oldtype.  Each displacement must equal
  // MPI_Aint_diff(member address, object address).
  const MPI_Datatype reflected = mp1::native_datatype<offset_probe>();

  MPI_Count ri = 0;
  MPI_Count ra = 0;
  MPI_Count rl = 0;
  MPI_Count rd = 0;
  int resized_combiner = -1;
  BOOST_REQUIRE(MPI_Type_get_envelope_c(reflected, &ri, &ra, &rl, &rd, &resized_combiner) ==
                MPI_SUCCESS);
  BOOST_REQUIRE(resized_combiner == MPI_COMBINER_RESIZED);

  contents_handle resized_child;
  {
    std::vector<int> integers(static_cast<std::size_t>(ri));
    std::vector<MPI_Aint> addresses(static_cast<std::size_t>(ra));
    std::vector<MPI_Count> large(static_cast<std::size_t>(rl));
    std::vector<MPI_Datatype> types(static_cast<std::size_t>(rd), MPI_DATATYPE_NULL);
    int dummy_integer = 0;
    MPI_Aint dummy_address = 0;
    MPI_Count dummy_large = 0;
    MPI_Datatype dummy_type = MPI_DATATYPE_NULL;
    BOOST_REQUIRE(MPI_Type_get_contents_c(reflected, ri, ra, rl, rd,
                                          integers.empty() ? &dummy_integer : integers.data(),
                                          addresses.empty() ? &dummy_address : addresses.data(),
                                          large.empty() ? &dummy_large : large.data(),
                                          types.empty() ? &dummy_type : types.data()) ==
                  MPI_SUCCESS);
    BOOST_REQUIRE(rd == 1);
    resized_child.reset(types[0]);
  }
  BOOST_REQUIRE(resized_child.get() != MPI_DATATYPE_NULL);

  const std::vector<MPI_Count> displacements = struct_displacements(resized_child.get());
  BOOST_REQUIRE(displacements.size() == 3u);

  offset_probe object{1, 2.0, 'a'};
  MPI_Aint base = 0;
  MPI_Aint first = 0;
  MPI_Aint second = 0;
  MPI_Aint third = 0;
  BOOST_REQUIRE(MPI_Get_address(&object, &base) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Get_address(&object.first, &first) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Get_address(&object.second, &second) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Get_address(&object.third, &third) == MPI_SUCCESS);

  const auto relative = [&base](MPI_Aint address) {
    return static_cast<MPI_Count>(
        mp1::detail::checked_address_diff(address, base, "test::reflection displacement"));
  };
  BOOST_TEST(displacements[0] == relative(first));
  BOOST_TEST(displacements[1] == relative(second));
  BOOST_TEST(displacements[2] == relative(third));
}

BOOST_AUTO_TEST_CASE(undefined_count_is_an_equality_test) {
  // MPI_UNDEFINED is implementation-defined and need not be negative.  The helper compares
  // with the sentinel; ordinary counts (including negative-looking ones that are not the
  // sentinel) are not undefined.
  BOOST_TEST(mp1::detail::is_undefined_count(static_cast<mp1::count_t>(MPI_UNDEFINED)));
  if (MPI_UNDEFINED != 0)
    BOOST_TEST(!mp1::detail::is_undefined_count(0));
  if (MPI_UNDEFINED != 1)
    BOOST_TEST(!mp1::detail::is_undefined_count(1));
  if (MPI_UNDEFINED != -1)
    BOOST_TEST(!mp1::detail::is_undefined_count(-1));
  static_assert(std::is_same_v<decltype(mp1::detail::is_undefined_count(0)), bool>);
  BOOST_TEST(mp1::detail::is_undefined_count(
      static_cast<mp1::count_t>(static_cast<MPI_Count>(MPI_UNDEFINED))));
}

BOOST_AUTO_TEST_CASE(aint_count_conversion_boundaries) {
  // The conversions use std::in_range, not a casted limit.  The exhaustive-boundary
  // branch is pinned only where the source range is wider than the destination type, so this
  // test stays valid when MPI_Aint and count_t have different widths.
  BOOST_TEST(mp1::detail::checked_aint_value(0, "test::checked_aint_value") == 0);
  BOOST_TEST(mp1::detail::checked_aint_value(-7, "test::checked_aint_value") == -7);
  if constexpr (sizeof(mp1::count_t) <= sizeof(MPI_Aint)) {
    BOOST_TEST(mp1::detail::checked_aint_value(std::numeric_limits<mp1::count_t>::max(),
                                               "test::checked_aint_value") ==
               static_cast<MPI_Aint>(std::numeric_limits<mp1::count_t>::max()));
  } else {
    BOOST_CHECK_THROW((void)mp1::detail::checked_aint_value(
                          std::numeric_limits<mp1::count_t>::max(), "test::checked_aint_value"),
                      std::overflow_error);
  }

  BOOST_TEST(mp1::detail::checked_mpi_count_from_aint(0, "test::checked_mpi_count_from_aint") ==
             0);
  BOOST_TEST(mp1::detail::checked_mpi_count_from_aint(static_cast<MPI_Aint>(sizeof(int)),
                                                      "test::checked_mpi_count_from_aint") ==
             static_cast<MPI_Count>(sizeof(int)));
  BOOST_CHECK_THROW((void)mp1::detail::checked_mpi_count_from_aint(
                        static_cast<MPI_Aint>(-1), "test::checked_mpi_count_from_aint"),
                    std::invalid_argument);
  if constexpr (sizeof(MPI_Aint) <= sizeof(MPI_Count)) {
    BOOST_TEST(mp1::detail::checked_mpi_count_from_aint(std::numeric_limits<MPI_Aint>::max(),
                                                        "test::checked_mpi_count_from_aint") ==
               static_cast<MPI_Count>(std::numeric_limits<MPI_Aint>::max()));
  } else {
    BOOST_CHECK_THROW(
        (void)mp1::detail::checked_mpi_count_from_aint(std::numeric_limits<MPI_Aint>::max(),
                                                       "test::checked_mpi_count_from_aint"),
        std::overflow_error);
  }
}
