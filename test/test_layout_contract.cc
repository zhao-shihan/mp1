// Contract test for mp1::layout<T>, mp1::layouts<T> and the datatype
// reflection hooks.
//
// The shared test harness links the Boost.Test main; this file defines no main and no
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// The reflected 119-member aggregate is the documented maximum. A generated 120-member TU is
// expected to FAIL to compile (the negative probe runs locally; it is not part
// of this executable, because a non-compiling source cannot be registered as a unit test).
//
// Every factory is checked against a raw MPI_Type_* construction with the same arguments
// ("differential" metadata: size, extent, lower/upper bound, true extent/lower bound, dense).

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/rma_engine.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <climits>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_LAYOUT_HAVE_WRAP 1
#else
#define MP1_LAYOUT_HAVE_WRAP 0
#endif


// ---------------------------------------------------------------------------------------------
// Global-scope reflected/built-in datatypes (the macros open namespace mp1).
// ---------------------------------------------------------------------------------------------

struct reflected_119 {
  int m0{0};
  int m1{0};
  int m2{0};
  int m3{0};
  int m4{0};
  int m5{0};
  int m6{0};
  int m7{0};
  int m8{0};
  int m9{0};
  int m10{0};
  int m11{0};
  int m12{0};
  int m13{0};
  int m14{0};
  int m15{0};
  int m16{0};
  int m17{0};
  int m18{0};
  int m19{0};
  int m20{0};
  int m21{0};
  int m22{0};
  int m23{0};
  int m24{0};
  int m25{0};
  int m26{0};
  int m27{0};
  int m28{0};
  int m29{0};
  int m30{0};
  int m31{0};
  int m32{0};
  int m33{0};
  int m34{0};
  int m35{0};
  int m36{0};
  int m37{0};
  int m38{0};
  int m39{0};
  int m40{0};
  int m41{0};
  int m42{0};
  int m43{0};
  int m44{0};
  int m45{0};
  int m46{0};
  int m47{0};
  int m48{0};
  int m49{0};
  int m50{0};
  int m51{0};
  int m52{0};
  int m53{0};
  int m54{0};
  int m55{0};
  int m56{0};
  int m57{0};
  int m58{0};
  int m59{0};
  int m60{0};
  int m61{0};
  int m62{0};
  int m63{0};
  int m64{0};
  int m65{0};
  int m66{0};
  int m67{0};
  int m68{0};
  int m69{0};
  int m70{0};
  int m71{0};
  int m72{0};
  int m73{0};
  int m74{0};
  int m75{0};
  int m76{0};
  int m77{0};
  int m78{0};
  int m79{0};
  int m80{0};
  int m81{0};
  int m82{0};
  int m83{0};
  int m84{0};
  int m85{0};
  int m86{0};
  int m87{0};
  int m88{0};
  int m89{0};
  int m90{0};
  int m91{0};
  int m92{0};
  int m93{0};
  int m94{0};
  int m95{0};
  int m96{0};
  int m97{0};
  int m98{0};
  int m99{0};
  int m100{0};
  int m101{0};
  int m102{0};
  int m103{0};
  int m104{0};
  int m105{0};
  int m106{0};
  int m107{0};
  int m108{0};
  int m109{0};
  int m110{0};
  int m111{0};
  int m112{0};
  int m113{0};
  int m114{0};
  int m115{0};
  int m116{0};
  int m117{0};
  int m118{0};
};

MP1_REFLECTION(reflected_119, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30,
               m31, m32, m33, m34, m35, m36, m37, m38, m39, m40, m41, m42, m43, m44, m45, m46,
               m47, m48, m49, m50, m51, m52, m53, m54, m55, m56, m57, m58, m59, m60, m61, m62,
               m63, m64, m65, m66, m67, m68, m69, m70, m71, m72, m73, m74, m75, m76, m77, m78,
               m79, m80, m81, m82, m83, m84, m85, m86, m87, m88, m89, m90, m91, m92, m93, m94,
               m95, m96, m97, m98, m99, m100, m101, m102, m103, m104, m105, m106, m107, m108,
               m109, m110, m111, m112, m113, m114, m115, m116, m117, m118)

struct reflected_empty {};

MP1_REFLECTION_EMPTY(reflected_empty)

template<class T>
struct reflected_wrapper {
  T value{};
};

MP1_REFLECTION_TEMPLATE((typename T), (reflected_wrapper<T>), value)

template<class T, class U>
struct reflected_pair {
  T first{};
  U second{};
};

MP1_REFLECTION_TEMPLATE((typename T, typename U), (reflected_pair<T, U>), first, second)

struct custom_int {
  int value{};
};

MP1_DATATYPE_TRAITS(custom_int, MPI_INT)


namespace {

  // -------------------------------------------------------------------------------------------
  // Raw MPI helpers and differential metadata comparison.
  // -------------------------------------------------------------------------------------------

  void require_mpi_ok(int rc, const char* what) {
    if (rc != MPI_SUCCESS)
      BOOST_FAIL(std::string{"raw MPI call failed: "} + what);
  }

  struct raw_metadata {
    MPI_Count size{0};
    MPI_Count lower_bound{0};
    MPI_Count extent{0};
    MPI_Count true_lower_bound{0};
    MPI_Count true_extent{0};
  };

  raw_metadata read_raw(MPI_Datatype type) {
    raw_metadata metadata;
    require_mpi_ok(MPI_Type_size_c(type, &metadata.size), "MPI_Type_size_c");
    require_mpi_ok(MPI_Type_get_extent_c(type, &metadata.lower_bound, &metadata.extent),
                   "MPI_Type_get_extent_c");
    require_mpi_ok(
        MPI_Type_get_true_extent_c(type, &metadata.true_lower_bound, &metadata.true_extent),
        "MPI_Type_get_true_extent_c");
    return metadata;
  }

  MPI_Datatype commit_raw(MPI_Datatype type) {
    require_mpi_ok(MPI_Type_commit(&type), "MPI_Type_commit");
    return type;
  }

  class raw_datatype {
  public:
    explicit raw_datatype(MPI_Datatype handle) noexcept : handle_{handle} {
    }

    raw_datatype(const raw_datatype&) = delete;
    raw_datatype& operator=(const raw_datatype&) = delete;

    raw_datatype(raw_datatype&& other) noexcept : handle_{other.handle_} {
      other.handle_ = MPI_DATATYPE_NULL;
    }

    ~raw_datatype() {
      if (handle_ != MPI_DATATYPE_NULL)
        (void)MPI_Type_free(&handle_);
    }

    [[nodiscard]] MPI_Datatype get() const noexcept {
      return handle_;
    }

  private:
    MPI_Datatype handle_{MPI_DATATYPE_NULL};
  };

  void compare_metadata(const raw_metadata& raw, mp1::count_t byte_size,
                        mp1::count_t byte_extent, mp1::count_t byte_lower_bound,
                        mp1::count_t true_extent, mp1::count_t true_lower_bound, bool dense,
                        const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(byte_size == static_cast<mp1::count_t>(raw.size));
      BOOST_TEST(byte_extent == static_cast<mp1::count_t>(raw.extent));
      BOOST_TEST(byte_lower_bound == static_cast<mp1::count_t>(raw.lower_bound));
      BOOST_TEST(true_extent == static_cast<mp1::count_t>(raw.true_extent));
      BOOST_TEST(true_lower_bound == static_cast<mp1::count_t>(raw.true_lower_bound));
      // packed implies no padding, but `size == true_extent` is not sufficient for packed
      // (duplicates and holes can cancel); the exact predicate is asserted per factory.
      if (dense)
        BOOST_TEST(raw.size == raw.true_extent);
    }
  }

  template<class T>
  void compare_layout(const mp1::layout<T>& actual, MPI_Datatype raw, const char* label) {
    const raw_metadata metadata = read_raw(raw);
    BOOST_TEST(actual.is_valid());
    BOOST_TEST(actual.native_handle() != MPI_DATATYPE_NULL);
    compare_metadata(metadata, actual.byte_size(), actual.byte_extent(),
                     actual.byte_lower_bound(), actual.true_byte_extent(),
                     actual.true_byte_lower_bound(), actual.dense(), label);
    BOOST_TEST(actual.byte_upper_bound() == actual.byte_lower_bound() + actual.byte_extent());
    BOOST_TEST(actual.true_byte_upper_bound() ==
               actual.true_byte_lower_bound() + actual.true_byte_extent());
  }

  template<class T, class = void>
  struct has_resize : std::false_type {};

  template<class T>
  struct has_resize<T, std::void_t<decltype(std::declval<T&>().resize(0, 0))>>
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

  // Namespace-scope invalid sentinels: construction and destruction must be MPI-free and must
  // remain safe when static destruction happens after the fixture finalized MPI.
  mp1::layout<int> static_invalid_layout;

  struct layout_contract_fixture {
    layout_contract_fixture() {
      mp1::init();
    }

    ~layout_contract_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a fixture destructor.
      }
    }
  };

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(layout_contract_fixture);

BOOST_AUTO_TEST_SUITE(layout_contract)

BOOST_AUTO_TEST_CASE(invalid_defaults) {
  static_assert(not has_resize<mp1::layout<int>>::value);

  mp1::layout<int> invalid;
  BOOST_TEST(not invalid.is_valid());
  BOOST_TEST(not invalid.dense());
  BOOST_TEST(invalid.native_handle() == MPI_DATATYPE_NULL);
  BOOST_CHECK_THROW((void)invalid.size(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.byte_size(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.byte_extent(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.byte_lower_bound(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.byte_upper_bound(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.true_byte_extent(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.true_byte_lower_bound(), std::invalid_argument);
  BOOST_CHECK_THROW((void)invalid.true_byte_upper_bound(), std::invalid_argument);
  BOOST_CHECK_THROW(invalid.byte_resize(0, 8), std::invalid_argument);

  mp1::layout<int> invalid_copy{invalid};
  BOOST_TEST(not invalid_copy.is_valid());
  BOOST_TEST(not invalid_copy.dense());
  mp1::layout<int> invalid_assigned;
  invalid_assigned = invalid;
  BOOST_TEST(not invalid_assigned.is_valid());

  BOOST_TEST(not static_invalid_layout.is_valid());
}

BOOST_AUTO_TEST_CASE(contiguous_and_empty_factories) {
  const mp1::layout<int> contiguous = mp1::layout<int>::contiguous(5);
  BOOST_TEST(contiguous.size() == 5);
  BOOST_TEST(contiguous.byte_size() == static_cast<mp1::count_t>(5 * sizeof(int)));
  BOOST_TEST(contiguous.byte_extent() == static_cast<mp1::count_t>(5 * sizeof(int)));
  BOOST_TEST(contiguous.byte_lower_bound() == 0);
  BOOST_TEST(contiguous.true_byte_extent() == contiguous.byte_size());
  BOOST_TEST(contiguous.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(5, mp1::native_datatype<int>(), &raw),
                 "MPI_Type_contiguous_c(5)");
  const raw_datatype raw_contiguous{commit_raw(raw)};
  compare_layout(contiguous, raw_contiguous.get(), "contiguous(5)");

  const mp1::layout<int> zero = mp1::layout<int>::contiguous(0);
  BOOST_TEST(zero.is_valid());
  BOOST_TEST(zero.size() == 0);
  BOOST_TEST(zero.byte_size() == 0);
  BOOST_TEST(zero.byte_extent() == 0);
  BOOST_TEST(zero.byte_lower_bound() == 0);
  BOOST_TEST(zero.true_byte_extent() == 0);
  BOOST_TEST(zero.dense());

  MPI_Datatype raw_zero = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(0, mp1::native_datatype<int>(), &raw_zero),
                 "MPI_Type_contiguous_c(0)");
  const raw_datatype raw_empty{commit_raw(raw_zero)};
  compare_layout(zero, raw_empty.get(), "empty");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(3);
  const mp1::layout<int> nested = mp1::layout<int>::contiguous(2, block);
  BOOST_TEST(nested.size() == 6);
  BOOST_TEST(nested.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(nested.byte_extent() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(nested.dense());

  MPI_Datatype raw_nested = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(2, block.native_handle(), &raw_nested),
                 "MPI_Type_contiguous_c(2, block)");
  const raw_datatype raw_nested_type{commit_raw(raw_nested)};
  compare_layout(nested, raw_nested_type.get(), "contiguous(2, block)");

  const mp1::layout<int> zero_nested = mp1::layout<int>::contiguous(0, block);
  BOOST_TEST(zero_nested.is_valid());
  BOOST_TEST(zero_nested.size() == 0);
  BOOST_TEST(zero_nested.byte_size() == 0);
  BOOST_TEST(zero_nested.dense());

  const std::vector<mp1::layout<int>> default_layouts(3);
  BOOST_TEST(default_layouts.size() == 3);
  for (const auto& entry : default_layouts)
    BOOST_TEST(not entry.is_valid());

  const std::vector<mp1::layout<int>> empty_layouts(2, mp1::layout<int>::contiguous(0));
  BOOST_TEST(empty_layouts.size() == 2);
  for (const auto& entry : empty_layouts) {
    BOOST_TEST(entry.is_valid());
    BOOST_TEST(entry.size() == 0);
    BOOST_TEST(entry.dense());
  }
  BOOST_TEST(empty_layouts[0].native_handle() != empty_layouts[1].native_handle());
}

BOOST_AUTO_TEST_CASE(vector_factories) {
  const mp1::layout<int> vector = mp1::layout<int>::vector(3, 2, 5);
  BOOST_TEST(vector.size() == 6);
  BOOST_TEST(vector.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(vector.byte_extent() == static_cast<mp1::count_t>(12 * sizeof(int)));
  BOOST_TEST(vector.true_byte_extent() == vector.byte_extent());
  BOOST_TEST(not vector.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_vector_c(3, 2, 5, mp1::native_datatype<int>(), &raw),
                 "MPI_Type_vector_c(3, 2, 5)");
  const raw_datatype raw_vector{commit_raw(raw)};
  compare_layout(vector, raw_vector.get(), "vector(3, 2, 5)");

  // The option struct is the target spelling; it selects the same constructor.
  const mp1::layout<int> options_vector = mp1::layout<int>::vector(
      mp1::vector_options<int>{.count = 3, .blocklength = 2, .stride = 5, .block = nullptr});
  compare_layout(options_vector, raw_vector.get(), "vector(options)");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> vector_block = mp1::layout<int>::vector(2, 3, 4, block);
  BOOST_TEST(vector_block.size() == 12);
  BOOST_TEST(vector_block.byte_size() == static_cast<mp1::count_t>(12 * sizeof(int)));
  BOOST_TEST(not vector_block.dense());

  MPI_Datatype raw_block = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_vector_c(2, 3, 4, block.native_handle(), &raw_block),
                 "MPI_Type_vector_c(2, 3, 4, block)");
  const raw_datatype raw_vector_block{commit_raw(raw_block)};
  compare_layout(vector_block, raw_vector_block.get(), "vector(2, 3, 4, block)");

  const mp1::layout<int> options_vector_block = mp1::layout<int>::vector(
      mp1::vector_options<int>{.count = 2, .blocklength = 3, .stride = 4, .block = &block});
  compare_layout(options_vector_block, raw_vector_block.get(), "vector(options, block)");

  const mp1::layout<int> strided = mp1::layout<int>::strided_vector(2, 3, 16);
  BOOST_TEST(strided.size() == 6);
  BOOST_TEST(strided.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(strided.byte_extent() == static_cast<mp1::count_t>(28));
  BOOST_TEST(not strided.dense());

  MPI_Datatype raw_strided = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_create_hvector_c(2, 3, 16, mp1::native_datatype<int>(), &raw_strided),
                 "MPI_Type_create_hvector_c(2, 3, 16)");
  const raw_datatype raw_strided_type{commit_raw(raw_strided)};
  compare_layout(strided, raw_strided_type.get(), "strided_vector(2, 3, 16)");

  const mp1::layout<int> options_strided =
      mp1::layout<int>::strided_vector(mp1::strided_vector_options<int>{
          .count = 2, .blocklength = 3, .byte_stride = 16, .block = nullptr});
  compare_layout(options_strided, raw_strided_type.get(), "strided_vector(options)");

  const mp1::layout<int> strided_block = mp1::layout<int>::strided_vector(2, 3, 32, block);
  BOOST_TEST(strided_block.size() == 12);
  BOOST_TEST(strided_block.byte_size() == static_cast<mp1::count_t>(12 * sizeof(int)));
  BOOST_TEST(not strided_block.dense());

  MPI_Datatype raw_strided_block = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_create_hvector_c(2, 3, 32, block.native_handle(), &raw_strided_block),
                 "MPI_Type_create_hvector_c(2, 3, 32, block)");
  const raw_datatype raw_strided_block_type{commit_raw(raw_strided_block)};
  compare_layout(strided_block, raw_strided_block_type.get(),
                 "strided_vector(2, 3, 32, block)");

  // The option-struct block form is the option-bag spelling of the same constructor; the
  // positional block path above is not a substitute for this comparison.
  const mp1::layout<int> options_strided_block =
      mp1::layout<int>::strided_vector(mp1::strided_vector_options<int>{
          .count = 2, .blocklength = 3, .byte_stride = 32, .block = &block});
  compare_layout(options_strided_block, raw_strided_block_type.get(),
                 "strided_vector(options, block)");

  BOOST_TEST(throws_invalid_argument([] { (void)mp1::layout<int>::vector(-1, 1, 1); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::layout<int>::strided_vector(1, 1, -4); }));
  const mp1::layout<int> invalid_block;
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::layout<int>::vector(1, 1, 1, invalid_block); }));
}

BOOST_AUTO_TEST_CASE(indexed_factories) {
  mp1::indexed_parameter parameter;
  parameter.add(2, 0);
  parameter.add(1, 4);
  const mp1::layout<int> indexed = mp1::layout<int>::indexed(parameter);
  BOOST_TEST(indexed.size() == 3);
  BOOST_TEST(indexed.byte_size() == static_cast<mp1::count_t>(3 * sizeof(int)));
  BOOST_TEST(indexed.byte_extent() == static_cast<mp1::count_t>(20));
  BOOST_TEST(not indexed.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  {
    const MPI_Count blocklengths[] = {2, 1};
    const MPI_Count offsets[] = {0, 4};
    require_mpi_ok(
        MPI_Type_indexed_c(2, blocklengths, offsets, mp1::native_datatype<int>(), &raw),
        "MPI_Type_indexed_c");
  }
  const raw_datatype raw_indexed{commit_raw(raw)};
  compare_layout(indexed, raw_indexed.get(), "indexed({2@0, 1@4})");

  mp1::indexed_parameter adjacent;
  adjacent.add(2, 0);
  adjacent.add(3, 2);
  const mp1::layout<int> dense_indexed = mp1::layout<int>::indexed(adjacent);
  BOOST_TEST(dense_indexed.size() == 5);
  BOOST_TEST(dense_indexed.byte_size() == static_cast<mp1::count_t>(5 * sizeof(int)));
  BOOST_TEST(dense_indexed.dense());

  MPI_Datatype raw_adjacent = MPI_DATATYPE_NULL;
  {
    const MPI_Count blocklengths[] = {2, 3};
    const MPI_Count offsets[] = {0, 2};
    require_mpi_ok(MPI_Type_indexed_c(2, blocklengths, offsets, mp1::native_datatype<int>(),
                                      &raw_adjacent),
                   "MPI_Type_indexed_c(adjacent)");
  }
  const raw_datatype raw_adjacent_type{commit_raw(raw_adjacent)};
  compare_layout(dense_indexed, raw_adjacent_type.get(), "indexed(adjacent)");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
  mp1::indexed_parameter block_parameter;
  block_parameter.add(2, 0);
  block_parameter.add(1, 3);
  const mp1::layout<int> indexed_block = mp1::layout<int>::indexed(block_parameter, block);
  BOOST_TEST(indexed_block.size() == 6);
  BOOST_TEST(indexed_block.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(not indexed_block.dense());

  MPI_Datatype raw_block = MPI_DATATYPE_NULL;
  {
    const MPI_Count blocklengths[] = {2, 1};
    const MPI_Count offsets[] = {0, 3};
    require_mpi_ok(
        MPI_Type_indexed_c(2, blocklengths, offsets, block.native_handle(), &raw_block),
        "MPI_Type_indexed_c(block)");
  }
  const raw_datatype raw_indexed_block{commit_raw(raw_block)};
  compare_layout(indexed_block, raw_indexed_block.get(), "indexed(block)");

  const mp1::layout<int> empty_indexed = mp1::layout<int>::indexed(mp1::indexed_parameter{});
  BOOST_TEST(empty_indexed.is_valid());
  BOOST_TEST(empty_indexed.size() == 0);
  BOOST_TEST(empty_indexed.byte_size() == 0);
  BOOST_TEST(empty_indexed.dense());

  BOOST_TEST(throws_invalid_argument([] {
    mp1::indexed_parameter bad;
    bad.add(-1, 0);
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::indexed_parameter bad;
    bad.add(1, -1);
  }));
}

BOOST_AUTO_TEST_CASE(hindexed_factories) {
  mp1::hindexed_parameter parameter;
  parameter.add(2, 0);
  parameter.add(3, 16);
  const mp1::layout<int> hindexed = mp1::layout<int>::hindexed(parameter);
  BOOST_TEST(hindexed.size() == 5);
  BOOST_TEST(hindexed.byte_size() == static_cast<mp1::count_t>(5 * sizeof(int)));
  BOOST_TEST(hindexed.byte_extent() == static_cast<mp1::count_t>(28));
  BOOST_TEST(not hindexed.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  {
    const int blocklengths[] = {2, 3};
    const MPI_Aint offsets[] = {0, 16};
    require_mpi_ok(
        MPI_Type_create_hindexed(2, blocklengths, offsets, mp1::native_datatype<int>(), &raw),
        "MPI_Type_create_hindexed");
  }
  const raw_datatype raw_hindexed{commit_raw(raw)};
  compare_layout(hindexed, raw_hindexed.get(), "hindexed({2@0, 3@16})");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
  mp1::hindexed_parameter block_parameter;
  block_parameter.add(2, 0);
  block_parameter.add(1, 24);
  const mp1::layout<int> hindexed_block = mp1::layout<int>::hindexed(block_parameter, block);
  BOOST_TEST(hindexed_block.size() == 6);
  BOOST_TEST(hindexed_block.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(not hindexed_block.dense());

  MPI_Datatype raw_block = MPI_DATATYPE_NULL;
  {
    const int blocklengths[] = {2, 1};
    const MPI_Aint offsets[] = {0, 24};
    require_mpi_ok(
        MPI_Type_create_hindexed(2, blocklengths, offsets, block.native_handle(), &raw_block),
        "MPI_Type_create_hindexed(block)");
  }
  const raw_datatype raw_hindexed_block{commit_raw(raw_block)};
  compare_layout(hindexed_block, raw_hindexed_block.get(), "hindexed(block)");

  BOOST_TEST(throws_invalid_argument([] {
    mp1::hindexed_parameter bad;
    bad.add(-1, 0);
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::hindexed_parameter bad;
    bad.add(1, -1);
  }));
}

BOOST_AUTO_TEST_CASE(block_indexed_factories) {
  mp1::indexed_block_parameter indexed_block_parameter;
  indexed_block_parameter.add(0);
  indexed_block_parameter.add(3);
  const mp1::layout<int> indexed_block =
      mp1::layout<int>::indexed_block(2, indexed_block_parameter);
  BOOST_TEST(indexed_block.size() == 4);
  BOOST_TEST(indexed_block.byte_size() == static_cast<mp1::count_t>(4 * sizeof(int)));
  BOOST_TEST(indexed_block.byte_extent() == static_cast<mp1::count_t>(20));
  BOOST_TEST(not indexed_block.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  {
    const MPI_Count offsets[] = {0, 3};
    require_mpi_ok(
        MPI_Type_create_indexed_block_c(2, 2, offsets, mp1::native_datatype<int>(), &raw),
        "MPI_Type_create_indexed_block_c");
  }
  const raw_datatype raw_indexed_block{commit_raw(raw)};
  compare_layout(indexed_block, raw_indexed_block.get(), "indexed_block(2, {0, 3})");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
  mp1::indexed_block_parameter indexed_block_with_layout_parameter;
  indexed_block_with_layout_parameter.add(0);
  indexed_block_with_layout_parameter.add(2);
  const mp1::layout<int> indexed_block_with_layout =
      mp1::layout<int>::indexed_block(1, indexed_block_with_layout_parameter, block);
  BOOST_TEST(indexed_block_with_layout.size() == 4);
  BOOST_TEST(indexed_block_with_layout.byte_size() ==
             static_cast<mp1::count_t>(4 * sizeof(int)));
  BOOST_TEST(not indexed_block_with_layout.dense());

  MPI_Datatype raw_block = MPI_DATATYPE_NULL;
  {
    const MPI_Count offsets[] = {0, 2};
    require_mpi_ok(
        MPI_Type_create_indexed_block_c(2, 1, offsets, block.native_handle(), &raw_block),
        "MPI_Type_create_indexed_block_c(block)");
  }
  const raw_datatype raw_indexed_block_with_layout{commit_raw(raw_block)};
  compare_layout(indexed_block_with_layout, raw_indexed_block_with_layout.get(),
                 "indexed_block(block)");

  mp1::hindexed_block_parameter hindexed_block_parameter;
  hindexed_block_parameter.add(0);
  hindexed_block_parameter.add(16);
  const mp1::layout<int> hindexed_block =
      mp1::layout<int>::hindexed_block(2, hindexed_block_parameter);
  BOOST_TEST(hindexed_block.size() == 4);
  BOOST_TEST(hindexed_block.byte_size() == static_cast<mp1::count_t>(4 * sizeof(int)));
  BOOST_TEST(hindexed_block.byte_extent() == static_cast<mp1::count_t>(24));
  BOOST_TEST(not hindexed_block.dense());

  MPI_Datatype raw_h = MPI_DATATYPE_NULL;
  {
    const MPI_Aint offsets[] = {0, 16};
    require_mpi_ok(
        MPI_Type_create_hindexed_block(2, 2, offsets, mp1::native_datatype<int>(), &raw_h),
        "MPI_Type_create_hindexed_block");
  }
  const raw_datatype raw_hindexed_block{commit_raw(raw_h)};
  compare_layout(hindexed_block, raw_hindexed_block.get(), "hindexed_block(2, {0, 16})");

  mp1::hindexed_block_parameter hindexed_block_with_layout_parameter;
  hindexed_block_with_layout_parameter.add(0);
  hindexed_block_with_layout_parameter.add(16);
  const mp1::layout<int> hindexed_block_with_layout =
      mp1::layout<int>::hindexed_block(1, hindexed_block_with_layout_parameter, block);
  BOOST_TEST(hindexed_block_with_layout.size() == 4);
  BOOST_TEST(hindexed_block_with_layout.byte_size() ==
             static_cast<mp1::count_t>(4 * sizeof(int)));
  BOOST_TEST(not hindexed_block_with_layout.dense());

  MPI_Datatype raw_h_block = MPI_DATATYPE_NULL;
  {
    const MPI_Aint offsets[] = {0, 16};
    require_mpi_ok(
        MPI_Type_create_hindexed_block(2, 1, offsets, block.native_handle(), &raw_h_block),
        "MPI_Type_create_hindexed_block(block)");
  }
  const raw_datatype raw_hindexed_block_with_layout{commit_raw(raw_h_block)};
  compare_layout(hindexed_block_with_layout, raw_hindexed_block_with_layout.get(),
                 "hindexed_block(block)");

  BOOST_TEST(throws_invalid_argument([] {
    mp1::indexed_block_parameter parameter;
    parameter.add(0);
    (void)mp1::layout<int>::indexed_block(-1, parameter);
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::hindexed_block_parameter parameter;
    parameter.add(MPI_Aint{0});
    (void)mp1::layout<int>::hindexed_block(-1, parameter);
  }));

  // Blocklength is count_t. An int literal still converts; MPI's hindexed-block C
  // binding takes an int, so a count_t value outside that range is rejected before MPI.
  mp1::indexed_block_parameter count_t_indexed_block_parameter;
  count_t_indexed_block_parameter.add(0);
  count_t_indexed_block_parameter.add(3);
  const mp1::layout<int> count_t_indexed_block =
      mp1::layout<int>::indexed_block(mp1::count_t{2}, count_t_indexed_block_parameter);
  BOOST_TEST(count_t_indexed_block.size() == 4);
  mp1::hindexed_block_parameter count_t_hindexed_block_parameter;
  count_t_hindexed_block_parameter.add(MPI_Aint{0});
  count_t_hindexed_block_parameter.add(MPI_Aint{2 * sizeof(int)});
  const mp1::layout<int> count_t_hindexed_block =
      mp1::layout<int>::hindexed_block(mp1::count_t{2}, count_t_hindexed_block_parameter);
  BOOST_TEST(count_t_hindexed_block.size() == 4);
  // 4 blocks of 2 ints; MPI_Type_create_hindexed_block blocklength is still int.
  BOOST_TEST(throws_invalid_argument([] {
    mp1::hindexed_block_parameter parameter;
    parameter.add(MPI_Aint{0});
    (void)mp1::layout<int>::hindexed_block(
        static_cast<mp1::count_t>(std::numeric_limits<int>::max()) + 1, parameter);
  }));
}

// MPI 4.1 §5.1.8 defines the true lower bound/extent over the typemap entries; an empty
// typemap leaves them undefined, and Open MPI 6.0.0rc2 returns a sentinel (true_lb = INT64_MAX,
// true_extent = 1) for some zero-byte derived layouts, which used to make the public
// true_byte_upper_bound() throw for a valid layout.  The wrapper reports the canonical empty
// geometry (0/0/0); byte_size() and the extent marker stay untouched.
BOOST_AUTO_TEST_CASE(zero_size_true_extent_is_total) {
  const auto check_empty = [](const auto& empty_layout) {
    BOOST_TEST(empty_layout.is_valid());
    BOOST_TEST(empty_layout.byte_size() == 0);
    BOOST_TEST(empty_layout.true_byte_lower_bound() == 0);
    BOOST_TEST(empty_layout.true_byte_extent() == 0);
    BOOST_TEST(empty_layout.true_byte_upper_bound() == 0);
    BOOST_TEST(empty_layout.byte_extent() >= 0);
  };

  check_empty(mp1::layout<int>::contiguous(0));
  check_empty(mp1::layout<int>::contiguous(3, mp1::layout<int>::contiguous(0)));
  // An empty block with a positive repetition count: the empty typemap is repeated but still
  // addresses no byte.
  check_empty(mp1::layout<int>::vector(2, 1, 1, mp1::layout<int>::contiguous(0)));
  check_empty(mp1::layout<int>::vector(2, 3, 2, mp1::layout<int>::contiguous(0)));
  // An empty reflected aggregate has an empty native typemap with extent 1.
  check_empty(mp1::layout<reflected_empty>::vector(2, 1, 1));
  check_empty(mp1::layout<reflected_empty>::contiguous(2));
}

BOOST_AUTO_TEST_CASE(subarray_factory) {
  mp1::subarray_parameter parameter;
  parameter.add(4, 2, 1);
  parameter.add(5, 3, 2);
  const mp1::layout<int> subarray = mp1::layout<int>::subarray(parameter);
  BOOST_TEST(subarray.size() == 6);
  BOOST_TEST(subarray.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(subarray.byte_extent() == static_cast<mp1::count_t>(80));
  BOOST_TEST(subarray.byte_lower_bound() == 0);
  BOOST_TEST(subarray.true_byte_lower_bound() == static_cast<mp1::count_t>(28));
  BOOST_TEST(subarray.true_byte_extent() == static_cast<mp1::count_t>(32));
  BOOST_TEST(not subarray.dense());

  MPI_Datatype raw = MPI_DATATYPE_NULL;
  {
    const int sizes[] = {4, 5};
    const int subsizes[] = {2, 3};
    const int starts[] = {1, 2};
    require_mpi_ok(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                            mp1::native_datatype<int>(), &raw),
                   "MPI_Type_create_subarray(row_major)");
  }
  const raw_datatype raw_subarray{commit_raw(raw)};
  compare_layout(subarray, raw_subarray.get(), "subarray(row_major)");

  mp1::subarray_parameter column_parameter;
  column_parameter.add(4, 2, 1);
  column_parameter.add(5, 3, 2);
  column_parameter.order(mp1::array_order::column_major);
  const mp1::layout<int> column_subarray = mp1::layout<int>::subarray(column_parameter);
  BOOST_TEST(column_subarray.size() == 6);
  BOOST_TEST(column_subarray.byte_size() == static_cast<mp1::count_t>(6 * sizeof(int)));
  BOOST_TEST(not column_subarray.dense());
  BOOST_TEST(column_subarray.true_byte_lower_bound() != subarray.true_byte_lower_bound());

  MPI_Datatype raw_column = MPI_DATATYPE_NULL;
  {
    const int sizes[] = {4, 5};
    const int subsizes[] = {2, 3};
    const int starts[] = {1, 2};
    require_mpi_ok(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_FORTRAN,
                                            mp1::native_datatype<int>(), &raw_column),
                   "MPI_Type_create_subarray(column_major)");
  }
  const raw_datatype raw_column_subarray{commit_raw(raw_column)};
  compare_layout(column_subarray, raw_column_subarray.get(), "subarray(column_major)");

  const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> subarray_block = mp1::layout<int>::subarray(parameter, block);
  BOOST_TEST(subarray_block.size() == 12);
  BOOST_TEST(subarray_block.byte_size() == static_cast<mp1::count_t>(12 * sizeof(int)));
  BOOST_TEST(not subarray_block.dense());

  MPI_Datatype raw_block = MPI_DATATYPE_NULL;
  {
    const int sizes[] = {4, 5};
    const int subsizes[] = {2, 3};
    const int starts[] = {1, 2};
    require_mpi_ok(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                            block.native_handle(), &raw_block),
                   "MPI_Type_create_subarray(block)");
  }
  const raw_datatype raw_subarray_block{commit_raw(raw_block)};
  compare_layout(subarray_block, raw_subarray_block.get(), "subarray(block)");

  // Open MPI rejects MPI_Type_create_subarray with a zero subsize; the portable fallback is a
  // zero type. No MPI constructor is called for this case.
  mp1::subarray_parameter zero_parameter;
  zero_parameter.add(4, 2, 1);
  zero_parameter.add(5, 0, 0);
  const mp1::layout<int> zero_subarray = mp1::layout<int>::subarray(zero_parameter);
  BOOST_TEST(zero_subarray.is_valid());
  BOOST_TEST(zero_subarray.size() == 0);
  BOOST_TEST(zero_subarray.byte_size() == 0);
  BOOST_TEST(zero_subarray.byte_extent() == 0);
  BOOST_TEST(zero_subarray.dense());
  BOOST_TEST(zero_subarray.native_handle() != MPI_DATATYPE_NULL);

  const mp1::layout<int> zero_subarray_block =
      mp1::layout<int>::subarray(zero_parameter, block);
  BOOST_TEST(zero_subarray_block.is_valid());
  BOOST_TEST(zero_subarray_block.size() == 0);
  BOOST_TEST(zero_subarray_block.byte_size() == 0);

  // A zero subsize dimension after several INT_MAX dimensions must select the zero fallback
  // before any count_t product overflow; MPI_Type_create_subarray is never called.
  mp1::subarray_parameter huge_zero_parameter;
  huge_zero_parameter.add(INT_MAX, INT_MAX, 0);
  huge_zero_parameter.add(INT_MAX, INT_MAX, 0);
  huge_zero_parameter.add(INT_MAX, INT_MAX, 0);
  huge_zero_parameter.add(INT_MAX, 0, 0);
  const mp1::layout<int> huge_zero_subarray = mp1::layout<int>::subarray(huge_zero_parameter);
  BOOST_TEST(huge_zero_subarray.is_valid());
  BOOST_TEST(huge_zero_subarray.size() == 0);
  BOOST_TEST(huge_zero_subarray.byte_size() == 0);
  BOOST_TEST(huge_zero_subarray.byte_extent() == 0);

  BOOST_TEST(throws_invalid_argument([] {
    mp1::subarray_parameter bad;
    bad.add(4, 5, 0);
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::subarray_parameter bad;
    bad.add(-1, 0, 0);
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::subarray_parameter bad;
    bad.add(4, 2, 3);
  }));

  // Large subsizes must not be truncated to int products. This type is larger than 2 GiB in
  // byte size and cannot be represented by the old int MPI_Type_size path.
  mp1::subarray_parameter large_parameter;
  large_parameter.add(INT_MAX, INT_MAX, 0);
  large_parameter.add(2, 1, 0);
  const mp1::layout<int> large_subarray = mp1::layout<int>::subarray(large_parameter);
  BOOST_TEST(large_subarray.size() == static_cast<mp1::count_t>(INT_MAX));
  BOOST_TEST(large_subarray.byte_size() ==
             static_cast<mp1::count_t>(INT_MAX) * static_cast<mp1::count_t>(sizeof(int)));
  BOOST_TEST(large_subarray.byte_extent() >
             static_cast<mp1::count_t>(INT_MAX) * static_cast<mp1::count_t>(sizeof(int)));

  MPI_Datatype raw_large = MPI_DATATYPE_NULL;
  {
    const int sizes[] = {INT_MAX, 2};
    const int subsizes[] = {INT_MAX, 1};
    const int starts[] = {0, 0};
    require_mpi_ok(MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C,
                                            mp1::native_datatype<int>(), &raw_large),
                   "MPI_Type_create_subarray(large)");
  }
  const raw_datatype raw_large_subarray{commit_raw(raw_large)};
  compare_layout(large_subarray, raw_large_subarray.get(), "subarray(large)");
}

BOOST_AUTO_TEST_CASE(copy_move_and_resize) {
  const mp1::layout<int> original = mp1::layout<int>::vector(3, 2, 5);

  mp1::layout<int> copied{original};
  BOOST_TEST(copied.native_handle() != original.native_handle());
  BOOST_TEST(copied.size() == original.size());
  BOOST_TEST(copied.byte_size() == original.byte_size());
  BOOST_TEST(copied.byte_extent() == original.byte_extent());
  BOOST_TEST(copied.byte_lower_bound() == original.byte_lower_bound());
  BOOST_TEST(copied.true_byte_extent() == original.true_byte_extent());
  BOOST_TEST(copied.true_byte_lower_bound() == original.true_byte_lower_bound());
  BOOST_TEST(copied.dense() == original.dense());

  mp1::layout<int> copy_assigned;
  copy_assigned = original;
  BOOST_TEST(copy_assigned.native_handle() != original.native_handle());
  BOOST_TEST(copy_assigned.byte_size() == original.byte_size());
  mp1::layout<int>* copy_self = &copy_assigned;
  copy_assigned = *copy_self;
  BOOST_TEST(copy_assigned.byte_size() == original.byte_size());

  const MPI_Datatype moved_handle = copied.native_handle();
  mp1::layout<int> moved{std::move(copied)};
  BOOST_TEST(moved.native_handle() == moved_handle);
  BOOST_TEST(moved.size() == original.size());
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not copied.is_valid());
  BOOST_CHECK_THROW((void)copied.size(), std::invalid_argument);

  mp1::layout<int> move_assigned;
  move_assigned = std::move(moved);
  BOOST_TEST(move_assigned.native_handle() == moved_handle);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not moved.is_valid());
  mp1::layout<int>* self = &move_assigned;
  move_assigned = std::move(*self);
  BOOST_TEST(move_assigned.native_handle() == moved_handle);

  mp1::layout<int> resized = mp1::layout<int>::contiguous(2);
  resized.byte_resize(-4, 16);
  BOOST_TEST(resized.size() == 2);
  BOOST_TEST(resized.byte_size() == static_cast<mp1::count_t>(2 * sizeof(int)));
  BOOST_TEST(resized.byte_lower_bound() == -4);
  BOOST_TEST(resized.byte_extent() == 16);
  BOOST_TEST(resized.byte_upper_bound() == 12);

  MPI_Datatype raw_resized = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_create_resized(mp1::native_datatype<int>(), -4, 16, &raw_resized),
                 "MPI_Type_create_resized");
  // The raw type must describe the same two ints, so resize a contiguous(2) raw type.
  (void)MPI_Type_free(&raw_resized);
  raw_resized = MPI_DATATYPE_NULL;
  {
    MPI_Datatype raw_base = MPI_DATATYPE_NULL;
    require_mpi_ok(MPI_Type_contiguous_c(2, mp1::native_datatype<int>(), &raw_base),
                   "MPI_Type_contiguous_c(resize base)");
    raw_base = commit_raw(raw_base);
    require_mpi_ok(MPI_Type_create_resized(raw_base, -4, 16, &raw_resized),
                   "MPI_Type_create_resized(contiguous)");
    (void)MPI_Type_free(&raw_base);
  }
  const raw_datatype raw_resized_type{commit_raw(raw_resized)};
  compare_layout(resized, raw_resized_type.get(), "byte_resize(-4, 16)");

  // A negative extent caches a self-overlapping, unusable layout. Reject it before
  // MPI_Type_create_resized; the previous state must stay intact.
  mp1::layout<int> resized_negative_extent = mp1::layout<int>::contiguous(2);
  const MPI_Datatype before_negative = resized_negative_extent.native_handle();
  BOOST_TEST(throws_invalid_argument([&] { resized_negative_extent.byte_resize(8, -8); }));
  BOOST_TEST(resized_negative_extent.native_handle() == before_negative);
  BOOST_TEST(resized_negative_extent.byte_extent() ==
             static_cast<mp1::count_t>(2 * sizeof(int)));

  // resized() is the non-mutating spelling.
  const mp1::layout<int> resized_copy = mp1::layout<int>::contiguous(2).resized(-4, 16);
  BOOST_TEST(resized_copy.byte_lower_bound() == -4);
  BOOST_TEST(resized_copy.byte_extent() == 16);
  BOOST_TEST(resized_copy.size() == 2);
}

BOOST_AUTO_TEST_CASE(dense_predicate) {
  BOOST_TEST(mp1::layout<int>::contiguous(3).dense());
  BOOST_TEST(mp1::layout<int>::contiguous(0).dense());
  BOOST_TEST(not mp1::layout<int>::vector(3, 2, 5).dense());

  const mp1::layout<int> resized_dense = [] {
    mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    layout.byte_resize(0, 8);
    return layout;
  }();
  BOOST_TEST(resized_dense.byte_size() == resized_dense.byte_extent());
  BOOST_TEST(resized_dense.dense());

  const mp1::layout<int> resized_holey = [] {
    mp1::layout<int> layout = mp1::layout<int>::vector(2, 1, 2);
    layout.byte_resize(0, 8);
    return layout;
  }();
  BOOST_TEST(resized_holey.byte_size() == resized_holey.byte_extent());
  BOOST_TEST(resized_holey.byte_size() != resized_holey.true_byte_extent());
  BOOST_TEST(not resized_holey.dense());

  mp1::hindexed_parameter offset_parameter;
  offset_parameter.add(1, 16);
  const mp1::layout<int> offset_dense = mp1::layout<int>::hindexed(offset_parameter);
  BOOST_TEST(offset_dense.true_byte_lower_bound() == 16);
  BOOST_TEST(offset_dense.true_byte_extent() == static_cast<mp1::count_t>(sizeof(int)));
  BOOST_TEST(offset_dense.dense());
}

BOOST_AUTO_TEST_CASE(reflection_and_datatype_traits) {
  const MPI_Datatype big_type = mp1::native_datatype<reflected_119>();
  raw_metadata big_metadata = read_raw(big_type);
  BOOST_TEST(big_metadata.size == static_cast<MPI_Count>(119 * sizeof(int)));
  BOOST_TEST(big_metadata.lower_bound == 0);
  BOOST_TEST(big_metadata.extent == static_cast<MPI_Count>(sizeof(reflected_119)));

  const mp1::layout<reflected_119> big_layout = mp1::layout<reflected_119>::contiguous(1);
  BOOST_TEST(big_layout.size() == 1);
  BOOST_TEST(big_layout.byte_size() == static_cast<mp1::count_t>(119 * sizeof(int)));
  BOOST_TEST(big_layout.byte_extent() == static_cast<mp1::count_t>(sizeof(reflected_119)));

  MPI_Datatype raw_big = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(1, big_type, &raw_big),
                 "MPI_Type_contiguous_c(reflected_119)");
  const raw_datatype raw_big_type{commit_raw(raw_big)};
  compare_layout(big_layout, raw_big_type.get(), "reflected_119 contiguous");

  reflected_119 values{};
  values.m0 = 17;
  values.m118 = -3;
  reflected_119 received{};
  require_mpi_ok(MPI_Sendrecv(&values, 1, big_type, 0, 700, &received, 1, big_type, 0, 700,
                              MPI_COMM_SELF, MPI_STATUS_IGNORE),
                 "MPI_Sendrecv(reflected_119)");
  BOOST_TEST(received.m0 == 17);
  BOOST_TEST(received.m118 == -3);

  const MPI_Datatype empty_type = mp1::native_datatype<reflected_empty>();
  raw_metadata empty_metadata = read_raw(empty_type);
  BOOST_TEST(empty_metadata.size == 0);
  BOOST_TEST(empty_metadata.extent == static_cast<MPI_Count>(sizeof(reflected_empty)));

  const MPI_Datatype tuple_type = mp1::native_datatype<std::tuple<>>();
  raw_metadata tuple_metadata = read_raw(tuple_type);
  BOOST_TEST(tuple_metadata.size == 0);
  BOOST_TEST(tuple_metadata.extent == static_cast<MPI_Count>(sizeof(std::tuple<>)));

  const MPI_Datatype wrapper_type = mp1::native_datatype<reflected_wrapper<double>>();
  raw_metadata wrapper_metadata = read_raw(wrapper_type);
  BOOST_TEST(wrapper_metadata.size == static_cast<MPI_Count>(sizeof(double)));
  BOOST_TEST(wrapper_metadata.extent ==
             static_cast<MPI_Count>(sizeof(reflected_wrapper<double>)));

  const MPI_Datatype pair_reflected_type = mp1::native_datatype<reflected_pair<int, double>>();
  raw_metadata pair_reflected_metadata = read_raw(pair_reflected_type);
  BOOST_TEST(pair_reflected_metadata.size ==
             static_cast<MPI_Count>(sizeof(int) + sizeof(double)));
  BOOST_TEST(pair_reflected_metadata.extent ==
             static_cast<MPI_Count>(sizeof(reflected_pair<int, double>)));

  BOOST_TEST(mp1::native_datatype<custom_int>() == MPI_INT);

  const MPI_Datatype pair_type = mp1::native_datatype<std::pair<int, double>>();
  raw_metadata pair_metadata = read_raw(pair_type);
  BOOST_TEST(pair_metadata.size == static_cast<MPI_Count>(sizeof(int) + sizeof(double)));
  BOOST_TEST(pair_metadata.extent == static_cast<MPI_Count>(sizeof(std::pair<int, double>)));

  const MPI_Datatype array_type = mp1::native_datatype<std::array<int, 3>>();
  raw_metadata array_metadata = read_raw(array_type);
  BOOST_TEST(array_metadata.size == static_cast<MPI_Count>(3 * sizeof(int)));
  BOOST_TEST(array_metadata.extent == static_cast<MPI_Count>(sizeof(std::array<int, 3>)));

  const MPI_Datatype c_array_type = mp1::native_datatype<int[3]>();
  raw_metadata c_array_metadata = read_raw(c_array_type);
  BOOST_TEST(c_array_metadata.size == static_cast<MPI_Count>(3 * sizeof(int)));
  BOOST_TEST(c_array_metadata.extent == static_cast<MPI_Count>(sizeof(int[3])));

  const mp1::layout<std::tuple<>> empty_tuple_layout = mp1::layout<std::tuple<>>::contiguous(0);
  BOOST_TEST(empty_tuple_layout.is_valid());
  BOOST_TEST(empty_tuple_layout.size() == 0);
  BOOST_TEST(empty_tuple_layout.byte_size() == 0);
  BOOST_TEST(empty_tuple_layout.dense());
}

BOOST_AUTO_TEST_CASE(span_from_contract) {
  // The checked pointer + count -> span conversion. The raw std::span
  // (pointer, size_type) constructor would turn a negative count_t into a huge size, so the
  // negative row must throw before any use.
  int values[4] = {1, 2, 3, 4};
  const std::span<int> span = mp1::span_from(values, 4);
  BOOST_TEST(span.size() == 4);
  BOOST_TEST(span.data() == values);

  const std::span<const int> const_span = mp1::span_from(static_cast<const int*>(values), 2);
  BOOST_TEST(const_span.size() == 2);
  BOOST_TEST(mp1::span_from<int>(nullptr, 0).empty());
  BOOST_TEST(mp1::span_from<int>(nullptr, 0).data() == nullptr);

  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::span_from(values, -1); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::span_from<int>(nullptr, -1); }));
}

BOOST_AUTO_TEST_CASE(public_layout_facts_and_from_raw) {
  // The public facts hint and the adopting from_raw contract.
  static_assert(std::is_aggregate_v<mp1::layout_facts>);
  const mp1::layout_facts default_facts{};
  BOOST_TEST(not default_facts.packed);
  BOOST_TEST(not default_facts.base_repetition);
  BOOST_TEST(not default_facts.has_overlapping_entries);

  // Adoption: an uncommitted raw datatype is committed, queried and owned by the layout.
  MPI_Datatype raw_uncommitted = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(4, mp1::native_datatype<int>(), &raw_uncommitted),
                 "MPI_Type_contiguous_c(from_raw)");
  {
    const mp1::layout<int> adopted = mp1::layout<int>::from_raw(raw_uncommitted, 4);
    BOOST_TEST(adopted.native_handle() != MPI_DATATYPE_NULL);
    BOOST_TEST(adopted.size() == 4);
    BOOST_TEST(adopted.byte_size() == static_cast<mp1::count_t>(4 * sizeof(int)));
    BOOST_TEST(adopted.byte_extent() == static_cast<mp1::count_t>(4 * sizeof(int)));
    // An absent hint proves nothing: no structural predicate may be claimed.
    BOOST_TEST(not adopted.dense());
    BOOST_TEST(not adopted.is_base_repetition());
    BOOST_TEST(not adopted.has_overlapping_entries());
  }

  // Adoption is ownership: the committed handle is freed exactly once when the layout dies.
#if MP1_LAYOUT_HAVE_WRAP
  if (mpi_wrap::supported()) {
    MPI_Datatype raw_owned = MPI_DATATYPE_NULL;
    require_mpi_ok(MPI_Type_contiguous_c(2, mp1::native_datatype<int>(), &raw_owned),
                   "MPI_Type_contiguous_c(from_raw ownership)");
    mpi_wrap::reset_counts();
    {
      const mp1::layout<int> owned = mp1::layout<int>::from_raw(raw_owned, 2);
      BOOST_TEST(owned.size() == 2);
      BOOST_TEST(mpi_wrap::count("MPI_Type_commit") == 1u);
    }
    BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 1u);
  }
#endif

  // The explicit facts form proves the structural predicates the caller knows.
  MPI_Datatype raw_hinted = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(4, mp1::native_datatype<int>(), &raw_hinted),
                 "MPI_Type_contiguous_c(from_raw hints)");
  const mp1::layout<int> hinted = mp1::layout<int>::from_raw(
      raw_hinted, 4,
      mp1::layout_facts{
          .packed = true, .base_repetition = true, .has_overlapping_entries = false});
  BOOST_TEST(hinted.dense());
  BOOST_TEST(hinted.is_base_repetition());

  // An already committed handle is adopted too (MPI_Type_commit is a documented no-op).
  MPI_Datatype raw_committed = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(2, mp1::native_datatype<int>(), &raw_committed),
                 "MPI_Type_contiguous_c(from_raw committed)");
  require_mpi_ok(MPI_Type_commit(&raw_committed), "MPI_Type_commit(from_raw committed)");
  const mp1::layout<int> committed = mp1::layout<int>::from_raw(raw_committed, 2);
  BOOST_TEST(committed.size() == 2);
  BOOST_TEST(committed.byte_size() == static_cast<mp1::count_t>(2 * sizeof(int)));

  // A dup of a predefined type reports MPI_COMBINER_DUP: it is a freeable derived handle and
  // must still be adopted (the predefined-handle rejection must not catch it).
  MPI_Datatype raw_dup = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_dup(MPI_INT, &raw_dup), "MPI_Type_dup(MPI_INT)");
  {
    const mp1::layout<int> dup = mp1::layout<int>::from_raw(raw_dup, 1);
    BOOST_TEST(dup.size() == 1);
    BOOST_TEST(dup.byte_size() == static_cast<mp1::count_t>(sizeof(int)));
  }

  // The designated aggregate spelling selects the public facts overload. Before the
  // overload-ambiguity fix this failed to compile because the private detail::layout_facts
  // overload also carries a `packed` member; this call is the regression.
  MPI_Datatype raw_designated = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(3, mp1::native_datatype<int>(), &raw_designated),
                 "MPI_Type_contiguous_c(from_raw designated)");
  const mp1::layout<int> designated =
      mp1::layout<int>::from_raw(raw_designated, 3, {.packed = true});
  BOOST_TEST(designated.size() == 3);
  BOOST_TEST(designated.dense());

  // The bare-braced default facts spelling is unambiguous after the two public overloads
  // collapsed into `from_raw(MPI_Datatype, count_t, layout_facts = {})`.
  MPI_Datatype raw_braced = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(2, mp1::native_datatype<int>(), &raw_braced),
                 "MPI_Type_contiguous_c(from_raw braced)");
  const mp1::layout<int> braced = mp1::layout<int>::from_raw(raw_braced, 2, {});
  BOOST_TEST(braced.size() == 2);
  BOOST_TEST(braced.byte_size() == static_cast<mp1::count_t>(2 * sizeof(int)));
  // An absent hint proves nothing.
  BOOST_TEST(not braced.dense());
  BOOST_TEST(not braced.is_base_repetition());

  // A proven overlap is adopted and reported; receive lowering rejects it.
  MPI_Datatype raw_overlap = MPI_DATATYPE_NULL;
  {
    const MPI_Count blocklengths[] = {1, 1};
    const MPI_Count offsets[] = {0, 0};
    require_mpi_ok(
        MPI_Type_indexed_c(2, blocklengths, offsets, mp1::native_datatype<int>(), &raw_overlap),
        "MPI_Type_indexed_c(from_raw overlap)");
  }
  const mp1::layout<int> overlapping = mp1::layout<int>::from_raw(
      raw_overlap, 2,
      mp1::layout_facts{
          .packed = false, .base_repetition = false, .has_overlapping_entries = true});
  BOOST_TEST(overlapping.has_overlapping_entries());

  // Rejections: null, predefined and negative. A predefined handle is never adopted/freed, so
  // MPI_INT stays usable afterwards.
  BOOST_TEST(
      throws_invalid_argument([] { (void)mp1::layout<int>::from_raw(MPI_DATATYPE_NULL, 1); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::layout<int>::from_raw(MPI_INT, 1); }));
  MPI_Count int_size{0};
  require_mpi_ok(MPI_Type_size_c(MPI_INT, &int_size), "MPI_Type_size_c(MPI_INT)");
  BOOST_TEST(int_size == static_cast<MPI_Count>(sizeof(int)));

  MPI_Datatype raw_negative = MPI_DATATYPE_NULL;
  require_mpi_ok(MPI_Type_contiguous_c(1, mp1::native_datatype<int>(), &raw_negative),
                 "MPI_Type_contiguous_c(from_raw negative)");
  BOOST_CHECK_THROW((void)mp1::layout<int>::from_raw(raw_negative, -1), std::invalid_argument);
  // A rejected handle is not adopted, so the caller still owns it.
  require_mpi_ok(MPI_Type_free(&raw_negative), "MPI_Type_free(rejected from_raw)");
}

BOOST_AUTO_TEST_CASE(predefined_datatype_rejection) {
  // from_raw rejects every predefined handle, not only MPI_COMBINER_NAMED.
  // MPI 4.1 §5.1.13 lists the combiners; §19.1 says the MPI_Type_create_f90_* results are
  // predefined, unnamed and "cannot be freed"; §6.9.4 says a value-index pair "cannot be
  // freed" and its decoders must handle both MPI_COMBINER_NAMED and MPI_COMBINER_VALUE_INDEX.
  // The adopting path commits, queries and frees the handle, so accepting one of these aborts
  // under MPI_ERRORS_ARE_FATAL (the pre-fix Open MPI behaviour for f90_real).
  //
  // Pin the classifier for every standard combiner: a provider that cannot return a
  // VALUE_INDEX or f90 handle still covers the decision that rejects it.
  static_assert(not mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_NAMED));
  static_assert(not mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_F90_REAL));
  static_assert(not mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_F90_COMPLEX));
  static_assert(not mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_F90_INTEGER));
  static_assert(not mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_VALUE_INDEX));
  static_assert(not mp1::detail::is_adoptable_derived_combiner(-1));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_DUP));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_CONTIGUOUS));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_VECTOR));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_HVECTOR));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_INDEXED));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_HINDEXED));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_INDEXED_BLOCK));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_HINDEXED_BLOCK));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_STRUCT));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_SUBARRAY));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_DARRAY));
  static_assert(mp1::detail::is_adoptable_derived_combiner(MPI_COMBINER_RESIZED));

  const auto combiner_of = [](MPI_Datatype type) {
    MPI_Count num_integers = 0;
    MPI_Count num_addresses = 0;
    MPI_Count num_large_counts = 0;
    MPI_Count num_datatypes = 0;
    int combiner = -1;
    require_mpi_ok(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses,
                                           &num_large_counts, &num_datatypes, &combiner),
                   "MPI_Type_get_envelope_c(predefined handle)");
    return combiner;
  };

  const auto rejected_and_intact = [&combiner_of](const char* label, MPI_Datatype type) {
    const int before = combiner_of(type);
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(throws_invalid_argument([&] { (void)mp1::layout<int>::from_raw(type, 1); }));
      // The rejection happens before commit/free, so the handle is unchanged and remains a
      // usable oldtype (MPI_INT stays usable is also asserted in the cases above).
      BOOST_TEST(combiner_of(type) == before);
      MPI_Count size = -1;
      require_mpi_ok(MPI_Type_size_c(type, &size), "MPI_Type_size_c(predefined handle)");
      BOOST_TEST(size > 0);
      MPI_Datatype derived = MPI_DATATYPE_NULL;
      require_mpi_ok(MPI_Type_contiguous_c(1, type, &derived),
                     "MPI_Type_contiguous_c(predefined oldtype)");
      const raw_datatype owned{commit_raw(derived)};
      BOOST_TEST(owned.get() != MPI_DATATYPE_NULL);
    }
  };

  // Named predefined handles (MPI_COMBINER_NAMED).
  BOOST_TEST(combiner_of(MPI_INT) == MPI_COMBINER_NAMED);
  rejected_and_intact("MPI_INT (NAMED)", MPI_INT);
  rejected_and_intact("MPI_2INT (NAMED)", MPI_2INT);

  // Predefined unnamed Fortran datatypes (MPI_COMBINER_F90_*): provider-conditional, because
  // MPICH 4.3.2 returns MPI_DATATYPE_NULL for these requests while Open MPI 6.0.0rc2 returns
  // the unnamed handle. A provider without the type is skipped, not asserted away.
  const auto check_f90 = [&combiner_of, &rejected_and_intact](const char* label, int rc,
                                                              MPI_Datatype type,
                                                              int expected_combiner) {
    if (rc != MPI_SUCCESS or type == MPI_DATATYPE_NULL) {
      BOOST_TEST_MESSAGE(label << ": provider returned MPI_DATATYPE_NULL (conditional skip)");
      return;
    }
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(combiner_of(type) == expected_combiner);
      rejected_and_intact(label, type);
    }
  };

  MPI_Datatype f90_real = MPI_DATATYPE_NULL;
  const int f90_real_rc = MPI_Type_create_f90_real(15, 307, &f90_real);
  check_f90("f90_real(15,307)", f90_real_rc, f90_real, MPI_COMBINER_F90_REAL);

  MPI_Datatype f90_complex = MPI_DATATYPE_NULL;
  const int f90_complex_rc = MPI_Type_create_f90_complex(15, 307, &f90_complex);
  check_f90("f90_complex(15,307)", f90_complex_rc, f90_complex, MPI_COMBINER_F90_COMPLEX);

  MPI_Datatype f90_integer = MPI_DATATYPE_NULL;
  const int f90_integer_rc = MPI_Type_create_f90_integer(15, &f90_integer);
  check_f90("f90_integer(15)", f90_integer_rc, f90_integer, MPI_COMBINER_F90_INTEGER);

  // Predefined unnamed value-index pairs (MPI_COMBINER_VALUE_INDEX). A provider may return a
  // named pair for the same (value, index) combination; both are predefined and both must be
  // rejected. Both pinned providers currently return only named pairs, so the VALUE_INDEX
  // combiner itself is pinned by the static_assert above and the probe log records the skip.
  bool saw_value_index = false;
  const MPI_Datatype value_types[] = {MPI_DOUBLE, MPI_FLOAT, MPI_INT};
  const MPI_Datatype index_types[] = {MPI_UINT64_T, MPI_INT64_T, MPI_INT};
  for (MPI_Datatype value_type : value_types) {
    for (MPI_Datatype index_type : index_types) {
      MPI_Datatype pair = MPI_DATATYPE_NULL;
      require_mpi_ok(MPI_Type_get_value_index(value_type, index_type, &pair),
                     "MPI_Type_get_value_index");
      if (pair == MPI_DATATYPE_NULL)
        continue;
      const int pair_combiner = combiner_of(pair);
      BOOST_TEST(
          (pair_combiner == MPI_COMBINER_NAMED or pair_combiner == MPI_COMBINER_VALUE_INDEX));
      if (pair_combiner == MPI_COMBINER_VALUE_INDEX)
        saw_value_index = true;
      rejected_and_intact("value-index pair", pair);
    }
  }
  if (not saw_value_index) {
    BOOST_TEST_MESSAGE(
        "value-index: provider returned no MPI_COMBINER_VALUE_INDEX handle; the combiner "
        "classifier static_assert above and the standalone probe cover that branch");
  }
}

BOOST_AUTO_TEST_CASE(value_index_datatype_is_borrowed_and_undecodable) {
  // MPI_COMBINER_VALUE_INDEX is a predefined unnamed value/index pair. MPI 4.1
  // §6.9.4/MPI_Type_get_value_index say it "is not a duplicate" and "cannot be freed", so a
  // walk that reads it out of MPI_Type_get_contents_c must not free it; it is also not a
  // derived map the layout type-map decoder models, so the decoder fails closed.
  // The provider may only expose named pairs (both pinned providers do), so the branch is
  // exercised where a VALUE_INDEX handle exists and skipped otherwise.
  const mp1::context ctx{"test::value_index"};
  const auto combiner_of = [](MPI_Datatype type) {
    MPI_Count num_integers = 0;
    MPI_Count num_addresses = 0;
    MPI_Count num_large_counts = 0;
    MPI_Count num_datatypes = 0;
    int combiner = -1;
    require_mpi_ok(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses,
                                           &num_large_counts, &num_datatypes, &combiner),
                   "MPI_Type_get_envelope_c(value-index pair)");
    return combiner;
  };

  const MPI_Datatype value_types[] = {MPI_DOUBLE, MPI_FLOAT,   MPI_INT,    MPI_LONG,
                                      MPI_SHORT,  MPI_INT64_T, MPI_INT32_T};
  const MPI_Datatype index_types[] = {MPI_INT64_T, MPI_UINT64_T, MPI_INT,
                                      MPI_LONG,    MPI_INT32_T,  MPI_UINT32_T};
  bool saw_value_index = false;
  for (MPI_Datatype value_type : value_types) {
    for (MPI_Datatype index_type : index_types) {
      MPI_Datatype pair = MPI_DATATYPE_NULL;
      if (MPI_Type_get_value_index(value_type, index_type, &pair) != MPI_SUCCESS or
          pair == MPI_DATATYPE_NULL) {
        continue;
      }
      if (combiner_of(pair) != MPI_COMBINER_VALUE_INDEX)
        continue;
      saw_value_index = true;
      BOOST_TEST_CONTEXT("value-index pair") {
        BOOST_TEST(not mp1::detail::type_map_detail::is_freeable(pair, ctx));
        BOOST_TEST(mp1::detail::rma_datatype_is_borrowed(pair));

        // A struct that contains the pair: the layout decoder must reject the unmodelled
        // constructor and leave the borrowed pair intact (freeing it would be erroneous).
        MPI_Datatype wrapper = MPI_DATATYPE_NULL;
        const MPI_Count blocklengths[] = {1};
        const MPI_Count displacements[] = {0};
        const MPI_Datatype children[] = {pair};
        require_mpi_ok(
            MPI_Type_create_struct_c(1, blocklengths, displacements, children, &wrapper),
            "MPI_Type_create_struct_c(value-index child)");
        require_mpi_ok(MPI_Type_commit(&wrapper), "MPI_Type_commit(value-index child)");
        std::vector<mp1::detail::type_map_interval> intervals;
        BOOST_TEST(not mp1::detail::decode_type_map(wrapper, intervals, ctx));
        BOOST_TEST(combiner_of(pair) == MPI_COMBINER_VALUE_INDEX);
        require_mpi_ok(MPI_Type_free(&wrapper), "MPI_Type_free(value-index child)");
      }
    }
  }
  if (not saw_value_index) {
    BOOST_TEST_MESSAGE(
        "value-index: provider returned no MPI_COMBINER_VALUE_INDEX handle; the borrowed "
        "classification is covered by the combiner arm");
  }
}

BOOST_AUTO_TEST_SUITE_END()
