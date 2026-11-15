// Reflection test: MP1_REFLECTION, MP1_REFLECTION_TEMPLATE,
// MP1_DATATYPE_TRAITS, the 119-member expansion limit, nested/array members and the
// reflected datatype's metadata versus a hand-built MPI type.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// the Boost.Test module macro. MPI is initialized by the global fixture below.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <type_traits>
#include <utility>


struct single_int {
  int value{0};
};

MP1_REFLECTION(single_int, value)

struct mixed {
  int i{0};
  double d{0.};
  std::array<int, 3> triple{};
  double c_array[2]{};
};

MP1_REFLECTION(mixed, i, d, triple, c_array)

template<class T>
struct wrapper {
  T value{};
};

MP1_REFLECTION_TEMPLATE((class T), (wrapper<T>), value)

template<class T, class U>
struct two_member {
  T first{};
  U second{};
};

MP1_REFLECTION_TEMPLATE((class T, class U), (two_member<T, U>), first, second)

struct nested {
  single_int first{};
  mixed second{};
  int tail{0};
};

MP1_REFLECTION(nested, first, second, tail)

struct custom_basic {
  int value{0};
};

MP1_DATATYPE_TRAITS(custom_basic, MPI_INT)

struct reflected_119 {
  int m0{0}, m1{0}, m2{0}, m3{0}, m4{0}, m5{0}, m6{0}, m7{0};
  int m8{0}, m9{0}, m10{0}, m11{0}, m12{0}, m13{0}, m14{0}, m15{0};
  int m16{0}, m17{0}, m18{0}, m19{0}, m20{0}, m21{0}, m22{0}, m23{0};
  int m24{0}, m25{0}, m26{0}, m27{0}, m28{0}, m29{0}, m30{0}, m31{0};
  int m32{0}, m33{0}, m34{0}, m35{0}, m36{0}, m37{0}, m38{0}, m39{0};
  int m40{0}, m41{0}, m42{0}, m43{0}, m44{0}, m45{0}, m46{0}, m47{0};
  int m48{0}, m49{0}, m50{0}, m51{0}, m52{0}, m53{0}, m54{0}, m55{0};
  int m56{0}, m57{0}, m58{0}, m59{0}, m60{0}, m61{0}, m62{0}, m63{0};
  int m64{0}, m65{0}, m66{0}, m67{0}, m68{0}, m69{0}, m70{0}, m71{0};
  int m72{0}, m73{0}, m74{0}, m75{0}, m76{0}, m77{0}, m78{0}, m79{0};
  int m80{0}, m81{0}, m82{0}, m83{0}, m84{0}, m85{0}, m86{0}, m87{0};
  int m88{0}, m89{0}, m90{0}, m91{0}, m92{0}, m93{0}, m94{0}, m95{0};
  int m96{0}, m97{0}, m98{0}, m99{0}, m100{0}, m101{0}, m102{0}, m103{0};
  int m104{0}, m105{0}, m106{0}, m107{0}, m108{0}, m109{0}, m110{0}, m111{0};
  int m112{0}, m113{0}, m114{0}, m115{0}, m116{0}, m117{0}, m118{0};
};

MP1_REFLECTION(reflected_119, m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27, m28, m29, m30,
               m31, m32, m33, m34, m35, m36, m37, m38, m39, m40, m41, m42, m43, m44, m45, m46,
               m47, m48, m49, m50, m51, m52, m53, m54, m55, m56, m57, m58, m59, m60, m61, m62,
               m63, m64, m65, m66, m67, m68, m69, m70, m71, m72, m73, m74, m75, m76, m77, m78,
               m79, m80, m81, m82, m83, m84, m85, m86, m87, m88, m89, m90, m91, m92, m93, m94,
               m95, m96, m97, m98, m99, m100, m101, m102, m103, m104, m105, m106, m107, m108,
               m109, m110, m111, m112, m113, m114, m115, m116, m117, m118)

// ---- basic-component trait ----------------------------------------------------------------
//
// The trait classifies a type at compile time for the accumulating RMA families (MPI
// 4.1 §12.3.4).  pair/tuple/array and MP1_REFLECTION expose
// struct_builder<T>::component_types and set the library-internal trust marker; a manual
// builder or MP1_DATATYPE_BASIC annotation is untrusted and validated once against its
// committed datatype before the first accumulating call.

struct empty_reflected {};

MP1_REFLECTION_EMPTY(empty_reflected)

struct custom_packed_basic {
  int value{0};
};

MP1_DATATYPE_TRAITS_PACKED(custom_packed_basic, MPI_INT)
MP1_DATATYPE_BASIC(custom_packed_basic, int)

struct holds_custom_packed {
  custom_packed_basic first{};
  custom_packed_basic second{};
};

MP1_REFLECTION(holds_custom_packed, first, second)

struct homogeneous_nested {
  single_int first{};
  single_int second{};
};

MP1_REFLECTION(homogeneous_nested, first, second)

// A manual struct_builder exposing component_types but not the library-internal
// component_types_trusted marker: always untrusted and validated once at run time.
struct manual_pair {
  int first{0};
  int second{0};
};

namespace mp1 {

  template<>
  class struct_builder<manual_pair> : public base_struct_builder<manual_pair> {
  public:
    using component_types = detail::type_list<int, int>;

    struct_builder() {
      manual_pair object{};
      this->register_object(object);
      this->register_element(object.first);
      this->register_element(object.second);
      this->define_struct();
    }
  };

}  // namespace mp1

static_assert(std::is_same_v<mp1::struct_builder<std::pair<int, int>>::component_types,
                             mp1::detail::type_list<int, int>>);
static_assert(std::is_same_v<mp1::struct_builder<std::tuple<int, double>>::component_types,
                             mp1::detail::type_list<int, double>>);
static_assert(std::is_same_v<mp1::struct_builder<std::array<int, 2>>::component_types,
                             mp1::detail::type_list<int>>);
static_assert(std::is_same_v<mp1::struct_builder<std::array<int, 0>>::component_types,
                             mp1::detail::type_list<>>);
static_assert(std::is_same_v<mp1::struct_builder<single_int>::component_types,
                             mp1::detail::type_list<int>>);
static_assert(std::is_same_v<mp1::struct_builder<empty_reflected>::component_types,
                             mp1::detail::type_list<>>);
static_assert(std::is_same_v<mp1::struct_builder<nested>::component_types,
                             mp1::detail::type_list<single_int, mixed, int>>);

// Trust markers: library-generated pair/tuple/array and MP1_REFLECTION are trusted; a manual
// builder is not.
static_assert(mp1::detail::component_types_trusted_v<std::pair<int, int>>);
static_assert(mp1::detail::component_types_trusted_v<std::tuple<int>>);
static_assert(mp1::detail::component_types_trusted_v<std::array<int, 3>>);
static_assert(mp1::detail::component_types_trusted_v<single_int>);
static_assert(mp1::detail::component_types_trusted_v<empty_reflected>);
static_assert(not mp1::detail::component_types_trusted_v<manual_pair>);

// Classification: predefined, homogeneous derived, mixed, empty and opaque.
static_assert(mp1::is_homogeneous_datatype_v<int>);
static_assert(std::is_same_v<mp1::basic_component_t<int>, int>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<int, int>>);
static_assert(std::is_same_v<mp1::basic_component_t<std::pair<int, int>>, int>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<std::pair<int, int>, int>>);
static_assert(mp1::is_homogeneous_datatype_v<std::tuple<int, int>>);
static_assert(mp1::is_homogeneous_datatype_v<std::array<int, 3>>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<std::array<int, 0>, int>>);
static_assert(mp1::is_homogeneous_datatype_v<single_int>);
static_assert(mp1::is_homogeneous_datatype_v<homogeneous_nested>);
static_assert(std::is_same_v<mp1::basic_component_t<homogeneous_nested>, int>);
static_assert(mp1::is_homogeneous_datatype_v<reflected_119>);
static_assert(std::is_same_v<mp1::basic_component_t<reflected_119>, int>);
static_assert(not mp1::is_homogeneous_datatype_v<std::array<int, 0>>);
static_assert(not mp1::is_homogeneous_datatype_v<empty_reflected>);
static_assert(not mp1::is_homogeneous_datatype_v<mixed>);
static_assert(not mp1::is_homogeneous_datatype_v<std::pair<int, double>>);
static_assert(mp1::detail::component_info<mixed>::status ==
              mp1::detail::component_status::heterogeneous);
static_assert(mp1::detail::component_info<std::array<int, 0>>::status ==
              mp1::detail::component_status::empty);
static_assert(mp1::detail::component_info<empty_reflected>::status ==
              mp1::detail::component_status::empty);

// Runtime validation is required only for untrusted custom/manual annotations, and it
// propagates through a containing homogeneous aggregate.
static_assert(not mp1::detail::component_info<int>::runtime_validate);
static_assert(not mp1::detail::component_info<std::pair<int, int>>::runtime_validate);
static_assert(not mp1::detail::component_info<single_int>::runtime_validate);
static_assert(not mp1::detail::component_info<mixed>::runtime_validate);
static_assert(mp1::detail::component_info<custom_packed_basic>::runtime_validate);
static_assert(mp1::detail::component_info<manual_pair>::runtime_validate);
static_assert(mp1::detail::component_info<
              std::pair<custom_packed_basic, custom_packed_basic>>::runtime_validate);
static_assert(
    mp1::detail::component_info<std::array<custom_packed_basic, 2>>::runtime_validate);
static_assert(mp1::detail::component_info<holds_custom_packed>::runtime_validate);
static_assert(std::is_same_v<mp1::basic_component_t<manual_pair>, int>);
static_assert(std::is_same_v<mp1::basic_component_t<custom_packed_basic>, int>);
static_assert(
    std::is_same_v<mp1::detail::type_list<int, int>,
                   mp1::detail::remove_void_t<mp1::detail::type_list<int, int, void>>>);

namespace {

  // The cached metadata of a layout<reflected T> is the reflected datatype's metadata.
  struct raw_metadata {
    MPI_Count size;
    MPI_Count lower_bound;
    MPI_Count extent;
  };

  raw_metadata read_raw(MPI_Datatype type) {
    raw_metadata result{0, 0, 0};
    BOOST_REQUIRE(MPI_Type_size_c(type, &result.size) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_get_extent_c(type, &result.lower_bound, &result.extent) ==
                  MPI_SUCCESS);
    return result;
  }

  // A hand-built type equivalent to MP1_REFLECTION(mixed, ...): one block per member, in
  // declaration order, then resized to sizeof(T).
  MPI_Datatype make_raw_mixed(mixed& object) {
    MPI_Aint base{0};
    MPI_Aint addresses[4]{0, 0, 0, 0};
    BOOST_REQUIRE(MPI_Get_address(&object, &base) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Get_address(&object.i, &addresses[0]) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Get_address(&object.d, &addresses[1]) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Get_address(object.triple.data(), &addresses[2]) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Get_address(&object.c_array[0], &addresses[3]) == MPI_SUCCESS);
    MPI_Count blocks[4]{1, 1, 3, 2};
    MPI_Count displacements[4]{0, 0, 0, 0};
    for (int i = 0; i < 4; ++i)
      displacements[i] = static_cast<MPI_Count>(addresses[i] - base);
    MPI_Datatype types[4]{MPI_INT, MPI_DOUBLE, MPI_INT, MPI_DOUBLE};
    MPI_Datatype temporary{MPI_DATATYPE_NULL};
    BOOST_REQUIRE(MPI_Type_create_struct_c(4, blocks, displacements, types, &temporary) ==
                  MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_commit(&temporary) == MPI_SUCCESS);
    MPI_Datatype result{MPI_DATATYPE_NULL};
    BOOST_REQUIRE(MPI_Type_create_resized(temporary, 0, static_cast<MPI_Aint>(sizeof(mixed)),
                                          &result) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_commit(&result) == MPI_SUCCESS);
    (void)MPI_Type_free(&temporary);
    return result;
  }

}  // namespace


struct mp1_reflection_fixture {
  mp1_reflection_fixture() {
    mp1::init();
  }

  ~mp1_reflection_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Another owner may finalize; a fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_reflection_fixture);

BOOST_AUTO_TEST_SUITE(reflection_test)

BOOST_AUTO_TEST_CASE(category_and_public_macros) {
  static_assert(std::is_same_v<typename mp1::datatype_traits<single_int>::data_type_category,
                               mp1::detail::basic_or_fixed_size_type>);
  static_assert(std::is_same_v<typename mp1::datatype_traits<wrapper<int>>::data_type_category,
                               mp1::detail::basic_or_fixed_size_type>);
  static_assert(
      std::is_same_v<typename mp1::datatype_traits<two_member<int, double>>::data_type_category,
                     mp1::detail::basic_or_fixed_size_type>);
  static_assert(std::is_same_v<typename mp1::datatype_traits<custom_basic>::data_type_category,
                               mp1::detail::basic_or_fixed_size_type>);
  static_assert(sizeof(custom_basic) == sizeof(int),
                "MP1_DATATYPE_TRAITS requires extent(sizeof(T))");

  BOOST_TEST(mp1::native_datatype<custom_basic>() == MPI_INT);
  const mp1::layout<custom_basic> custom_layout = mp1::layout<custom_basic>::contiguous(2);
  BOOST_TEST(custom_layout.size() == 2);
  BOOST_TEST(custom_layout.byte_size() == static_cast<mp1::count_t>(2 * sizeof(int)));
  BOOST_TEST(custom_layout.byte_extent() == static_cast<mp1::count_t>(2 * sizeof(int)));

  // std::pair is built in by the datatype layer and needs no user macro.
  const mp1::layout<std::pair<int, double>> pair_layout =
      mp1::layout<std::pair<int, double>>::contiguous(1);
  BOOST_TEST(pair_layout.byte_size() ==
             static_cast<mp1::count_t>(sizeof(int) + sizeof(double)));
  BOOST_TEST(pair_layout.byte_extent() ==
             static_cast<mp1::count_t>(sizeof(std::pair<int, double>)));
}

BOOST_AUTO_TEST_CASE(reflected_members_metadata_and_roundtrip) {
  const mp1::layout<mixed> layout = mp1::layout<mixed>::contiguous(1);
  BOOST_TEST(layout.size() == 1);
  // Every member is registered once: 1 int + 1 double + 3 ints + 2 doubles.
  BOOST_TEST(layout.byte_size() ==
             static_cast<mp1::count_t>(4 * sizeof(int) + 3 * sizeof(double)));
  BOOST_TEST(layout.byte_extent() == static_cast<mp1::count_t>(sizeof(mixed)));
  BOOST_TEST(layout.true_byte_lower_bound() == 0);
  BOOST_TEST(layout.true_byte_extent() == static_cast<mp1::count_t>(sizeof(mixed)));
  BOOST_TEST(not layout.dense());

  mixed object{};
  MPI_Datatype raw = make_raw_mixed(object);
  const raw_metadata raw_info = read_raw(raw);
  BOOST_TEST(layout.byte_size() == raw_info.size);
  BOOST_TEST(layout.byte_extent() == raw_info.extent);
  BOOST_TEST(layout.byte_lower_bound() == raw_info.lower_bound);
  (void)MPI_Type_free(&raw);

  mixed sent{};
  sent.i = 17;
  sent.d = 2.5;
  sent.triple = {1, 2, 3};
  sent.c_array[0] = 4.5;
  sent.c_array[1] = 6.25;
  mixed received{};
  BOOST_REQUIRE(MPI_Sendrecv(&sent, 1, mp1::native_datatype<mixed>(), 0, 701, &received, 1,
                             mp1::native_datatype<mixed>(), 0, 701, MPI_COMM_SELF,
                             MPI_STATUS_IGNORE) == MPI_SUCCESS);
  BOOST_TEST(received.i == 17);
  BOOST_TEST(received.d == 2.5);
  BOOST_TEST(received.triple == (std::array<int, 3>{1, 2, 3}));
  BOOST_TEST(received.c_array[0] == 4.5);
  BOOST_TEST(received.c_array[1] == 6.25);
}

BOOST_AUTO_TEST_CASE(template_and_nested_reflection) {
  const mp1::layout<wrapper<double>> wrapper_layout =
      mp1::layout<wrapper<double>>::contiguous(1);
  BOOST_TEST(wrapper_layout.byte_size() == static_cast<mp1::count_t>(sizeof(double)));
  BOOST_TEST(wrapper_layout.byte_extent() ==
             static_cast<mp1::count_t>(sizeof(wrapper<double>)));

  const mp1::layout<two_member<int, double>> pair_layout =
      mp1::layout<two_member<int, double>>::contiguous(1);
  BOOST_TEST(pair_layout.byte_size() ==
             static_cast<mp1::count_t>(sizeof(int) + sizeof(double)));
  BOOST_TEST(pair_layout.byte_extent() ==
             static_cast<mp1::count_t>(sizeof(two_member<int, double>)));

  // Nested reflection adds the members' payloads (4 + 40 + 4) and keeps the outer struct's
  // padded extent, so the outer type is necessarily holey.
  const mp1::layout<nested> nested_layout = mp1::layout<nested>::contiguous(1);
  BOOST_TEST(nested_layout.byte_size() == static_cast<mp1::count_t>(4 + 40 + 4));
  BOOST_TEST(nested_layout.byte_extent() == static_cast<mp1::count_t>(sizeof(nested)));
  BOOST_TEST(not nested_layout.dense());

  nested sent{};
  sent.first.value = -3;
  sent.second.i = 9;
  sent.second.d = 1.25;
  sent.second.triple = {7, 8, 9};
  sent.second.c_array[0] = 0.5;
  sent.second.c_array[1] = 0.75;
  sent.tail = 123;
  nested received{};
  BOOST_REQUIRE(MPI_Sendrecv(&sent, 1, mp1::native_datatype<nested>(), 0, 702, &received, 1,
                             mp1::native_datatype<nested>(), 0, 702, MPI_COMM_SELF,
                             MPI_STATUS_IGNORE) == MPI_SUCCESS);
  BOOST_TEST(received.first.value == -3);
  BOOST_TEST(received.second.i == 9);
  BOOST_TEST(received.second.d == 1.25);
  BOOST_TEST(received.second.triple == (std::array<int, 3>{7, 8, 9}));
  BOOST_TEST(received.second.c_array[1] == 0.75);
  BOOST_TEST(received.tail == 123);
}

BOOST_AUTO_TEST_CASE(reflected_119_member_limit) {
  // The preprocessor expansion supports exactly 119 named members. All 119 must reach the
  // generated MPI type: 119 ints of payload, resized to sizeof(reflected_119).
  const mp1::layout<reflected_119> layout = mp1::layout<reflected_119>::contiguous(1);
  BOOST_TEST(layout.size() == 1);
  BOOST_TEST(layout.byte_size() == static_cast<mp1::count_t>(119 * sizeof(int)));
  BOOST_TEST(layout.byte_extent() == static_cast<mp1::count_t>(sizeof(reflected_119)));
  BOOST_TEST(layout.true_byte_extent() == static_cast<mp1::count_t>(sizeof(reflected_119)));
  BOOST_TEST(layout.dense());

  reflected_119 sent{};
  sent.m0 = 11;
  sent.m58 = -22;
  sent.m118 = 33;
  reflected_119 received{};
  BOOST_REQUIRE(MPI_Sendrecv(&sent, 1, mp1::native_datatype<reflected_119>(), 0, 703, &received,
                             1, mp1::native_datatype<reflected_119>(), 0, 703, MPI_COMM_SELF,
                             MPI_STATUS_IGNORE) == MPI_SUCCESS);
  BOOST_TEST(received.m0 == 11);
  BOOST_TEST(received.m58 == -22);
  BOOST_TEST(received.m118 == 33);

  // 120 members must be a compile-time error; that negative case is a compile-fail probe, not
  // a runtime assertion. The positive 119-member expansion above is the limit test here.
}

BOOST_AUTO_TEST_SUITE_END()
