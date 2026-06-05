#define BOOST_TEST_MODULE reflection
#include "boost/test/included/unit_test.hpp"
#include "test_helper.hpp"

#include <type_traits>

BOOST_AUTO_TEST_CASE(reflection_compile_time_checks) {
  // Verify non-template MPLR_REFLECTION
  static_assert(
      std::is_same_v<typename mplr::detail::datatype_traits<tuple>::data_type_category,
                     mplr::detail::basic_or_fixed_size_type>,
      "tuple should be recognized as a valid struct type");

  // Verify single template param MPLR_REFLECTION_TEMPLATE
  static_assert(
      std::is_same_v<typename mplr::detail::datatype_traits<wrapper<int>>::data_type_category,
                     mplr::detail::basic_or_fixed_size_type>,
      "wrapper<int> should be recognized as a valid struct type");

  static_assert(std::is_same_v<
                    typename mplr::detail::datatype_traits<wrapper<double>>::data_type_category,
                    mplr::detail::basic_or_fixed_size_type>,
                "wrapper<double> should be recognized as a valid struct type");

  // Verify multiple template params MPLR_REFLECTION_TEMPLATE
  static_assert(
      std::is_same_v<
          typename mplr::detail::datatype_traits<pair_like<int, double>>::data_type_category,
          mplr::detail::basic_or_fixed_size_type>,
      "pair_like<int, double> should be recognized as a valid struct type");

  static_assert(
      std::is_same_v<
          typename mplr::detail::datatype_traits<pair_like<float, char>>::data_type_category,
          mplr::detail::basic_or_fixed_size_type>,
      "pair_like<float, char> should be recognized as a valid struct type");

  BOOST_TEST(true);
}
