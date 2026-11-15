// Public-API compile check: the layout/datatype target inventory. Pins, at compile time:
//
//   * the 18 layout factories (14 base/block shapes + 2 from_raw + 2 resize); the former
//     `layout<T>::empty()` spelling is removed -- `contiguous(0)` is the zero layout.
//   * the 17 parameter declarations (5 builders x {default ctor, explicit std::span ctor,
//     add()}, plus the subarray order setter/getter) and their count_t/MPI_Aint element types,
//   * the public layout_facts aggregate (three structural predicates, all false by default),
//   * the six public datatype/reflection macros, used here so each expands,
//   * span_from's exact signature.
//
// The positional vector/strided_vector overloads and the builder initializer-list/range
// constructors are transitional spellings; they are deliberately not pinned here.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#if !defined(MP1_DATATYPE_TRAITS) || !defined(MP1_DATATYPE_TRAITS_PACKED) || \
    !defined(MP1_DATATYPE_BASIC) || !defined(MP1_REFLECTION) ||              \
    !defined(MP1_REFLECTION_EMPTY) || !defined(MP1_REFLECTION_TEMPLATE)
#error "mp1: the six public datatype/reflection macros must be defined"
#endif

namespace {

  struct plain_empty {};

  struct plain_point {
    int x = 0;
    double y = 0;
  };

  template<class T>
  struct plain_box {
    T value{};
  };

  struct traits_mapped {
    int value = 0;
  };

  struct traits_packed {
    int value = 0;
  };

}  // namespace

MP1_DATATYPE_TRAITS(traits_mapped, MPI_BYTE)
MP1_DATATYPE_TRAITS_PACKED(traits_packed, MPI_BYTE)
MP1_DATATYPE_BASIC(traits_packed, int)
MP1_REFLECTION(plain_point, x, y)
MP1_REFLECTION_EMPTY(plain_empty)
MP1_REFLECTION_TEMPLATE((class T), (plain_box<T>), value)

namespace {

  // Reference the macro-provided trait members so the inventory is checked, not just
  // expanded (and so Clang's -Wunused-const-variable stays quiet under -Werror probes).
  static_assert(mp1::datatype_traits<traits_mapped>::is_predefined);
  static_assert(mp1::datatype_traits<traits_packed>::byte_contiguous);
  static_assert(mp1::basic_component_traits<traits_packed>::provided);
  static_assert(mp1::struct_builder<plain_point>::component_types_trusted);
  static_assert(mp1::struct_builder<plain_empty>::component_types_trusted);
  static_assert(mp1::struct_builder<plain_box<int>>::component_types_trusted);
  static_assert(mp1::is_predefined_datatype_v<traits_mapped>);
  static_assert(!mp1::is_predefined_datatype_v<traits_packed>);
  static_assert(mp1::is_predefined_datatype_v<int>);

  using layout_i = mp1::layout<int>;
  using count_t = mp1::count_t;

  // ---- 18 layout factories ---------------------------------------------------------------

  using contiguous_fn = layout_i (*)(count_t);
  using contiguous_block_fn = layout_i (*)(count_t, const layout_i&);
  using vector_options_fn = layout_i (*)(mp1::vector_options<int>);
  using strided_vector_options_fn = layout_i (*)(mp1::strided_vector_options<int>);
  using indexed_fn = layout_i (*)(const mp1::indexed_parameter&);
  using indexed_block_layout_fn = layout_i (*)(const mp1::indexed_parameter&, const layout_i&);
  using hindexed_fn = layout_i (*)(const mp1::hindexed_parameter&);
  using hindexed_block_layout_fn = layout_i (*)(const mp1::hindexed_parameter&,
                                                const layout_i&);
  using indexed_block_fn = layout_i (*)(count_t, const mp1::indexed_block_parameter&);
  using indexed_block_block_fn = layout_i (*)(count_t, const mp1::indexed_block_parameter&,
                                              const layout_i&);
  using hindexed_block_fn = layout_i (*)(count_t, const mp1::hindexed_block_parameter&);
  using hindexed_block_block_fn = layout_i (*)(count_t, const mp1::hindexed_block_parameter&,
                                               const layout_i&);
  using subarray_fn = layout_i (*)(const mp1::subarray_parameter&);
  using subarray_block_fn = layout_i (*)(const mp1::subarray_parameter&, const layout_i&);
  using from_raw_facts_fn = layout_i (*)(MPI_Datatype, count_t, mp1::layout_facts);
  using byte_resize_fn = void (layout_i::*)(count_t, count_t);
  using resized_fn = layout_i (layout_i::*)(count_t, count_t) const;

  static_assert(std::is_same_v<decltype(static_cast<contiguous_fn>(&layout_i::contiguous)),
                               contiguous_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<contiguous_block_fn>(&layout_i::contiguous)),
                     contiguous_block_fn>);
  static_assert(std::is_same_v<decltype(static_cast<vector_options_fn>(&layout_i::vector)),
                               vector_options_fn>);
  static_assert(std::is_same_v<
                decltype(static_cast<strided_vector_options_fn>(&layout_i::strided_vector)),
                strided_vector_options_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_fn>(&layout_i::indexed)), indexed_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_block_layout_fn>(&layout_i::indexed)),
                     indexed_block_layout_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_fn>(&layout_i::hindexed)), hindexed_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_block_layout_fn>(&layout_i::hindexed)),
                     hindexed_block_layout_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_block_fn>(&layout_i::indexed_block)),
                     indexed_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_block_block_fn>(&layout_i::indexed_block)),
                     indexed_block_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_block_fn>(&layout_i::hindexed_block)),
                     hindexed_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_block_block_fn>(&layout_i::hindexed_block)),
                     hindexed_block_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<subarray_fn>(&layout_i::subarray)), subarray_fn>);
  static_assert(std::is_same_v<decltype(static_cast<subarray_block_fn>(&layout_i::subarray)),
                               subarray_block_fn>);
  static_assert(std::is_same_v<decltype(static_cast<from_raw_facts_fn>(&layout_i::from_raw)),
                               from_raw_facts_fn>);
  static_assert(std::is_same_v<decltype(static_cast<byte_resize_fn>(&layout_i::byte_resize)),
                               byte_resize_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<resized_fn>(&layout_i::resized)), resized_fn>);

  // The option structs carry the block as a member, so one declaration per MPI family.
  static_assert(std::is_aggregate_v<mp1::vector_options<int>>);
  static_assert(std::is_aggregate_v<mp1::strided_vector_options<int>>);
  static_assert(std::is_same_v<decltype(mp1::vector_options<int>::count), count_t>);
  static_assert(std::is_same_v<decltype(mp1::vector_options<int>::blocklength), count_t>);
  static_assert(std::is_same_v<decltype(mp1::vector_options<int>::stride), count_t>);
  static_assert(
      std::is_same_v<decltype(mp1::vector_options<int>::block), const mp1::layout<int>*>);
  static_assert(
      std::is_same_v<decltype(mp1::strided_vector_options<int>::byte_stride), MPI_Aint>);

  // ---- 17 parameter declarations ---------------------------------------------------------

  static_assert(std::is_default_constructible_v<mp1::indexed_parameter>);
  static_assert(std::is_constructible_v<mp1::indexed_parameter,
                                        std::span<const std::pair<count_t, count_t>>>);
  static_assert(!std::is_convertible_v<std::span<const std::pair<count_t, count_t>>,
                                       mp1::indexed_parameter>);
  using indexed_add_fn = void (mp1::indexed_parameter::*)(count_t, count_t);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_add_fn>(&mp1::indexed_parameter::add)),
                     indexed_add_fn>);

  static_assert(std::is_default_constructible_v<mp1::hindexed_parameter>);
  static_assert(std::is_constructible_v<mp1::hindexed_parameter,
                                        std::span<const std::pair<count_t, MPI_Aint>>>);
  static_assert(!std::is_convertible_v<std::span<const std::pair<count_t, MPI_Aint>>,
                                       mp1::hindexed_parameter>);
  using hindexed_add_fn = void (mp1::hindexed_parameter::*)(count_t, MPI_Aint);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_add_fn>(&mp1::hindexed_parameter::add)),
                     hindexed_add_fn>);

  static_assert(std::is_default_constructible_v<mp1::indexed_block_parameter>);
  static_assert(
      std::is_constructible_v<mp1::indexed_block_parameter, std::span<const count_t>>);
  static_assert(!std::is_convertible_v<std::span<const count_t>, mp1::indexed_block_parameter>);
  using indexed_block_add_fn = void (mp1::indexed_block_parameter::*)(count_t);
  static_assert(std::is_same_v<
                decltype(static_cast<indexed_block_add_fn>(&mp1::indexed_block_parameter::add)),
                indexed_block_add_fn>);

  static_assert(std::is_default_constructible_v<mp1::hindexed_block_parameter>);
  static_assert(
      std::is_constructible_v<mp1::hindexed_block_parameter, std::span<const MPI_Aint>>);
  static_assert(
      !std::is_convertible_v<std::span<const MPI_Aint>, mp1::hindexed_block_parameter>);
  using hindexed_block_add_fn = void (mp1::hindexed_block_parameter::*)(MPI_Aint);
  static_assert(std::is_same_v<decltype(static_cast<hindexed_block_add_fn>(
                                   &mp1::hindexed_block_parameter::add)),
                               hindexed_block_add_fn>);

  static_assert(std::is_default_constructible_v<mp1::subarray_parameter>);
  static_assert(std::is_constructible_v<mp1::subarray_parameter,
                                        std::span<const std::array<count_t, 3>>>);
  static_assert(
      !std::is_convertible_v<std::span<const std::array<count_t, 3>>, mp1::subarray_parameter>);
  using subarray_add_fn = void (mp1::subarray_parameter::*)(count_t, count_t, count_t);
  static_assert(
      std::is_same_v<decltype(static_cast<subarray_add_fn>(&mp1::subarray_parameter::add)),
                     subarray_add_fn>);
  using order_set_fn = void (mp1::subarray_parameter::*)(mp1::array_order) noexcept;
  using order_get_fn = mp1::array_order (mp1::subarray_parameter::*)() const noexcept;
  static_assert(
      std::is_same_v<decltype(static_cast<order_set_fn>(&mp1::subarray_parameter::order)),
                     order_set_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<order_get_fn>(&mp1::subarray_parameter::order)),
                     order_get_fn>);

  // ---- public layout_facts -----------------------------------------------------------------

  static_assert(std::is_aggregate_v<mp1::layout_facts>);
  static_assert(std::is_same_v<decltype(mp1::layout_facts::packed), bool>);
  static_assert(std::is_same_v<decltype(mp1::layout_facts::base_repetition), bool>);
  static_assert(std::is_same_v<decltype(mp1::layout_facts::has_overlapping_entries), bool>);
  static_assert(!mp1::layout_facts{}.packed);
  static_assert(!mp1::layout_facts{}.base_repetition);
  static_assert(!mp1::layout_facts{}.has_overlapping_entries);

  // The designated third argument must select the public facts overload. The private
  // adopting overload also takes a `detail::layout_facts` aggregate, so this spelling must
  // remain unambiguous.
  [[maybe_unused]] void from_raw_designated_spelling(MPI_Datatype type) {
    (void)layout_i::from_raw(type, 1, {.packed = true});
    (void)layout_i::from_raw(type, 1, {.has_overlapping_entries = true});
  }

  // ---- span_from --------------------------------------------------------------------------

  static_assert(
      std::is_same_v<decltype(&mp1::span_from<int>), std::span<int> (*)(int*, count_t)>);
  static_assert(std::is_same_v<decltype(&mp1::span_from<const int>),
                               std::span<const int> (*)(const int*, count_t)>);

}  // namespace

int main() {
  return 0;
}
