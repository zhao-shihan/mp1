// Public-API compile check: the vocabulary types and their object properties.  Compile-only;
// nothing here is linked or executed.
//
// Pins:
//   * `region<T>`: non-owning two-word class, not an aggregate, not default-constructible,
//     copy/move constructible, copy/move assignment deleted, the deleted rvalue-layout
//     constructor, the accessor types, CTAD from the pointer, and the explicit-T null
//     spelling (`region<int>{nullptr, l}`);
//   * `blocks`: aggregate with exactly the two `std::span<const count_t>` members, empty by
//     value-initialization, designated initialization with lvalue and temporary arrays;
//   * `capacity`: the explicit upper-bound value type (no implicit count_t conversion);
//   * `empty_data`: the explicit zero-count message span (const, empty, no MPI call);
//   * `in_place_t`/`in_place`: the stateless sentinel and its position-concept rows;
//   * the async/persistent plan hook `detail::capture_region`: pointer value + owning
//     duplicate of the layout, and its element types for const and non-const regions;
//   * the removed transitional spellings are covered by the compile-fail probes.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  // ---- region<T>: type properties ---------------------------------------------------------
  static_assert(!std::is_aggregate_v<mp1::region<int>>,
                "region is a class, not an aggregate: the deleted rvalue-layout constructor "
                "is the point");
  static_assert(!std::is_default_constructible_v<mp1::region<int>>);
  static_assert(std::is_copy_constructible_v<mp1::region<int>>);
  static_assert(std::is_move_constructible_v<mp1::region<int>>);
  static_assert(std::is_nothrow_copy_constructible_v<mp1::region<int>>);
  static_assert(std::is_nothrow_move_constructible_v<mp1::region<int>>);
  static_assert(!std::is_copy_assignable_v<mp1::region<int>>);
  static_assert(!std::is_move_assignable_v<mp1::region<int>>);
  static_assert(!std::is_copy_assignable_v<mp1::region<const int>>);
  static_assert(!std::is_move_assignable_v<mp1::region<const int>>);

  static_assert(std::is_constructible_v<mp1::region<int>, int*, const mp1::layout<int>&>);
  static_assert(!std::is_constructible_v<mp1::region<int>, int*, const mp1::layout<int>&&>);
  static_assert(std::is_constructible_v<mp1::region<const int>, int*, const mp1::layout<int>&>);
  static_assert(
      !std::is_constructible_v<mp1::region<const int>, int*, const mp1::layout<int>&&>);
  // Explicit T makes the null-pointer spelling legal; CTAD cannot deduce T from nullptr (the
  // negative compile-fail probe region_ctad_nullptr.cc pins that).
  static_assert(
      std::is_constructible_v<mp1::region<int>, std::nullptr_t, const mp1::layout<int>&>);
  static_assert(
      !std::is_constructible_v<mp1::region<int>, std::nullptr_t, const mp1::layout<int>&&>);

  // Accessor signatures: T* / const layout<remove_const_t<T>>&, noexcept.
  using region_data_pointer = int* (mp1::region<int>::*)() const noexcept;
  using region_map_pointer = const mp1::layout<int>& (mp1::region<int>::*)() const noexcept;
  static_assert(std::is_same_v<decltype(&mp1::region<int>::data), region_data_pointer>);
  static_assert(std::is_same_v<decltype(&mp1::region<int>::map), region_map_pointer>);
  static_assert(std::is_same_v<decltype(std::declval<const mp1::region<const int>&>().data()),
                               const int*>);

  // CTAD deduces T from the pointer argument (and keeps constness).
  template<class T>
  inline constexpr bool region_ctad_type =
      std::is_same_v<decltype(mp1::region{
                         std::declval<T*>(),
                         std::declval<const mp1::layout<std::remove_const_t<T>>&>()}),
                     mp1::region<T>>;
  static_assert(region_ctad_type<int>);
  static_assert(region_ctad_type<const int>);
  static_assert(region_ctad_type<std::array<int, 3>>);

  // ---- blocks: the aggregate (counts, displacements) pair ---------------------------------
  static_assert(std::is_aggregate_v<mp1::blocks>);
  static_assert(std::is_default_constructible_v<mp1::blocks>);
  static_assert(std::is_same_v<decltype(mp1::blocks::counts), std::span<const mp1::count_t>>);
  static_assert(
      std::is_same_v<decltype(mp1::blocks::displacements), std::span<const mp1::count_t>>);
  static_assert(std::is_constructible_v<mp1::blocks, std::span<const mp1::count_t>,
                                        std::span<const mp1::count_t>>);
  static_assert(std::is_copy_constructible_v<mp1::blocks>);
  static_assert(std::is_copy_assignable_v<mp1::blocks>);

  // ---- capacity: explicit upper bound ----------------------------------------------------
  static_assert(!std::is_convertible_v<mp1::count_t, mp1::capacity>);
  static_assert(std::is_constructible_v<mp1::capacity, mp1::count_t>);
  static_assert(std::is_trivially_copyable_v<mp1::capacity>);
  static_assert(noexcept(mp1::capacity{mp1::count_t{0}}));
  static_assert(noexcept(std::declval<const mp1::capacity&>().value()));

  // ---- empty_data: the explicit zero-count message ---------------------------------------
  static_assert(std::is_same_v<decltype(mp1::empty_data), const std::span<std::byte>>);
  static_assert(mp1::empty_data.empty());
  static_assert(mp1::empty_data.size() == 0);
  static_assert(mp1::empty_data.data() == nullptr);

  // ---- in_place_t: the stateless sentinel -------------------------------------------------
  static_assert(std::is_empty_v<mp1::in_place_t>);
  static_assert(!std::is_aggregate_v<mp1::in_place_t>);
  static_assert(std::is_default_constructible_v<mp1::in_place_t>);
  static_assert(!std::is_convertible_v<mp1::in_place_t, int>);
  static_assert(std::is_same_v<decltype(mp1::in_place), const mp1::in_place_t>);

  // ---- the plan hook: pointer + owning duplicate of the layout ----------------------------
  template<class T>
  using capture_of = decltype(mp1::detail::capture_region(std::declval<mp1::region<T>>()));
  static_assert(std::is_same_v<capture_of<int>, mp1::detail::region_capture<int>>);
  static_assert(std::is_same_v<capture_of<const int>, mp1::detail::region_capture<const int>>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<int>>().data), int*>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<const int>>().data),
                     const int*>);
  static_assert(std::is_same_v<decltype(std::declval<mp1::detail::region_capture<int>>().map),
                               mp1::layout<int>>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<const int>>().map),
                     mp1::layout<int>>);

  // ---- removed transitional spellings ----------------------------------------------------
  // `counts`, `displacements`, `layouts<T>`, `layout_view<T>` and `make_layout_span` are
  // not part of the API; the dense descriptor is `blocks`, the typed one is
  // `std::span<const layout<T>>` + `std::span<const count_t>`.  Compile-fail probes pin the
  // removals, including the `block_count` removal.

  // ---- positive spellings (compiled, never run) ------------------------------------------
  [[maybe_unused]] void region_spellings(int* data, const int* const_data,
                                         const mp1::layout<int>& map) {
    mp1::region r{data, map};  // CTAD -> region<int>
    mp1::region<int> explicit_t{data, map};
    mp1::region<const int> const_region{const_data, map};
    mp1::region<int> null_data{nullptr, map};
    static_assert(std::is_same_v<decltype(r), mp1::region<int>>);
    static_assert(std::is_same_v<decltype(explicit_t), mp1::region<int>>);
    static_assert(std::is_same_v<decltype(const_region), mp1::region<const int>>);
    const mp1::region<int> copy{r};
    const mp1::region<int> moved{std::move(explicit_t)};
    (void)null_data;
    (void)copy;
    (void)moved;
  }

  mp1::count_t blocks_total(mp1::blocks value) {
    mp1::count_t total = 0;
    for (const mp1::count_t count : value.counts)
      total += count;
    return total;
  }

  [[maybe_unused]] void blocks_spellings(std::span<const mp1::count_t> counts,
                                         std::span<const mp1::count_t> displacements) {
    mp1::blocks designated{.counts = counts, .displacements = displacements};
    mp1::blocks empty{};
    (void)designated;
    (void)empty;
    // A temporary vector is bound by the spans for the whole full-expression only.  Storing
    // the blocks object beyond it would dangle, so this is the maximal
    // temporary spelling that is pinned.
    (void)blocks_total(mp1::blocks{
        .counts = std::vector<mp1::count_t>{1, 2},
        .displacements = std::vector<mp1::count_t>{0, 1},
    });
  }

  [[maybe_unused]] void capacity_and_in_place_spellings() {
    const mp1::capacity bound{4};
    (void)bound.value();
    mp1::in_place_t sentinel{};
    (void)mp1::in_place;
    (void)sentinel;
  }

  [[maybe_unused]] void capture_spelling(int* data, const mp1::layout<int>& map) {
    mp1::detail::region_capture<int> capture{
        mp1::detail::capture_region(mp1::region{data, map})};
    (void)capture;
  }

}  // namespace
