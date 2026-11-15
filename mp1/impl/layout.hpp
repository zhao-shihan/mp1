#if !(defined MP1_IMPL_LAYOUT_HPP)

#define MP1_IMPL_LAYOUT_HPP

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/fwd.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1 {

  // MPI_Type_create_subarray order.
  enum class array_order : int { row_major = MPI_ORDER_C, column_major = MPI_ORDER_FORTRAN };

  namespace detail {

    [[nodiscard]] inline MPI_Count checked_mpi_count_from_aint(MPI_Aint value,
                                                               const context& ctx) {
      if (value < 0) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "byte stride {} is negative",
                                               static_cast<long long>(value));
      }
      if (not std::in_range<MPI_Count>(value)) {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "byte stride does not fit MPI_Count");
      }
      return static_cast<MPI_Count>(value);
    }

    [[nodiscard]] inline MPI_Aint checked_aint_value(count_t value, const context& ctx) {
      if (not std::in_range<MPI_Aint>(value)) {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "value does not fit MPI_Aint");
      }
      return static_cast<MPI_Aint>(value);
    }

    [[nodiscard]] inline overlap_state combine_overlap(overlap_state left,
                                                       overlap_state right) noexcept {
      if (left == overlap_state::overlap or right == overlap_state::overlap)
        return overlap_state::overlap;
      if (left == overlap_state::unknown or right == overlap_state::unknown)
        return overlap_state::unknown;
      return overlap_state::disjoint;
    }

    // Recorded layout facts. The byte_* fields are descriptive geometry queried from MPI; the
    // structural fields are proven by the factory from its construction arguments. Only the
    // structural fields select a lowering.
    struct layout_facts {
      count_t byte_size{0};
      count_t byte_lower_bound{0};
      count_t byte_extent{0};
      count_t true_lb{0};
      count_t true_extent{0};
      bool packed{true};
      bool base_repetition{true};
      overlap_state overlap{overlap_state::disjoint};
      bool valid{false};
    };

    // Structural facts of one "element" of a repetition: a base datatype or a child layout.
    struct map_element_facts {
      count_t size{0};
      count_t byte_size{0};
      count_t true_extent{0};
      count_t extent{0};
      count_t true_lb{0};
      bool packed{true};
      bool base_repetition{true};
      overlap_state overlap{overlap_state::disjoint};
    };

    struct repeated_facts {
      count_t byte_size{0};
      count_t true_extent{0};
      bool packed{true};
      bool base_repetition{true};
      overlap_state overlap{overlap_state::disjoint};
    };

    // `count` copies of `child` at `stride` bytes. This is the common algebra behind
    // contiguous/vector/strided_vector and each indexed entry.
    [[nodiscard]] inline repeated_facts repeat_facts(const map_element_facts& child,
                                                     count_t count, count_t stride,
                                                     count_t base_extent, const context& ctx) {
      if (count < 0)
        throw_invalid(ctx.operation, "negative repetition count");
      if (stride < 0)
        throw_invalid(ctx.operation, "negative byte stride");
      if (count == 0 or child.byte_size == 0)
        return repeated_facts{};
      repeated_facts result;
      result.byte_size = checked_count_mul(count, child.byte_size, ctx);
      result.true_extent =
          checked_count_add(checked_count_mul(count - 1, stride, ctx), child.true_extent, ctx);
      result.packed = child.packed and (count <= 1 or stride == child.byte_size);
      const count_t natural = checked_count_mul(child.size, base_extent, ctx);
      result.base_repetition = child.base_repetition and (count <= 1 or stride == natural);
      overlap_state overlap = child.overlap;
      if (overlap != overlap_state::overlap) {
        if (count <= 1) {
          // The child's own overlap analysis is the answer.
        } else if (stride == 0) {
          overlap = overlap_state::overlap;
        } else if (overlap == overlap_state::disjoint and stride >= child.true_extent) {
          overlap = overlap_state::disjoint;
        } else if (overlap == overlap_state::disjoint and child.packed) {
          // A packed child tiles one contiguous interval in map order, so a stride shorter
          // than its true extent necessarily places some bytes of a later copy inside an
          // earlier copy.  This proves the vector(2,2,1) / stride < blocklength case
          // exactly; a holey child stays `unknown`.
          overlap = overlap_state::overlap;
        } else {
          overlap = overlap_state::unknown;
        }
      }
      result.overlap = overlap;
      return result;
    }

    struct indexed_entry_facts {
      count_t blocklength{0};
      count_t byte_displacement{0};
    };

    struct map_interval {
      count_t start{0};
      count_t end{0};
    };

    [[nodiscard]] inline bool intervals_intersect(std::vector<map_interval> intervals) {
      if (intervals.size() < 2)
        return false;
      std::sort(intervals.begin(), intervals.end(),
                [](const map_interval& left, const map_interval& right) {
                  return left.start < right.start;
                });
      count_t furthest = intervals[0].end;
      for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].start < furthest)
          return true;
        furthest = std::max(furthest, intervals[i].end);
      }
      return false;
    }

    // Map-order walk of indexed/hindexed/indexed_block/hindexed_block entries. Entries are
    // examined in parameter order, so packed/base_repetition are O(entries) with no sort.
    // Overlap is proven only from an exact interval image. One entry's image is exact only
    // when the child map is itself the gapless ascending interval
    // [true_lb, true_lb + byte_size) and consecutive copies are adjacent
    // (child.extent == child.byte_size); then the whole span is one interval. A holey child
    // or a gapped repetition stride makes a bounding interval a superset of the map, so an
    // intersection is reported `unknown`, never `overlap`: only the exact map proves that a
    // receive would write one location twice (MPI 4.1 §5.1.11). A bounding box that does not
    // intersect any other still proves disjointness.
    [[nodiscard]] inline layout_facts walk_indexed_facts(
        const std::vector<indexed_entry_facts>& entries, const map_element_facts& child,
        count_t base_extent, bool check_overlap, const context& ctx) {
      layout_facts facts;
      bool packed = true;
      bool have_packed = false;
      count_t packed_cursor{0};
      bool base_repetition = true;
      bool have_base = false;
      count_t base_cursor{0};
      overlap_state seen = overlap_state::disjoint;
      std::vector<map_interval> bounds;
      std::vector<map_interval> exact_intervals;
      bool all_exact = true;
      if (check_overlap) {
        bounds.reserve(entries.size());
        exact_intervals.reserve(entries.size());
      }
      for (const indexed_entry_facts& entry : entries) {
        if (entry.blocklength == 0 or child.byte_size == 0)
          continue;
        const repeated_facts span =
            repeat_facts(child, entry.blocklength, child.extent, base_extent, ctx);
        const count_t start = checked_count_add(entry.byte_displacement, child.true_lb, ctx);
        if (packed) {
          // span.packed already handles blocklength == 1 (a single copy is contiguous even
          // when the child's extent has holes) and blocklength > 1 (copies must be adjacent).
          if (not span.packed) {
            packed = false;
          } else if (have_packed and start != packed_cursor) {
            packed = false;
          } else {
            have_packed = true;
            packed_cursor = checked_count_add(start, span.byte_size, ctx);
          }
        }
        if (base_repetition) {
          if (not span.base_repetition) {
            base_repetition = false;
          } else if (have_base and start != base_cursor) {
            base_repetition = false;
          } else {
            have_base = true;
            const count_t span_elements = checked_count_mul(entry.blocklength, child.size, ctx);
            base_cursor = checked_count_add(
                start, checked_count_mul(span_elements, base_extent, ctx), ctx);
          }
        }
        seen = combine_overlap(seen, span.overlap);
        if (check_overlap and span.byte_size != 0) {
          const bool exact_span = child.packed and child.byte_size == child.true_extent and
                                  (entry.blocklength <= 1 or child.extent == child.byte_size);
          if (exact_span) {
            const map_interval image{start, checked_count_add(start, span.byte_size, ctx)};
            bounds.push_back(image);
            exact_intervals.push_back(image);
          } else {
            all_exact = false;
            bounds.push_back(
                map_interval{start, checked_count_add(start, span.true_extent, ctx)});
          }
        }
      }
      facts.packed = packed;
      facts.base_repetition = base_repetition;
      if (seen == overlap_state::overlap) {
        facts.overlap = overlap_state::overlap;
        return facts;
      }
      if (not check_overlap) {
        facts.overlap = bounds.size() <= 1 ? seen : overlap_state::unknown;
        return facts;
      }
      // Exact images can prove overlap even when some other entry only has a bound.
      if (intervals_intersect(exact_intervals)) {
        facts.overlap = overlap_state::overlap;
        return facts;
      }
      if (not all_exact and intervals_intersect(bounds)) {
        // At least one entry's image is only a superset of its map: the intersection may be
        // spurious, so do not claim a proven overlap.
        facts.overlap = overlap_state::unknown;
        return facts;
      }
      // All exact images are disjoint (or all bounding boxes are), so the map is disjoint.
      facts.overlap = overlap_state::disjoint;
      return facts;
    }

    // MPI_Type_create_subarray visits the selected positions in the requested array order.
    // The selected positions are one consecutive run of linear element indices iff their
    // number equals (last selected linear index - first selected linear index + 1). This
    // recognizes a single-row/1xK tile whose inner range is partial, not only tiles that
    // select every non-outer dimension in full. A geometry whose linear-index arithmetic
    // cannot be shown exactly (overflow) is conservatively reported as not linear.
    [[nodiscard]] inline bool subarray_elements_are_linear(const std::vector<int>& sizes,
                                                           const std::vector<int>& subsizes,
                                                           const std::vector<int>& starts,
                                                           array_order order) noexcept {
      if (sizes.size() <= 1)
        return true;
      const std::size_t rank = sizes.size();
      if (subsizes.size() != rank or starts.size() != rank)
        return false;
      const count_t limit = std::numeric_limits<count_t>::max();
      // Element strides in memory order: row-major stores the last dimension contiguously,
      // column-major the first.
      std::vector<count_t> strides(rank);
      if (order == array_order::row_major) {
        count_t accumulator{1};
        for (std::size_t d = rank; d > 0; --d) {
          strides[d - 1] = accumulator;
          if (sizes[d - 1] > 0 and accumulator > limit / sizes[d - 1])
            return false;
          accumulator *= sizes[d - 1];
        }
      } else {
        count_t accumulator{1};
        for (std::size_t d = 0; d < rank; ++d) {
          strides[d] = accumulator;
          if (sizes[d] > 0 and accumulator > limit / sizes[d])
            return false;
          accumulator *= sizes[d];
        }
      }
      count_t first{0};
      count_t last{0};
      count_t count{1};
      for (std::size_t d = 0; d < rank; ++d) {
        if (subsizes[d] <= 0)
          return false;
        const count_t start = starts[d];
        const count_t final = start + static_cast<count_t>(subsizes[d]) - 1;
        const count_t stride = strides[d];
        if ((start != 0 and stride > limit / start) or
            (final != 0 and stride > limit / final) or
            static_cast<count_t>(subsizes[d]) > limit / count)
          return false;
        const count_t first_offset = start * stride;
        const count_t last_offset = final * stride;
        if (first > limit - first_offset or last > limit - last_offset)
          return false;
        first += first_offset;
        last += last_offset;
        count *= static_cast<count_t>(subsizes[d]);
      }
      return last >= first and last - first == count - 1;
    }

    // `repetitions` is the number of selected subarray positions (the product of the
    // subsizes), NOT the total base-element count: MPI_Type_create_subarray repeats the
    // oldtype once per selected position.
    [[nodiscard]] inline layout_facts subarray_facts(
        const std::vector<int>& sizes, const std::vector<int>& subsizes,
        const std::vector<int>& starts, array_order order, const map_element_facts& child,
        count_t repetitions, count_t base_extent, const context& ctx) {
      layout_facts facts;
      if (repetitions == 0 or child.byte_size == 0)
        return facts;
      const bool linear = subarray_elements_are_linear(sizes, subsizes, starts, order);
      // repetitions == 1 needs no repetition, so a child whose extent has holes is still a
      // packed/base-repetition fact; repeat_facts already encodes that distinction.
      const repeated_facts repeated =
          linear ? repeat_facts(child, repetitions, child.extent, base_extent, ctx)
                 : repeated_facts{};
      facts.packed = linear and repeated.packed;
      facts.base_repetition = linear and repeated.base_repetition;
      if (child.overlap == overlap_state::overlap) {
        facts.overlap = overlap_state::overlap;
      } else if (repetitions <= 1) {
        facts.overlap = child.overlap;
      } else if (not linear) {
        facts.overlap = overlap_state::unknown;
      } else {
        facts.overlap = repeated.overlap;
      }
      return facts;
    }

    [[nodiscard]] inline layout_facts query_layout_geometry(MPI_Datatype type,
                                                            layout_facts facts,
                                                            const context& ctx) {
      MPI_Count size{0};
      MPI_Count lower_bound{0};
      MPI_Count extent{0};
      MPI_Count true_lb{0};
      MPI_Count true_extent{0};
      check_mpi(MPI_Type_size_c(type, &size), ctx);
      check_mpi(MPI_Type_get_extent_c(type, &lower_bound, &extent), ctx);
      check_mpi(MPI_Type_get_true_extent_c(type, &true_lb, &true_extent), ctx);
      facts.byte_size = checked_count(size, ctx);
      facts.byte_lower_bound = checked_count(lower_bound, ctx);
      facts.byte_extent = checked_count(extent, ctx);
      facts.true_lb = checked_count(true_lb, ctx);
      facts.true_extent = checked_count(true_extent, ctx);
      // MPI 4.1 §5.1.8 defines the true lower bound/extent as the minimum/maximum over the
      // typemap entries, which is not defined for an empty typemap.  A provider may return a
      // sentinel there (Open MPI 6.0.0rc2 returns true_lb = INT64_MAX, true_extent = 1 for
      // some zero-byte derived types), making the public true_byte_upper_bound() throw for a
      // valid layout.  The empty map addresses no byte, so normalize to (0, 0) and keep the
      // provider's byte extent marker (a resized empty type may carry a nonzero extent).
      if (facts.byte_size == 0) {
        facts.true_lb = 0;
        facts.true_extent = 0;
      }
      facts.valid = true;
      return facts;
    }

    // The adopting from_raw path may commit, dup and free the handle, so it accepts only a
    // handle the application provably owns: a derived datatype, decoded from its envelope.
    // MPI 4.1 §5.1.13 (Type Envelopes and Contents) defines exactly the combiners below for
    // derived datatypes; MPI_Type_dup of a predefined type reports MPI_COMBINER_DUP and is
    // adopted (and freed) normally.
    //
    // Everything else is a predefined handle, a process-global constant that must never be
    // freed:
    //  - MPI_COMBINER_NAMED: a named predefined datatype (§5.1.13).
    //  - MPI_COMBINER_F90_REAL/COMPLEX/INTEGER: MPI_Type_create_f90_real/complex/integer
    //    return predefined *unnamed* datatypes; they "cannot be freed" and "do not need to be
    //    committed" (§19.1, Example 19.1 advice to users, and the implementor advice that the
    //    application is not allowed to free the returned handles).
    //  - MPI_COMBINER_VALUE_INDEX: MPI_Type_get_value_index may return a predefined unnamed
    //    pair handle; "This type cannot be freed", and decoders "must therefore handle both
    //    MPI_COMBINER_NAMED and MPI_COMBINER_VALUE_INDEX" (§6.9.4, Example 6.17 advice).
    //  - An implementation-specific combiner is refused too: freeability cannot be proven, and
    //    a false reject throws std::invalid_argument, whereas freeing a predefined handle
    //    aborts under MPI_ERRORS_ARE_FATAL. The cost is one envelope query on the from_raw
    //    user path only (internal factories pass their already-owned handles straight to the
    //    private adopting path).
    //
    // The combiner predicate is separate so a unit test can pin the full predefined set even
    // on a provider that has no f90/value-index handle to offer.
    [[nodiscard]] constexpr bool is_adoptable_derived_combiner(int combiner) noexcept {
      switch (combiner) {
        case MPI_COMBINER_DUP:
        case MPI_COMBINER_CONTIGUOUS:
        case MPI_COMBINER_VECTOR:
        case MPI_COMBINER_HVECTOR:
        case MPI_COMBINER_INDEXED:
        case MPI_COMBINER_HINDEXED:
        case MPI_COMBINER_INDEXED_BLOCK:
        case MPI_COMBINER_HINDEXED_BLOCK:
        case MPI_COMBINER_STRUCT:
        case MPI_COMBINER_SUBARRAY:
        case MPI_COMBINER_DARRAY:
        case MPI_COMBINER_RESIZED:
          return true;
        case MPI_COMBINER_NAMED:
        case MPI_COMBINER_F90_REAL:
        case MPI_COMBINER_F90_COMPLEX:
        case MPI_COMBINER_F90_INTEGER:
        case MPI_COMBINER_VALUE_INDEX:
        default:
          return false;
      }
    }

    [[nodiscard]] inline bool is_adoptable_derived_datatype(MPI_Datatype type) {
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner{MPI_COMBINER_NAMED};
      check_mpi(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses, &num_large_counts,
                                        &num_datatypes, &combiner),
                "MPI_Type_get_envelope_c(layout<T>::from_raw)");
      return is_adoptable_derived_combiner(combiner);
    }

    [[nodiscard]] inline count_t layout_byte_upper_bound(const layout_facts& facts,
                                                         const context& ctx) {
      return checked_count_add(facts.byte_lower_bound, facts.byte_extent, ctx);
    }

    [[nodiscard]] inline count_t layout_true_byte_upper_bound(const layout_facts& facts,
                                                              const context& ctx) {
      return checked_count_add(facts.true_lb, facts.true_extent, ctx);
    }

  }  // namespace detail

  namespace detail {
    // Internal lowering guard: rejects an invalid layout before a lowering uses it. The
    // public layout<T> API has no factory that derives a layout from raw addresses.
    template<class T>
    void require_relative(const layout<T>& l, const context& ctx);
  }  // namespace detail


  //--------------------------------------------------------------------------
  // Parameter types. They own their data, validate local contracts in add(),
  // and are consumed by the owning layout factories below.
  //--------------------------------------------------------------------------

  //--------------------------------------------------------------------------
  // Structural facts the caller can prove only at construction. from_raw accepts them
  // as hints; a member left false means "not proven", never a false proof. The geometry
  // (size, extent, true extent, lower bound) is always queried from the committed handle.
  //--------------------------------------------------------------------------

  struct layout_facts {
    bool packed = false;                   // proven: the map tiles [lb, lb+byte_size) once
    bool base_repetition = false;          // proven: size() copies of native_datatype<T>()
    bool has_overlapping_entries = false;  // proven: the map overlaps (a receive rejects it)
  };

  //--------------------------------------------------------------------------
  // Adjacent-integer factory options. The block form of vector and
  // strided_vector is a member of the option struct, so one declaration per MPI family; a
  // nullptr block selects the base datatype.
  //--------------------------------------------------------------------------

  template<class T>
  struct vector_options {
    count_t count{0};
    count_t blocklength{0};
    count_t stride{0};                 // old-type extent units, MPI_Type_vector
    const layout<T>* block = nullptr;  // optional oldtype
  };

  template<class T>
  struct strided_vector_options {
    count_t count{0};
    count_t blocklength{0};
    MPI_Aint byte_stride{0};           // bytes, MPI_Type_create_hvector
    const layout<T>* block = nullptr;  // optional oldtype
  };

  //--------------------------------------------------------------------------
  // Parameter types. They own their data, validate local contracts in add(),
  // and are consumed by the owning layout factories below. The target declarations are the
  // default constructor, the explicit std::span constructor (explicit element type) and
  // add(); the constructors are not initializer-list constructors, so a braced listing is
  // rejected (a braced-init-list does not convert to std::span).
  //--------------------------------------------------------------------------

  namespace detail {

    // One template and five tags replace the five former parameter classes. The tag selects
    // the stored entry type, the span constructor's element type, the
    // add() overloads and the local validation; the public names below are aliases, so every
    // documented builder spelling (default construction, named contiguous range, add(),
    // subarray order) keeps its exact signature and member-pointer type.
    struct indexed_parameter_tag {};
    struct hindexed_parameter_tag {};
    struct indexed_block_parameter_tag {};
    struct hindexed_block_parameter_tag {};
    struct subarray_parameter_tag {};

    template<class Tag>
    struct parameter_traits;

    template<>
    struct parameter_traits<indexed_parameter_tag> {
      using entry_type = std::pair<count_t, count_t>;
      static constexpr const char* operation = "indexed_parameter::add";
    };

    template<>
    struct parameter_traits<hindexed_parameter_tag> {
      using entry_type = std::pair<count_t, MPI_Aint>;
      static constexpr const char* operation = "hindexed_parameter::add";
    };

    template<>
    struct parameter_traits<indexed_block_parameter_tag> {
      using entry_type = count_t;
      static constexpr const char* operation = "indexed_block_parameter::add";
    };

    template<>
    struct parameter_traits<hindexed_block_parameter_tag> {
      using entry_type = MPI_Aint;
      static constexpr const char* operation = "hindexed_block_parameter::add";
    };

    template<>
    struct parameter_traits<subarray_parameter_tag> {
      using entry_type = std::array<count_t, 3>;
      static constexpr const char* operation = "subarray_parameter::add";
    };

    // The array order is subarray-only state; the empty primary keeps every other builder
    // exactly the size of its entry vector (no per-builder order byte).
    template<class Tag>
    struct parameter_order {};

    template<>
    struct parameter_order<subarray_parameter_tag> {
      array_order value{array_order::row_major};
    };

    template<class Tag>
    class parameter_builder {
    public:
      using entry_type = typename parameter_traits<Tag>::entry_type;

      parameter_builder() = default;

      explicit parameter_builder(std::span<const entry_type> list) {
        entries_.reserve(list.size());
        for (const entry_type& entry : list) {
          if constexpr (std::is_same_v<Tag, subarray_parameter_tag>)
            add(entry[0], entry[1], entry[2]);
          else if constexpr (std::is_same_v<Tag, indexed_parameter_tag> or
                             std::is_same_v<Tag, hindexed_parameter_tag>)
            add(entry.first, entry.second);
          else
            add(entry);
        }
      }

      void add(count_t blocklength, count_t displacement)
      requires(std::is_same_v<Tag, indexed_parameter_tag>)
      {
        require_pair(blocklength, displacement);
      }

      // On a platform where MPI_Aint and count_t are the same type this overload and the
      // indexed one share a parameter list; the tags keep exactly one of them viable.
      void add(count_t blocklength, MPI_Aint displacement)
      requires(std::is_same_v<Tag, hindexed_parameter_tag>)
      {
        require_pair(blocklength, displacement);
      }

      void add(count_t displacement)
      requires(std::is_same_v<Tag, indexed_block_parameter_tag>)
      {
        require_displacement(displacement);
      }

      void add(MPI_Aint displacement)
      requires(std::is_same_v<Tag, hindexed_block_parameter_tag>)
      {
        require_displacement(displacement);
      }

      void add(count_t size, count_t subsize, count_t start)
      requires(std::is_same_v<Tag, subarray_parameter_tag>)
      {
        if (size < 0)
          throw_invalid(parameter_traits<Tag>::operation, "negative dimension size");
        if (subsize < 0 or subsize > size)
          throw_invalid(parameter_traits<Tag>::operation, "subsize outside [0, size]");
        if (start < 0 or start > size - subsize)
          throw_invalid(parameter_traits<Tag>::operation, "start outside [0, size - subsize]");
        entries_.push_back({size, subsize, start});
      }

      void order(array_order new_order) noexcept
      requires(std::is_same_v<Tag, subarray_parameter_tag>)
      {
        order_.value = new_order;
      }

      [[nodiscard]] array_order order() const noexcept
      requires(std::is_same_v<Tag, subarray_parameter_tag>)
      {
        return order_.value;
      }

    private:
      template<class Displacement>
      void require_pair(count_t blocklength, Displacement displacement) {
        if (blocklength < 0)
          throw_invalid(parameter_traits<Tag>::operation, "negative blocklength");
        if (displacement < 0)
          throw_invalid(parameter_traits<Tag>::operation, "negative displacement");
        entries_.emplace_back(blocklength, displacement);
      }

      template<class Displacement>
      void require_displacement(Displacement displacement) {
        if (displacement < 0)
          throw_invalid(parameter_traits<Tag>::operation, "negative displacement");
        entries_.push_back(displacement);
      }

      std::vector<entry_type> entries_;
      [[no_unique_address]] parameter_order<Tag> order_{};

      template<class>
      friend class ::mp1::layout;
    };

  }  // namespace detail

  using indexed_parameter = detail::parameter_builder<detail::indexed_parameter_tag>;
  using hindexed_parameter = detail::parameter_builder<detail::hindexed_parameter_tag>;
  using indexed_block_parameter =
      detail::parameter_builder<detail::indexed_block_parameter_tag>;
  using hindexed_block_parameter =
      detail::parameter_builder<detail::hindexed_block_parameter_tag>;
  using subarray_parameter = detail::parameter_builder<detail::subarray_parameter_tag>;

  //--------------------------------------------------------------------------
  // layout<T>: one owning committed MPI_Datatype plus the facts its factories
  // proved at construction. The geometry accessors are descriptive; dense() and
  // is_base_repetition() are the structural facts used by lowerings.
  //--------------------------------------------------------------------------

  template<class T>
  class layout {
  public:
    layout() noexcept = default;

    /// An owning copy duplicates the MPI datatype (MPI_Type_dup, MPI 4.1 §5.1.10) and frees
    /// its own handle at destruction.  Blocking operations take a layout by reference (a
    /// `region` or a `std::span<const layout<T>>` descriptor) and never copy it; copy only when
    /// a true independent owner is needed, and prefer moving an rvalue (a move transfers the
    /// handle, no dup).
    layout(const layout& other) {
      if (not other.is_valid())
        return;
      detail::require_mpi("mp1: layout copy");
      detail::check_mpi(MPI_Type_dup(other.handle_, &handle_), "MPI_Type_dup(layout)");
      size_ = other.size_;
      facts_ = other.facts_;
    }

    layout(layout&& other) noexcept {
      swap(other);
    }

    layout& operator=(const layout& other) {
      layout copy{other};
      swap(copy);
      return *this;
    }

    layout& operator=(layout&& other) noexcept {
      if (this != &other) {
        layout moved{std::move(other)};
        swap(moved);
      }
      return *this;
    }

    ~layout() noexcept {
      release();
    }

    // ---- factories: each constructs one MPI_Type_* object and commits it; the private
    // adopting path then queries size, extent and true extent, so every factory costs five
    // MPI calls (no predefined-handle envelope query: the factory created the handle) ----

    [[nodiscard]] static layout contiguous(count_t count) {
      detail::require_mpi("mp1::layout<T>::contiguous");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::contiguous");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_contiguous_c(mpi_count, base.type, &type),
                        "MPI_Type_contiguous_c");
      const detail::repeated_facts repeated = detail::repeat_facts(
          base_element_facts(), count, base.extent, base.extent, "layout<T>::contiguous");
      return adopt_derived(type, count, make_structural_facts(repeated));
    }

    [[nodiscard]] static layout contiguous(count_t count, const layout& block) {
      detail::require_mpi("mp1::layout<T>::contiguous(block)");
      require_valid_block(block, "layout<T>::contiguous");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::contiguous");
      const count_t elements =
          detail::checked_count_mul(count, block.size(), "layout<T>::contiguous");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_contiguous_c(mpi_count, block.handle_, &type),
                        "MPI_Type_contiguous_c");
      const detail::repeated_facts repeated =
          detail::repeat_facts(element_facts_of(block), count, block.facts_.byte_extent,
                               base.extent, "layout<T>::contiguous");
      return adopt_derived(type, elements, make_structural_facts(repeated));
    }

    [[nodiscard]] static layout vector(count_t count, count_t blocklength, count_t stride) {
      detail::require_mpi("mp1::layout<T>::vector");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::vector");
      const MPI_Count mpi_blocklength =
          detail::checked_mpi_count(blocklength, "layout<T>::vector");
      const MPI_Count mpi_stride = detail::checked_mpi_count(stride, "layout<T>::vector");
      const count_t elements =
          detail::checked_count_mul(count, blocklength, "layout<T>::vector");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_vector_c(mpi_count, mpi_blocklength, mpi_stride, base.type, &type),
          "MPI_Type_vector_c");
      // MPI_Type_vector's stride is in units of extent(oldtype); the facts helper takes bytes.
      const count_t outer_byte_stride =
          detail::checked_count_mul(stride, base.extent, "layout<T>::vector");
      return adopt_derived(
          type, elements,
          make_repetition_facts(base_element_facts(), blocklength, count, base.extent,
                                outer_byte_stride, "layout<T>::vector"));
    }

    [[nodiscard]] static layout vector(count_t count, count_t blocklength, count_t stride,
                                       const layout& block) {
      detail::require_mpi("mp1::layout<T>::vector(block)");
      require_valid_block(block, "layout<T>::vector");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::vector");
      const MPI_Count mpi_blocklength =
          detail::checked_mpi_count(blocklength, "layout<T>::vector");
      const MPI_Count mpi_stride = detail::checked_mpi_count(stride, "layout<T>::vector");
      count_t elements = detail::checked_count_mul(count, blocklength, "layout<T>::vector");
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::vector");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_vector_c(mpi_count, mpi_blocklength, mpi_stride, block.handle_, &type),
          "MPI_Type_vector_c");
      // One outer vector block is already `blocklength` copies of `block` at extent(block);
      // the outer repetition is `count` groups whose stride is in extent(block) units.
      const count_t outer_byte_stride =
          detail::checked_count_mul(stride, block.facts_.byte_extent, "layout<T>::vector");
      return adopt_derived(
          type, elements,
          make_repetition_facts(element_facts_of(block), blocklength, count, base.extent,
                                outer_byte_stride, "layout<T>::vector"));
    }

    [[nodiscard]] static layout strided_vector(count_t count, count_t blocklength,
                                               MPI_Aint byte_stride) {
      detail::require_mpi("mp1::layout<T>::strided_vector");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::strided_vector");
      const MPI_Count mpi_blocklength =
          detail::checked_mpi_count(blocklength, "layout<T>::strided_vector");
      const MPI_Count mpi_stride =
          detail::checked_mpi_count_from_aint(byte_stride, "layout<T>::strided_vector");
      const count_t elements =
          detail::checked_count_mul(count, blocklength, "layout<T>::strided_vector");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_hvector_c(mpi_count, mpi_blocklength, mpi_stride, base.type, &type),
          "MPI_Type_create_hvector_c");
      const count_t stride_bytes =
          detail::checked_count(mpi_stride, "layout<T>::strided_vector");
      return adopt_derived(
          type, elements,
          make_repetition_facts(base_element_facts(), blocklength, count, base.extent,
                                stride_bytes, "layout<T>::strided_vector"));
    }

    [[nodiscard]] static layout strided_vector(count_t count, count_t blocklength,
                                               MPI_Aint byte_stride, const layout& block) {
      detail::require_mpi("mp1::layout<T>::strided_vector(block)");
      require_valid_block(block, "layout<T>::strided_vector");
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      const MPI_Count mpi_count = detail::checked_mpi_count(count, "layout<T>::strided_vector");
      const MPI_Count mpi_blocklength =
          detail::checked_mpi_count(blocklength, "layout<T>::strided_vector");
      const MPI_Count mpi_stride =
          detail::checked_mpi_count_from_aint(byte_stride, "layout<T>::strided_vector");
      count_t elements =
          detail::checked_count_mul(count, blocklength, "layout<T>::strided_vector");
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::strided_vector");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_hvector_c(mpi_count, mpi_blocklength, mpi_stride,
                                                  block.handle_, &type),
                        "MPI_Type_create_hvector_c");
      // The hvector stride argument is already in bytes; pass it through unchanged.
      const count_t stride_bytes =
          detail::checked_count(mpi_stride, "layout<T>::strided_vector");
      return adopt_derived(
          type, elements,
          make_repetition_facts(element_facts_of(block), blocklength, count, base.extent,
                                stride_bytes, "layout<T>::strided_vector"));
    }

    // ---- target option-struct spellings: the block form is a member, so
    // one declaration per MPI family. The positional overloads above remain as transitional
    // shims for the transitional spelling. ----

    [[nodiscard]] static layout vector(vector_options<T> options) {
      if (options.block == nullptr)
        return vector(options.count, options.blocklength, options.stride);
      return vector(options.count, options.blocklength, options.stride, *options.block);
    }

    [[nodiscard]] static layout strided_vector(strided_vector_options<T> options) {
      if (options.block == nullptr)
        return strided_vector(options.count, options.blocklength, options.byte_stride);
      return strided_vector(options.count, options.blocklength, options.byte_stride,
                            *options.block);
    }

    [[nodiscard]] static layout indexed(const indexed_parameter& parameter) {
      detail::require_mpi("mp1::layout<T>::indexed");
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      count_t elements{0};
      std::vector<MPI_Count> blocklengths;
      std::vector<MPI_Count> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      blocklengths.reserve(entries.size());
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const auto& entry : entries) {
        elements = detail::checked_count_add(elements, entry.first, "layout<T>::indexed");
        blocklengths.push_back(static_cast<MPI_Count>(entry.first));
        offsets.push_back(static_cast<MPI_Count>(entry.second));
        fact_entries.push_back(detail::indexed_entry_facts{
            entry.first,
            detail::checked_count_mul(entry.second, base.extent, "layout<T>::indexed")});
      }
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_indexed_c(static_cast<MPI_Count>(entries.size()), blocklengths.data(),
                             offsets.data(), base.type, &type),
          "MPI_Type_indexed_c");
      return adopt_derived(type, elements,
                           detail::walk_indexed_facts(fact_entries, base_element_facts(),
                                                      base.extent, true, "layout<T>::indexed"));
    }

    [[nodiscard]] static layout indexed(const indexed_parameter& parameter,
                                        const layout& block) {
      detail::require_mpi("mp1::layout<T>::indexed(block)");
      require_valid_block(block, "layout<T>::indexed");
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      count_t elements{0};
      std::vector<MPI_Count> blocklengths;
      std::vector<MPI_Count> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      blocklengths.reserve(entries.size());
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const auto& entry : entries) {
        elements = detail::checked_count_add(elements, entry.first, "layout<T>::indexed");
        blocklengths.push_back(static_cast<MPI_Count>(entry.first));
        offsets.push_back(static_cast<MPI_Count>(entry.second));
        fact_entries.push_back(detail::indexed_entry_facts{
            entry.first, detail::checked_count_mul(entry.second, block.facts_.byte_extent,
                                                   "layout<T>::indexed")});
      }
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::indexed");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_indexed_c(static_cast<MPI_Count>(entries.size()), blocklengths.data(),
                             offsets.data(), block.handle_, &type),
          "MPI_Type_indexed_c");
      return adopt_derived(type, elements,
                           detail::walk_indexed_facts(fact_entries, element_facts_of(block),
                                                      base.extent, true, "layout<T>::indexed"));
    }

    [[nodiscard]] static layout hindexed(const hindexed_parameter& parameter) {
      detail::require_mpi("mp1::layout<T>::hindexed");
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      if (entries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::hindexed", "too many blocks for int");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      count_t elements{0};
      std::vector<int> blocklengths;
      std::vector<MPI_Aint> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      blocklengths.reserve(entries.size());
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const auto& entry : entries) {
        elements = detail::checked_count_add(elements, entry.first, "layout<T>::hindexed");
        blocklengths.push_back(checked_blocklength_int(entry.first, "layout<T>::hindexed"));
        offsets.push_back(entry.second);
        fact_entries.push_back(detail::indexed_entry_facts{
            entry.first, detail::checked_count(static_cast<MPI_Count>(entry.second),
                                               "layout<T>::hindexed")});
      }
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_hindexed(static_cast<int>(entries.size()), blocklengths.data(),
                                   offsets.data(), base.type, &type),
          "MPI_Type_create_hindexed");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, base_element_facts(), base.extent, true,
                                     "layout<T>::hindexed"));
    }

    [[nodiscard]] static layout hindexed(const hindexed_parameter& parameter,
                                         const layout& block) {
      detail::require_mpi("mp1::layout<T>::hindexed(block)");
      require_valid_block(block, "layout<T>::hindexed");
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      if (entries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::hindexed", "too many blocks for int");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      count_t elements{0};
      std::vector<int> blocklengths;
      std::vector<MPI_Aint> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      blocklengths.reserve(entries.size());
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const auto& entry : entries) {
        elements = detail::checked_count_add(elements, entry.first, "layout<T>::hindexed");
        blocklengths.push_back(checked_blocklength_int(entry.first, "layout<T>::hindexed"));
        offsets.push_back(entry.second);
        fact_entries.push_back(detail::indexed_entry_facts{
            entry.first, detail::checked_count(static_cast<MPI_Count>(entry.second),
                                               "layout<T>::hindexed")});
      }
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::hindexed");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_hindexed(static_cast<int>(entries.size()), blocklengths.data(),
                                   offsets.data(), block.handle_, &type),
          "MPI_Type_create_hindexed");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, element_facts_of(block), base.extent, true,
                                     "layout<T>::hindexed"));
    }

    [[nodiscard]] static layout indexed_block(count_t blocklength,
                                              const indexed_block_parameter& parameter) {
      detail::require_mpi("mp1::layout<T>::indexed_block");
      if (blocklength < 0) {
        throw_invalid("layout<T>::indexed_block", "negative blocklength");
      }
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<MPI_Count> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const count_t offset : entries) {
        offsets.push_back(static_cast<MPI_Count>(offset));
        fact_entries.push_back(detail::indexed_entry_facts{
            blocklength,
            detail::checked_count_mul(offset, base.extent, "layout<T>::indexed_block")});
      }
      const count_t elements = detail::checked_count_mul(
          static_cast<count_t>(entries.size()), blocklength, "layout<T>::indexed_block");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_indexed_block_c(static_cast<MPI_Count>(entries.size()),
                                                        static_cast<MPI_Count>(blocklength),
                                                        offsets.data(), base.type, &type),
                        "MPI_Type_create_indexed_block_c");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, base_element_facts(), base.extent, true,
                                     "layout<T>::indexed_block"));
    }

    [[nodiscard]] static layout indexed_block(count_t blocklength,
                                              const indexed_block_parameter& parameter,
                                              const layout& block) {
      detail::require_mpi("mp1::layout<T>::indexed_block(block)");
      require_valid_block(block, "layout<T>::indexed_block");
      if (blocklength < 0) {
        throw_invalid("layout<T>::indexed_block", "negative blocklength");
      }
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<MPI_Count> offsets;
      std::vector<detail::indexed_entry_facts> fact_entries;
      offsets.reserve(entries.size());
      fact_entries.reserve(entries.size());
      for (const count_t offset : entries) {
        offsets.push_back(static_cast<MPI_Count>(offset));
        fact_entries.push_back(detail::indexed_entry_facts{
            blocklength, detail::checked_count_mul(offset, block.facts_.byte_extent,
                                                   "layout<T>::indexed_block")});
      }
      count_t elements = detail::checked_count_mul(static_cast<count_t>(entries.size()),
                                                   blocklength, "layout<T>::indexed_block");
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::indexed_block");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_indexed_block_c(static_cast<MPI_Count>(entries.size()),
                                                        static_cast<MPI_Count>(blocklength),
                                                        offsets.data(), block.handle_, &type),
                        "MPI_Type_create_indexed_block_c");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, element_facts_of(block), base.extent, true,
                                     "layout<T>::indexed_block"));
    }

    [[nodiscard]] static layout hindexed_block(count_t blocklength,
                                               const hindexed_block_parameter& parameter) {
      detail::require_mpi("mp1::layout<T>::hindexed_block");
      if (blocklength < 0) {
        throw_invalid("layout<T>::hindexed_block", "negative blocklength");
      }
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      if (entries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::hindexed_block", "too many blocks for int");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<detail::indexed_entry_facts> fact_entries;
      fact_entries.reserve(entries.size());
      for (const MPI_Aint offset : entries) {
        fact_entries.push_back(detail::indexed_entry_facts{
            blocklength, detail::checked_count(static_cast<MPI_Count>(offset),
                                               "layout<T>::hindexed_block")});
      }
      const count_t elements = detail::checked_count_mul(
          static_cast<count_t>(entries.size()), blocklength, "layout<T>::hindexed_block");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_hindexed_block(
                            static_cast<int>(entries.size()),
                            checked_blocklength_int(blocklength, "layout<T>::hindexed_block"),
                            entries.data(), base.type, &type),
                        "MPI_Type_create_hindexed_block");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, base_element_facts(), base.extent, true,
                                     "layout<T>::hindexed_block"));
    }

    [[nodiscard]] static layout hindexed_block(count_t blocklength,
                                               const hindexed_block_parameter& parameter,
                                               const layout& block) {
      detail::require_mpi("mp1::layout<T>::hindexed_block(block)");
      require_valid_block(block, "layout<T>::hindexed_block");
      if (blocklength < 0) {
        throw_invalid("layout<T>::hindexed_block", "negative blocklength");
      }
      const auto& entries = parameter.entries_;
      if (entries.empty())
        return contiguous(0);
      if (entries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::hindexed_block", "too many blocks for int");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<detail::indexed_entry_facts> fact_entries;
      fact_entries.reserve(entries.size());
      for (const MPI_Aint offset : entries) {
        fact_entries.push_back(detail::indexed_entry_facts{
            blocklength, detail::checked_count(static_cast<MPI_Count>(offset),
                                               "layout<T>::hindexed_block")});
      }
      count_t elements = detail::checked_count_mul(static_cast<count_t>(entries.size()),
                                                   blocklength, "layout<T>::hindexed_block");
      elements = detail::checked_count_mul(elements, block.size(), "layout<T>::hindexed_block");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_hindexed_block(
                            static_cast<int>(entries.size()),
                            checked_blocklength_int(blocklength, "layout<T>::hindexed_block"),
                            entries.data(), block.handle_, &type),
                        "MPI_Type_create_hindexed_block");
      return adopt_derived(
          type, elements,
          detail::walk_indexed_facts(fact_entries, element_facts_of(block), base.extent, true,
                                     "layout<T>::hindexed_block"));
    }

    [[nodiscard]] static layout subarray(const subarray_parameter& parameter) {
      detail::require_mpi("mp1::layout<T>::subarray");
      const auto& dimensions = parameter.entries_;
      if (dimensions.empty())
        return contiguous(0);
      if (dimensions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::subarray", "too many dimensions");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<int> sizes;
      std::vector<int> subsizes;
      std::vector<int> starts;
      sizes.reserve(dimensions.size());
      subsizes.reserve(dimensions.size());
      starts.reserve(dimensions.size());
      bool zero_subsize = false;
      for (const auto& dimension : dimensions) {
        const count_t size = dimension[0];
        const count_t subsize = dimension[1];
        const count_t start = dimension[2];
        if (size < 0 or subsize < 0 or subsize > size or start < 0 or start > size - subsize) {
          throw_invalid("layout<T>::subarray", "invalid subarray geometry");
        }
        if (size > static_cast<count_t>(std::numeric_limits<int>::max()) or
            subsize > static_cast<count_t>(std::numeric_limits<int>::max()) or
            start > static_cast<count_t>(std::numeric_limits<int>::max())) {
          throw_invalid("layout<T>::subarray", "dimension does not fit int");
        }
        if (subsize == 0)
          zero_subsize = true;
        sizes.push_back(static_cast<int>(size));
        subsizes.push_back(static_cast<int>(subsize));
        starts.push_back(static_cast<int>(start));
      }
      // The zero fallback wins over any product overflow: MPI_Type_create_subarray must never
      // see a zero subsize (Open MPI rejects it), and size() is defined as zero in this case.
      if (zero_subsize)
        return contiguous(0);
      count_t elements{1};
      for (const int subsize : subsizes)
        elements = detail::checked_count_mul(elements, subsize, "layout<T>::subarray");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_subarray(static_cast<int>(dimensions.size()), sizes.data(),
                                   subsizes.data(), starts.data(),
                                   static_cast<int>(parameter.order()), base.type, &type),
          "MPI_Type_create_subarray");
      return adopt_derived(type, elements,
                           detail::subarray_facts(sizes, subsizes, starts, parameter.order(),
                                                  base_element_facts(), elements, base.extent,
                                                  "layout<T>::subarray"));
    }

    [[nodiscard]] static layout subarray(const subarray_parameter& parameter,
                                         const layout& block) {
      detail::require_mpi("mp1::layout<T>::subarray(block)");
      require_valid_block(block, "layout<T>::subarray");
      const auto& dimensions = parameter.entries_;
      if (dimensions.empty())
        return contiguous(0);
      if (dimensions.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid("layout<T>::subarray", "too many dimensions");
      }
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      std::vector<int> sizes;
      std::vector<int> subsizes;
      std::vector<int> starts;
      sizes.reserve(dimensions.size());
      subsizes.reserve(dimensions.size());
      starts.reserve(dimensions.size());
      bool zero_subsize = false;
      for (const auto& dimension : dimensions) {
        const count_t size = dimension[0];
        const count_t subsize = dimension[1];
        const count_t start = dimension[2];
        if (size < 0 or subsize < 0 or subsize > size or start < 0 or start > size - subsize) {
          throw_invalid("layout<T>::subarray", "invalid subarray geometry");
        }
        if (size > static_cast<count_t>(std::numeric_limits<int>::max()) or
            subsize > static_cast<count_t>(std::numeric_limits<int>::max()) or
            start > static_cast<count_t>(std::numeric_limits<int>::max())) {
          throw_invalid("layout<T>::subarray", "dimension does not fit int");
        }
        if (subsize == 0)
          zero_subsize = true;
        sizes.push_back(static_cast<int>(size));
        subsizes.push_back(static_cast<int>(subsize));
        starts.push_back(static_cast<int>(start));
      }
      if (zero_subsize)
        return contiguous(0);
      count_t repetitions{1};
      for (const int subsize : subsizes)
        repetitions = detail::checked_count_mul(repetitions, subsize, "layout<T>::subarray");
      const count_t elements =
          detail::checked_count_mul(repetitions, block.size(), "layout<T>::subarray");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_subarray(static_cast<int>(dimensions.size()), sizes.data(),
                                   subsizes.data(), starts.data(),
                                   static_cast<int>(parameter.order()), block.handle_, &type),
          "MPI_Type_create_subarray");
      return adopt_derived(type, elements,
                           detail::subarray_facts(sizes, subsizes, starts, parameter.order(),
                                                  element_facts_of(block), repetitions,
                                                  base.extent, "layout<T>::subarray"));
    }

    // ---- validity and recorded facts (no MPI in the accessors) ----

    [[nodiscard]] bool is_valid() const noexcept {
      return handle_ != MPI_DATATYPE_NULL;
    }

    // packed: the map, in map order, tiles [true_lb, true_lb + byte_size) exactly once in
    // increasing address order. This is a map fact, not a transfer predicate: a dense() map
    // that is not a native-T repetition (for example a stride on a padded T) is staged.
    [[nodiscard]] bool dense() const noexcept {
      return is_valid() and facts_.packed;
    }

    // base_repetition: the map, in map order, is exactly size() copies of
    // native_datatype<T>() at extent(native_datatype<T>()) stride. This is the predicate for
    // the flat-reduction substitution and for the direct single-block variable-collective
    // lowering.
    [[nodiscard]] bool is_base_repetition() const noexcept {
      return is_valid() and facts_.base_repetition;
    }

    [[nodiscard]] bool has_overlapping_entries() const noexcept {
      return is_valid() and facts_.overlap == detail::overlap_state::overlap;
    }

    [[nodiscard]] count_t size() const {
      require_valid("layout<T>::size");
      return size_;
    }

    [[nodiscard]] count_t byte_size() const {
      require_valid("layout<T>::byte_size");
      return facts_.byte_size;
    }

    [[nodiscard]] count_t byte_extent() const {
      require_valid("layout<T>::byte_extent");
      return facts_.byte_extent;
    }

    [[nodiscard]] count_t byte_lower_bound() const {
      require_valid("layout<T>::byte_lower_bound");
      return facts_.byte_lower_bound;
    }

    [[nodiscard]] count_t byte_upper_bound() const {
      require_valid("layout<T>::byte_upper_bound");
      return detail::layout_byte_upper_bound(facts_, "layout<T>::byte_upper_bound");
    }

    [[nodiscard]] count_t true_byte_lower_bound() const {
      require_valid("layout<T>::true_byte_lower_bound");
      return facts_.true_lb;
    }

    [[nodiscard]] count_t true_byte_extent() const {
      require_valid("layout<T>::true_byte_extent");
      return facts_.true_extent;
    }

    // The true bounds are the typemap's minimum/maximum byte addresses (MPI 4.1 §5.1.8).
    // For a layout with byte_size() == 0 the typemap is empty and MPI leaves those values
    // undefined, so the wrapper reports the canonical empty geometry: true lower bound 0,
    // true extent 0 and true upper bound 0.  This makes true_byte_upper_bound() total over
    // every valid layout instead of exposing a provider sentinel or an overflow throw.  The
    // provider's byte_lower_bound()/byte_extent() (the extent marker) are left unchanged.
    [[nodiscard]] count_t true_byte_upper_bound() const {
      require_valid("layout<T>::true_byte_upper_bound");
      return detail::layout_true_byte_upper_bound(facts_, "layout<T>::true_byte_upper_bound");
    }

    // Marker-only resize: the type map is unchanged, so map facts (packed/base_repetition/
    // overlap) are preserved. A negative extent is rejected; lb may be negative. MPI 4.1
    // §5.1.6: markers affect extent, never the typemap entries.
    void byte_resize(count_t lb, count_t extent) {
      if (not is_valid()) {
        throw_invalid("layout<T>::byte_resize", "invalid layout");
      }
      if (extent < 0) {
        throw_invalid("layout<T>::byte_resize", "negative extent");
      }
      detail::require_mpi("mp1::layout<T>::byte_resize");
      const MPI_Aint mpi_lb = detail::checked_aint_value(lb, "layout<T>::byte_resize");
      const MPI_Aint mpi_extent = detail::checked_aint_value(extent, "layout<T>::byte_resize");
      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(MPI_Type_create_resized(handle_, mpi_lb, mpi_extent, &type),
                        "MPI_Type_create_resized");
      try {
        detail::check_mpi(MPI_Type_commit(&type), "MPI_Type_commit");
        const detail::layout_facts updated =
            detail::query_layout_geometry(type, facts_, "layout<T>::byte_resize");
        detail::release_handle(handle_, MPI_DATATYPE_NULL, true, &detail::free_datatype);
        handle_ = type;
        facts_ = updated;
      } catch (...) {
        if (type != MPI_DATATYPE_NULL)
          (void)MPI_Type_free(&type);
        throw;
      }
    }

    // The non-mutating spelling of byte_resize: the common case avoids rebinding a shared
    // layout object.
    [[nodiscard]] layout resized(count_t lb, count_t extent) const {
      layout copy{*this};
      copy.byte_resize(lb, extent);
      return copy;
    }

    // ---- from_raw: adopt a raw datatype ----
    //
    // This is the one documented raw-adoption escape. There is no general raw-datatype
    // bridge or transferable datatype owner in 1.0; every other typed operation must use a
    // layout produced by a public factory. `from_raw` transfers ownership of one committed
    // or uncommitted derived handle into the layout.
    //
    // Preconditions (the first three are checked; the type map is the caller's obligation):
    //  * the handle must be a derived datatype. A predefined handle is process-global and
    //    freeing it aborts under MPI_ERRORS_ARE_FATAL, so MPI_DATATYPE_NULL, the named
    //    predefined datatypes (MPI_COMBINER_NAMED), the predefined unnamed Fortran datatypes
    //    of MPI_Type_create_f90_real/complex/integer (MPI_COMBINER_F90_*) and the predefined
    //    unnamed pair of the value-index query (combiner MPI_COMBINER_VALUE_INDEX) are
    //    rejected with std::invalid_argument. The rejected path performs one read-only envelope
    //    query and never commits, dups or frees, so ownership stays with the caller.
    //  * `elements` must be non-negative; a negative value throws std::invalid_argument.
    //  * `elements` is the application count returned by size(), never an MPI count. It is
    //    not validated against the handle's geometry.
    //  * the type map must be an occurrence of T. mp1 does not validate the leaf basic type
    //    in 1.0; a handle whose map is not over T is caller misuse and the behavior of a
    //    later typed operation is the same as passing the wrong datatype to MPI.
    //  * the facts hint may only claim the three structural predicates MPI cannot query; a
    //    member left false means "not proven", never a false proof.
    //
    // Ownership and failure: ownership transfers on entry. MPI_Type_commit is called (a
    // documented no-op for an already committed datatype, MPI 4.1 §5.1.9 "Commit and Free").
    // If adoption fails after that stage (commit or geometry query), the handle has already
    // been freed and the caller must not free it again; if the call rejects the handle before
    // adoption (null, negative count, predefined/underived combiner), ownership stays with the
    // caller, who must still free it. On success the caller must not free or reuse the
    // handle; the layout frees it exactly once.
    //
    // One overload, so every spelling is unambiguous: `from_raw(t, n)`,
    // `from_raw(t, n, mp1::layout_facts{...})`, the designated
    // `from_raw(t, n, {.packed = true})` and the bare `from_raw(t, n, {})` all compile.
    [[nodiscard]] static layout from_raw(MPI_Datatype type, count_t elements,
                                         layout_facts facts = {}) {
      detail::require_mpi("mp1::layout<T>::from_raw");
      if (type == MPI_DATATYPE_NULL) {
        throw_invalid("layout<T>::from_raw", "MPI_DATATYPE_NULL handle");
      }
      if (elements < 0) {
        throw_invalid("layout<T>::from_raw", "negative application count");
      }
      if (not detail::is_adoptable_derived_datatype(type)) {
        throw_invalid("layout<T>::from_raw",
                      "a predefined named or unnamed datatype (or a handle whose combiner is "
                      "not a derived-datatype constructor) is process-global or not provably "
                      "owned, so it must not be adopted by a layout");
      }
      detail::layout_facts proofs;
      proofs.packed = facts.packed;
      proofs.base_repetition = facts.base_repetition;
      proofs.overlap = facts.has_overlapping_entries ? detail::overlap_state::overlap
                                                     : detail::overlap_state::unknown;
      return adopt_derived(type, elements, proofs);
    }

    [[nodiscard]] MPI_Datatype native_handle() const noexcept {
      return handle_;
    }

    void swap(layout& other) noexcept {
      std::swap(handle_, other.handle_);
      std::swap(size_, other.size_);
      std::swap(facts_, other.facts_);
    }

  private:
    layout(MPI_Datatype handle, count_t elements, const detail::layout_facts& facts) noexcept
        : handle_{handle}, size_{elements}, facts_{facts} {
    }

    // MPI_Type_create_hindexed / MPI_Type_create_hindexed_block take the blocklength as an
    // int in the C binding; count_t is wider, so reject an out-of-range value before the call.
    [[nodiscard]] static int checked_blocklength_int(count_t value, const context& ctx) {
      if (value < 0 or value > static_cast<count_t>(std::numeric_limits<int>::max())) {
        throw_invalid(ctx.operation, "blocklength does not fit int");
      }
      return static_cast<int>(value);
    }

    [[nodiscard]] static detail::map_element_facts base_element_facts() {
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      return detail::map_element_facts{1,           base.size,    base.true_extent,
                                       base.extent, base.true_lb, base.packed,
                                       true,        base.overlap};
    }

    [[nodiscard]] static detail::map_element_facts element_facts_of(const layout& l) noexcept {
      return detail::map_element_facts{
          l.size_,          l.facts_.byte_size, l.facts_.true_extent,     l.facts_.byte_extent,
          l.facts_.true_lb, l.facts_.packed,    l.facts_.base_repetition, l.facts_.overlap};
    }

    [[nodiscard]] static detail::layout_facts make_structural_facts(
        const detail::repeated_facts& repeated) noexcept {
      detail::layout_facts facts;
      facts.packed = repeated.packed;
      facts.base_repetition = repeated.base_repetition;
      facts.overlap = repeated.overlap;
      return facts;
    }

    // Common two-level algebra for vector/strided_vector:
    //   * `blocklength` copies of `child`, one child extent apart (the inner repetition MPI
    //     builds inside one vector block), then
    //   * `count` such groups, one `outer_byte_stride` apart (the outer repetition).
    // `outer_byte_stride` is always in BYTES for both families. MPI_Type_vector counts its
    // stride in old-type extents, so that caller converts to bytes before calling this
    // helper; MPI_Type_create_hvector takes bytes directly.
    [[nodiscard]] static detail::layout_facts make_repetition_facts(
        const detail::map_element_facts& child, count_t blocklength, count_t count,
        count_t base_extent, count_t outer_byte_stride, const context& ctx) {
      const detail::repeated_facts inner =
          detail::repeat_facts(child, blocklength, child.extent, base_extent, ctx);
      const detail::map_element_facts group{
          detail::checked_count_mul(blocklength, child.size, ctx),
          inner.byte_size,
          inner.true_extent,
          detail::checked_count_mul(blocklength, child.extent, ctx),
          child.true_lb,
          inner.packed,
          inner.base_repetition,
          inner.overlap};
      const detail::repeated_facts outer =
          detail::repeat_facts(group, count, outer_byte_stride, base_extent, ctx);
      return make_structural_facts(outer);
    }

    // Internal adopting path for a handle the library has just created or already owns: the
    // caller has established that it is a derived datatype, so the public predefined-handle
    // envelope query is not repeated on factory paths. It is kept out of the public overload
    // set so `from_raw` has exactly one spelling.
    [[nodiscard]] static layout adopt_derived(MPI_Datatype type, count_t elements,
                                              detail::layout_facts facts) {
      MPI_Datatype committed = type;
      try {
        detail::check_mpi(MPI_Type_commit(&committed), "MPI_Type_commit");
        const detail::layout_facts complete =
            detail::query_layout_geometry(committed, facts, "layout<T>::query");
        return layout{committed, elements, complete};
      } catch (...) {
        if (committed != MPI_DATATYPE_NULL)
          (void)MPI_Type_free(&committed);
        throw;
      }
    }

    static void require_valid_block(const layout& block, const context& ctx) {
      if (not block.is_valid()) {
        throw_invalid(ctx.operation, "block layout is invalid");
      }
    }

    void require_valid(const context& ctx) const {
      if (not is_valid()) {
        throw_invalid(ctx.operation, "invalid layout (no MPI datatype)");
      }
    }

    void release() noexcept {
      detail::release_handle(handle_, MPI_DATATYPE_NULL, true, &detail::free_datatype);
    }

    MPI_Datatype handle_{MPI_DATATYPE_NULL};
    count_t size_{0};
    detail::layout_facts facts_{};
  };

}  // namespace mp1

#endif
