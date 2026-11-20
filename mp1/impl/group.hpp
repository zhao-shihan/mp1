#if !(defined MP1_IMPL_GROUP_HPP)

#define MP1_IMPL_GROUP_HPP

#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/fwd.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>


namespace mp1 {

  // MPI_Comm_compare / MPI_Group_compare result. Declared here because the communicator and
  // group headers both need it and no earlier header defines it.
  enum class equality_type : int {
    identical = MPI_IDENT,
    congruent = MPI_CONGRUENT,
    similar = MPI_SIMILAR,
    unequal = MPI_UNEQUAL
  };

  /// One MPI_Group_range_incl/excl triplet.  The field order is MPI's (first, last, stride).
  ///
  /// The default {0, 0, 1} is a valid one-rank range, so value-initialized arrays/vectors never
  /// contain indeterminate fields.  `stride` must be nonzero; `first > last` is legal with a
  /// negative stride.  Every emitted rank must be a valid, distinct rank of the group (MPI 4.1
  /// §7.3.2); mp1 validates stride, direction and the computed endpoints locally but does not
  /// expand overlapping/duplicate ranges (that would defeat the range form's scalability).
  struct rank_range {
    int first{0};
    int last{0};
    int stride{1};
  };

  namespace detail {
    struct group_factory;

    /// MPI 4.1 §7.3.2: the ranks of MPI_Group_incl/excl must be valid and all distinct.
    inline void require_distinct_ranks(std::span<const int> ranks, const context& ctx) {
      if (ranks.size() < 2)
        return;
      std::vector<int> sorted{ranks.begin(), ranks.end()};
      std::sort(sorted.begin(), sorted.end());
      const auto duplicate = std::adjacent_find(sorted.begin(), sorted.end());
      if (duplicate != sorted.end()) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "duplicate rank {}", *duplicate);
      }
    }

  }  // namespace detail

  /// Owning wrapper around an MPI_Group. A default-constructed group is invalid
  /// (MPI_GROUP_NULL), like every other owning handle; group::empty() is the valid, empty,
  /// non-owned MPI_GROUP_EMPTY group. A moved-from group is invalid.
  ///
  /// Group equality is MPI_IDENT: the same members in the same order.  It is MPI-free when
  /// either operand is invalid or both native handles are equal; distinct valid groups call
  /// MPI_Group_compare and therefore require active MPI. compare() reports only
  /// identical/similar/unequal; MPI_Group_compare cannot return MPI_CONGRUENT.
  class group {
  public:
    /// MPI_GROUP_NULL: invalid, owning nothing. No MPI call.
    group() noexcept = default;

    group(group&& other) noexcept : handle_{other.handle_} {
      other.handle_ = MPI_GROUP_NULL;
    }

    group& operator=(group&& other) noexcept {
      if (this != &other) {
        release();
        handle_ = other.handle_;
        other.handle_ = MPI_GROUP_NULL;
      }
      return *this;
    }

    group(const group&) = delete;
    group& operator=(const group&) = delete;

    /// Frees iff the handle is neither MPI_GROUP_NULL nor MPI_GROUP_EMPTY and MPI is active.
    /// After finalize the handle is abandoned and counted. No MPI_Group_compare is performed.
    ~group() noexcept {
      release();
    }

    /// MPI_GROUP_EMPTY: valid, empty, not owned. No MPI call.
    [[nodiscard]] static group empty() noexcept {
      return group{MPI_GROUP_EMPTY};
    }

    /// \return true if the handle is not MPI_GROUP_NULL (the empty group is valid)
    [[nodiscard]] bool is_valid() const noexcept {
      return handle_ != MPI_GROUP_NULL;
    }

    /// \return true if this is the valid, empty MPI_GROUP_EMPTY group; no MPI call
    [[nodiscard]] bool is_empty() const noexcept {
      return handle_ == MPI_GROUP_EMPTY;
    }

    /// \return number of members; 0 for the empty group. Requires active MPI.
    [[nodiscard]] int size() const {
      require_usable("mp1::group::size");
      int result{0};
      detail::check_mpi(MPI_Group_size(handle_, &result), "MPI_Group_size");
      return result;
    }

    /// \return rank of the calling process in the group, or MPI_UNDEFINED
    [[nodiscard]] int rank() const {
      require_usable("mp1::group::rank");
      int result{MPI_UNDEFINED};
      detail::check_mpi(MPI_Group_rank(handle_, &result), "MPI_Group_rank");
      return result;
    }

    /// MPI_Group_translate_ranks for a single rank; MPI_UNDEFINED when absent from other.
    ///
    /// MPI_PROC_NULL is passed through unchanged (MPI 4.1 permits it in the ranks array)
    /// without being range-checked against this group.
    [[nodiscard]] int translate(int rank, const group& other) const {
      require_usable("mp1::group::translate");
      if (not other.is_valid())
        throw_invalid("mp1::group::translate", "invalid other group");
      if (rank == MPI_PROC_NULL)
        return MPI_PROC_NULL;
      if (rank < 0 or rank >= size()) {
        detail::fail_as<std::invalid_argument>({"mp1::group::translate"},
                                               std::source_location::current(),
                                               "rank {} is outside [0,{})", rank, size());
      }
      int result{MPI_UNDEFINED};
      detail::check_mpi(MPI_Group_translate_ranks(handle_, 1, &rank, other.handle_, &result),
                        "MPI_Group_translate_ranks");
      return result;
    }

    /// MPI_Group_compare; groups return identical/similar/unequal from MPI. Requires active
    /// MPI and two valid groups.
    [[nodiscard]] equality_type compare(const group& other) const {
      require_usable("mp1::group::compare");
      if (not other.is_valid())
        throw_invalid("mp1::group::compare", "invalid other group");
      int result{MPI_UNEQUAL};
      detail::check_mpi(MPI_Group_compare(handle_, other.handle_, &result),
                        "MPI_Group_compare");
      return static_cast<equality_type>(result);
    }

    /// MPI_IDENT equality. Two invalid groups are equal and an invalid/valid pair is unequal
    /// without any MPI call; two valid groups with the same native handle are equal without
    /// any MPI call. Distinct valid groups require active MPI.
    [[nodiscard]] bool operator==(const group& other) const {
      if (not is_valid() or not other.is_valid())
        return not is_valid() and not other.is_valid();
      if (handle_ == other.handle_)
        return true;
      return compare(other) == equality_type::identical;
    }

    [[nodiscard]] bool operator!=(const group& other) const {
      return not(*this == other);
    }

    /// \return non-owning native handle
    [[nodiscard]] MPI_Group native_handle() const noexcept {
      return handle_;
    }

  private:
    explicit group(MPI_Group handle) noexcept : handle_{handle} {
    }

    static bool owned(MPI_Group handle) noexcept {
      return handle != MPI_GROUP_NULL and handle != MPI_GROUP_EMPTY;
    }

    static void free_group(MPI_Group& handle) noexcept {
      (void)MPI_Group_free(&handle);
    }

    void release() noexcept {
      detail::release_handle(handle_, MPI_GROUP_NULL, owned(handle_), &free_group);
    }

    void require_usable(const context& ctx) const {
      if (not is_valid())
        throw_invalid(ctx.operation, "invalid group");
      detail::require_mpi(ctx);
    }

    MPI_Group handle_{MPI_GROUP_NULL};

    friend struct detail::group_factory;
  };

  namespace detail {

    // Single construction point for group factories and get_group().
    struct group_factory {
      static group make(MPI_Group handle) noexcept {
        return group{handle};
      }
    };

    [[nodiscard]] inline std::size_t group_rank_count(std::span<const int> ranks,
                                                      const context& ctx) {
      if (ranks.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid(ctx.operation, "more than INT_MAX ranks");
      }
      return ranks.size();
    }

    [[nodiscard]] inline std::size_t group_range_count(std::span<const rank_range> ranges,
                                                       const context& ctx) {
      if (ranges.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid(ctx.operation, "more than INT_MAX ranges");
      }
      return ranges.size();
    }

    /// Validate one (first, last, stride) triplet without enumerating the progression.
    ///
    /// MPI 4.1 §7.3.2: stride must be nonzero, first must be a valid rank, first > last is legal
    /// with a negative stride, and every emitted rank must be a valid rank of the group.  The
    /// last value itself may be outside the group when it is not emitted (for example
    /// (0, 99, 100) emits only rank 0).  Duplicate/overlapping ranges are not expanded: the
    /// standard makes distinctness an erroneous-program obligation, and expanding would turn
    /// the range form into group_incl.
    inline void require_valid_range(const rank_range& range, int group_size,
                                    const context& ctx) {
      if (range.stride == 0) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "range stride is zero");
      }
      if (range.first < 0 or range.first >= group_size) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "range first rank {} is outside [0,{})",
                                               range.first, group_size);
      }
      if (range.first != range.last and (range.last > range.first) != (range.stride > 0)) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "range (first {}, last {}, stride {}) direction does not match its stride",
            range.first, range.last, range.stride);
      }
      const long long gap = static_cast<long long>(range.last) - range.first;
      const long long steps = gap / range.stride;
      const long long last_emitted = static_cast<long long>(range.first) + steps * range.stride;
      if (last_emitted < 0 or last_emitted >= group_size) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "range (first {}, last {}, stride {}) emits rank {} outside [0,{})", range.first,
            range.last, range.stride, last_emitted, group_size);
      }
    }

    inline void require_valid_ranges(const group& g, std::span<const rank_range> ranges,
                                     const context& ctx) {
      if (ranges.empty())
        return;
      const int group_size = g.size();
      for (const rank_range& range : ranges)
        require_valid_range(range, group_size, ctx);
    }

    // MPI_Group_range_* takes `int ranges[][3]`.  A real array of int[3] elements keeps every
    // access an access to a genuine int[3] object; a flat vector plus a cast would alias an
    // incompatible type.  The deleter makes the allocation RAII-safe.
    struct triple_deleter {
      void operator()(int (*p)[3]) const noexcept {
        delete[] p;
      }
    };

    using triple_array = std::unique_ptr<int[3], triple_deleter>;

    [[nodiscard]] inline triple_array make_triples(std::span<const rank_range> ranges) {
      if (ranges.empty())
        return {};
      triple_array triples{new int[ranges.size()][3]};
      for (std::size_t i = 0; i < ranges.size(); ++i) {
        triples.get()[i][0] = ranges[i].first;
        triples.get()[i][1] = ranges[i].last;
        triples.get()[i][2] = ranges[i].stride;
      }
      return triples;
    }

  }  // namespace detail

  /// Local group of a communicator (MPI_Comm_group). Defined in communicator.hpp because it
  /// needs the complete mpi_communicator type.
  [[nodiscard]] group get_group(const mpi_communicator& c);

  [[nodiscard]] inline group group_union(const group& a, const group& b) {
    if (not a.is_valid() or not b.is_valid())
      throw_invalid("mp1::group_union", "invalid group");
    detail::require_mpi("mp1::group_union");
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_union(a.native_handle(), b.native_handle(), &result),
                      "MPI_Group_union");
    return detail::group_factory::make(result);
  }

  [[nodiscard]] inline group group_intersection(const group& a, const group& b) {
    if (not a.is_valid() or not b.is_valid())
      throw_invalid("mp1::group_intersection", "invalid group");
    detail::require_mpi("mp1::group_intersection");
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_intersection(a.native_handle(), b.native_handle(), &result),
                      "MPI_Group_intersection");
    return detail::group_factory::make(result);
  }

  [[nodiscard]] inline group group_difference(const group& a, const group& b) {
    if (not a.is_valid() or not b.is_valid())
      throw_invalid("mp1::group_difference", "invalid group");
    detail::require_mpi("mp1::group_difference");
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_difference(a.native_handle(), b.native_handle(), &result),
                      "MPI_Group_difference");
    return detail::group_factory::make(result);
  }

  /// MPI_Group_incl. Local; returns an owning group.  `ranks` is a contiguous rank range
  /// (std::array, std::vector, C array or std::span); every rank must be valid and distinct and
  /// is validated before the MPI call (MPI 4.1 §7.3.2).  An empty span yields MPI_GROUP_EMPTY.
  [[nodiscard]] inline group group_incl(const group& g, std::span<const int> ranks) {
    if (not g.is_valid())
      throw_invalid("mp1::group_incl", "invalid group");
    detail::require_mpi("mp1::group_incl");
    const int group_size = g.size();
    for (const int rank : ranks) {
      if (rank < 0 or rank >= group_size) {
        detail::fail_as<std::invalid_argument>({"mp1::group_incl"},
                                               std::source_location::current(),
                                               "rank {} is outside [0,{})", rank, group_size);
      }
    }
    detail::require_distinct_ranks(ranks, "mp1::group_incl");
    const int count = static_cast<int>(detail::group_rank_count(ranks, "group_incl"));
    int dummy{0};
    const int* data = ranks.empty() ? &dummy : ranks.data();
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_incl(g.native_handle(), count, data, &result),
                      "MPI_Group_incl");
    return detail::group_factory::make(result);
  }

  /// MPI_Group_excl. Local; returns an owning group.  `ranks` is a contiguous rank range and
  /// every rank must be valid and distinct (MPI 4.1 §7.3.2); an empty span returns a group
  /// identical to g.
  [[nodiscard]] inline group group_excl(const group& g, std::span<const int> ranks) {
    if (not g.is_valid())
      throw_invalid("mp1::group_excl", "invalid group");
    detail::require_mpi("mp1::group_excl");
    const int group_size = g.size();
    for (const int rank : ranks) {
      if (rank < 0 or rank >= group_size) {
        detail::fail_as<std::invalid_argument>({"mp1::group_excl"},
                                               std::source_location::current(),
                                               "rank {} is outside [0,{})", rank, group_size);
      }
    }
    detail::require_distinct_ranks(ranks, "mp1::group_excl");
    const int count = static_cast<int>(detail::group_rank_count(ranks, "group_excl"));
    int dummy{0};
    const int* data = ranks.empty() ? &dummy : ranks.data();
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_excl(g.native_handle(), count, data, &result),
                      "MPI_Group_excl");
    return detail::group_factory::make(result);
  }

  /// MPI_Group_range_incl. Local; returns an owning group.
  ///
  /// `ranges` is a contiguous range of rank_range triplets (std::array, std::vector or
  /// std::span).  mp1 validates stride, direction and the computed endpoints locally and does
  /// not enumerate the progression; duplicate/overlapping ranges are an MPI-erroneous program
  /// (MPI 4.1 §7.3.2) that is not diagnosed locally.  An empty span yields MPI_GROUP_EMPTY.
  [[nodiscard]] inline group group_range_incl(const group& g,
                                              std::span<const rank_range> ranges) {
    if (not g.is_valid())
      throw_invalid("mp1::group_range_incl", "invalid group");
    detail::require_mpi("mp1::group_range_incl");
    const int count = static_cast<int>(detail::group_range_count(ranges, "group_range_incl"));
    detail::require_valid_ranges(g, ranges, "mp1::group_range_incl");
    const detail::triple_array triples = detail::make_triples(ranges);
    int dummy[3]{};
    int (*native_ranges)[3] = triples ? triples.get() : &dummy;
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_range_incl(g.native_handle(), count, native_ranges, &result),
                      "MPI_Group_range_incl");
    return detail::group_factory::make(result);
  }

  /// MPI_Group_range_excl. Local; returns an owning group.  The validation and no-enumeration
  /// contract are those of group_range_incl; an empty span returns a group identical to g.
  [[nodiscard]] inline group group_range_excl(const group& g,
                                              std::span<const rank_range> ranges) {
    if (not g.is_valid())
      throw_invalid("mp1::group_range_excl", "invalid group");
    detail::require_mpi("mp1::group_range_excl");
    const int count = static_cast<int>(detail::group_range_count(ranges, "group_range_excl"));
    detail::require_valid_ranges(g, ranges, "mp1::group_range_excl");
    const detail::triple_array triples = detail::make_triples(ranges);
    int dummy[3]{};
    int (*native_ranges)[3] = triples ? triples.get() : &dummy;
    MPI_Group result{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Group_range_excl(g.native_handle(), count, native_ranges, &result),
                      "MPI_Group_range_excl");
    return detail::group_factory::make(result);
  }

}  // namespace mp1

#endif
