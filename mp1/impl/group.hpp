#if !(defined MP1_IMPL_GROUP_HPP)

#define MP1_IMPL_GROUP_HPP

#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/fwd.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
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

  namespace detail {
    struct group_factory;

    /// MPI 4.1 §7.3.2: the ranks of MPI_Group_incl/excl must be valid and all distinct.
    inline void require_distinct_ranks(const std::vector<int>& ranks, const context& ctx) {
      if (ranks.size() < 2)
        return;
      std::vector<int> sorted{ranks};
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

    [[nodiscard]] inline std::size_t group_rank_count(const std::vector<int>& ranks,
                                                      const context& ctx) {
      if (ranks.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw_invalid(ctx.operation, "more than INT_MAX ranks");
      }
      return ranks.size();
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

  [[nodiscard]] inline group group_incl(const group& g, const std::vector<int>& ranks) {
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

  [[nodiscard]] inline group group_excl(const group& g, const std::vector<int>& ranks) {
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

}  // namespace mp1

#endif
