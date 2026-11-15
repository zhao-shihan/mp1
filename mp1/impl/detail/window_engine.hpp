#if !(defined MP1_DETAIL_WINDOW_ENGINE_HPP)

#define MP1_DETAIL_WINDOW_ENGINE_HPP

// Internal helpers for window.hpp: the metadata snapshot taken when a borrowed MPI_Win handle
// becomes a view, and the small validity/epoch preconditions shared by every window operation.
// They live here instead of in the public header so window.hpp exposes only the window types
// and operations.  Nothing here performs a
// data-movement or epoch call: query_window_metadata only reads MPI_Win attributes and the
// window group, exactly like the view constructor it serves.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/mpi_call.hpp"

#include <stdexcept>
#include <string>


namespace mp1::detail {

  /// Snapshot of the immutable MPI_Win attributes of a valid handle.
  struct window_metadata {
    void* base{nullptr};
    MPI_Aint byte_size{0};
    int disp_unit{0};
    int create_flavor{0};
    int memory_model{0};
    int rank{0};
    int group_size{0};
  };

  /// Cache MPI_WIN_BASE/SIZE/DISP_UNIT/CREATE_FLAVOR/MODEL plus the local rank and size of the
  /// window group.  The handle must be valid and MPI must be active; any missing attribute or
  /// failed query throws (the caller keeps ownership of a handle it has not adopted yet).
  [[nodiscard]] inline window_metadata query_window_metadata(MPI_Win handle) {
    require_mpi_checked("mp1::mpi_window");

    window_metadata metadata;

    void* base{nullptr};
    int flag{0};
    check_mpi(MPI_Win_get_attr(handle, MPI_WIN_BASE, &base, &flag),
              "MPI_Win_get_attr(MPI_WIN_BASE)");
    if (flag == 0)
      detail::fail_with<std::runtime_error>("mp1::mpi_window", std::source_location::current(),
                                            "MPI_WIN_BASE is not set");
    metadata.base = base;

    MPI_Aint* size{nullptr};
    check_mpi(MPI_Win_get_attr(handle, MPI_WIN_SIZE, &size, &flag),
              "MPI_Win_get_attr(MPI_WIN_SIZE)");
    if (flag == 0)
      detail::fail_with<std::runtime_error>("mp1::mpi_window", std::source_location::current(),
                                            "MPI_WIN_SIZE is not set");
    metadata.byte_size = *size;

    int* unit{nullptr};
    check_mpi(MPI_Win_get_attr(handle, MPI_WIN_DISP_UNIT, &unit, &flag),
              "MPI_Win_get_attr(MPI_WIN_DISP_UNIT)");
    if (flag == 0)
      detail::fail_with<std::runtime_error>("mp1::mpi_window", std::source_location::current(),
                                            "MPI_WIN_DISP_UNIT is not set");
    metadata.disp_unit = *unit;

    int* flavor{nullptr};
    check_mpi(MPI_Win_get_attr(handle, MPI_WIN_CREATE_FLAVOR, &flavor, &flag),
              "MPI_Win_get_attr(MPI_WIN_CREATE_FLAVOR)");
    if (flag == 0)
      detail::fail_with<std::runtime_error>("mp1::mpi_window", std::source_location::current(),
                                            "MPI_WIN_CREATE_FLAVOR is not set");
    metadata.create_flavor = *flavor;

    int* model{nullptr};
    check_mpi(MPI_Win_get_attr(handle, MPI_WIN_MODEL, &model, &flag),
              "MPI_Win_get_attr(MPI_WIN_MODEL)");
    if (flag == 0)
      detail::fail_with<std::runtime_error>("mp1::mpi_window", std::source_location::current(),
                                            "MPI_WIN_MODEL is not set");
    metadata.memory_model = *model;

    MPI_Group group{MPI_GROUP_NULL};
    check_mpi(MPI_Win_get_group(handle, &group), "MPI_Win_get_group");
    try {
      int size_count{0};
      int local_rank{MPI_UNDEFINED};
      check_mpi(MPI_Group_size(group, &size_count), "MPI_Group_size");
      check_mpi(MPI_Group_rank(group, &local_rank), "MPI_Group_rank");
      metadata.group_size = size_count;
      metadata.rank = local_rank;
    } catch (...) {
      (void)MPI_Group_free(&group);
      throw;
    }
    (void)MPI_Group_free(&group);
    return metadata;
  }

  /// The one validity precondition used by every cached accessor.
  inline void require_window_valid(bool valid, const context& ctx) {
    if (not valid)
      throw_invalid(ctx.operation, "invalid window");
  }

  /// Validity plus active MPI: the operations that make an MPI call below the check.
  inline void require_window_active(bool valid, const context& ctx) {
    require_window_valid(valid, ctx);
    require_mpi(ctx);
  }

  /// Dynamic-window precondition: valid, active MPI, and MPI_WIN_FLAVOR_DYNAMIC.
  inline void require_window_dynamic(bool valid, int create_flavor, const context& ctx) {
    require_window_active(valid, ctx);
    if (create_flavor != MPI_WIN_FLAVOR_DYNAMIC)
      throw_invalid(ctx.operation, "window is not dynamic");
  }

  /// Factory preconditions: valid, active MPI, and a non-inter intracommunicator.
  inline void require_window_factory_operand(const mpi_communicator& comm, const context& ctx) {
    require_valid(comm, ctx);
    require_mpi(ctx);
    if (comm.is_inter())
      throw_invalid(ctx.operation, "intercommunicator is not supported");
  }

  /// Factory preconditions: size is a non-negative byte count.
  inline void require_window_size(MPI_Aint size, const context& ctx) {
    if (size < 0) {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "size {} is negative",
                                             static_cast<long long>(size));
    }
  }

  /// Factory preconditions: disp_unit is positive.
  inline void require_window_disp_unit(int disp_unit, const context& ctx) {
    if (disp_unit <= 0) {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "disp_unit {} is not positive", disp_unit);
    }
  }

}  // namespace mp1::detail

#endif
