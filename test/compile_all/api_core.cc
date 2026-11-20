// Public-API compile check: environment, error, status, info, group.
//
// This translation unit includes the umbrella and names every public type and
// free function in the family; it is compile-only (an OBJECT library) so no
// MPI program is run.  A member that is advertised but not declared, or a
// header missing from the umbrella, fails the build.  Keep this file in sync
// with the public surface; topology/neighbor/grid and layout<void> are removed
// and do not appear here.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

void api_environment(int& argc, char**& argv) {
  bool flag = mp1::initialized();
  flag = mp1::finalized();
  flag = mp1::available();
  (void)flag;

  const std::pair<int, int> version = mp1::get_version();
  const std::string library = mp1::get_library_version();
  (void)version;
  (void)library;

  mp1::info hardware = mp1::get_hw_resource_info();
  (void)hardware;

  mp1::init();
  mp1::init(argc, argv);
  mp1::init(mp1::threading_mode::single);
  mp1::init(argc, argv, mp1::threading_mode::multiple);
  const mp1::threading_mode queried = mp1::query_thread();
  const mp1::threading_mode offered = mp1::provided();
  (void)queried;
  (void)offered;
  flag = mp1::is_thread_main();
  const std::string name = mp1::processor_name();
  const double now = mp1::wtime();
  const double tick = mp1::wtick();
  flag = mp1::wtime_is_global();
  (void)name;
  (void)now;
  (void)tick;
  (void)flag;
  mp1::finalize();
}

void api_buffer() {
  // Process-buffer attach/detach is count_t-sized (MPI_Buffer_attach_c/detach_c).
  mp1::buffer_attach(nullptr, 0);
  mp1::buffer_attach(mp1::buffer_automatic, std::numeric_limits<mp1::count_t>::max());
  mp1::buffer_attach(nullptr, sizeof(int) * 1024);
  const std::pair<void*, mp1::count_t> detached = mp1::buffer_detach();
  (void)detached;
  mp1::buffer_flush();
  mp1::request r1 = mp1::buffer_iflush();
  (void)r1;
  const mp1::mpi_communicator c;
  mp1::attach_buffer(c, nullptr, 0);
  const std::pair<void*, mp1::count_t> detached_comm = mp1::detach_buffer(c);
  (void)detached_comm;
  mp1::flush_buffer(c);
  mp1::request r2 = mp1::iflush_buffer(c);
  (void)r2;
}

template<class T>
concept has_context_member = requires(T value) { value.context(); };

void api_error() {
  mp1::mpi_error with_operation(MPI_SUCCESS, "mp1::api_probe");
  mp1::mpi_error multi(MPI_ERR_IN_STATUS, "mp1::api_probe", 2);
  const int code = with_operation.code();
  const int error_class = with_operation.error_class();
  const std::string class_name = with_operation.class_name();
  const std::string_view operation = with_operation.operation();
  const int slot = with_operation.slot();
  const std::source_location where = with_operation.where();
  const bool is_class = with_operation.is_class(MPI_SUCCESS);
  const char* message = with_operation.what();
  (void)code;
  (void)error_class;
  (void)class_name;
  (void)operation;
  (void)slot;
  (void)where;
  (void)is_class;
  (void)message;
  (void)multi;

  // The target error layer is exact: operation()/where() replace context(), and no public
  // entry takes a source_location or a `what` parameter.
  static_assert(std::is_same_v<decltype(with_operation.operation()), std::string_view>);
  static_assert(std::is_same_v<decltype(with_operation.where()), std::source_location>);
  static_assert(!has_context_member<mp1::mpi_error>);
  static_assert(std::is_constructible_v<mp1::context, std::string_view>);
  static_assert(std::is_constructible_v<mp1::context, const char*>);
  static_assert(std::is_same_v<decltype(mp1::fail(std::declval<const mp1::context&>(),
                                                  std::source_location::current(), "{}", 1)),
                               void>);
  static_assert(std::is_same_v<decltype(mp1::require(true, std::declval<const mp1::context&>(),
                                                     std::source_location::current(), "{}", 1)),
                               void>);
  static_assert(
      std::is_same_v<decltype(mp1::throw_invalid("mp1::api_probe", "message")), void>);

  // The error layer public helpers (compiled, never executed).
  const mp1::context ctx{"mp1::api_probe"};
  if (false) {
    mp1::fail(ctx, std::source_location::current(), "value {}", 1);
    mp1::require(true, ctx, std::source_location::current(), "message");
    mp1::throw_invalid("mp1::api_probe", "message");
  }
}

void api_tag() {
  // The P2P tag vocabulary: aggregate value 0, wildcard and guaranteed minimum bound.
  static_assert(std::is_aggregate_v<mp1::tag>);
  static_assert(mp1::tag{}.value == 0);
  static_assert(mp1::tag{1} < mp1::tag{2});
  static_assert(mp1::tag{1} == mp1::tag{1});
  static_assert(mp1::any_tag == mp1::tag{MPI_ANY_TAG});
  static_assert(mp1::tag_ub_min == mp1::tag{32767});
  const mp1::tag value = mp1::tag{7};
  const mp1::tag wildcard = mp1::any_tag;
  (void)value;
  (void)wildcard;
}

void api_status() {
  const mp1::status s;
  const int source = s.source();
  const mp1::tag tag = s.tag();
  const int error = s.error();
  const bool cancelled = s.is_cancelled();
  const mp1::layout<int> l = mp1::layout<int>::contiguous(1);
  const mp1::count_t count = s.get_count<int>();
  const mp1::count_t layout_count = s.get_count(l);
  const mp1::count_t elements = s.get_elements<int>();
  const mp1::count_t layout_elements = s.get_elements(l);
  const MPI_Status& native = s.native_handle();
  (void)source;
  (void)tag;
  (void)error;
  (void)cancelled;
  (void)count;
  (void)layout_count;
  (void)elements;
  (void)layout_elements;
  (void)native;
}

void api_info() {
  mp1::info empty;
  mp1::info table;
  table.set("key", "value");
  table.remove("key");
  const std::optional<std::string> value = table.value("key");
  const int keys = table.size();
  const std::string key = table.key(0);
  const MPI_Info native = table.native_handle();
  (void)value;
  (void)keys;
  (void)key;
  (void)native;

  mp1::info copy{empty};
  mp1::info moved{std::move(copy)};
  mp1::infos tables;
  tables.push_back(std::move(moved));
}

void api_group(const mp1::mpi_communicator& world) {
  mp1::group g = mp1::get_group(world);
  const bool valid = g.is_valid();
  const int size = g.size();
  const int rank = g.rank();
  const int translated = g.translate(0, g);
  const mp1::equality_type equality = g.compare(g);
  const MPI_Group native = g.native_handle();
  (void)valid;
  (void)size;
  (void)rank;
  (void)translated;
  (void)equality;
  (void)native;

  const mp1::group included = mp1::group_incl(g, std::array{0});
  const mp1::group excluded = mp1::group_excl(g, std::array{0});
  const mp1::group united = mp1::group_union(included, excluded);
  const mp1::group intersected = mp1::group_intersection(included, excluded);
  const mp1::group different = mp1::group_difference(united, intersected);
  // MPI_Group_range_incl/excl: one MPI-shaped (first, last, stride) triplet list.
  const std::array<mp1::rank_range, 1> ranges{mp1::rank_range{0, 0, 1}};
  const mp1::group range_included = mp1::group_range_incl(g, ranges);
  const mp1::group range_excluded = mp1::group_range_excl(g, ranges);
  (void)range_included;
  (void)range_excluded;
  (void)different;
}

int main() {
  return 0;
}
