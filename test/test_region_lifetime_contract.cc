// Region lifetime and ownership.
//
// The region is two words (data pointer + borrowed layout reference).  A blocking call borrows
// the layout; a nonblocking/persistent initiation duplicates it (MPI_Type_dup, MPI 4.1
// §5.1.10) so the caller's layout may die when the call returns, while the data buffer may not
// (§2.2, §4.4 rule 3).  The deleted rvalue-layout constructor makes `region{p, layout<T>{}}` a
// compile error; the named-layout one-object std::array spelling is the positive form.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_RMA_LIFETIME_WRAP 1
#else
#define MP1_RMA_LIFETIME_WRAP 0
#endif


namespace {

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  [[nodiscard]] int next_rank(int rank, int size) {
    return (rank + 1) % size;
  }

  [[nodiscard]] int previous_rank(int rank, int size) {
    return (rank - 1 + size) % size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  [[nodiscard]] int combiner_of(MPI_Datatype type) {
    int num_integers{0};
    int num_addresses{0};
    int num_datatypes{0};
    int combiner{MPI_COMBINER_NAMED};
    (void)MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
    return combiner;
  }

#if MP1_RMA_LIFETIME_WRAP
  // MPI_Datatype is an integer handle on MPICH and a pointer on Open MPI.  The comparison
  // token is the same integer form mpi_wrap records; it is never converted back to a handle.
  template<class Handle>
  [[nodiscard]] std::uintptr_t handle_token(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>)
      return reinterpret_cast<std::uintptr_t>(handle);
    else
      return static_cast<std::uintptr_t>(handle);
  }
#endif

}  // namespace


struct mp1_region_lifetime_fixture {
  mp1_region_lifetime_fixture() {
    mp1::init();
  }

  ~mp1_region_lifetime_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_region_lifetime_fixture);

BOOST_AUTO_TEST_SUITE(region_lifetime_contract)

BOOST_AUTO_TEST_CASE(region_shape_and_special_members) {
  BOOST_TEST(true);
  using region_type = mp1::region<int>;
  static_assert(std::is_copy_constructible_v<region_type>);
  static_assert(std::is_move_constructible_v<region_type>);
  static_assert(not std::is_copy_assignable_v<region_type>,
                "region holds a reference member; assignment is deleted");
  static_assert(not std::is_move_assignable_v<region_type>);
  static_assert(not std::is_default_constructible_v<region_type>);
  static_assert(not std::is_constructible_v<region_type, int*, mp1::layout<int>&&>,
                "the rvalue-layout constructor is the local dangling guard");

  int data[2] = {1, 2};
  mp1::layout<int> map = mp1::layout<int>::contiguous(2);
  mp1::region<int> r{&data[0], map};
  BOOST_TEST(r.data() == &data[0]);
  BOOST_TEST(r.map().native_handle() == map.native_handle());
  BOOST_TEST(r.map().size() == 2);

  // Copying the region copies the borrow, not the data or the layout.
  mp1::region<int> copy = r;
  BOOST_TEST(copy.data() == r.data());
  BOOST_TEST(copy.map().native_handle() == map.native_handle());

  // A const region is a send-only spelling.
  const mp1::region<const int> const_region{&data[0], map};
  BOOST_TEST(const_region.data() == &data[0]);
  static_assert(not mp1::recv_sink<mp1::region<const int>>);
  static_assert(mp1::send_source<mp1::region<const int>>);
  static_assert(mp1::recv_sink<mp1::region<int>>);
}

BOOST_AUTO_TEST_CASE(blocking_borrow) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  constexpr int tag = 2101;

  // The layout lvalue is in the caller's frame and must outlive the blocking call.
  mp1::layout<int> map = mp1::layout<int>::contiguous(2);
  int data[3] = {0, 0, -1};

  if (rank == 0) {
    data[0] = 41;
    data[1] = 1;
    const mp1::region<const int> send_buffer{&data[0], map};
    mp1::send(world, send_buffer, 1, mp1::tag{tag});
  } else if (rank == 1) {
    mp1::region<int> recv_buffer{&data[0], map};
    (void)mp1::recv(world, recv_buffer, 0, mp1::tag{tag});
    BOOST_TEST(data[0] == 41);
    BOOST_TEST(data[1] == 1);
    BOOST_TEST(data[2] == -1);
  }
  world_barrier();
  BOOST_TEST(map.is_valid());
}

BOOST_AUTO_TEST_CASE(async_duplicates_layout) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  constexpr int tag = 2102;

  int data[3] = {7, 8, 9};
  std::optional<mp1::request> request;

  if (rank == 0) {
    // The initiating layout dies while the request owns a duplicate (§2.2 rule 2).
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(3);
    request.emplace(mp1::isend(world, mp1::region{&data[0], scoped}, 1, mp1::tag{tag}));
    BOOST_TEST(request->is_valid());
  }

  if (rank == 0) {
    (void)request->wait();
  } else if (rank == 1) {
    int received[3] = {0, 0, 0};
    mp1::layout<int> recv_map = mp1::layout<int>::contiguous(3);
    (void)mp1::recv(world, mp1::region{&received[0], recv_map}, 0, mp1::tag{tag});
    BOOST_TEST(received[0] == 7);
    BOOST_TEST(received[1] == 8);
    BOOST_TEST(received[2] == 9);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(async_receive_duplicates_layout) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  constexpr int tag = 2103;

  int received[2] = {0, 0};
  std::optional<mp1::request> request;
  if (rank == 1) {
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(2);
    request.emplace(mp1::irecv(world, mp1::region{&received[0], scoped}, 0, mp1::tag{tag}));
  }
  if (rank == 0) {
    const std::array<int, 2> sent{{11, 12}};
    mp1::send(world, sent, 1, mp1::tag{tag});
  } else if (rank == 1) {
    (void)request->wait();
    BOOST_TEST(received[0] == 11);
    BOOST_TEST(received[1] == 12);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(persistent_duplicates_layout) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  constexpr int tag = 2104;

  std::optional<mp1::request> request;
  int value = 0;
  if (rank == 1) {
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(1);
    request.emplace(mp1::recv_init(world, mp1::region{&value, scoped}, 0, mp1::tag{tag}));
  }
  if (rank == 0) {
    mp1::send(world, 33, 1, mp1::tag{tag});
  } else if (rank == 1) {
    request->start();
    (void)request->wait();
    BOOST_TEST(value == 33);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(persistent_send_duplicates_layout) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const int next = next_rank(rank, size);
  const int previous = previous_rank(rank, size);
  constexpr int tag = 2106;

  int data[3] = {rank * 10, rank * 10 + 1, rank * 10 + 2};
  std::optional<mp1::request> request;
  {
    // The initiating layout dies before start(); the persistent request must own the
    // duplicate and use it for every cycle.
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(3);
    request.emplace(mp1::send_init(world, mp1::region<int>{&data[0], scoped}, next, mp1::tag{tag}));
  }
  BOOST_TEST(request->is_valid());

  for (int cycle = 0; cycle < 2; ++cycle) {
    request->start();
    int received[3] = {-1, -1, -1};
    mp1::layout<int> recv_map = mp1::layout<int>::contiguous(3);
    (void)mp1::recv(world, mp1::region<int>{&received[0], recv_map}, previous, mp1::tag{tag});
    BOOST_TEST(received[0] == previous * 10);
    BOOST_TEST(received[1] == previous * 10 + 1);
    BOOST_TEST(received[2] == previous * 10 + 2);
    (void)request->wait();
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(combined_duplicates_layouts) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const int next = next_rank(rank, size);
  const int previous = previous_rank(rank, size);
  constexpr int tag = 2107;

  int send_data[2] = {rank * 10, rank * 10 + 1};
  int recv_data[2] = {-1, -1};
  std::optional<mp1::request> request;
  {
    // Both region sides duplicate their layout into the one combined request, so both
    // initiating layouts may die before wait().
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(2);
    request.emplace(mp1::isendrecv(world, mp1::region<int>{&send_data[0], scoped}, next, mp1::tag{tag},
                                   mp1::region<int>{&recv_data[0], scoped}, previous, mp1::tag{tag}));
  }
  (void)request->wait();
  BOOST_TEST(recv_data[0] == previous * 10);
  BOOST_TEST(recv_data[1] == previous * 10 + 1);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(combined_replace_duplicates_layout) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const int next = next_rank(rank, size);
  const int previous = previous_rank(rank, size);
  constexpr int tag = 2108;

  int data[2] = {rank * 10, rank * 10 + 1};
  std::optional<mp1::request> request;
  {
    // The single replace buffer is both send and receive; the duplicated layout is the one
    // datatype for both directions and outlives the initiating layout.
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(2);
    request.emplace(mp1::isendrecv_replace(world, mp1::region<int>{&data[0], scoped}, next, mp1::tag{tag},
                                           previous, mp1::tag{tag}));
  }
  (void)request->wait();
  BOOST_TEST(data[0] == previous * 10);
  BOOST_TEST(data[1] == previous * 10 + 1);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(named_layout_one_object_array) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  constexpr int tag = 2105;

  // The one-object spelling: the whole std::array is one occurrence of a layout over the
  // array type.  The named layout is required (a temporary layout is rejected by the deleted
  // rvalue-layout constructor; see test/compile_fail/region_array_temporary_layout.cc).
  using array_type = std::array<int, 3>;
  mp1::layout<array_type> map = mp1::layout<array_type>::contiguous(1);
  array_type values{{5, 6, 7}};

  if (rank == 0) {
    mp1::send(world, mp1::region<const array_type>{&values, map}, 1, mp1::tag{tag});
  } else if (rank == 1) {
    array_type received{{0, 0, 0}};
    (void)mp1::recv(world, mp1::region<array_type>{&received, map}, 0, mp1::tag{tag});
    BOOST_TEST(received == values);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(empty_region_is_one_zero_byte_occurrence) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  constexpr int tag = 2109;

  // Count 1 with a zero-size datatype is a legal zero-byte message; MPI_PROC_NULL never
  // applies the datatype at all.
  mp1::layout<int> empty_map = mp1::layout<int>::contiguous(0);
  int dummy = 0;
  mp1::send(world, mp1::region<int>{&dummy, empty_map}, MPI_PROC_NULL);
  (void)mp1::recv(world, mp1::region<int>{&dummy, empty_map}, MPI_PROC_NULL);
  BOOST_TEST(empty_map.is_valid());
  BOOST_TEST(empty_map.size() == 0);
  BOOST_TEST(empty_map.byte_size() == 0);
  // A zero-size layout is still a derived datatype; a duplicated handle (the request path)
  // reports a non-NAMED combiner on this provider.
  mp1::layout<int> duplicate = empty_map;
  BOOST_TEST(combiner_of(duplicate.native_handle()) != MPI_COMBINER_NAMED);

  if (size < 2)
    return;
  const int next = next_rank(rank, size);
  const int previous = previous_rank(rank, size);

  // An active zero-byte message: null data with count 1 of a zero-size datatype is legal,
  // so the datatype is applied for real instead of being bypassed by MPI_PROC_NULL.
  if (rank == 0) {
    mp1::send(world, mp1::region<int>{nullptr, empty_map}, 1, mp1::tag{tag});
  } else if (rank == 1) {
    const mp1::status received = mp1::recv(world, mp1::region<int>{nullptr, empty_map}, 0, mp1::tag{tag});
    BOOST_TEST(received.get_count(empty_map) == 0);
  }

  // Combined forms with both directions active and an empty region: the zero-byte message
  // still matches on the ring, and sendrecv_replace applies the layout in both directions.
  (void)mp1::sendrecv(world, mp1::region<int>{nullptr, empty_map}, next, mp1::tag{tag + 1},
                      mp1::region<int>{nullptr, empty_map}, previous, mp1::tag{tag + 1});
  (void)mp1::sendrecv_replace(world, mp1::region<int>{nullptr, empty_map}, next, mp1::tag{tag + 2},
                              previous, mp1::tag{tag + 2});

  // The request-owned duplicate of the zero-size layout is used by the nonblocking path.
  mp1::request recv_request =
      mp1::irecv(world, mp1::region<int>{nullptr, empty_map}, previous, mp1::tag{tag + 3});
  mp1::request send_request =
      mp1::isend(world, mp1::region<int>{nullptr, empty_map}, next, mp1::tag{tag + 3});
  (void)send_request.wait();
  (void)recv_request.wait();
  world_barrier();
}

BOOST_AUTO_TEST_CASE(rma_request_duplicates_layouts) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world.rank();
  constexpr int elements = 8;

  mp1::window w = mp1::window::allocate(
      world, {.size = elements * static_cast<MPI_Aint>(sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = 0;
  target[2] = 20;
  target[3] = 21;
  mp1::fence(w);
  world_barrier();

  // The four request-based RMA families duplicate every layout they pass to the native
  // request (the origin/result `region` map and the target layout) into a request-owned
  // payload, so the caller's layout objects may die as soon as the call returns; only the
  // data buffers must outlive completion (MPI 4.1 §5.1.9 /
  // §12.3.5).  Each initiation is wrapped in an inner scope so the caller layouts are
  // destroyed before wait(); the wrap counters then distinguish the caller's own handles from
  // the request-owned duplicates: one MPI_Type_dup per caller layout at initiation, the
  // caller handles freed by their destructors, and exactly the same number of duplicates
  // freed by the payload at completion.
  int rput_origin[2] = {11, 13};
  std::optional<mp1::request> request;
#if MP1_RMA_LIFETIME_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  {
    mp1::layout<int> origin_map = mp1::layout<int>::contiguous(2);
    mp1::layout<int> target_map = mp1::layout<int>::contiguous(2);
#if MP1_RMA_LIFETIME_WRAP
    const std::uintptr_t caller_origin = handle_token(origin_map.native_handle());
    const std::uintptr_t caller_target = handle_token(target_map.native_handle());
#endif
    request.emplace(
        mp1::rput(w, mp1::region<const int>{rput_origin, origin_map}, rank, 0, target_map));
#if MP1_RMA_LIFETIME_WRAP
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 2);
    const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
    BOOST_TEST(duplicate != 0);
    BOOST_TEST(duplicate != caller_origin);
    BOOST_TEST(duplicate != caller_target);
#endif
  }
#if MP1_RMA_LIFETIME_WRAP
  // The initiating layouts died; their own handles are freed, the request-owned duplicates
  // are not (completion has not happened yet).
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 2);
#endif
  (void)request->wait();
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 4);
#endif
  mp1::flush(w, rank);
  mp1::unlock(w, rank);
  world_barrier();
  BOOST_TEST(*static_cast<const volatile int*>(target + 0) == 11);
  BOOST_TEST(*static_cast<const volatile int*>(target + 1) == 13);

  // rget: a region result and a layout target; the result data buffer stays alive.
  int rget_result[2] = {-1, -1};
#if MP1_RMA_LIFETIME_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  {
    mp1::layout<int> result_map = mp1::layout<int>::contiguous(2);
    mp1::layout<int> target_map = mp1::layout<int>::contiguous(2);
    request.emplace(
        mp1::rget(w, mp1::region<int>{rget_result, result_map}, rank, 2, target_map));
#if MP1_RMA_LIFETIME_WRAP
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 2);
#endif
  }
  (void)request->wait();
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 4);
#endif
  mp1::flush(w, rank);
  mp1::unlock(w, rank);
  world_barrier();
  BOOST_TEST(rget_result[0] == 20);
  BOOST_TEST(rget_result[1] == 21);

  // raccumulate: a region origin and a layout target.
  int raccumulate_origin[2] = {2, 3};
#if MP1_RMA_LIFETIME_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  {
    mp1::layout<int> origin_map = mp1::layout<int>::contiguous(2);
    mp1::layout<int> target_map = mp1::layout<int>::contiguous(2);
    request.emplace(mp1::raccumulate(w, mp1::region<const int>{raccumulate_origin, origin_map},
                                     rank, 4, target_map, MPI_SUM));
#if MP1_RMA_LIFETIME_WRAP
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 2);
#endif
  }
  (void)request->wait();
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 4);
#endif
  mp1::flush(w, rank);
  mp1::unlock(w, rank);
  world_barrier();
  BOOST_TEST(*static_cast<const volatile int*>(target + 4) == 2);
  BOOST_TEST(*static_cast<const volatile int*>(target + 5) == 3);

  // rget_accumulate: three independent caller layouts (origin region, result region, target
  // layout), all destroyed before completion.
  int rget_accumulate_origin[2] = {1, 1};
  int rget_accumulate_result[2] = {-1, -1};
#if MP1_RMA_LIFETIME_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  {
    mp1::layout<int> origin_map = mp1::layout<int>::contiguous(2);
    mp1::layout<int> result_map = mp1::layout<int>::contiguous(2);
    mp1::layout<int> target_map = mp1::layout<int>::contiguous(2);
    request.emplace(mp1::rget_accumulate(
        w, mp1::region<const int>{rget_accumulate_origin, origin_map},
        mp1::region<int>{rget_accumulate_result, result_map}, rank, 6, target_map, MPI_SUM));
#if MP1_RMA_LIFETIME_WRAP
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 3);
#endif
  }
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 3);
#endif
  (void)request->wait();
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 6);
#endif
  mp1::flush(w, rank);
  mp1::unlock(w, rank);
  world_barrier();
  BOOST_TEST(rget_accumulate_result[0] == 0);
  BOOST_TEST(rget_accumulate_result[1] == 0);
  BOOST_TEST(*static_cast<const volatile int*>(target + 6) == 1);
  BOOST_TEST(*static_cast<const volatile int*>(target + 7) == 1);

  // A flat request shape has no layout: no request-owned payload and no datatype duplicate.
  int flat_result[2] = {-1, -1};
#if MP1_RMA_LIFETIME_WRAP
  mpi_wrap::reset_counts();
#endif
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  request.emplace(mp1::rget(w, std::span<int>{flat_result, 2}, rank, 0, 2));
  (void)request->wait();
#if MP1_RMA_LIFETIME_WRAP
  BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == 0);
#endif
  mp1::flush(w, rank);
  mp1::unlock(w, rank);
  world_barrier();
  BOOST_TEST(flat_result[0] == 11);
  BOOST_TEST(flat_result[1] == 13);
  w.free();
}

BOOST_AUTO_TEST_SUITE_END()
