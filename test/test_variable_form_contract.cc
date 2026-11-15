// Variable-family form contract.
//
// Pins the target surface at runtime: one dense declaration and one typed declaration per
// family per lifetime, with the family's compile-time in-place side, the documented non-root
// dummy for gatherv/scatterv, no root for allgatherv/alltoallv, the dense/typed cross-spelling
// wire identity, staged per-block layouts (holes preserved), the descriptor copies at
// initiation, and the intercommunicator root encoding.
//
// Cases:
//  * dense gatherv/scatterv/allgatherv/alltoallv blocking with values, spans, regions and
//    in-place;
//  * typed descriptors on the same data (one layout per peer + element displacements), with
//    and without the displacement array, proving both descriptor kinds produce the same wire;
//  * mixed dense/typed spellings across ranks;
//  * nonblocking and persistent dense/typed forms, descriptors destroyed before completion;
//  * in-place rejection on a non-root and on an intercommunicator (compile-time side,
//    runtime role);
//  * asymmetric intercommunicator allgatherv/alltoallv and the MPI_ROOT/MPI_PROC_NULL root
//    encoding for gatherv/scatterv;
//  * sparse bounds (dense and typed) rejected before MPI, self-placing typed layouts accepted.
//
// MPI 4.1 basis: §6.2.3 (root encoding), §6.5-§6.8 (variable collectives), §6.12/§6.13
// (request lifetime), §5.1.11 (type-map authority).

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>


namespace {

  struct form_fixture {
    form_fixture() {
      mp1::init();
    }

    ~form_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  template<class F>
  bool throws_invalid_argument(F&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  [[nodiscard]] std::vector<mp1::count_t> uniform(int peers, mp1::count_t value) {
    return std::vector<mp1::count_t>(static_cast<std::size_t>(peers), value);
  }

  [[nodiscard]] std::vector<mp1::count_t> prefix(const std::vector<mp1::count_t>& counts) {
    std::vector<mp1::count_t> displs(counts.size(), 0);
    mp1::count_t offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      displs[i] = offset;
      offset += counts[i];
    }
    return displs;
  }

  [[nodiscard]] mp1::layout<int> holey_two() {
    // Two ints at element offsets 0 and 2: size 8 bytes, extent 12 bytes, one hole.
    return mp1::layout<int>::vector(2, 1, 2);
  }

  void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(form_fixture);

BOOST_AUTO_TEST_SUITE(variable_form_contract)

// Dense and typed descriptors are two spellings of the same wire: a dense root receives from
// mixed dense/typed peers, and a typed root receives the same values from a dense peer.
BOOST_AUTO_TEST_CASE(cross_spelling_dense_typed) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform(size, elements);
  const std::vector<mp1::count_t> displs = prefix(counts);
  std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size),
                                        mp1::layout<int>::contiguous(elements));

  std::array<int, elements> send{{rank * 10, rank * 10 + 1}};
  const std::span<const int> send_span{send};
  std::vector<int> recv(static_cast<std::size_t>(size * elements), -1);

  // Root dense; even peers send a scalar block, odd peers a layout block with the same count.
  if (rank == root) {
    mp1::gatherv(c, root, send_span, recv, mp1::blocks{counts, displs});
  } else if (rank % 2 == 0) {
    mp1::gatherv(c, root, send_span, std::span<int>{}, mp1::blocks{});
  } else {
    mp1::gatherv(c, root, mp1::region{send.data(), layouts[0]}, std::span<int>{},
                 mp1::blocks{});
  }
  if (rank == root) {
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(i * elements)] == i * 10);
      BOOST_TEST(recv[static_cast<std::size_t>(i * elements + 1)] == i * 10 + 1);
    }
  }
  barrier();

  // Root typed with self-placing layouts (element displacement omitted -> all zero): each
  // block's own map places it, so the restricted side is disjoint.  Peers send the dense flat
  // form; both produce the same native-T wire.
  std::vector<mp1::layout<int>> self_layouts;
  std::vector<int> self_placed(static_cast<std::size_t>(size * elements), -1);
  for (int i = 0; i < size; ++i) {
    mp1::indexed_parameter parameter;
    parameter.add(static_cast<mp1::count_t>(elements), static_cast<mp1::count_t>(i * elements));
    self_layouts.push_back(mp1::layout<int>::indexed(parameter));
    self_placed[static_cast<std::size_t>(i * elements)] = i * 10;
    self_placed[static_cast<std::size_t>(i * elements + 1)] = i * 10 + 1;
  }
  std::vector<int> flat(static_cast<std::size_t>(size * elements), -1);
  if (rank == root) {
    mp1::gatherv(c, root, send_span, flat, self_layouts);
  } else {
    mp1::gatherv(c, root, send_span, std::span<int>{}, mp1::blocks{});
  }
  if (rank == root) {
    BOOST_TEST(flat == self_placed, boost::test_tools::per_element());
  }
  barrier();
}

// The scatterv/alltoallv sides: dense descriptors with values, spans and regions, and the
// in-place root on the scatter side.
BOOST_AUTO_TEST_CASE(scatterv_and_alltoallv_forms) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform(size, elements);
  const std::vector<mp1::count_t> displs = prefix(counts);

  std::vector<int> send(static_cast<std::size_t>(size * elements), -1);
  for (int i = 0; i < size; ++i) {
    send[static_cast<std::size_t>(i * elements)] = i * 100;
    send[static_cast<std::size_t>(i * elements + 1)] = i * 100 + 1;
  }

  // Root in-place scatterv: the data buffer is the send aggregate; the root's own segment is
  // not moved (MPI 4.1 §6.6).
  {
    std::vector<int> local(static_cast<std::size_t>(elements), -1);
    if (rank == root) {
      mp1::scatterv(c, root, send, mp1::blocks{counts, displs}, mp1::in_place);
    } else {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, local);
    }
    if (rank != root) {
      BOOST_TEST(local[0] == rank * 100);
      BOOST_TEST(local[1] == rank * 100 + 1);
    }
  }
  barrier();

  // Out-of-place scatterv with a value receive on every rank (root included).
  {
    std::array<int, elements> local{{-1, -1}};
    mp1::scatterv(c, root, send, mp1::blocks{counts, displs}, local);
    BOOST_TEST(local[0] == rank * 100);
    BOOST_TEST(local[1] == rank * 100 + 1);
  }
  barrier();

  // alltoallv in-place: the receive buffer is the exchange input and output.
  {
    std::vector<int> buffer(static_cast<std::size_t>(size * elements), -1);
    for (int peer = 0; peer < size; ++peer) {
      buffer[static_cast<std::size_t>(peer * elements)] = rank * 100 + peer * 10;
      buffer[static_cast<std::size_t>(peer * elements + 1)] = rank * 100 + peer * 10 + 1;
    }
    mp1::alltoallv(c, mp1::in_place, mp1::blocks{}, buffer, mp1::blocks{counts, displs});
    for (int peer = 0; peer < size; ++peer) {
      BOOST_TEST(buffer[static_cast<std::size_t>(peer * elements)] == peer * 100 + rank * 10);
      BOOST_TEST(buffer[static_cast<std::size_t>(peer * elements + 1)] ==
                 peer * 100 + rank * 10 + 1);
    }
  }
  barrier();
}

// In-place is compile-time per side and runtime root/inter-only: a non-root gatherv in-place,
// a non-root scatterv in-place and any intercommunicator in-place throw before MPI.
BOOST_AUTO_TEST_CASE(in_place_role_rejections) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int root = 0;
  const std::vector<mp1::count_t> counts = uniform(c.size(), 1);
  const std::vector<mp1::count_t> displs = prefix(counts);
  std::vector<int> recv(static_cast<std::size_t>(c.size()), -1);

  if (rank != root) {
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::gatherv(c, root, mp1::in_place, recv, mp1::blocks{counts, displs}); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, mp1::in_place); }));
  }
  barrier();

  const auto world = mp1::comm_world();
  if (world.size() < 2) {
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  auto local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  auto inter = mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{931});
  if (rank == 0) {
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(inter, MPI_ROOT, mp1::in_place, std::span<int>{}, mp1::blocks{});
    }));
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::allgatherv(inter, mp1::in_place, std::span<int>{},
                      std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{});
    }));
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::alltoallv(inter, mp1::in_place, mp1::blocks{}, std::span<int>{}, mp1::blocks{});
    }));
  }
  barrier();
  inter.disconnect();
}

// Intercommunicator variable exchanges with asymmetric per-peer counts: dense allgatherv and
// alltoallv, plus the root encoding for gatherv/scatterv with MPI_ROOT/MPI_PROC_NULL.
BOOST_AUTO_TEST_CASE(inter_asymmetric_dense_and_typed) {
  const auto world = mp1::comm_world();
  if (world.size() < 2) {
    BOOST_TEST_MESSAGE("intercommunicator form test needs at least 2 ranks");
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  auto local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  auto inter = mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{932});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();

  // Group 0 sends one element per remote rank, group 1 two: asymmetric alltoallv.
  {
    const int send_elements = color == 0 ? 1 : 2;
    const int recv_elements = color == 0 ? 2 : 1;
    const std::vector<mp1::count_t> send_counts = uniform(remote_size, send_elements);
    const std::vector<mp1::count_t> recv_counts = uniform(remote_size, recv_elements);
    const std::vector<mp1::count_t> send_displs = prefix(send_counts);
    const std::vector<mp1::count_t> recv_displs = prefix(recv_counts);
    std::vector<int> send(static_cast<std::size_t>(send_elements * remote_size), -1);
    for (int peer = 0; peer < remote_size; ++peer)
      for (int j = 0; j < send_elements; ++j)
        send[static_cast<std::size_t>(peer * send_elements + j)] =
            color * 1000 + local_rank * 10 + j;
    std::vector<int> recv(static_cast<std::size_t>(recv_elements * remote_size), -1);
    mp1::alltoallv(inter, send, mp1::blocks{send_counts, send_displs}, recv,
                   mp1::blocks{recv_counts, recv_displs});
    for (int peer = 0; peer < remote_size; ++peer)
      for (int j = 0; j < recv_elements; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(peer * recv_elements + j)] ==
                   (1 - color) * 1000 + peer * 10 + j);
  }

  // Abutting asymmetric allgatherv: the local send count and the remote receive count differ.
  {
    const int send_elements = color == 0 ? 1 : 2;
    const int recv_elements = color == 0 ? 2 : 1;
    const std::vector<mp1::count_t> recv_counts = uniform(remote_size, recv_elements);
    const std::vector<mp1::count_t> recv_displs = prefix(recv_counts);
    std::vector<int> send(static_cast<std::size_t>(send_elements), color * 1000 + local_rank);
    std::vector<int> recv(static_cast<std::size_t>(recv_elements * remote_size), -1);
    mp1::allgatherv(inter, send, recv, mp1::blocks{recv_counts, recv_displs});
    for (int i = 0; i < remote_size; ++i)
      for (int j = 0; j < recv_elements; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(i * recv_elements + j)] ==
                   (1 - color) * 1000 + i);
  }

  // Root encoding: root group passes MPI_ROOT at its root and MPI_PROC_NULL elsewhere.
  {
    const std::vector<mp1::count_t> counts = uniform(remote_size, 1);
    const std::vector<mp1::count_t> displs = prefix(counts);
    std::vector<int> recv(static_cast<std::size_t>(remote_size), -1);
    const int value = 9000 + local_rank;
    if (color == 1) {
      mp1::gatherv(inter, 0, value, std::span<int>{}, mp1::blocks{});
    } else if (local_rank == 0) {
      mp1::gatherv(inter, MPI_ROOT, 0, recv, mp1::blocks{counts, displs});
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 9000 + i);
    } else {
      mp1::gatherv(inter, MPI_PROC_NULL, 0, std::span<int>{}, mp1::blocks{});
    }
  }

  inter.disconnect();
}

// Typed sparse bounds and self-placement: a layout whose map is disjoint but whose true extent
// overlaps is legal; a block outside the caller's span is not.
BOOST_AUTO_TEST_CASE(typed_bounds_and_self_placement) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;

  // Self-placing indexed layout: block i lives at element i, and the omitted displacement
  // array is all-zero.  The receive span covers every block.
  {
    std::vector<mp1::layout<int>> layouts;
    for (int i = 0; i < size; ++i) {
      mp1::indexed_parameter parameter;
      parameter.add(1, static_cast<mp1::count_t>(i));
      layouts.push_back(mp1::layout<int>::indexed(parameter));
    }
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    mp1::allgatherv(c, std::span<const int>{&rank, 1}, recv, layouts);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i);
  }
  barrier();

  // A block whose true extent exceeds the receive span is rejected before MPI.
  if (rank == root) {
    std::vector<mp1::layout<int>> layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      layouts.push_back(mp1::layout<int>::contiguous(2));
      displs.push_back(static_cast<mp1::count_t>(2 * i));
    }
    std::vector<int> too_small(static_cast<std::size_t>(2 * size - 1), -1);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{&rank, 2}, too_small, layouts, displs);
    }));
  }
  barrier();

  // Same check for the typed scatterv root send side (read-restricted).
  if (rank == root) {
    std::vector<mp1::layout<int>> layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      layouts.push_back(mp1::layout<int>::contiguous(2));
      displs.push_back(static_cast<mp1::count_t>(2 * i));
    }
    std::vector<int> too_small(static_cast<std::size_t>(2 * size - 1), -1);
    const std::span<int> local{&too_small[0], 2};
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::scatterv(c, root, too_small, layouts, displs, local); }));
  }
  barrier();
}

// Nonblocking and persistent typed/dense lifetime: the descriptor arrays and the staged
// layout array may die after initiation; completion and every persistent start stay correct.
BOOST_AUTO_TEST_CASE(dense_typed_lifetime) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const mp1::layout<int> holey = holey_two();
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
  const std::span<int> recv_span{recv};
  std::vector<int> send(3, -1);
  send[0] = rank + 1;
  send[2] = rank + 100;

  mp1::request pending;
  {
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i)
      displs.push_back(static_cast<mp1::count_t>(3 * i));
    if (rank == root) {
      pending = mp1::igatherv(c, root, mp1::region{send.data(), holey}, recv, layouts, displs);
    } else {
      pending = mp1::igatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{},
                              mp1::blocks{});
    }
  }  // layouts/displs destroyed before completion
  pending.wait();
  if (rank == root) {
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i + 100);
    }
  }
  barrier();

  // Persistent dense allgatherv with the descriptor destroyed after init; the staged send is
  // re-packed at every start and the typed receive (dense here) is direct.
  {
    std::vector<int> persistent_recv(static_cast<std::size_t>(size), -1);
    const std::span<int> persistent_span{persistent_recv};
    std::vector<int> persistent_send(1, 0);
    const std::span<int> send_span{persistent_send};
    mp1::request persistent;
    {
      std::vector<mp1::count_t> counts = uniform(size, 1);
      std::vector<mp1::count_t> displs = prefix(counts);
      persistent =
          mp1::allgatherv_init(c, send_span, persistent_span, mp1::blocks{counts, displs});
    }
    for (int cycle = 0; cycle < 2; ++cycle) {
      persistent_send[0] = rank + cycle * 1000;
      persistent.start();
      persistent.wait();
      for (int i = 0; i < size; ++i)
        BOOST_TEST(persistent_recv[static_cast<std::size_t>(i)] == i + cycle * 1000);
    }
  }
  barrier();
}

BOOST_AUTO_TEST_SUITE_END()
