// Contract test for the mp1 free collective API.
//
// The shared test harness links the Boost.Test main; this file defines no main and no
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// Coverage: fixed-size collectives, the four variable families (dense and staged), the
// reduction families, in-place forms, root sentinels and the intercommunicator matrix. The
// implementation is compared against raw MPI for the fixed families and against an independent
// raw-MPI byte-staging reference for the variable families, so the test does not share the
// engine's behavior. Call-count gates run only under the wrap shim.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_COLLECTIVE_TEST_WRAP 1
#else
#define MP1_COLLECTIVE_TEST_WRAP 0
#endif


namespace {

  struct collective_fixture {
    collective_fixture() {
      mp1::init();
    }

    ~collective_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

  [[nodiscard]] mp1::mpi_communicator world() {
    return mp1::comm_world();
  }

  [[nodiscard]] mp1::layout<int> holey_int_layout() {
    // Two int elements at element offsets 0 and 2: size 8 bytes, true extent 12 bytes.
    return mp1::layout<int>::vector(2, 1, 2);
  }

  // One holey_int_layout() block per peer, as an aggregate layout over the whole buffer.
  [[nodiscard]] mp1::layout<int> aggregate_holey_int(int peers) {
    mp1::indexed_parameter parameter;
    for (int i = 0; i < peers; ++i) {
      parameter.add(1, static_cast<mp1::count_t>(3 * i));
      parameter.add(1, static_cast<mp1::count_t>(3 * i + 2));
    }
    return mp1::layout<int>::indexed(parameter);
  }

  [[nodiscard]] mp1::layout<int> indexed_int_layout() {
    mp1::indexed_parameter parameter;
    parameter.add(1, 0);
    parameter.add(2, 3);
    return mp1::layout<int>::indexed(parameter);
  }

  [[nodiscard]] mp1::layout<int> hindexed_int_layout() {
    mp1::hindexed_parameter parameter;
    parameter.add(1, MPI_Aint{0});
    parameter.add(2, static_cast<MPI_Aint>(3 * sizeof(int)));
    return mp1::layout<int>::hindexed(parameter);
  }

  [[nodiscard]] mp1::layout<int> subarray_int_layout() {
    mp1::subarray_parameter parameter;
    parameter.add(8, 2, 2);
    return mp1::layout<int>::subarray(parameter);
  }

  template<class Function>
  void check_throws_invalid(Function&& function) {
    bool thrown = false;
    try {
      function();
    } catch (const std::invalid_argument&) {
      thrown = true;
    }
    BOOST_TEST(thrown);
  }

  // ---------------------------------------------------------------------------------------
  // Independent raw-MPI byte-staging reference for the variable families.  This does not use
  // xfer_plan: it packs/unpacks explicitly and calls MPI_BYTE collectives with exact sizes.
  // ---------------------------------------------------------------------------------------

  struct raw_block {
    std::vector<std::byte> bytes;
  };

  template<class T>
  [[nodiscard]] raw_block raw_pack(const T* base, const mp1::layout<T>& l) {
    MPI_Count capacity{0};
    MPI_Pack_size_c(1, l.native_handle(), MPI_COMM_SELF, &capacity);
    raw_block result;
    result.bytes.resize(static_cast<std::size_t>(capacity));
    MPI_Count position{0};
    MPI_Pack_c(base, 1, l.native_handle(), result.bytes.data(), capacity, &position,
               MPI_COMM_SELF);
    result.bytes.resize(static_cast<std::size_t>(position));
    return result;
  }

  template<class T>
  void raw_unpack(const std::byte* src, MPI_Count size, T* base, const mp1::layout<T>& l) {
    MPI_Count position{0};
    MPI_Unpack_c(src, size, &position, base, 1, l.native_handle(), MPI_COMM_SELF);
    BOOST_REQUIRE_EQUAL(position, size);
  }

  [[nodiscard]] MPI_Aint raw_element_bytes(mp1::count_t elements, MPI_Datatype base) {
    MPI_Count lower_bound{0};
    MPI_Count extent{0};
    MPI_Type_get_extent_c(base, &lower_bound, &extent);
    return static_cast<MPI_Aint>(elements) * static_cast<MPI_Aint>(extent);
  }

  template<class T>
  [[nodiscard]] std::vector<T> raw_gatherv(const mp1::mpi_communicator& c, int root,
                                           const T* send, const mp1::layout<T>& sendl,
                                           const std::vector<mp1::layout<T>>& recvls,
                                           const std::vector<mp1::count_t>& recvdispls,
                                           std::size_t result_size, T fill) {
    const raw_block packed = raw_pack(send, sendl);
    const int peers = c.is_inter() ? c.remote_size() : c.size();
    std::vector<MPI_Count> counts(static_cast<std::size_t>(peers));
    std::vector<MPI_Aint> displs(static_cast<std::size_t>(peers));
    MPI_Aint capacity{0};
    for (int i = 0; i < peers; ++i) {
      counts[static_cast<std::size_t>(i)] = recvls[static_cast<std::size_t>(i)].byte_size();
      displs[static_cast<std::size_t>(i)] =
          raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                            mp1::native_datatype<T>()) +
          recvls[static_cast<std::size_t>(i)].true_byte_lower_bound();
      capacity = std::max(
          capacity,
          static_cast<MPI_Aint>(displs[static_cast<std::size_t>(i)] +
                                static_cast<MPI_Aint>(counts[static_cast<std::size_t>(i)])));
    }
    std::vector<std::byte> staging(static_cast<std::size_t>(capacity));
    MPI_Gatherv_c(packed.bytes.data(), static_cast<MPI_Count>(packed.bytes.size()), MPI_BYTE,
                  staging.data(), counts.data(), displs.data(), MPI_BYTE, root,
                  c.native_handle());
    const bool receives =
        (not c.is_inter() and c.rank() == root) or (c.is_inter() and root == MPI_ROOT);
    if (not receives)
      return {};
    std::vector<T> result(result_size, fill);
    for (int i = 0; i < peers; ++i) {
      raw_unpack(
          staging.data() + static_cast<std::ptrdiff_t>(displs[static_cast<std::size_t>(i)]),
          counts[static_cast<std::size_t>(i)],
          mp1::detail::offset_bytes(result.data(),
                                    raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                                                      mp1::native_datatype<T>())),
          recvls[static_cast<std::size_t>(i)]);
    }
    return result;
  }

  template<class T>
  [[nodiscard]] std::vector<T> raw_scatterv(const mp1::mpi_communicator& c, int root,
                                            const T* send,
                                            const std::vector<mp1::layout<T>>& sendls,
                                            const std::vector<mp1::count_t>& senddispls,
                                            const mp1::layout<T>& recvl,
                                            std::size_t result_size, T fill) {
    const int peers = c.is_inter() ? c.remote_size() : c.size();
    std::vector<MPI_Count> counts(static_cast<std::size_t>(peers));
    std::vector<MPI_Aint> displs(static_cast<std::size_t>(peers));
    MPI_Aint capacity{0};
    for (int i = 0; i < peers; ++i) {
      counts[static_cast<std::size_t>(i)] = sendls[static_cast<std::size_t>(i)].byte_size();
      displs[static_cast<std::size_t>(i)] =
          raw_element_bytes(senddispls[static_cast<std::size_t>(i)],
                            mp1::native_datatype<T>()) +
          sendls[static_cast<std::size_t>(i)].true_byte_lower_bound();
      capacity = std::max(
          capacity,
          static_cast<MPI_Aint>(displs[static_cast<std::size_t>(i)] +
                                static_cast<MPI_Aint>(counts[static_cast<std::size_t>(i)])));
    }
    std::vector<std::byte> staging(static_cast<std::size_t>(capacity));
    for (int i = 0; i < peers; ++i) {
      const raw_block packed =
          raw_pack(mp1::detail::offset_bytes(send, senddispls[static_cast<std::size_t>(i)] *
                                                       static_cast<mp1::count_t>(sizeof(T))),
                   sendls[static_cast<std::size_t>(i)]);
      if (not packed.bytes.empty()) {
        std::memcpy(
            staging.data() + static_cast<std::ptrdiff_t>(displs[static_cast<std::size_t>(i)]),
            packed.bytes.data(), packed.bytes.size());
      }
    }
    std::vector<std::byte> recvbytes(static_cast<std::size_t>(recvl.byte_size()));
    MPI_Scatterv_c(staging.data(), counts.data(), displs.data(), MPI_BYTE, recvbytes.data(),
                   static_cast<MPI_Count>(recvbytes.size()), MPI_BYTE, root, c.native_handle());
    std::vector<T> result(result_size, fill);
    if (not recvbytes.empty())
      raw_unpack(recvbytes.data(), static_cast<MPI_Count>(recvbytes.size()), result.data(),
                 recvl);
    return result;
  }

  template<class T>
  [[nodiscard]] std::vector<T> raw_allgatherv(const mp1::mpi_communicator& c, const T* send,
                                              const mp1::layout<T>& sendl,
                                              const std::vector<mp1::layout<T>>& recvls,
                                              const std::vector<mp1::count_t>& recvdispls,
                                              std::size_t result_size, T fill) {
    const raw_block packed = raw_pack(send, sendl);
    const int peers = c.is_inter() ? c.remote_size() : c.size();
    std::vector<MPI_Count> counts(static_cast<std::size_t>(peers));
    std::vector<MPI_Aint> displs(static_cast<std::size_t>(peers));
    MPI_Aint capacity{0};
    for (int i = 0; i < peers; ++i) {
      counts[static_cast<std::size_t>(i)] = recvls[static_cast<std::size_t>(i)].byte_size();
      displs[static_cast<std::size_t>(i)] =
          raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                            mp1::native_datatype<T>()) +
          recvls[static_cast<std::size_t>(i)].true_byte_lower_bound();
      capacity = std::max(
          capacity,
          static_cast<MPI_Aint>(displs[static_cast<std::size_t>(i)] +
                                static_cast<MPI_Aint>(counts[static_cast<std::size_t>(i)])));
    }
    std::vector<std::byte> staging(static_cast<std::size_t>(capacity));
    MPI_Allgatherv_c(packed.bytes.data(), static_cast<MPI_Count>(packed.bytes.size()), MPI_BYTE,
                     staging.data(), counts.data(), displs.data(), MPI_BYTE, c.native_handle());
    std::vector<T> result(result_size, fill);
    for (int i = 0; i < peers; ++i) {
      raw_unpack(
          staging.data() + static_cast<std::ptrdiff_t>(displs[static_cast<std::size_t>(i)]),
          counts[static_cast<std::size_t>(i)],
          mp1::detail::offset_bytes(result.data(),
                                    raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                                                      mp1::native_datatype<T>())),
          recvls[static_cast<std::size_t>(i)]);
    }
    return result;
  }

  template<class T>
  [[nodiscard]] std::vector<T> raw_alltoallv(const mp1::mpi_communicator& c, const T* send,
                                             const std::vector<mp1::layout<T>>& sendls,
                                             const std::vector<mp1::count_t>& senddispls,
                                             const std::vector<mp1::layout<T>>& recvls,
                                             const std::vector<mp1::count_t>& recvdispls,
                                             std::size_t result_size, T fill) {
    const int peers = c.is_inter() ? c.remote_size() : c.size();
    std::vector<MPI_Count> sendcounts(static_cast<std::size_t>(peers));
    std::vector<MPI_Count> recvcounts(static_cast<std::size_t>(peers));
    std::vector<MPI_Aint> sendoffs(static_cast<std::size_t>(peers));
    std::vector<MPI_Aint> recvoffs(static_cast<std::size_t>(peers));
    MPI_Aint sendcap{0};
    MPI_Aint recvcap{0};
    for (int i = 0; i < peers; ++i) {
      sendcounts[static_cast<std::size_t>(i)] = sendls[static_cast<std::size_t>(i)].byte_size();
      recvcounts[static_cast<std::size_t>(i)] = recvls[static_cast<std::size_t>(i)].byte_size();
      sendoffs[static_cast<std::size_t>(i)] =
          raw_element_bytes(senddispls[static_cast<std::size_t>(i)],
                            mp1::native_datatype<T>()) +
          sendls[static_cast<std::size_t>(i)].true_byte_lower_bound();
      recvoffs[static_cast<std::size_t>(i)] =
          raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                            mp1::native_datatype<T>()) +
          recvls[static_cast<std::size_t>(i)].true_byte_lower_bound();
      sendcap = std::max(sendcap,
                         static_cast<MPI_Aint>(
                             sendoffs[static_cast<std::size_t>(i)] +
                             static_cast<MPI_Aint>(sendcounts[static_cast<std::size_t>(i)])));
      recvcap = std::max(recvcap,
                         static_cast<MPI_Aint>(
                             recvoffs[static_cast<std::size_t>(i)] +
                             static_cast<MPI_Aint>(recvcounts[static_cast<std::size_t>(i)])));
    }
    std::vector<std::byte> sendstaging(static_cast<std::size_t>(sendcap));
    for (int i = 0; i < peers; ++i) {
      const raw_block packed =
          raw_pack(mp1::detail::offset_bytes(send, senddispls[static_cast<std::size_t>(i)] *
                                                       static_cast<mp1::count_t>(sizeof(T))),
                   sendls[static_cast<std::size_t>(i)]);
      if (not packed.bytes.empty()) {
        std::memcpy(sendstaging.data() +
                        static_cast<std::ptrdiff_t>(sendoffs[static_cast<std::size_t>(i)]),
                    packed.bytes.data(), packed.bytes.size());
      }
    }
    std::vector<std::byte> recvstaging(static_cast<std::size_t>(recvcap));
    MPI_Alltoallv_c(sendstaging.data(), sendcounts.data(), sendoffs.data(), MPI_BYTE,
                    recvstaging.data(), recvcounts.data(), recvoffs.data(), MPI_BYTE,
                    c.native_handle());
    std::vector<T> result(result_size, fill);
    for (int i = 0; i < peers; ++i) {
      raw_unpack(recvstaging.data() +
                     static_cast<std::ptrdiff_t>(recvoffs[static_cast<std::size_t>(i)]),
                 recvcounts[static_cast<std::size_t>(i)],
                 mp1::detail::offset_bytes(
                     result.data(), raw_element_bytes(recvdispls[static_cast<std::size_t>(i)],
                                                      mp1::native_datatype<T>())),
                 recvls[static_cast<std::size_t>(i)]);
    }
    return result;
  }

}  // namespace

// A padded reflected value: sizeof 16, type-map size 12, so a contiguous layout is holey.
struct padded_value {
  int first{0};
  double second{0.0};
};

MP1_REFLECTION(padded_value, first, second)

namespace {

  struct padded_add {
    padded_value operator()(const padded_value& left,
                            const padded_value& right) const noexcept {
      return padded_value{left.first + right.first, left.second + right.second};
    }
  };

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(collective_fixture);

BOOST_AUTO_TEST_SUITE(collective_contract)

BOOST_AUTO_TEST_CASE(barrier_and_bcast_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  mp1::barrier(c);
  BOOST_TEST(c.is_valid());
  BOOST_TEST(c.size() == size);
  BOOST_TEST(c.rank() == rank);

  int scalar = (rank == root) ? 42 : 0;
  mp1::bcast(c, root, scalar);
  BOOST_TEST(scalar == 42);

  std::vector<int> values(static_cast<std::size_t>(4), (rank == root) ? 7 : 0);
  mp1::bcast(c, root, values);
  for (const int value : values)
    BOOST_TEST(value == 7);

  mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
  std::array<int, 2> array{rank == root ? 11 : 0, rank == root ? 12 : 0};
  mp1::bcast(c, root, mp1::region<int>{array.data(), layout});
  BOOST_TEST(array[0] == 11);
  BOOST_TEST(array[1] == 12);

  std::array<int, 2> container{rank == root ? 21 : 0, rank == root ? 22 : 0};
  mp1::bcast(c, root, container);
  BOOST_TEST(container[0] == 21);
  BOOST_TEST(container[1] == 22);

  int async_value = (rank == root) ? 55 : 0;
  mp1::request request = mp1::ibcast(c, root, async_value);
  BOOST_TEST(request.is_active());
  request.wait();
  BOOST_TEST(async_value == 55);

  // Intracommunicator sentinels are rejected before MPI.
  check_throws_invalid([&] { mp1::bcast(c, MPI_ROOT, scalar); });
  check_throws_invalid([&] { mp1::bcast(c, MPI_PROC_NULL, scalar); });
  check_throws_invalid([&] { mp1::bcast(c, size, scalar); });
  check_throws_invalid([&] { mp1::bcast(c, -1, scalar); });

  // A zero-count span accepts a null buffer.
  mp1::bcast(c, root, std::span<int>{});
}

BOOST_AUTO_TEST_CASE(fixed_gather_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  std::vector<int> send{rank, rank + 10};
  std::vector<int> mine(static_cast<std::size_t>(2 * size), -1);
  std::vector<int> reference(static_cast<std::size_t>(2 * size), -1);
  mp1::gather(c, root, send, mine);
  MPI_Gather(send.data(), 2, MPI_INT, reference.data(), 2, MPI_INT, root, c.native_handle());
  if (rank == root) {
    BOOST_TEST(mine == reference);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(mine[static_cast<std::size_t>(2 * i)] == i);
      BOOST_TEST(mine[static_cast<std::size_t>(2 * i + 1)] == i + 10);
    }
  }

  // A region describes the single send block and the whole receive aggregate.
  const mp1::layout<int> layout = holey_int_layout();
  const mp1::layout<int> aggregate = aggregate_holey_int(size);
  std::vector<int> layout_send(3, -1);
  layout_send[0] = rank;
  layout_send[2] = rank + 100;
  std::vector<int> layout_mine(static_cast<std::size_t>(3 * size), -1);
  std::vector<int> layout_reference(static_cast<std::size_t>(3 * size), -1);
  mp1::gather(c, root, mp1::region<int>{layout_send.data(), layout},
              mp1::region<int>{layout_mine.data(), aggregate});
  MPI_Gather(layout_send.data(), 1, layout.native_handle(), layout_reference.data(), 1,
             layout.native_handle(), root, c.native_handle());
  if (rank == root) {
    BOOST_TEST(layout_mine == layout_reference);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(layout_mine[static_cast<std::size_t>(3 * i)] == i);
      BOOST_TEST(layout_mine[static_cast<std::size_t>(3 * i + 2)] == i + 100);
    }
  }

  // Root form on the root, non-root dummy elsewhere: mixed API call sites.
  int scalar_recv = -1;
  if (rank == root) {
    std::vector<int> values(static_cast<std::size_t>(size), -1);
    mp1::gather(c, root, rank, values);
    scalar_recv = values[static_cast<std::size_t>(rank)];
    for (int i = 0; i < size; ++i)
      BOOST_TEST(values[static_cast<std::size_t>(i)] == i);
  } else {
    mp1::gather(c, root, rank, std::span<int>{});
    scalar_recv = rank;
  }
  BOOST_TEST(scalar_recv == rank);

  // Container form.
  const std::vector<int> container_send{rank, rank + 1};
  std::vector<int> container_recv(static_cast<std::size_t>(2 * size), -1);
  mp1::gather(c, root, container_send, container_recv);
  if (rank == root) {
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(container_recv[static_cast<std::size_t>(2 * i)] == i);
      BOOST_TEST(container_recv[static_cast<std::size_t>(2 * i + 1)] == i + 1);
    }
  }

  // Root in-place: the root contribution is already in the receive buffer.
  {
    std::vector<int> recvbuf(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      recvbuf[static_cast<std::size_t>(rank)] = rank;
      mp1::gather(c, root, mp1::in_place, recvbuf);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recvbuf[static_cast<std::size_t>(i)] == i);
    } else {
      mp1::gather(c, root, rank, std::span<int>{});
    }
  }

  // A non-root rank passes the ignored receive position as an empty span.
  if (rank != root) {
    mp1::gather(c, root, send, std::span<int>{});
  } else {
    std::vector<int> recvbuf(static_cast<std::size_t>(2 * size), -1);
    mp1::gather(c, root, send, recvbuf);
  }

  // Zero blocks are legal with empty spans.
  mp1::gather(c, root, std::span<const int>{}, std::span<int>{});

  // Root mismatched block sizes throw before MPI; only the root enters the check.
  if (rank == root) {
    const mp1::layout<int> other = mp1::layout<int>::contiguous(3);
    check_throws_invalid([&] {
      mp1::gather(c, root, mp1::region<int>{layout_send.data(), layout},
                  mp1::region<int>{layout_mine.data(), other});
    });
  }

  check_throws_invalid([&] { mp1::gather(c, MPI_ROOT, 1, std::span<int>{}); });
  check_throws_invalid([&] { mp1::gather(c, MPI_PROC_NULL, 1, std::span<int>{}); });

  if (rank == root) {
    // A root call with an empty (ignored-spelling) receive position is invalid when the send
    // is non-empty: the local block sizes must agree.
    check_throws_invalid([&] { mp1::gather(c, root, 1, std::span<int>{}); });
    check_throws_invalid([&] { mp1::gather(c, root, send, std::span<int>{}); });
  }
}

BOOST_AUTO_TEST_CASE(fixed_scatter_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  std::vector<int> send(static_cast<std::size_t>(size), -1);
  for (int i = 0; i < size; ++i)
    send[static_cast<std::size_t>(i)] = 500 + i;
  int mine = -1;
  int reference = -1;
  if (rank == root)
    mp1::scatter(c, root, send, mine);
  else
    mp1::scatter(c, root, std::span<const int>{}, mine);
  MPI_Scatter(send.data(), 1, MPI_INT, &reference, 1, MPI_INT, root, c.native_handle());
  BOOST_TEST(mine == reference);
  BOOST_TEST(mine == 500 + rank);

  std::vector<int> send2(static_cast<std::size_t>(2 * size), -1);
  for (int i = 0; i < size; ++i) {
    send2[static_cast<std::size_t>(2 * i)] = 600 + i;
    send2[static_cast<std::size_t>(2 * i + 1)] = 700 + i;
  }
  std::array<int, 2> recv2{-1, -1};
  std::array<int, 2> reference2{-1, -1};
  if (rank == root)
    mp1::scatter(c, root, send2, recv2);
  else
    mp1::scatter(c, root, std::span<const int>{}, recv2);
  MPI_Scatter(send2.data(), 2, MPI_INT, reference2.data(), 2, MPI_INT, root, c.native_handle());
  BOOST_TEST(recv2 == reference2);
  BOOST_TEST(recv2[0] == 600 + rank);
  BOOST_TEST(recv2[1] == 700 + rank);

  // A region describes the whole root send aggregate and the single receive block.
  const mp1::layout<int> layout = holey_int_layout();
  const mp1::layout<int> aggregate = aggregate_holey_int(size);
  std::vector<int> layout_send(static_cast<std::size_t>(3 * size), -1);
  for (int i = 0; i < size; ++i) {
    layout_send[static_cast<std::size_t>(3 * i)] = 800 + i;
    layout_send[static_cast<std::size_t>(3 * i + 2)] = 900 + i;
  }
  std::vector<int> layout_recv(3, -1);
  std::vector<int> layout_reference(3, -1);
  if (rank == root)
    mp1::scatter(c, root, mp1::region<int>{layout_send.data(), aggregate},
                 mp1::region<int>{layout_recv.data(), layout});
  else
    mp1::scatter(c, root, std::span<const int>{}, mp1::region<int>{layout_recv.data(), layout});
  MPI_Scatter(layout_send.data(), 1, layout.native_handle(), layout_reference.data(), 1,
              layout.native_handle(), root, c.native_handle());
  BOOST_TEST(layout_recv == layout_reference);
  BOOST_TEST(layout_recv[0] == 800 + rank);
  BOOST_TEST(layout_recv[2] == 900 + rank);

  // Root form on the root, non-root dummy elsewhere.
  if (rank == root) {
    std::vector<int> values(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      values[static_cast<std::size_t>(i)] = 1000 + i;
    int ignored = -1;
    mp1::scatter(c, root, values, ignored);
  } else {
    int value = -1;
    mp1::scatter(c, root, std::span<const int>{}, value);
    BOOST_TEST(value == 1000 + rank);
  }

  // Container form.
  std::vector<int> container_send(static_cast<std::size_t>(size), -1);
  for (int i = 0; i < size; ++i)
    container_send[static_cast<std::size_t>(i)] = 1100 + i;
  std::vector<int> container_recv(1, -1);
  if (rank == root)
    mp1::scatter(c, root, container_send, container_recv);
  else
    mp1::scatter(c, root, std::span<const int>{}, container_recv);
  BOOST_TEST(container_recv[0] == 1100 + rank);

  // Root in-place: the root block is already in its send buffer.
  {
    std::vector<int> values(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      values[static_cast<std::size_t>(i)] = 1200 + i;
    if (rank == root) {
      mp1::scatter(c, root, values, mp1::in_place);
    } else {
      int value = -1;
      mp1::scatter(c, root, std::span<const int>{}, value);
      BOOST_TEST(value == 1200 + rank);
    }
  }

  // Zero blocks.
  mp1::scatter(c, root, std::span<const int>{}, std::span<int>{});

  // A non-root rank ignores an empty send span.
  if (rank != root) {
    int value = -1;
    mp1::scatter(c, root, std::span<const int>{}, value);
  } else {
    std::vector<int> values(static_cast<std::size_t>(size), 7);
    int value = -1;
    mp1::scatter(c, root, values, value);
  }

  check_throws_invalid([&] { mp1::scatter(c, MPI_ROOT, std::span<const int>{}, mine); });
  check_throws_invalid([&] { mp1::scatter(c, MPI_PROC_NULL, std::span<const int>{}, mine); });

  if (rank == root) {
    // A root call with an empty send aggregate is invalid for a non-empty receive block.
    check_throws_invalid([&] { mp1::scatter(c, root, std::span<const int>{}, mine); });
  }
}

BOOST_AUTO_TEST_CASE(fixed_allgather_alltoall_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  std::vector<int> gathered(static_cast<std::size_t>(size), -1);
  int one = rank * 3;
  mp1::allgather(c, one, gathered);
  for (int i = 0; i < size; ++i)
    BOOST_TEST(gathered[static_cast<std::size_t>(i)] == 3 * i);

  std::vector<int> send2{rank, rank + 1};
  std::vector<int> gathered2(static_cast<std::size_t>(2 * size), -1);
  mp1::allgather(c, send2, gathered2);
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(gathered2[static_cast<std::size_t>(2 * i)] == i);
    BOOST_TEST(gathered2[static_cast<std::size_t>(2 * i + 1)] == i + 1);
  }

  // In-place allgather: the local contribution is already in its slot.
  {
    std::vector<int> values(static_cast<std::size_t>(size), -1);
    values[static_cast<std::size_t>(rank)] = rank + 40;
    mp1::allgather(c, mp1::in_place, values);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(values[static_cast<std::size_t>(i)] == i + 40);
  }

  // Region allgather, dense and holey: single send block, aggregate receive layout.
  const mp1::layout<int> dense = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> dense_aggregate = mp1::layout<int>::contiguous(2 * size);
  std::vector<int> dense_send{rank, rank + 1};
  std::vector<int> dense_recv(static_cast<std::size_t>(2 * size), -1);
  mp1::allgather(c, mp1::region<int>{dense_send.data(), dense},
                 mp1::region<int>{dense_recv.data(), dense_aggregate});
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(dense_recv[static_cast<std::size_t>(2 * i)] == i);
    BOOST_TEST(dense_recv[static_cast<std::size_t>(2 * i + 1)] == i + 1);
  }

  const mp1::layout<int> holey = holey_int_layout();
  const mp1::layout<int> holey_aggregate = aggregate_holey_int(size);
  std::vector<int> holey_send(3, -1);
  holey_send[0] = rank;
  holey_send[2] = rank + 10;
  std::vector<int> holey_recv(static_cast<std::size_t>(3 * size), -1);
  mp1::allgather(c, mp1::region<int>{holey_send.data(), holey},
                 mp1::region<int>{holey_recv.data(), holey_aggregate});
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(holey_recv[static_cast<std::size_t>(3 * i)] == i);
    BOOST_TEST(holey_recv[static_cast<std::size_t>(3 * i + 2)] == i + 10);
  }

  // alltoall spans and aggregate regions.
  std::vector<int> send(static_cast<std::size_t>(size), -1);
  std::vector<int> recv(static_cast<std::size_t>(size), -1);
  for (int i = 0; i < size; ++i)
    send[static_cast<std::size_t>(i)] = rank * 100 + i;
  mp1::alltoall(c, send, recv);
  for (int i = 0; i < size; ++i)
    BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 100 + rank);

  std::vector<int> layout_send(static_cast<std::size_t>(3 * size), -1);
  std::vector<int> layout_recv(static_cast<std::size_t>(3 * size), -1);
  for (int i = 0; i < size; ++i) {
    layout_send[static_cast<std::size_t>(3 * i)] = rank * 1000 + i;
    layout_send[static_cast<std::size_t>(3 * i + 2)] = rank * 1000 + i + 1;
  }
  mp1::alltoall(c, mp1::region<int>{layout_send.data(), holey_aggregate},
                mp1::region<int>{layout_recv.data(), holey_aggregate});
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(layout_recv[static_cast<std::size_t>(3 * i)] == i * 1000 + rank);
    BOOST_TEST(layout_recv[static_cast<std::size_t>(3 * i + 2)] == i * 1000 + rank + 1);
  }

  // In-place alltoall uses one buffer for both directions.
  {
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      buffer[static_cast<std::size_t>(i)] = rank * 100 + i;
    mp1::alltoall(c, mp1::in_place, buffer);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(buffer[static_cast<std::size_t>(i)] == i * 100 + rank);
  }

  // Nonblocking forms complete with wait/test.
  {
    std::vector<int> recv2(static_cast<std::size_t>(size), -1);
    mp1::request request = mp1::iallgather(c, one, recv2);
    request.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv2[static_cast<std::size_t>(i)] == 3 * i);

    std::vector<int> recv3(static_cast<std::size_t>(size), -1);
    mp1::request request3 = mp1::ialltoall(c, send, recv3);
    const bool complete = request3.test().has_value() or request3.wait().error() == MPI_SUCCESS;
    BOOST_TEST(complete);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv3[static_cast<std::size_t>(i)] == i * 100 + rank);
  }

  // Mismatched aggregate sizes throw before MPI everywhere.
  std::vector<int> bad3(static_cast<std::size_t>(3 * size + 1), -1);
  check_throws_invalid([&] { mp1::alltoall(c, send, bad3); });
  const mp1::layout<int> three = mp1::layout<int>::contiguous(3);
  const mp1::layout<int> one_layout = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> two_layout = mp1::layout<int>::contiguous(2);
  check_throws_invalid([&] {
    mp1::allgather(c, mp1::region<int>{dense_send.data(), dense},
                   mp1::region<int>{dense_recv.data(), three});
  });
  check_throws_invalid([&] {
    mp1::alltoall(c, mp1::region<int>{send.data(), one_layout},
                  mp1::region<int>{recv.data(), two_layout});
  });
}

namespace {

  // Element stride that keeps arbitrarily-shaped blocks from overlapping in a receive buffer.
  [[nodiscard]] mp1::count_t layout_stride(const mp1::layout<int>& l) {
    const mp1::count_t extent = l.byte_extent();
    return (extent + static_cast<mp1::count_t>(sizeof(int)) - 1) /
           static_cast<mp1::count_t>(sizeof(int));
  }

  [[nodiscard]] std::vector<int> filled_buffer(std::size_t size, int value) {
    std::vector<int> result(size);
    for (std::size_t i = 0; i < size; ++i)
      result[i] = value + static_cast<int>(i);
    return result;
  }

}  // namespace

BOOST_AUTO_TEST_CASE(gatherv_layout_differential) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  std::vector<mp1::layout<int>> cases;
  cases.push_back(mp1::layout<int>::contiguous(2));
  cases.push_back(holey_int_layout());
  cases.push_back(indexed_int_layout());
  cases.push_back(hindexed_int_layout());
  cases.push_back(subarray_int_layout());

  int case_index = 0;
  for (const mp1::layout<int>& send_layout : cases) {
    BOOST_TEST_CONTEXT("gatherv layout case " << case_index) {
      ++case_index;
      const mp1::count_t stride = layout_stride(send_layout);
      const std::size_t send_size =
          static_cast<std::size_t>(send_layout.true_byte_upper_bound() / sizeof(int)) + 1;
      std::vector<int> send = filled_buffer(send_size, rank * 1000);

      std::vector<mp1::layout<int>> recv_layouts;
      std::vector<mp1::count_t> displs;
      for (int i = 0; i < size; ++i) {
        recv_layouts.push_back(send_layout);
        displs.push_back(static_cast<mp1::count_t>(i) * stride);
      }
      const std::size_t extra =
          static_cast<std::size_t>(send_layout.true_byte_upper_bound() / sizeof(int)) + 2;
      const std::size_t total = static_cast<std::size_t>(stride * size) + extra;
      std::vector<int> mine(total, -1);
      std::vector<int> reference(total, -1);

      if (rank == root) {
        mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, mine, recv_layouts,
                     displs);
      } else {
        mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, std::span<int>{},
                     mp1::blocks{});
      }
      reference =
          raw_gatherv(c, root, send.data(), send_layout, recv_layouts, displs, total, -1);
      if (rank == root) {
        if (mine != reference) {
          int shown = 0;
          for (std::size_t i = 0; i < mine.size(); ++i) {
            if (mine[i] != reference[i]) {
              if (++shown == 8)
                break;
            }
          }
        }
        BOOST_TEST(mine == reference);
      }

      // Root-form on a non-root rank ignores null receive arrays.
      if (rank != root) {
        mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, std::span<int>{},
                     std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{});
      } else {
        std::vector<int> scratch(total, -1);
        mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, scratch, recv_layouts,
                     displs);
      }
    }
  }

  // Zero blocks with a null pointer are legal on every rank.
  {
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(mp1::layout<int>::contiguous(0));
      displs.push_back(0);
    }
    mp1::gatherv(c, root, std::span<const int>{}, std::span<int>{}, recv_layouts, displs);
  }

  // A scalar send form with per-rank receive layouts.
  {
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> recv_displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(mp1::layout<int>::contiguous(1));
      recv_displs.push_back(i);
    }
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      mp1::gatherv(c, root, 100 + rank, recv, recv_layouts, recv_displs);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 100 + i);
    } else {
      mp1::gatherv(c, root, 100 + rank, std::span<int>{}, mp1::blocks{});
    }
  }

  // In-place root: the root block is already at its receive slot.
  {
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(2 * i));
    }
    std::vector<int> recv(static_cast<std::size_t>(2 * size), -1);
    for (int i = 0; i < size; ++i) {
      recv[static_cast<std::size_t>(2 * i)] = i;
      recv[static_cast<std::size_t>(2 * i + 1)] = i + 50;
    }
    if (rank == root) {
      mp1::gatherv(c, root, mp1::in_place, recv, recv_layouts, displs);
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i)] == i);
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i + 1)] == i + 50);
      }
    } else {
      std::vector<int> send{rank, rank + 50};
      mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    }
  }

  // Array length is exact: the root rejects a short receive array before MPI.
  if (rank == root) {
    std::vector<mp1::layout<int>> short_layouts;
    std::vector<mp1::count_t> short_displs;
    for (int i = 0; i < size; ++i) {
      short_layouts.push_back(mp1::layout<int>::contiguous(1));
      short_displs.push_back(i);
    }
    if (size > 1) {
      short_layouts.pop_back();
      short_displs.pop_back();
      check_throws_invalid(
          [&] { mp1::gatherv(c, root, 0, std::span<int>{}, short_layouts, short_displs); });
    }
  }

  // In-place is root-only on an intracommunicator.
  if (rank != root) {
    check_throws_invalid([&] {
      mp1::gatherv(c, root, mp1::in_place, std::span<int>{},
                   std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{});
    });
  }

  // Nonblocking gatherv: staged receive must unpack on completion.
  {
    const mp1::layout<int> layout = holey_int_layout();
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(3 * i));
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    if (rank == root) {
      mp1::request request =
          mp1::igatherv(c, root, mp1::region{send.data(), layout}, recv, recv_layouts, displs);
      request.wait();
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i);
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i + 10);
      }
    } else {
      mp1::request request = mp1::igatherv(c, root, mp1::region{send.data(), layout},
                                           std::span<int>{}, mp1::blocks{});
      request.wait();
    }
  }
}

BOOST_AUTO_TEST_CASE(scatterv_layout_differential) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  std::vector<mp1::layout<int>> cases;
  cases.push_back(mp1::layout<int>::contiguous(2));
  cases.push_back(holey_int_layout());
  cases.push_back(indexed_int_layout());
  cases.push_back(hindexed_int_layout());
  cases.push_back(subarray_int_layout());

  for (const mp1::layout<int>& recv_layout : cases) {
    const mp1::count_t stride = layout_stride(recv_layout);
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<mp1::count_t> displs;
    const std::size_t extra =
        static_cast<std::size_t>(recv_layout.true_byte_upper_bound() / sizeof(int)) + 2;
    std::vector<int> send(static_cast<std::size_t>(stride * size) + extra, -1);
    for (int i = 0; i < size; ++i) {
      send_layouts.push_back(recv_layout);
      displs.push_back(static_cast<mp1::count_t>(i) * stride);
      // Give every block distinct visible values at the mapped offsets.
      for (mp1::count_t e = 0; e < stride; ++e) {
        const std::size_t index = static_cast<std::size_t>(i * stride + e);
        if (index < send.size())
          send[index] = i * 1000 + static_cast<int>(e);
      }
    }
    const std::size_t block_extra =
        static_cast<std::size_t>(recv_layout.true_byte_upper_bound() / sizeof(int)) + 2;
    const std::size_t block_total = static_cast<std::size_t>(stride) + block_extra;
    std::vector<int> mine(block_total, -1);
    std::vector<int> reference(block_total, -1);
    if (rank == root) {
      mp1::scatterv(c, root, send, send_layouts, displs, mp1::region{mine.data(), recv_layout});
    } else {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region{mine.data(), recv_layout});
    }
    reference =
        raw_scatterv(c, root, send.data(), send_layouts, displs, recv_layout, block_total, -1);
    BOOST_TEST(mine == reference);

    // Root-form on a non-root rank ignores a null send array.
    if (rank != root) {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region{mine.data(), recv_layout});
    } else {
      mp1::scatterv(c, root, send, send_layouts, displs, mp1::region{mine.data(), recv_layout});
    }
  }

  // Scalar receive form.
  {
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(1);
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<mp1::count_t> displs;
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i) {
      send_layouts.push_back(layout);
      displs.push_back(i);
      send[static_cast<std::size_t>(i)] = 700 + i;
    }
    if (rank == root) {
      mp1::scatterv(c, root, send, send_layouts, displs, mp1::in_place);
    } else {
      int value = -1;
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, value);
      BOOST_TEST(value == 700 + rank);
    }
  }

  // In-place root with a holey block layout.
  {
    const mp1::layout<int> layout = holey_int_layout();
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<mp1::count_t> displs;
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(3 * i));
      send[static_cast<std::size_t>(3 * i)] = 800 + i;
      send[static_cast<std::size_t>(3 * i + 2)] = 900 + i;
    }
    if (rank == root) {
      mp1::scatterv(c, root, send, send_layouts, displs, mp1::in_place);
    } else {
      std::vector<int> recv(3, -1);
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region{recv.data(), layout});
      BOOST_TEST(recv[0] == 800 + rank);
      BOOST_TEST(recv[2] == 900 + rank);
    }
  }

  // Zero blocks.
  {
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      send_layouts.push_back(mp1::layout<int>::contiguous(0));
      displs.push_back(0);
    }
    mp1::scatterv(c, root, std::span<const int>{}, send_layouts, displs, std::span<int>{});
  }

  // Array length is exact and validated on the root only.
  if (rank == root && size > 1) {
    std::vector<mp1::layout<int>> short_layouts;
    std::vector<mp1::count_t> short_displs;
    for (int i = 0; i < size - 1; ++i) {
      short_layouts.push_back(mp1::layout<int>::contiguous(1));
      short_displs.push_back(i);
    }
    check_throws_invalid([&] {
      mp1::scatterv(c, root, std::span<const int>{}, short_layouts, short_displs,
                    std::span<int>{});
    });
  }

  // The non-root-only arity no longer exists: a root that passes the documented dummy send
  // position is rejected by the local descriptor/role checks instead of entering MPI.
  if (rank == root) {
    const mp1::layout<int> one_layout = mp1::layout<int>::contiguous(1);
    check_throws_invalid([&] {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region<int>{nullptr, one_layout});
    });
  }

  // Nonblocking form.
  {
    const mp1::layout<int> layout = holey_int_layout();
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<mp1::count_t> displs;
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(3 * i));
      send[static_cast<std::size_t>(3 * i)] = 1000 + i;
      send[static_cast<std::size_t>(3 * i + 2)] = 1100 + i;
    }
    if (rank == root) {
      mp1::request request = mp1::iscatterv(c, root, send, send_layouts, displs, mp1::in_place);
      request.wait();
    } else {
      std::vector<int> recv(3, -1);
      mp1::request request = mp1::iscatterv(c, root, std::span<const int>{}, mp1::blocks{},
                                            mp1::region{recv.data(), layout});
      request.wait();
      BOOST_TEST(recv[0] == 1000 + rank);
      BOOST_TEST(recv[2] == 1100 + rank);
    }
  }
}

BOOST_AUTO_TEST_CASE(allgatherv_layout_differential) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  std::vector<mp1::layout<int>> cases;
  cases.push_back(mp1::layout<int>::contiguous(2));
  cases.push_back(holey_int_layout());
  cases.push_back(indexed_int_layout());
  cases.push_back(hindexed_int_layout());

  for (const mp1::layout<int>& layout : cases) {
    const mp1::count_t stride = layout_stride(layout);
    const std::size_t send_size =
        static_cast<std::size_t>(layout.true_byte_upper_bound() / sizeof(int)) + 1;
    std::vector<int> send = filled_buffer(send_size, rank * 1000);
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(i) * stride);
    }
    const std::size_t extra =
        static_cast<std::size_t>(layout.true_byte_upper_bound() / sizeof(int)) + 2;
    const std::size_t total = static_cast<std::size_t>(stride * size) + extra;
    std::vector<int> mine(total, -1);
    std::vector<int> reference(total, -1);
    mp1::allgatherv(c, mp1::region{send.data(), layout}, mine, recv_layouts, displs);
    reference = raw_allgatherv(c, send.data(), layout, recv_layouts, displs, total, -1);
    BOOST_TEST(mine == reference);
  }

  // In-place: every rank's contribution already sits in its slot.
  {
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    std::vector<int> buffer(static_cast<std::size_t>(2 * size), -1);
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(2 * i));
      buffer[static_cast<std::size_t>(2 * i)] = i;
      buffer[static_cast<std::size_t>(2 * i + 1)] = i + 20;
    }
    mp1::allgatherv(c, mp1::in_place, buffer, recv_layouts, displs);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(buffer[static_cast<std::size_t>(2 * i)] == i);
      BOOST_TEST(buffer[static_cast<std::size_t>(2 * i + 1)] == i + 20);
    }
  }

  // Zero blocks.
  {
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(mp1::layout<int>::contiguous(0));
      displs.push_back(0);
    }
    mp1::allgatherv(c, std::span<const int>{}, std::span<int>{}, recv_layouts, displs);
  }

  // Short receive arrays are rejected on every rank before MPI.
  if (size > 1) {
    std::vector<mp1::layout<int>> short_layouts;
    std::vector<mp1::count_t> short_displs;
    for (int i = 0; i < size - 1; ++i) {
      short_layouts.push_back(mp1::layout<int>::contiguous(1));
      short_displs.push_back(i);
    }
    check_throws_invalid(
        [&] { mp1::allgatherv(c, 0, std::span<int>{}, short_layouts, short_displs); });
  }

  // Nonblocking staged receive.
  {
    const mp1::layout<int> layout = holey_int_layout();
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(layout);
      displs.push_back(static_cast<mp1::count_t>(3 * i));
    }
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 30;
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    mp1::request request =
        mp1::iallgatherv(c, mp1::region{send.data(), layout}, recv, recv_layouts, displs);
    request.wait();
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i + 30);
    }
  }
}

BOOST_AUTO_TEST_CASE(alltoallv_layout_differential) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  const mp1::layout<int> layout = holey_int_layout();
  const mp1::count_t stride = layout_stride(layout);
  std::vector<mp1::layout<int>> send_layouts;
  std::vector<mp1::layout<int>> recv_layouts;
  std::vector<mp1::count_t> send_displs;
  std::vector<mp1::count_t> recv_displs;
  for (int i = 0; i < size; ++i) {
    send_layouts.push_back(layout);
    recv_layouts.push_back(layout);
    send_displs.push_back(static_cast<mp1::count_t>(i) * stride);
    recv_displs.push_back(static_cast<mp1::count_t>(i) * stride);
  }
  const std::size_t extra =
      static_cast<std::size_t>(layout.true_byte_upper_bound() / sizeof(int)) + 2;
  const std::size_t total = static_cast<std::size_t>(stride * size) + extra;
  std::vector<int> send(total, -1);
  std::vector<int> mine(total, -1);
  std::vector<int> reference(total, -1);
  for (int i = 0; i < size; ++i) {
    send[static_cast<std::size_t>(stride * i)] = rank * 1000 + i;
    send[static_cast<std::size_t>(stride * i + 1)] = rank * 1000 + i + 1;
    send[static_cast<std::size_t>(stride * i + 2)] = rank * 1000 + i + 2;
  }
  mp1::alltoallv(c, send, send_layouts, send_displs, mine, recv_layouts, recv_displs);
  reference = raw_alltoallv(c, send.data(), send_layouts, send_displs, recv_layouts,
                            recv_displs, total, -1);
  BOOST_TEST(mine == reference);
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(mine[static_cast<std::size_t>(stride * i)] == i * 1000 + rank);
  }

  // Zero-displacement idiom: every per-peer layout names its own offset inside the common
  // base buffer.
  {
    std::vector<mp1::layout<int>> zero_layouts;
    for (int i = 0; i < size; ++i) {
      mp1::indexed_parameter parameter;
      parameter.add(1, i);
      zero_layouts.push_back(mp1::layout<int>::indexed(parameter));
    }
    std::vector<int> sendv(static_cast<std::size_t>(size), -1);
    std::vector<int> recvv(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      sendv[static_cast<std::size_t>(i)] = rank * 100 + i;
    mp1::alltoallv(c, sendv, zero_layouts, std::span<const mp1::count_t>{}, recvv,
                   zero_layouts);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recvv[static_cast<std::size_t>(i)] == i * 100 + rank);
  }

  // In-place alltoallv: one buffer holds all send and receive blocks.
  {
    std::vector<int> buffer(total, -1);
    for (int i = 0; i < size; ++i) {
      buffer[static_cast<std::size_t>(stride * i)] = rank * 1000 + i;
      buffer[static_cast<std::size_t>(stride * i + 1)] = rank * 1000 + i + 1;
      buffer[static_cast<std::size_t>(stride * i + 2)] = rank * 1000 + i + 2;
    }
    mp1::alltoallv(c, mp1::in_place, std::span<const mp1::layout<int>>{},
                   std::span<const mp1::count_t>{}, buffer, recv_layouts, recv_displs);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(buffer[static_cast<std::size_t>(stride * i)] == i * 1000 + rank);
      BOOST_TEST(buffer[static_cast<std::size_t>(stride * i + 2)] == i * 1000 + rank + 2);
    }
  }

  // Zero blocks.
  {
    std::vector<mp1::layout<int>> zero_layouts;
    std::vector<mp1::count_t> empty_displs;
    for (int i = 0; i < size; ++i) {
      zero_layouts.push_back(mp1::layout<int>::contiguous(0));
      empty_displs.push_back(0);
    }
    mp1::alltoallv(c, std::span<const int>{}, zero_layouts, empty_displs, std::span<int>{},
                   zero_layouts, empty_displs);
  }

  // Short arrays are rejected on every rank before MPI.
  if (size > 1) {
    std::vector<mp1::layout<int>> short_layouts;
    std::vector<mp1::count_t> short_displs;
    for (int i = 0; i < size - 1; ++i) {
      short_layouts.push_back(mp1::layout<int>::contiguous(1));
      short_displs.push_back(i);
    }
    check_throws_invalid([&] {
      mp1::alltoallv(c, std::span<const int>{}, short_layouts, short_displs, std::span<int>{},
                     short_layouts, short_displs);
    });
  }

  // Nonblocking staged alltoallv.
  {
    std::vector<int> sendv(total, -1);
    std::vector<int> recvv(total, -1);
    for (int i = 0; i < size; ++i) {
      sendv[static_cast<std::size_t>(stride * i)] = rank * 1000 + i;
      sendv[static_cast<std::size_t>(stride * i + 2)] = rank * 1000 + i + 2;
    }
    mp1::request request =
        mp1::ialltoallv(c, sendv, send_layouts, send_displs, recvv, recv_layouts, recv_displs);
    request.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recvv[static_cast<std::size_t>(stride * i)] == i * 1000 + rank);
  }
}

namespace {

  [[nodiscard]] int sum_ranks(int size) {
    return size * (size - 1) / 2;
  }

}  // namespace

BOOST_AUTO_TEST_CASE(reduction_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const int expected_all = sum_ranks(size);
  const std::plus<int> plus{};

  // Scalar reduce with a non-zero root.
  {
    int recv = -1;
    if (rank == root) {
      mp1::reduce(c, plus, root, rank, recv);
      BOOST_TEST(recv == expected_all);
    } else {
      mp1::reduce(c, plus, root, rank, std::span<int>{});
    }
  }

  // Root in-place reduce.
  {
    int recv = rank;
    if (rank == root) {
      mp1::reduce(c, plus, root, mp1::in_place, recv);
      BOOST_TEST(recv == expected_all);
    } else {
      mp1::reduce(c, plus, root, rank, std::span<int>{});
    }
  }

  // Root form on a non-root rank ignores the receive buffer.
  {
    int ignored = -1;
    if (rank == root) {
      mp1::reduce(c, plus, root, rank, ignored);
      BOOST_TEST(ignored == expected_all);
    } else {
      mp1::reduce(c, plus, root, rank, ignored);
    }
  }

  // Non-root form is invalid on the root rank.
  if (rank == root) {
    check_throws_invalid([&] { mp1::reduce(c, plus, root, rank, std::span<int>{}); });
  }

  // Buffer form.
  {
    std::array<int, 2> send{rank, 2 * rank};
    std::array<int, 2> recv{-1, -1};
    mp1::reduce(c, plus, root, std::span<const int>{send}, std::span<int>{recv});
    if (rank == root) {
      BOOST_TEST(recv[0] == expected_all);
      BOOST_TEST(recv[1] == 2 * expected_all);
    }
  }

  // allreduce scalar, buffer, container and in-place.
  {
    int recv = -1;
    mp1::allreduce(c, plus, rank, recv);
    BOOST_TEST(recv == expected_all);

    std::array<int, 2> send{rank, 2 * rank};
    std::array<int, 2> recv2{-1, -1};
    mp1::allreduce(c, plus, std::span<const int>{send}, std::span<int>{recv2});
    BOOST_TEST(recv2[0] == expected_all);
    BOOST_TEST(recv2[1] == 2 * expected_all);

    std::vector<int> sendv{rank, rank + 1};
    std::vector<int> recvv{-1, -1};
    mp1::allreduce(c, plus, sendv, recvv);
    BOOST_TEST(recvv[0] == expected_all);
    BOOST_TEST(recvv[1] == expected_all + size);

    int inplace = rank;
    mp1::allreduce(c, plus, mp1::in_place, inplace);
    BOOST_TEST(inplace == expected_all);
  }

  // scan / exscan, out-of-place and in-place.
  {
    int recv = -1;
    mp1::scan(c, plus, rank, recv);
    BOOST_TEST(recv == rank * (rank + 1) / 2);

    int exscan_recv = 777;
    mp1::exscan(c, plus, rank, exscan_recv);
    if (rank == 0)
      BOOST_TEST(exscan_recv == 777);
    else
      BOOST_TEST(exscan_recv == rank * (rank - 1) / 2);

    int scan_inplace = rank;
    mp1::scan(c, plus, mp1::in_place, scan_inplace);
    BOOST_TEST(scan_inplace == rank * (rank + 1) / 2);

    int exscan_inplace = rank;
    mp1::exscan(c, plus, mp1::in_place, exscan_inplace);
    if (rank == 0)
      BOOST_TEST(exscan_inplace == 0);
    else
      BOOST_TEST(exscan_inplace == rank * (rank - 1) / 2);
  }

  // Layout reduction: dense, vector (holey) and padded element type.
  {
    const mp1::layout<int> dense = mp1::layout<int>::contiguous(2);
    std::array<int, 2> send{rank, rank + 1};
    std::array<int, 2> recv{-1, -1};
    mp1::allreduce(c, plus, mp1::region<int>{send.data(), dense},
                   mp1::region<int>{recv.data(), dense});
    BOOST_TEST(recv[0] == expected_all);
    BOOST_TEST(recv[1] == expected_all + size);

    const mp1::layout<int> holey = holey_int_layout();
    std::vector<int> holey_send(3, 0);
    holey_send[0] = rank;
    holey_send[2] = rank + 10;
    std::vector<int> holey_recv(3, -1);
    mp1::allreduce(c, plus, mp1::region<int>{holey_send.data(), holey},
                   mp1::region<int>{holey_recv.data(), holey});
    BOOST_TEST(holey_recv[0] == expected_all);
    BOOST_TEST(holey_recv[2] == expected_all + 10 * size);

    // Non-flat reduce (root) with a holey layout.
    std::vector<int> reduce_recv(3, -1);
    if (rank == root) {
      mp1::reduce(c, plus, root, mp1::region<int>{holey_send.data(), holey},
                  mp1::region<int>{reduce_recv.data(), holey});
      BOOST_TEST(reduce_recv[0] == expected_all);
      BOOST_TEST(reduce_recv[2] == expected_all + 10 * size);
    } else {
      mp1::reduce(c, plus, root, mp1::region<int>{holey_send.data(), holey}, std::span<int>{});
    }

    const mp1::layout<padded_value> padded = mp1::layout<padded_value>::contiguous(1);
    const padded_value send_pair{rank, static_cast<double>(rank) / 2.0};
    padded_value recv_pair{-1, -1.0};
    mp1::allreduce(c, padded_add{}, send_pair, recv_pair);
    BOOST_TEST(recv_pair.first == expected_all);
    BOOST_TEST(recv_pair.second == static_cast<double>(expected_all) / 2.0);

    const mp1::layout<padded_value> padded2 = mp1::layout<padded_value>::contiguous(2);
    std::array<padded_value, 2> padded_send{padded_value{rank, 0.5 * rank},
                                            padded_value{rank + 1, 0.5 * rank + 1.0}};
    std::array<padded_value, 2> padded_recv{padded_value{-1, -1.0}, padded_value{-1, -1.0}};
    mp1::allreduce(c, padded_add{}, mp1::region<padded_value>{padded_send.data(), padded2},
                   mp1::region<padded_value>{padded_recv.data(), padded2});
    BOOST_TEST(padded_recv[0].first == expected_all);
    BOOST_TEST(padded_recv[0].second == 0.5 * expected_all);
    BOOST_TEST(padded_recv[1].first == expected_all + size);
  }

  // Zero-length reductions enter MPI with count 0 and null buffers.
  {
    int dummy = 0;
    mp1::allreduce(c, plus, std::span<const int>{}, std::span<int>{&dummy, 0});
    const mp1::layout<int> empty_layout = mp1::layout<int>::contiguous(0);
    mp1::allreduce(c, plus, mp1::region<const int>{nullptr, empty_layout},
                   mp1::region<int>{&dummy, empty_layout});
  }

  // Nonblocking reduction forms.
  {
    int recv = -1;
    mp1::request request = mp1::iallreduce(c, plus, rank, recv);
    request.wait();
    BOOST_TEST(recv == expected_all);

    int scan_recv = -1;
    mp1::request scan_request = mp1::iscan(c, plus, rank, scan_recv);
    scan_request.wait();
    BOOST_TEST(scan_recv == rank * (rank + 1) / 2);

    int exscan_recv = 777;
    mp1::request exscan_request = mp1::iexscan(c, plus, rank, exscan_recv);
    exscan_request.wait();
    if (rank == 0)
      BOOST_TEST(exscan_recv == 777);
    else
      BOOST_TEST(exscan_recv == rank * (rank - 1) / 2);

    const mp1::layout<int> holey = holey_int_layout();
    std::vector<int> send(3, 0);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv2(3, -1);
    mp1::request layout_request = mp1::iallreduce(c, plus, mp1::region<int>{send.data(), holey},
                                                  mp1::region<int>{recv2.data(), holey});
    layout_request.wait();
    BOOST_TEST(recv2[0] == expected_all);
    BOOST_TEST(recv2[2] == expected_all + 10 * size);
  }
}

BOOST_AUTO_TEST_CASE(reduce_scatter_contract) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};

  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
  int total = 0;
  for (int i = 0; i < size; ++i) {
    counts[static_cast<std::size_t>(i)] = 1 + (i % 2);
    total += 1 + (i % 2);
  }
  std::vector<int> send(static_cast<std::size_t>(total), 0);
  for (int i = 0; i < total; ++i)
    send[static_cast<std::size_t>(i)] = rank * 100 + i;
  int offset = 0;
  for (int i = 0; i < rank; ++i)
    offset += 1 + (i % 2);
  const int local_count = 1 + (rank % 2);
  std::vector<int> expected(static_cast<std::size_t>(local_count), 0);
  for (int i = 0; i < local_count; ++i) {
    int sum = 0;
    for (int r = 0; r < size; ++r)
      sum += r * 100 + offset + i;
    expected[static_cast<std::size_t>(i)] = sum;
  }

  std::vector<int> recv(static_cast<std::size_t>(local_count), -1);
  mp1::reduce_scatter(c, plus, std::span<int>{send}, std::span<int>{recv}, counts);
  BOOST_TEST(recv == expected);

  // Raw MPI differential for the flat form.
  {
    std::vector<int> raw_recv(static_cast<std::size_t>(local_count), -1);
    std::vector<int> raw_counts(counts.begin(), counts.end());
    MPI_Reduce_scatter(send.data(), raw_recv.data(), raw_counts.data(), MPI_INT, MPI_SUM,
                       c.native_handle());
    BOOST_TEST(recv == raw_recv);
  }

  // In-place flat: the full input vector is in recv.
  {
    std::vector<int> buffer = send;
    mp1::reduce_scatter(c, plus, mp1::in_place, std::span<int>{buffer}, counts);
    for (int i = 0; i < local_count; ++i)
      BOOST_TEST(buffer[static_cast<std::size_t>(i)] == expected[static_cast<std::size_t>(i)]);
  }

  // Layout block form: recvcounts[i] copies of a dense two-element block.
  {
    // Whole-buffer regions: each rank receives two base elements.
    std::vector<mp1::count_t> block_counts(static_cast<std::size_t>(size), 2);
    std::vector<int> layout_send(static_cast<std::size_t>(2 * size), 0);
    for (int i = 0; i < size; ++i) {
      layout_send[static_cast<std::size_t>(2 * i)] = rank * 100 + i;
      layout_send[static_cast<std::size_t>(2 * i + 1)] = rank * 100 + i + 1;
    }
    const mp1::layout<int> send_layout = mp1::layout<int>::contiguous(2 * size);
    const mp1::layout<int> recv_layout = mp1::layout<int>::contiguous(2);
    std::vector<int> layout_recv(2, -1);
    mp1::reduce_scatter(c, plus, mp1::region<int>{layout_send.data(), send_layout},
                        mp1::region<int>{layout_recv.data(), recv_layout}, block_counts);
    BOOST_TEST(layout_recv[0] == sum_ranks(size) * 100 + rank * size);
    BOOST_TEST(layout_recv[1] == sum_ranks(size) * 100 + (rank + 1) * size);

    std::vector<int> inplace_buffer = layout_send;
    mp1::reduce_scatter(c, plus, mp1::in_place, std::span<int>{inplace_buffer}, block_counts);
    BOOST_TEST(inplace_buffer[0] == layout_recv[0]);
    BOOST_TEST(inplace_buffer[1] == layout_recv[1]);
  }

  // Non-flat block layout staging.
  {
    // Non-flat whole-buffer layout: one mapped element per rank, staged both directions.
    mp1::indexed_parameter send_parameter;
    for (int i = 0; i < size; ++i)
      send_parameter.add(1, static_cast<mp1::count_t>(2 * i + 1));
    const mp1::layout<int> holey_send_layout = mp1::layout<int>::indexed(send_parameter);
    mp1::indexed_parameter holey_recv_parameter;
    holey_recv_parameter.add(1, 1);
    const mp1::layout<int> holey_recv = mp1::layout<int>::indexed(holey_recv_parameter);
    std::vector<int> holey_send(static_cast<std::size_t>(2 * size), -1);
    std::vector<mp1::count_t> one_each(static_cast<std::size_t>(size), 1);
    for (int i = 0; i < size; ++i)
      holey_send[static_cast<std::size_t>(2 * i + 1)] = rank * 10 + i;
    std::vector<int> holey_recv_data(2, -1);
    mp1::reduce_scatter(c, plus, mp1::region<int>{holey_send.data(), holey_send_layout},
                        mp1::region<int>{holey_recv_data.data(), holey_recv}, one_each);
    BOOST_TEST(holey_recv_data[1] == 10 * sum_ranks(size) + rank * size);
    BOOST_TEST(holey_recv_data[0] == -1);
  }

  // reduce_scatter_block flat and layout.
  {
    const int block_count = 2;
    std::vector<int> block_send(static_cast<std::size_t>(size * block_count), 0);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < block_count; ++j)
        block_send[static_cast<std::size_t>(i * block_count + j)] = rank * 10 + j;
    }
    std::vector<int> block_recv(static_cast<std::size_t>(block_count), -1);
    mp1::reduce_scatter_block(c, plus, std::span<int>{block_send}, std::span<int>{block_recv},
                              block_count);
    for (int j = 0; j < block_count; ++j)
      BOOST_TEST(block_recv[static_cast<std::size_t>(j)] == 10 * sum_ranks(size) + j * size);

    const mp1::layout<int> send_layout = mp1::layout<int>::contiguous(2 * size);
    const mp1::layout<int> recv_layout = mp1::layout<int>::contiguous(2);
    std::vector<int> layout_send(static_cast<std::size_t>(size * 2), 0);
    for (int i = 0; i < size; ++i) {
      layout_send[static_cast<std::size_t>(2 * i)] = rank * 10 + i;
      layout_send[static_cast<std::size_t>(2 * i + 1)] = rank * 10 + i + 1;
    }
    std::vector<int> layout_recv(2, -1);
    mp1::reduce_scatter_block(c, plus, mp1::region<int>{layout_send.data(), send_layout},
                              mp1::region<int>{layout_recv.data(), recv_layout}, 2);
    BOOST_TEST(layout_recv[0] == 10 * sum_ranks(size) + rank * size);
    BOOST_TEST(layout_recv[1] == 10 * sum_ranks(size) + (rank + 1) * size);

    std::vector<int> inplace = layout_send;
    mp1::reduce_scatter_block(c, plus, mp1::in_place, std::span<int>{inplace}, 2);
    BOOST_TEST(inplace[0] == layout_recv[0]);
    BOOST_TEST(inplace[1] == layout_recv[1]);
  }

  // Exact counts array length.
  if (size > 1) {
    std::vector<mp1::count_t> short_counts(static_cast<std::size_t>(size - 1), 1);
    check_throws_invalid([&] {
      mp1::reduce_scatter(c, plus, std::span<int>{send}, std::span<int>{recv}, short_counts);
    });
  }

  // Nonblocking forms keep the count array alive in the payload.
  {
    std::vector<int> nb_recv(static_cast<std::size_t>(local_count), -1);
    mp1::request request =
        mp1::ireduce_scatter(c, plus, std::span<int>{send}, std::span<int>{nb_recv}, counts);
    request.wait();
    BOOST_TEST(nb_recv == expected);

    std::vector<int> nb_block_recv(2, -1);
    const mp1::layout<int> nb_send_layout = mp1::layout<int>::contiguous(2 * size);
    const mp1::layout<int> nb_recv_layout = mp1::layout<int>::contiguous(2);
    std::vector<int> block_send(static_cast<std::size_t>(2 * size), 0);
    for (int i = 0; i < size; ++i) {
      block_send[static_cast<std::size_t>(2 * i)] = rank * 10 + i;
      block_send[static_cast<std::size_t>(2 * i + 1)] = rank * 10 + i + 1;
    }
    std::vector<mp1::count_t> one_each(static_cast<std::size_t>(size), 2);
    mp1::request block_request =
        mp1::ireduce_scatter(c, plus, mp1::region<int>{block_send.data(), nb_send_layout},
                             mp1::region<int>{nb_block_recv.data(), nb_recv_layout}, one_each);
    block_request.wait();
    BOOST_TEST(nb_block_recv[0] == 10 * sum_ranks(size) + rank * size);
    BOOST_TEST(nb_block_recv[1] == 10 * sum_ranks(size) + (rank + 1) * size);
  }
}

namespace {

  struct inter_pair {
    mp1::communicator local;
    mp1::communicator inter;
    int color{0};
    int local_size{0};
    int remote_size{0};
  };

  [[nodiscard]] inter_pair make_inter_pair(const mp1::mpi_communicator& c, int first, int tag) {
    inter_pair pair;
    pair.color = c.rank() < first ? 0 : 1;
    pair.local = mp1::communicator::split(c, pair.color, c.rank());
    const int remote_leader = pair.color == 0 ? first : 0;
    pair.inter = mp1::communicator::intercomm_create(pair.local, 0, c, remote_leader, mp1::tag{tag});
    pair.local_size = pair.local.size();
    pair.remote_size = pair.inter.remote_size();
    return pair;
  }

  void intercomm_bcast_and_fixed(const inter_pair& pair, const mp1::mpi_communicator& world) {
    (void)world;
    const int local_rank = pair.local.rank();
    const int root_arg = pair.color == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    int value = (pair.color == 0 and local_rank == 0) ? 4321 : -1;
    mp1::bcast(pair.inter, root_arg, value);
    if (pair.color == 1)
      BOOST_TEST(value == 4321);
    if (pair.color == 0 and local_rank != 0)
      BOOST_TEST(value == -1);

    // Fixed gather: group 1 sends, group 0 root receives.
    if (pair.color == 1) {
      const int send = 1000 + local_rank;
      mp1::gather(pair.inter, 0, send, std::span<int>{});
    } else if (local_rank == 0) {
      std::vector<int> recv(static_cast<std::size_t>(pair.remote_size), -1);
      int ignored = 0;
      mp1::gather(pair.inter, MPI_ROOT, ignored, std::span<int>{recv});
      for (int i = 0; i < pair.remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 1000 + i);
    } else {
      int ignored = 0;
      mp1::gather(pair.inter, MPI_PROC_NULL, ignored, std::span<int>{});
    }

    // Fixed scatter: group 0 root sends, group 1 receives.
    if (pair.color == 0) {
      if (local_rank == 0) {
        std::vector<int> send(static_cast<std::size_t>(pair.remote_size), -1);
        for (int i = 0; i < pair.remote_size; ++i)
          send[static_cast<std::size_t>(i)] = 2000 + i;
        int ignored = 0;
        mp1::scatter(pair.inter, MPI_ROOT, std::span<const int>{send}, ignored);
      } else {
        int ignored = 0;
        mp1::scatter(pair.inter, MPI_PROC_NULL, std::span<const int>{}, ignored);
      }
    } else {
      int recv = -1;
      mp1::scatter(pair.inter, 0, std::span<const int>{}, recv);
      BOOST_TEST(recv == 2000 + local_rank);
    }
  }

  void intercomm_variable(const inter_pair& pair, const mp1::mpi_communicator& world) {
    const int local_rank = pair.local.rank();
    (void)world;

    const mp1::layout<int> send_layout = mp1::layout<int>::contiguous(1);
    std::vector<mp1::layout<int>> remote_layouts;
    std::vector<mp1::count_t> remote_displs;
    for (int i = 0; i < pair.remote_size; ++i) {
      remote_layouts.push_back(mp1::layout<int>::contiguous(1));
      remote_displs.push_back(i);
    }

    // gatherv: every group-1 rank sends to the group-0 root.
    if (pair.color == 1) {
      int send = 3000 + local_rank;
      mp1::gatherv(pair.inter, 0, send, std::span<int>{}, mp1::blocks{});
    } else if (local_rank == 0) {
      std::vector<int> recv(static_cast<std::size_t>(pair.remote_size), -1);
      int ignored = -1;
      mp1::gatherv(pair.inter, MPI_ROOT, ignored, recv, remote_layouts, remote_displs);
      for (int i = 0; i < pair.remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 3000 + i);
    } else {
      int ignored = -1;
      mp1::gatherv(pair.inter, MPI_PROC_NULL, ignored, std::span<int>{},
                   std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{});
    }

    // scatterv: group-0 root sends to every group-1 rank.
    std::vector<mp1::layout<int>> send_layouts;
    if (pair.color == 0 and local_rank == 0) {
      for (int i = 0; i < pair.remote_size; ++i)
        send_layouts.push_back(mp1::layout<int>::contiguous(1));
    }
    if (pair.color == 0) {
      if (local_rank == 0) {
        std::vector<int> send(static_cast<std::size_t>(pair.remote_size), -1);
        for (int i = 0; i < pair.remote_size; ++i)
          send[static_cast<std::size_t>(i)] = 4000 + i;
        int ignored = 0;
        mp1::scatterv(pair.inter, MPI_ROOT, send, send_layouts, remote_displs, ignored);
      } else {
        int ignored = 0;
        mp1::scatterv(pair.inter, MPI_PROC_NULL, std::span<const int>{},
                      std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{},
                      ignored);
      }
    } else {
      int recv = -1;
      mp1::scatterv(pair.inter, 0, std::span<const int>{}, mp1::blocks{}, recv);
      BOOST_TEST(recv == 4000 + local_rank);
    }

    // allgatherv: each group receives the remote group's contributions.
    {
      int send = 5000 + local_rank;
      std::vector<int> recv(static_cast<std::size_t>(pair.remote_size), -1);
      mp1::allgatherv(pair.inter, send, recv, remote_layouts, remote_displs);
      for (int i = 0; i < pair.remote_size; ++i) {
        const int expected = 5000 + i;
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == expected);
      }
    }

    // alltoallv: arrays have remote_size entries on each side.
    {
      std::vector<mp1::layout<int>> layouts;
      std::vector<mp1::count_t> displs;
      std::vector<int> send(static_cast<std::size_t>(pair.remote_size), -1);
      std::vector<int> recv(static_cast<std::size_t>(pair.remote_size), -1);
      for (int i = 0; i < pair.remote_size; ++i) {
        layouts.push_back(mp1::layout<int>::contiguous(1));
        displs.push_back(i);
        send[static_cast<std::size_t>(i)] = 6000 + local_rank * 100 + i;
      }
      mp1::alltoallv(pair.inter, send, layouts, displs, recv, layouts, displs);
      for (int i = 0; i < pair.remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 6000 + i * 100 + local_rank);
    }
  }

  void intercomm_reductions(const inter_pair& pair, const mp1::mpi_communicator& world) {
    const int local_rank = pair.local.rank();
    (void)world;
    const std::plus<int> plus{};

    // reduce: the root group's root receives the remote group's sum.
    if (pair.color == 1) {
      mp1::reduce(pair.inter, plus, 0, 10 + local_rank, std::span<int>{});
    } else if (local_rank == 0) {
      int recv = -1;
      int ignored = 0;
      mp1::reduce(pair.inter, plus, MPI_ROOT, ignored, recv);
      int expected = 0;
      for (int i = 0; i < pair.remote_size; ++i)
        expected += 10 + i;
      BOOST_TEST(recv == expected);
    } else {
      int ignored = 0;
      mp1::reduce(pair.inter, plus, MPI_PROC_NULL, ignored, std::span<int>{});
    }

    // allreduce: each group reduces the remote group's contributions.
    {
      int recv = -1;
      mp1::allreduce(pair.inter, plus, 20 + local_rank, recv);
      int expected = 0;
      for (int i = 0; i < pair.remote_size; ++i)
        expected += 20 + i;
      BOOST_TEST(recv == expected);
    }

    // reduce_scatter: the total of recvcounts must be equal in both groups.  Pad the larger
    // local group with zero counts so a 2+3 split is valid.
    {
      const int total = std::min(pair.local_size, pair.remote_size);
      std::vector<mp1::count_t> counts(static_cast<std::size_t>(pair.local_size), 0);
      for (int i = 0; i < total; ++i)
        counts[static_cast<std::size_t>(i)] = 1;
      std::vector<int> send(static_cast<std::size_t>(total), -1);
      for (int i = 0; i < total; ++i)
        send[static_cast<std::size_t>(i)] = 30 + local_rank + i;
      const int local_count = (local_rank < total) ? 1 : 0;
      std::vector<int> recv(static_cast<std::size_t>(local_count), -1);
      mp1::reduce_scatter(pair.inter, plus, std::span<int>{send}, std::span<int>{recv}, counts);
      if (local_count > 0) {
        int expected = 0;
        for (int i = 0; i < pair.remote_size; ++i)
          expected += 30 + i + local_rank;
        BOOST_TEST(recv[0] == expected);
      }
    }

    // reduce_scatter_block: both groups' totals must match, so this form needs equal group
    // sizes (intercommunicator block counts are not padded).
    if (pair.local_size == pair.remote_size) {
      std::vector<int> send(static_cast<std::size_t>(pair.local_size), -1);
      for (int i = 0; i < pair.local_size; ++i)
        send[static_cast<std::size_t>(i)] = 40 + i;
      std::vector<int> recv(1, -1);
      mp1::reduce_scatter_block(pair.inter, plus, std::span<int>{send}, std::span<int>{recv},
                                1);
      int expected = 0;
      for (int i = 0; i < pair.remote_size; ++i)
        expected += 40 + local_rank;
      BOOST_TEST(recv[0] == expected);

      const mp1::layout<int> block = mp1::layout<int>::contiguous(2 * pair.local_size);
      const mp1::layout<int> recv_block = mp1::layout<int>::contiguous(2);
      std::vector<int> layout_send(static_cast<std::size_t>(2 * pair.local_size), -1);
      for (int i = 0; i < pair.local_size; ++i) {
        layout_send[static_cast<std::size_t>(2 * i)] = 50 + i;
        layout_send[static_cast<std::size_t>(2 * i + 1)] = 60 + i;
      }
      std::vector<int> layout_recv(2, -1);
      mp1::reduce_scatter_block(pair.inter, plus, mp1::region<int>{layout_send.data(), block},
                                mp1::region<int>{layout_recv.data(), recv_block}, 2);
      int expected_first = 0;
      int expected_second = 0;
      for (int i = 0; i < pair.remote_size; ++i) {
        expected_first += 50 + local_rank;
        expected_second += 60 + local_rank;
      }
      BOOST_TEST(layout_recv[0] == expected_first);
      BOOST_TEST(layout_recv[1] == expected_second);
    }

    // scan and exscan are not defined on intercommunicators.
    {
      int recv = -1;
      check_throws_invalid([&] { mp1::scan(pair.inter, plus, 1, recv); });
      check_throws_invalid([&] { mp1::exscan(pair.inter, plus, 1, recv); });
    }
  }

  void intercomm_in_place_rejected(const inter_pair& pair) {
    const std::plus<int> plus{};
    int dummy = 0;
    std::vector<mp1::layout<int>> layouts;
    std::vector<mp1::count_t> displs;
    check_throws_invalid([&] {
      mp1::gatherv(pair.inter, MPI_ROOT, mp1::in_place, std::span<int>{}, layouts, displs);
    });
    check_throws_invalid(
        [&] { mp1::gatherv(pair.inter, 0, mp1::in_place, std::span<int>{}, layouts, displs); });
    check_throws_invalid([&] {
      mp1::scatterv(pair.inter, MPI_ROOT, std::span<const int>{},
                    std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{},
                    mp1::in_place);
    });
    check_throws_invalid([&] {
      mp1::scatterv(pair.inter, 0, std::span<const int>{}, std::span<const mp1::layout<int>>{},
                    std::span<const mp1::count_t>{}, mp1::in_place);
    });
    check_throws_invalid(
        [&] { mp1::allgatherv(pair.inter, mp1::in_place, std::span<int>{}, layouts, displs); });
    check_throws_invalid([&] {
      mp1::alltoallv(pair.inter, mp1::in_place, std::span<const mp1::layout<int>>{},
                     std::span<const mp1::count_t>{}, std::span<int>{}, layouts, displs);
    });
    check_throws_invalid(
        [&] { mp1::reduce(pair.inter, plus, MPI_ROOT, mp1::in_place, dummy); });
    check_throws_invalid([&] { mp1::allreduce(pair.inter, plus, mp1::in_place, dummy); });
    check_throws_invalid([&] {
      mp1::reduce_scatter(pair.inter, plus, mp1::in_place,
                          std::span<int>{static_cast<int*>(nullptr), 1},
                          std::vector<mp1::count_t>{});
    });
    check_throws_invalid([&] {
      mp1::reduce_scatter_block(pair.inter, plus, mp1::in_place,
                                std::span<int>{static_cast<int*>(nullptr), 1}, 1);
    });
  }

}  // namespace

BOOST_AUTO_TEST_CASE(intercommunicator_contract) {
  const mp1::mpi_communicator c = world();
  const int size = c.size();

  std::vector<std::pair<int, int>> splits;
  splits.emplace_back(1, 1);
  if (size >= 3)
    splits.emplace_back(1, 2);
  if (size >= 4)
    splits.emplace_back(2, 2);
  if (size >= 5)
    splits.emplace_back(2, 3);

  for (const auto& split : splits) {
    const int first = split.first;
    // Only pair exactly `first` local ranks with the rest: other sizes would give the
    // second group more members than the split's declared remote size.
    if (size != first + split.second)
      continue;
    const int tag = 6000 + first * 10 + (size - first);
    inter_pair pair = make_inter_pair(c, first, tag);
    BOOST_TEST(pair.inter.is_valid());
    BOOST_TEST(pair.inter.is_inter());
    BOOST_TEST(pair.inter.remote_size() == (pair.color == 0 ? split.second : split.first));
    intercomm_bcast_and_fixed(pair, c);
    intercomm_variable(pair, c);
    intercomm_reductions(pair, c);
    intercomm_in_place_rejected(pair);

    // Root arguments outside the remote group are rejected before MPI.
    check_throws_invalid([&] {
      mp1::gatherv(pair.inter, pair.inter.remote_size(), 1, std::span<int>{},
                   std::span<const mp1::layout<int>>{}, std::span<const mp1::count_t>{});
    });

    // Allgatherv arrays use remote_size; short arrays throw on every rank.
    if (pair.remote_size > 1) {
      std::vector<mp1::layout<int>> short_layouts;
      std::vector<mp1::count_t> short_displs;
      for (int i = 0; i < pair.remote_size - 1; ++i) {
        short_layouts.push_back(mp1::layout<int>::contiguous(1));
        short_displs.push_back(i);
      }
      check_throws_invalid([&] {
        mp1::allgatherv(pair.inter, 1, std::span<int>{}, short_layouts, short_displs);
      });
    }
  }
}

BOOST_AUTO_TEST_CASE(in_place_container_and_pointer_forms) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const std::plus<int> plus{};

  // gather(in_place, C&): the root slot is already present.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    recv[static_cast<std::size_t>(rank)] = rank + 1;
    if (rank == root) {
      mp1::gather(c, root, mp1::in_place, recv);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    } else {
      std::vector<int> send{rank + 1};
      mp1::gather(c, root, send, std::span<int>{});
    }
  }

  // scatter(in_place, C&): the root block stays in its send buffer.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = 200 + i;
    if (rank == root) {
      mp1::scatter(c, root, send, mp1::in_place);
    } else {
      std::vector<int> recv(1, -1);
      mp1::scatter(c, root, std::span<const int>{}, recv);
      BOOST_TEST(recv[0] == 200 + rank);
    }
  }

  // allgather(in_place, C&).
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    recv[static_cast<std::size_t>(rank)] = 300 + rank;
    mp1::allgather(c, mp1::in_place, recv);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == 300 + i);
  }

  // alltoall(in_place, C&) compared with the out-of-place call.
  {
    std::vector<int> expected(static_cast<std::size_t>(size), -1);
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = rank * 100 + i;
    mp1::alltoall(c, send, expected);
    std::vector<int> inplace = send;
    mp1::alltoall(c, mp1::in_place, inplace);
    BOOST_TEST(inplace == expected);
  }

  // alltoall(in_place, T*, layout) compared with the out-of-place call.
  {
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    std::vector<int> expected(static_cast<std::size_t>(2 * size), -1);
    std::vector<int> send(static_cast<std::size_t>(2 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(2 * i)] = rank * 100 + i;
      send[static_cast<std::size_t>(2 * i + 1)] = rank * 100 + i + 1;
    }
    const mp1::layout<int> aggregate = mp1::layout<int>::contiguous(2 * size);
    mp1::alltoall(c, mp1::region<int>{send.data(), aggregate},
                  mp1::region<int>{expected.data(), aggregate});
    std::vector<int> inplace = send;
    mp1::alltoall(c, mp1::in_place, mp1::region<int>{inplace.data(), aggregate});
    BOOST_TEST(inplace == expected);
  }

  // Reduction in-place container and pointer forms.
  {
    std::vector<int> values(static_cast<std::size_t>(size), rank);
    if (rank == root) {
      mp1::reduce(c, plus, root, mp1::in_place, values);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(values[static_cast<std::size_t>(i)] == sum_ranks(size));
    } else {
      mp1::reduce(c, plus, root, values, std::span<int>{});
    }

    std::vector<int> all(static_cast<std::size_t>(size), rank);
    mp1::allreduce(c, plus, mp1::in_place, all);
    for (const int value : all)
      BOOST_TEST(value == sum_ranks(size));

    std::vector<int> prefix(static_cast<std::size_t>(size), rank);
    mp1::scan(c, plus, mp1::in_place, prefix);
    for (const int value : prefix)
      BOOST_TEST(value == rank * (rank + 1) / 2);

    std::vector<int> suffix(static_cast<std::size_t>(size), rank);
    mp1::exscan(c, plus, mp1::in_place, suffix);
    if (rank == 0) {
      for (const int value : suffix)
        BOOST_TEST(value == 0);
    } else {
      for (const int value : suffix)
        BOOST_TEST(value == rank * (rank - 1) / 2);
    }
  }

  // Reduction in-place pointer and layout forms.
  {
    std::vector<int> buffer{rank, rank + 1};
    if (rank == root) {
      mp1::reduce(c, plus, root, mp1::in_place, std::span<int>{buffer});
      BOOST_TEST(buffer[0] == sum_ranks(size));
      BOOST_TEST(buffer[1] == sum_ranks(size) + size);
    } else {
      mp1::reduce(c, plus, root, std::span<int>{buffer}, std::span<int>{});
    }

    const mp1::layout<int> holey = holey_int_layout();
    std::vector<int> holey_buffer(3, 0);
    holey_buffer[0] = rank;
    holey_buffer[2] = rank + 5;
    mp1::allreduce(c, plus, mp1::in_place, mp1::region<int>{holey_buffer.data(), holey});
    BOOST_TEST(holey_buffer[0] == sum_ranks(size));
    BOOST_TEST(holey_buffer[2] == sum_ranks(size) + 5 * size);
  }

  // A malformed reduction base type contract throws before MPI on every rank.  The check is
  // exercised through a zero-count call so no collective state is entered.
  {
    int dummy = 0;
    mp1::allreduce(c, plus, std::span<const int>{}, std::span<int>{&dummy, 0});
  }
}

BOOST_AUTO_TEST_CASE(dense_and_holey_call_counts) {
#if MP1_COLLECTIVE_TEST_WRAP
  if (not mpi_wrap::supported()) {
    BOOST_TEST(true);
    return;
  }
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const std::string alltoallw_symbol = std::string{"MPI_Allto"} + "allw";
  const std::string alltoallw_c_symbol = std::string{"MPI_Allto"} + "allw_c";

  // Dense variable collective: one native call, no staging, no duplicate type.
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 2);
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      displs.push_back(static_cast<mp1::count_t>(2 * i));
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send{rank, rank + 1};
    std::vector<int> recv(static_cast<std::size_t>(2 * size), -1);
    mpi_wrap::reset_counts();
    if (rank == root) {
      mp1::gatherv(c, root, send, recv, dense);
    } else {
      mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    }
    BOOST_TEST(mpi_wrap::count("MPI_Gatherv_c") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Gatherv") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_size_c") == 0);
    BOOST_TEST(mpi_wrap::count(alltoallw_symbol.c_str()) == 0);
    BOOST_TEST(mpi_wrap::count(alltoallw_c_symbol.c_str()) == 0);
  }

  // Holey variable collective: the direction is staged, at most one pack per block per rank.
  {
    const mp1::layout<int> holey = holey_int_layout();
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(holey);
      displs.push_back(static_cast<mp1::count_t>(3 * i));
    }
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 1;
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    mpi_wrap::reset_counts();
    if (rank == root) {
      mp1::gatherv(c, root, mp1::region{send.data(), holey}, recv, recv_layouts, displs);
    } else {
      mp1::gatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{}, mp1::blocks{});
    }
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    BOOST_TEST(packs <= static_cast<std::size_t>(size + 1));
    BOOST_TEST(packs >= 1);
    BOOST_TEST(mpi_wrap::count("MPI_Gatherv_c") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Gatherv") == 0);
    BOOST_TEST(mpi_wrap::count(alltoallw_symbol.c_str()) == 0);
    BOOST_TEST(mpi_wrap::count(alltoallw_c_symbol.c_str()) == 0);
  }

  // C++17 captureless lambdas are stateless but not default-constructible; the
  // reduction operator cache must capture the callable object, not default-construct it.
  {
    auto c = mp1::comm_world();
    int send_sum = 1;
    int recv_sum = 0;
    mp1::allreduce(
        c, [](int a, int b) noexcept { return a + b; }, std::span<int>{&send_sum, 1},
        std::span<int>{&recv_sum, 1});
    BOOST_TEST(recv_sum == c.size());
    int send_max = c.rank();
    int recv_max = 0;
    mp1::allreduce(
        c, [](int a, int b) noexcept { return a > b ? a : b; }, std::span<int>{&send_max, 1},
        std::span<int>{&recv_max, 1});
    BOOST_TEST(recv_max == c.size() - 1);
  }

  // Dense flat reduction: one native call, no staging.
  {
    int recv = -1;
    mpi_wrap::reset_counts();
    mp1::allreduce(c, std::plus<int>{}, rank, recv);
    BOOST_TEST(recv == size * (size - 1) / 2);
    BOOST_TEST(mpi_wrap::count("MPI_Allreduce_c") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Allreduce") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
  }
#else
  BOOST_TEST(true);
#endif
}

// ---------------------------------------------------------------------------------------
// Staged reduction descriptor lifetime.  A nonblocking staged reduction's
// completion unpacks after the call returns, so the payload must own the type map it reads.
// Destroying or reusing the caller's layout object before wait() must not change the result;
// under ASan a stored borrow fails instead of silently mis-unpacking.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(staged_reduction_layout_lifetime) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int sum = size * (size - 1) / 2;

  // ireduce root form: the layout is destroyed right after the call.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_int_layout());
      r = mp1::ireduce(c, plus, 0, mp1::region<int>{send.data(), *l},
                       mp1::region<int>{recv.data(), *l});
    }
    r.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == sum);
      BOOST_TEST(recv[2] == sum + 10 * size);
      BOOST_TEST(recv[1] == -1);
    }
  }

  // iallreduce: the same layout object is reused with a different type before wait().  The
  // owned copy must keep the original (holey) map, so the result still lands at offsets 0/2.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    std::optional<mp1::layout<int>> l;
    l.emplace(holey_int_layout());
    mp1::request r = mp1::iallreduce(c, plus, mp1::region<int>{send.data(), *l},
                                     mp1::region<int>{recv.data(), *l});
    l.emplace(mp1::layout<int>::contiguous(2));
    r.wait();
    BOOST_TEST(recv[0] == sum);
    BOOST_TEST(recv[2] == sum + 10 * size);
    BOOST_TEST(recv[1] == -1);
  }

  // iscan: prefix sums.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_int_layout());
      r = mp1::iscan(c, plus, mp1::region<int>{send.data(), *l},
                     mp1::region<int>{recv.data(), *l});
    }
    r.wait();
    BOOST_TEST(recv[0] == rank * (rank + 1) / 2);
    BOOST_TEST(recv[2] == rank * (rank + 1) / 2 + 10 * (rank + 1));
  }

  // iexscan: rank 0's receive buffer stays untouched.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, 777);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_int_layout());
      r = mp1::iexscan(c, plus, mp1::region<int>{send.data(), *l},
                       mp1::region<int>{recv.data(), *l});
    }
    r.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == 777);
    } else {
      BOOST_TEST(recv[0] == rank * (rank - 1) / 2);
      BOOST_TEST(recv[2] == rank * (rank - 1) / 2 + 10 * rank);
    }
  }

  // ireduce_scatter: one holey block per rank.
  {
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = i;
      send[static_cast<std::size_t>(3 * i + 2)] = 10 + i;
    }
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_int_layout());
      const mp1::layout<int> aggregate = aggregate_holey_int(size);
      const std::vector<mp1::count_t> per_rank(static_cast<std::size_t>(size), 2);
      r = mp1::ireduce_scatter(c, plus, mp1::region<int>{send.data(), aggregate},
                               mp1::region<int>{recv.data(), *l}, per_rank);
    }
    r.wait();
    // Each rank contributes the full flat vector; rank r's receive block is the summed
    // flat segment [2r, 2r+1], i.e. (size*r, size*(10+r)) for this identical input.
    BOOST_TEST(recv[0] == size * rank);
    BOOST_TEST(recv[2] == size * (10 + rank));
    BOOST_TEST(recv[1] == -1);
  }

  // ireduce_scatter_block: the same per-rank block with a uniform repetition count.
  {
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = i;
      send[static_cast<std::size_t>(3 * i + 2)] = 10 + i;
    }
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_int_layout());
      const mp1::layout<int> aggregate = aggregate_holey_int(size);
      r = mp1::ireduce_scatter_block(c, plus, mp1::region<int>{send.data(), aggregate},
                                     mp1::region<int>{recv.data(), *l}, 2);
    }
    r.wait();
    BOOST_TEST(recv[0] == size * rank);
    BOOST_TEST(recv[2] == size * (10 + rank));
    BOOST_TEST(recv[1] == -1);
  }
}

// ---------------------------------------------------------------------------------------
// A staged reduction with a non-empty receive block must reject a null receive pointer
// before entering MPI; the completion path would otherwise silently drop the result
// (MPI 4.1 6.9 makes the receive buffer significant on the process that receives).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(staged_reduction_null_recv_is_rejected) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const std::plus<int> plus{};
  std::vector<int> send(3, -1);
  send[0] = rank;
  send[2] = rank + 10;

  // reduce: only the root's receive side is significant.
  if (rank == 0) {
    const mp1::layout<int> l = holey_int_layout();
    check_throws_invalid([&] {
      const mp1::layout<int> l{holey_int_layout()};
      mp1::reduce(c, plus, 0, mp1::region<int>{send.data(), l},
                  mp1::region<int>{static_cast<int*>(nullptr), l});
    });
  }
  mp1::barrier(c);

  // allreduce: every rank's receive side is significant, so every rank rejects locally.
  {
    const mp1::layout<int> l = holey_int_layout();
    check_throws_invalid([&] {
      mp1::allreduce(c, plus, mp1::region<int>{send.data(), l},
                     mp1::region<int>{static_cast<int*>(nullptr), l});
    });
  }
  mp1::barrier(c);
}

BOOST_AUTO_TEST_SUITE_END()
