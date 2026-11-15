// Typed managed-reduction image: the callback datatype passed to MPI for an mp1-managed
// functor is the native typed map of T at extent sizeof(T), not an MPI_BYTE full-object
// image.  The image is process-lifetime, its geometry preserves T's native true lower bound,
// and MPI receives the image handle (never the old byte image) for the callback path.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>


namespace {

  struct mpi_fixture {
    mpi_fixture() {
      mp1::init();
    }

    ~mpi_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  // A reflected element type with internal padding: the native map is int@0 + double@8,
  // size 12, extent 16.  The image must carry exactly those two typed leaves.
  struct padded_value {
    int first{0};
    double second{0.0};
  };

  struct int_add {
    int operator()(int left, int right) const noexcept {
      return left + right;
    }
  };

  struct padded_add {
    padded_value operator()(const padded_value& left,
                            const padded_value& right) const noexcept {
      return padded_value{left.first + right.first, left.second + right.second};
    }
  };

  // Capture the datatype argument MPI receives.  The test executable defines the MPI
  // entry point itself (the profiling-interface idiom) and forwards to the PMPI entry
  // point; mpi_wrap's --wrap counter for this symbol is not used by this test.
  MPI_Datatype last_allreduce_datatype = MPI_DATATYPE_NULL;
  MPI_Count last_allreduce_count = 0;
  int allreduce_calls = 0;

  void reset_allreduce_capture() {
    last_allreduce_datatype = MPI_DATATYPE_NULL;
    last_allreduce_count = 0;
    allreduce_calls = 0;
  }

  [[nodiscard]] std::vector<MPI_Datatype> leaf_datatypes(MPI_Datatype type) {
    std::vector<MPI_Datatype> leaves;
    std::vector<MPI_Datatype> pending{type};
    while (not pending.empty()) {
      const MPI_Datatype current = pending.back();
      pending.pop_back();
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner = MPI_COMBINER_NAMED;
      BOOST_REQUIRE(MPI_Type_get_envelope_c(current, &num_integers, &num_addresses,
                                            &num_large_counts, &num_datatypes,
                                            &combiner) == MPI_SUCCESS);
      if (combiner == MPI_COMBINER_NAMED or num_datatypes == 0) {
        leaves.push_back(current);
        continue;
      }
      std::vector<int> integers(static_cast<std::size_t>(num_integers));
      std::vector<MPI_Aint> addresses(static_cast<std::size_t>(num_addresses));
      std::vector<MPI_Count> large_counts(static_cast<std::size_t>(num_large_counts));
      std::vector<MPI_Datatype> children(static_cast<std::size_t>(num_datatypes));
      BOOST_REQUIRE(MPI_Type_get_contents_c(current, num_integers, num_addresses,
                                            num_large_counts, num_datatypes, integers.data(),
                                            addresses.data(), large_counts.data(),
                                            children.data()) == MPI_SUCCESS);
      for (const MPI_Datatype child : children) {
        if (child != MPI_DATATYPE_NULL)
          pending.push_back(child);
      }
    }
    return leaves;
  }

  [[nodiscard]] bool contains_datatype(const std::vector<MPI_Datatype>& types,
                                       MPI_Datatype wanted) {
    for (const MPI_Datatype type : types) {
      if (type == wanted)
        return true;
    }
    return false;
  }

}  // namespace

MP1_REFLECTION(padded_value, first, second)

// The test defines the MPI entry point so it can observe the exact datatype handle mp1
// passes to the reduction.  MPI 4.1 §6.9.5 gives the user-op callback that datatype; the
// managed callback path must use reduction_image_datatype<T>().
extern "C" int MPI_Allreduce_c(const void* sendbuf, void* recvbuf, MPI_Count count,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  last_allreduce_datatype = datatype;
  last_allreduce_count = count;
  ++allreduce_calls;
  return PMPI_Allreduce_c(sendbuf, recvbuf, count, datatype, op, comm);
}

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(reduction_image_test)

BOOST_AUTO_TEST_CASE(predefined_element_image_geometry_is_typed) {
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<int>();
  BOOST_TEST(image != MPI_DATATYPE_NULL);

  MPI_Count size = 0;
  MPI_Count lower_bound = -1;
  MPI_Count extent = 0;
  MPI_Count true_lb = -1;
  MPI_Count true_extent = 0;
  BOOST_REQUIRE(MPI_Type_size_c(image, &size) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Type_get_extent_c(image, &lower_bound, &extent) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Type_get_true_extent_c(image, &true_lb, &true_extent) == MPI_SUCCESS);
  BOOST_TEST(size == static_cast<MPI_Count>(sizeof(int)));
  BOOST_TEST(lower_bound == 0);
  BOOST_TEST(extent == static_cast<MPI_Count>(sizeof(int)));
  BOOST_TEST(true_lb == 0);
  BOOST_TEST(true_extent == static_cast<MPI_Count>(sizeof(int)));

  // The old image was MPI_COMBINER_CONTIGUOUS over a single MPI_BYTE.  The typed image is a
  // resized view of the native datatype; its child is MPI_INT, not MPI_BYTE.
  MPI_Count num_integers{0};
  MPI_Count num_addresses{0};
  MPI_Count num_large_counts{0};
  MPI_Count num_datatypes{0};
  int combiner = MPI_COMBINER_NAMED;
  BOOST_REQUIRE(MPI_Type_get_envelope_c(image, &num_integers, &num_addresses, &num_large_counts,
                                        &num_datatypes, &combiner) == MPI_SUCCESS);
  BOOST_TEST(combiner == MPI_COMBINER_RESIZED);
  BOOST_REQUIRE(num_datatypes == 1);
  std::vector<int> integers(static_cast<std::size_t>(num_integers));
  std::vector<MPI_Aint> addresses(static_cast<std::size_t>(num_addresses));
  std::vector<MPI_Count> large_counts(static_cast<std::size_t>(num_large_counts));
  std::vector<MPI_Datatype> children(static_cast<std::size_t>(num_datatypes));
  BOOST_REQUIRE(MPI_Type_get_contents_c(image, num_integers, num_addresses, num_large_counts,
                                        num_datatypes, integers.data(), addresses.data(),
                                        large_counts.data(), children.data()) == MPI_SUCCESS);
  BOOST_TEST(children[0] == MPI_INT);

  const std::vector<MPI_Datatype> leaves = leaf_datatypes(image);
  BOOST_REQUIRE(leaves.size() == 1);
  BOOST_TEST(leaves[0] == MPI_INT);
  BOOST_TEST(not contains_datatype(leaves, MPI_BYTE));

  // One process-lifetime committed handle per T.
  BOOST_TEST(image == mp1::detail::reduction_image_datatype<int>());
}

BOOST_AUTO_TEST_CASE(reflected_element_image_preserves_native_leaf_map) {
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<padded_value>();
  MPI_Count size = 0;
  MPI_Count lower_bound = -1;
  MPI_Count extent = 0;
  MPI_Count true_lb = -1;
  MPI_Count true_extent = 0;
  BOOST_REQUIRE(MPI_Type_size_c(image, &size) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Type_get_extent_c(image, &lower_bound, &extent) == MPI_SUCCESS);
  BOOST_REQUIRE(MPI_Type_get_true_extent_c(image, &true_lb, &true_extent) == MPI_SUCCESS);

  // The native map is int@0 + double@8: size 12, true extent 16, extent sizeof(T) == 16.
  // Padding bytes 4..7 are not part of the map.
  BOOST_TEST(size == 12);
  BOOST_TEST(lower_bound == 0);
  BOOST_TEST(extent == static_cast<MPI_Count>(sizeof(padded_value)));
  BOOST_TEST(true_lb == 0);
  BOOST_TEST(true_extent == static_cast<MPI_Count>(sizeof(padded_value)));

  const std::vector<MPI_Datatype> image_leaves = leaf_datatypes(image);
  const std::vector<MPI_Datatype> native_leaves =
      leaf_datatypes(mp1::native_datatype<padded_value>());
  BOOST_TEST(image_leaves.size() == 2);
  BOOST_TEST(native_leaves.size() == 2);
  BOOST_TEST(contains_datatype(image_leaves, MPI_INT));
  BOOST_TEST(contains_datatype(image_leaves, MPI_DOUBLE));
  BOOST_TEST(not contains_datatype(image_leaves, MPI_BYTE));

  // Heterogeneous reasoning: the image's type signature is exactly the native type
  // signature, so MPI 4.1 §3.3.2 conversion applies per typed leaf.  No heterogeneous
  // provider is available in this matrix; the signature equality is the local proof.
  BOOST_TEST(image_leaves.size() == native_leaves.size());
  for (const MPI_Datatype leaf : native_leaves)
    BOOST_TEST(contains_datatype(image_leaves, leaf));
}

BOOST_AUTO_TEST_CASE(byte_element_keeps_native_byte_leaf) {
  // std::byte maps to MPI_BYTE natively.  A managed callback over std::byte therefore has a
  // byte leaf because that is the element's native type, not because mp1 added a byte
  // full-object image or padding block.
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<std::byte>();
  const std::vector<MPI_Datatype> leaves = leaf_datatypes(image);
  BOOST_REQUIRE(leaves.size() == 1);
  BOOST_TEST(leaves[0] == MPI_BYTE);
  BOOST_TEST(mp1::native_datatype<std::byte>() == MPI_BYTE);
}

BOOST_AUTO_TEST_CASE(managed_functor_passes_image_handle_to_mpi) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size - 1) / 2;
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<int>();

  reset_allreduce_capture();
  int recv = -1;
  mp1::allreduce(comm, int_add{}, rank, recv);
  BOOST_TEST(recv == expected);
  BOOST_TEST(allreduce_calls == 1);
  BOOST_TEST(last_allreduce_count == 1);
  BOOST_TEST(last_allreduce_datatype == image);

  // A predefined mapped functor keeps native_datatype<T>() (MPI_INT) and never creates the
  // callback image.  This also keeps the predefined fast path unchanged.
  reset_allreduce_capture();
  recv = -1;
  mp1::allreduce(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);
  BOOST_TEST(allreduce_calls == 1);
  BOOST_TEST(last_allreduce_datatype == MPI_INT);

  // A raw caller MPI_Op keeps the caller's native datatype contract.
  MPI_Op raw = MPI_OP_NULL;
  BOOST_REQUIRE(MPI_Op_create(
                    [](void* in, void* inout, int* len, MPI_Datatype*) {
                      const auto* input = static_cast<const int*>(in);
                      auto* output = static_cast<int*>(inout);
                      for (int i = 0; i < *len; ++i)
                        output[i] += input[i];
                    },
                    0, &raw) == MPI_SUCCESS);
  reset_allreduce_capture();
  recv = -1;
  mp1::allreduce(comm, raw, rank, recv);
  BOOST_TEST(recv == expected);
  BOOST_TEST(allreduce_calls == 1);
  BOOST_TEST(last_allreduce_datatype == MPI_INT);
  BOOST_REQUIRE(MPI_Op_free(&raw) == MPI_SUCCESS);
}

BOOST_AUTO_TEST_CASE(managed_functor_reduction_uses_typed_image_for_reflected_type) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<padded_value>();

  reset_allreduce_capture();
  const padded_value send{rank, static_cast<double>(rank)};
  padded_value recv{-1, -1.0};
  mp1::allreduce(comm, padded_add{}, send, recv);
  BOOST_TEST(recv.first == size * (size - 1) / 2);
  BOOST_TEST(recv.second == static_cast<double>(size * (size - 1) / 2));
  BOOST_TEST(allreduce_calls == 1);
  BOOST_TEST(last_allreduce_datatype == image);
}

BOOST_AUTO_TEST_CASE(managed_reduce_scatter_full_span_image) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;
  const std::vector<mp1::count_t> recvcounts(static_cast<std::size_t>(size), 1);

  // padded_value's native map is int@0 + double@8: true extent 16 == extent sizeof(T), so the
  // callback image spans the whole object and reduce_scatter keeps the typed image.
  std::vector<padded_value> send(static_cast<std::size_t>(size));
  for (int j = 0; j < size; ++j) {
    send[static_cast<std::size_t>(j)] =
        padded_value{rank + 1 + 10 * j, 0.5 * (rank + 1 + 10 * j)};
  }
  std::vector<padded_value> recv(1, padded_value{-1, -1.0});
  mp1::reduce_scatter(comm, padded_add{}, std::span<const padded_value>{send},
                      std::span<padded_value>{recv}, recvcounts);
  BOOST_TEST(recv[0].first == expected + 10 * size * rank);
  BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected + 10 * size * rank));

  // The block family uses the same image and count.
  std::vector<padded_value> recv_block(1, padded_value{-1, -1.0});
  mp1::reduce_scatter_block(comm, padded_add{}, std::span<const padded_value>{send},
                            std::span<padded_value>{recv_block}, 1);
  BOOST_TEST(recv_block[0].first == expected + 10 * size * rank);
}

BOOST_AUTO_TEST_SUITE_END()
