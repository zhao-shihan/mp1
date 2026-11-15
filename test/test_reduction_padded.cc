// Typed managed-reduction image for an element type whose native datatype has a non-zero
// true lower bound and unmapped bytes.  The image preserves the native typed map (an int leaf
// at its C++ offset), is resized to extent sizeof(T), transmits only the mapped int, and
// leaves the unlisted member and padding of a direct receive buffer untouched.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>


// Native map: one MPI_INT at byte 8, resized to extent sizeof(padded_int) == 16.  The
// native true lower bound is therefore 8 and the true extent is 4; bytes 0..7 and 12..15 are
// not part of the type map.  This mirrors a reflected aggregate whose first listed member is
// not at offset 0.  The managed callback image shifts the native block down by 8 so its own
// true lower bound is 0, and the reduction buffer pointer is shifted up by 8: the mapped int
// still lands at the C++ offset and the callback subtracts the same offset before T indexing.
struct padded_int {
  std::uint64_t lead{0};
  int value{0};
};

static_assert(sizeof(padded_int) == 16);
static_assert(offsetof(padded_int, value) == 8);

template<>
struct mp1::datatype_traits<padded_int> {
  using data_type_category = mp1::detail::basic_or_fixed_size_type;

  static MPI_Datatype get_datatype() {
    static const MPI_Datatype type = [] {
      MPI_Datatype block = MPI_DATATYPE_NULL;
      MPI_Datatype resized = MPI_DATATYPE_NULL;
      int blocklength = 1;
      MPI_Aint displacement = static_cast<MPI_Aint>(offsetof(padded_int, value));
      MPI_Datatype child = MPI_INT;
      mp1::detail::check_mpi(
          MPI_Type_create_struct(1, &blocklength, &displacement, &child, &block),
          {"MPI_Type_create_struct"});
      mp1::detail::check_mpi(MPI_Type_create_resized(
                                 block, 0, static_cast<MPI_Aint>(sizeof(padded_int)), &resized),
                             {"MPI_Type_create_resized"});
      mp1::detail::check_mpi(MPI_Type_commit(&resized), {"MPI_Type_commit"});
      (void)MPI_Type_free(&block);
      return resized;
    }();
    return type;
  }
};

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

  struct padded_int_add {
    padded_int operator()(const padded_int& left, const padded_int& right) const noexcept {
      // The functor contract: only listed members are read and written.  `lead` and the
      // padding bytes are not part of the image and are never inspected here.
      return padded_int{0, left.value + right.value};
    }
  };

  constexpr int kBeforeCanary = 0x12345678;
  constexpr int kAfterCanary = 0x5a5a5a5a;
  constexpr std::uint64_t kSendLead = 0x1111111111111111ULL;
  constexpr std::uint64_t kRecvLead = 0x2222222222222222ULL;
  constexpr unsigned char kPadByte = 0xa5;

  struct guarded_pair {
    int before{kBeforeCanary};
    std::array<padded_int, 2> values{};
    int after{kAfterCanary};
  };

  struct guarded_triple {
    int before{kBeforeCanary};
    std::array<padded_int, 3> values{};
    int after{kAfterCanary};
  };

  // `lead` is the unlisted member; bytes 12..15 are alignment padding.  Both are outside the
  // native type map and must not be rewritten by a typed receive.
  void seed_unmapped_bytes(padded_int& value, std::uint64_t lead) {
    value.lead = lead;
    auto* bytes = reinterpret_cast<unsigned char*>(&value);
    for (std::size_t i = 12; i < sizeof(padded_int); ++i)
      bytes[i] = kPadByte;
  }

  void require_unmapped_bytes(const padded_int& value, std::uint64_t lead, const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(value.lead == lead);
      const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
      for (std::size_t i = 12; i < sizeof(padded_int); ++i)
        BOOST_TEST(bytes[i] == kPadByte);
    }
  }

  void require_canaries(const guarded_pair& block, const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(block.before == kBeforeCanary);
      BOOST_TEST(block.after == kAfterCanary);
    }
  }

  void require_canaries(const guarded_triple& block, const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(block.before == kBeforeCanary);
      BOOST_TEST(block.after == kAfterCanary);
    }
  }

  void require_typed_int_leaf(MPI_Datatype type) {
    std::vector<MPI_Datatype> pending{type};
    std::size_t leaves = 0;
    bool saw_byte = false;
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
        ++leaves;
        if (current == MPI_BYTE)
          saw_byte = true;
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
    BOOST_TEST(leaves == 1);
    BOOST_TEST(not saw_byte);
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(reduction_padded_test)

BOOST_AUTO_TEST_CASE(image_geometry_preserves_native_true_lower_bound) {
  const MPI_Datatype image = mp1::detail::reduction_image_datatype<padded_int>();
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
  BOOST_TEST(extent == static_cast<MPI_Count>(sizeof(padded_int)));
  // The image normalizes the true lower bound to 0 by placing the native block at
  // displacement -8; the buffer pointer carries the compensating +8 offset.
  BOOST_TEST(true_lb == 0);
  BOOST_TEST(true_extent == static_cast<MPI_Count>(sizeof(int)));
  BOOST_TEST(mp1::detail::reduction_image_buffer_offset<padded_int>() == 8);
  BOOST_TEST(mp1::detail::datatype_info_of<padded_int>().true_lb == 8);

  // One committed process-lifetime handle; the native map is wrapped, not replaced.
  BOOST_TEST(image != mp1::native_datatype<padded_int>());
  BOOST_TEST(image == mp1::detail::reduction_image_datatype<padded_int>());
  require_typed_int_leaf(image);
}

BOOST_AUTO_TEST_CASE(direct_flat_allreduce_preserves_unlisted_member_and_padding) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;

  guarded_pair send;
  seed_unmapped_bytes(send.values[0], kSendLead);
  seed_unmapped_bytes(send.values[1], kSendLead);
  send.values[0].value = rank + 1;
  send.values[1].value = 10 * (rank + 1);

  guarded_pair recv;
  for (auto& value : recv.values) {
    value.value = -1;
    seed_unmapped_bytes(value, kRecvLead);
  }

  mp1::allreduce(comm, padded_int_add{}, std::span<const padded_int>{send.values},
                 std::span<padded_int>{recv.values});

  BOOST_TEST(recv.values[0].value == expected);
  BOOST_TEST(recv.values[1].value == 10 * expected);
  // A typed receive writes only the mapped int; the unlisted member and the padding bytes of
  // the caller's direct receive buffer are untouched.
  require_unmapped_bytes(recv.values[0], kRecvLead, "direct recv[0] unmapped bytes");
  require_unmapped_bytes(recv.values[1], kRecvLead, "direct recv[1] unmapped bytes");
  require_canaries(send, "direct send canaries");
  require_canaries(recv, "direct recv canaries");

  // In-place allreduce keeps the same preservation rule; the buffer is both input and result.
  guarded_pair in_place;
  seed_unmapped_bytes(in_place.values[0], kSendLead);
  seed_unmapped_bytes(in_place.values[1], kSendLead);
  in_place.values[0].value = rank + 1;
  in_place.values[1].value = 10 * (rank + 1);
  mp1::allreduce(comm, padded_int_add{}, mp1::in_place, std::span<padded_int>{in_place.values});
  BOOST_TEST(in_place.values[0].value == expected);
  BOOST_TEST(in_place.values[1].value == 10 * expected);
  require_unmapped_bytes(in_place.values[0], kSendLead, "in-place[0] unmapped bytes");
  require_unmapped_bytes(in_place.values[1], kSendLead, "in-place[1] unmapped bytes");
  require_canaries(in_place, "in-place canaries");
}

BOOST_AUTO_TEST_CASE(staged_layout_preserves_unlisted_member_and_layout_holes) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;

  // count 2, blocklength 1, stride 2: values[0] and values[2] are mapped, values[1] is a
  // layout hole.  Both directions stage through flat T slots.
  const mp1::layout<padded_int> holey = mp1::layout<padded_int>::vector(2, 1, 2);
  BOOST_TEST(not holey.is_base_repetition());

  guarded_triple send;
  for (auto& value : send.values)
    seed_unmapped_bytes(value, kSendLead);
  send.values[0].value = rank + 1;
  send.values[1].value = -7;
  send.values[2].value = 10 * (rank + 1);
  send.values[1].lead = 0x3333333333333333ULL;

  guarded_triple recv;
  for (auto& value : recv.values) {
    value.value = -1;
    seed_unmapped_bytes(value, kRecvLead);
  }

  mp1::allreduce(comm, padded_int_add{},
                 mp1::region<const padded_int>{send.values.data(), holey},
                 mp1::region<padded_int>{recv.values.data(), holey});

  BOOST_TEST(recv.values[0].value == expected);
  BOOST_TEST(recv.values[2].value == 10 * expected);
  // The layout hole is never packed or unpacked.
  BOOST_TEST(recv.values[1].value == -1);
  BOOST_TEST(recv.values[1].lead == kRecvLead);
  // Stage packing copies only the native map's int; completion unpacks only that int, so the
  // caller's unlisted member and padding survive the staged round trip.
  require_unmapped_bytes(recv.values[0], kRecvLead, "staged recv[0] unmapped bytes");
  require_unmapped_bytes(recv.values[2], kRecvLead, "staged recv[2] unmapped bytes");
  require_canaries(send, "staged send canaries");
  require_canaries(recv, "staged recv canaries");
}

BOOST_AUTO_TEST_CASE(nonblocking_and_persistent_reductions) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;

  // Nonblocking direct span.
  {
    guarded_pair send;
    seed_unmapped_bytes(send.values[0], kSendLead);
    seed_unmapped_bytes(send.values[1], kSendLead);
    send.values[0].value = rank + 1;
    send.values[1].value = 10 * (rank + 1);
    guarded_pair recv;
    for (auto& value : recv.values) {
      value.value = -1;
      seed_unmapped_bytes(value, kRecvLead);
    }
    mp1::request request =
        mp1::iallreduce(comm, padded_int_add{}, std::span<const padded_int>{send.values},
                        std::span<padded_int>{recv.values});
    request.wait();
    BOOST_TEST(recv.values[0].value == expected);
    BOOST_TEST(recv.values[1].value == 10 * expected);
    require_unmapped_bytes(recv.values[0], kRecvLead, "iallreduce recv[0] unmapped bytes");
    require_unmapped_bytes(recv.values[1], kRecvLead, "iallreduce recv[1] unmapped bytes");
  }

  // Persistent allreduce over the staged layout: each start re-packs the current send values;
  // each completion unpacks only the mapped int.
  {
    const mp1::layout<padded_int> holey = mp1::layout<padded_int>::vector(2, 1, 2);
    guarded_triple send;
    for (auto& value : send.values)
      seed_unmapped_bytes(value, kSendLead);
    send.values[0].value = rank + 1;
    send.values[1].value = -7;
    send.values[2].value = 10 * (rank + 1);
    guarded_triple recv;
    for (auto& value : recv.values) {
      value.value = -1;
      seed_unmapped_bytes(value, kRecvLead);
    }
    mp1::request request = mp1::allreduce_init(
        comm, padded_int_add{}, mp1::region<const padded_int>{send.values.data(), holey},
        mp1::region<padded_int>{recv.values.data(), holey});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send.values[0].value = rank + 1 + cycle;
      send.values[2].value = 10 * (rank + 1 + cycle);
      request.start();
      request.wait();
      BOOST_TEST(recv.values[0].value == expected + cycle * size);
      BOOST_TEST(recv.values[2].value == 10 * (expected + cycle * size));
      BOOST_TEST(recv.values[1].value == -1);
      require_unmapped_bytes(recv.values[0], kRecvLead, "persistent recv[0] unmapped bytes");
      require_unmapped_bytes(recv.values[1], kRecvLead, "persistent hole unmapped bytes");
      require_unmapped_bytes(recv.values[2], kRecvLead, "persistent recv[2] unmapped bytes");
    }
    require_canaries(send, "persistent send canaries");
    require_canaries(recv, "persistent recv canaries");
  }
}

BOOST_AUTO_TEST_CASE(reduce_scatter_rejects_gapped_callback_image) {
  // padded_int has a non-zero native true lower bound and trailing padding, so its callback
  // image does not span sizeof(T).  Managed reduce_scatter is rejected locally on every rank;
  // the typed image remains usable for the other reduction families.
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int size = comm.size();
  const std::vector<mp1::count_t> recvcounts(static_cast<std::size_t>(size), 1);

  std::vector<padded_int> send(static_cast<std::size_t>(size), padded_int{});
  padded_int recv{};
  seed_unmapped_bytes(recv, kRecvLead);
  BOOST_CHECK_THROW(
      mp1::reduce_scatter(comm, padded_int_add{}, std::span<const padded_int>{send},
                          std::span<padded_int>{&recv, 1}, recvcounts),
      std::invalid_argument);
  BOOST_CHECK_THROW(
      (void)mp1::ireduce_scatter(comm, padded_int_add{}, std::span<const padded_int>{send},
                                 std::span<padded_int>{&recv, 1}, recvcounts),
      std::invalid_argument);
  BOOST_CHECK_THROW(
      (void)mp1::reduce_scatter_init(comm, padded_int_add{}, std::span<const padded_int>{send},
                                     std::span<padded_int>{&recv, 1}, recvcounts),
      std::invalid_argument);
  BOOST_CHECK_THROW(
      mp1::reduce_scatter_block(comm, padded_int_add{}, std::span<const padded_int>{send},
                                std::span<padded_int>{&recv, 1}, 1),
      std::invalid_argument);
}

BOOST_AUTO_TEST_SUITE_END()
