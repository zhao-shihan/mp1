// Typed native wire for the variable collectives (gatherv/scatterv/allgatherv/alltoallv).
//
// The wire of every used typed position is (buffer, n elements, native_datatype<T>()): the
// type is a function of T only, the count is an element count and a displacement is an extent
// unit.  MPI 4.1 5.1.11 makes a count repeat the datatype at its extent stride, which is
// exactly sizeof(T) (enforced by the library), so a padded/derived T is correct in the dense
// spelling and a repeated packed-but-spaced T cannot be lowered to a flat byte range.
//
// Cases:
//  * the tail-padded Widget facts and the four families on that type (dense counts,
//    layout and pointer+count spellings, counts 0..3, mixed spellings across ranks);
//  * raw-MPI peers with the native basic type and with a hand-built type of the same basic
//    component sequence (proves the wire signature and that no padding is on the wire);
//  * std::byte dense/layout (MPI_BYTE is the native type there) and holey layouts with
//    canaries (a receive must not write outside the type map);
//  * in-place root/all-rank forms, three persistent cycles, a nonblocking staged receive whose
//    descriptors are destroyed before completion;
//  * a nonzero true lower bound, a dense-but-not-native-repetition Widget map, a custom
//    zero-size native datatype (plan normalization plus the portable zero-count call), an
//    intercommunicator alltoallv and the existing intercommunicator exchange, all with the
//    tail-padded type;
//  * the dense-path allocation budget (0 allocations up to K peers, the exact array spill above
//    K, when the wrap counter is available).
//
// MPI 4.1 citations: 5.1.7/5.1.8 (extent/true extent), 5.1.11 (extent-stride repetition, type
// signature, receive must not touch bytes outside the map), 5.2 (pack/unpack), 6.5-6.8
// (variable collectives, extent-unit displacements), 6.12/6.13 (nonblocking/persistent).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_TYPED_HAVE_WRAP 1
#else
#define MP1_TYPED_HAVE_WRAP 0
#endif


namespace {

  // The tail-padded shape: a reflected struct with tail padding.  My reflection-derived
  // datatype is the map {uint32 at 0, uint8 at 4}, so MPI_Type_size_c == 5 while the extent is
  // sizeof == 8.
  struct Widget {
    std::uint32_t id;
    std::uint8_t flag;
  };

  static_assert(sizeof(Widget) == 8, "Widget must be the tail-padded 8-byte shape");

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

  [[nodiscard]] std::vector<mp1::count_t> uniform_counts(int peers, mp1::count_t elements) {
    return std::vector<mp1::count_t>(static_cast<std::size_t>(peers), elements);
  }

  [[nodiscard]] std::vector<mp1::count_t> prefix_displacements(
      const std::vector<mp1::count_t>& counts) {
    std::vector<mp1::count_t> displs(counts.size(), 0);
    mp1::count_t offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      displs[i] = offset;
      offset += counts[i];
    }
    return displs;
  }

  [[nodiscard]] Widget widget_value(int rank, int index) {
    return Widget{static_cast<std::uint32_t>(rank * 100 + index),
                  static_cast<std::uint8_t>(rank * 10 + index)};
  }

  [[nodiscard]] mp1::layout<int> holey_int_layout() {
    // Two ints at element offsets 0 and 2: size 8 bytes, extent 12 bytes, one hole.
    return mp1::layout<int>::vector(2, 1, 2);
  }

  [[nodiscard]] mp1::layout<Widget> holey_widget_layout() {
    // One Widget element at stride 2 extents: a non-native repetition that must stage.
    return mp1::layout<Widget>::vector(1, 1, 2);
  }

  // A native datatype whose true lower bound is nonzero: only the second member is reflected,
  // so MPI_Type_create_struct starts the map at byte 4.  The resize keeps extent == sizeof(T),
  // the library invariant, and the true extent is the 4 reflected bytes.  MPI applies the true
  // lower bound itself (MPI 4.1 5.1.11 addr = buf + extent*i + disp_j); no wrapper pointer
  // shift may add it a second time.
  struct shifted_widget {
    std::uint32_t padding;
    std::uint32_t value;
  };

  static_assert(sizeof(shifted_widget) == 8, "shifted_widget must be an 8-byte object");

  // A custom zero-size native datatype (MPI_Type_size_c == 0, extent resized to sizeof(T)).
  // Only a trait can produce one; every block is empty by the n * size == 0 rule, and no byte
  // may be written through it.
  struct zero_size_value {
    std::uint8_t tag;
  };

  static_assert(sizeof(zero_size_value) == 1, "zero_size_value must be a 1-byte object");

  [[nodiscard]] MPI_Datatype zero_size_native_datatype() {
    // Process-lifetime committed handle: freeing it after MPI_Finalize would be invalid, and
    // the trait is queried lazily inside the tests, after mp1::init().
    static const MPI_Datatype handle = [] {
      MPI_Datatype base = MPI_DATATYPE_NULL;
      MPI_Datatype resized = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_contiguous(0, MPI_BYTE, &base) == MPI_SUCCESS);
      const int create = MPI_Type_create_resized(
          base, 0, static_cast<MPI_Aint>(sizeof(zero_size_value)), &resized);
      (void)MPI_Type_free(&base);
      BOOST_REQUIRE(create == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&resized) == MPI_SUCCESS);
      return resized;
    }();
    return handle;
  }

  // The packed-but-spaced Widget map used below: dense() (it tiles 10 bytes) but not a
  // base_repetition (the second copy starts at byte 5, not at an extent stride).  Under the
  // typed wire it must stage; a flat byte lowering would have sent/read the wrong image.
  [[nodiscard]] mp1::layout<Widget> packed_spaced_widget_layout() {
    mp1::hindexed_parameter parameter;
    parameter.add(1, 0);
    parameter.add(1, 5);
    return mp1::layout<Widget>::hindexed(parameter);
  }

  // Read/write the Widget map entries of a raw byte image.  The block layout above maps the
  // second Widget copy to bytes [5,10), so a raw unsigned-char view is the only portable way to
  // inspect it (a Widget lvalue at byte 5 would be misaligned and is not what MPI sees).
  [[nodiscard]] std::uint32_t load_u32(const std::vector<Widget>& buffer,
                                       std::size_t byte_offset) {
    std::uint32_t value{0};
    std::memcpy(&value, reinterpret_cast<const unsigned char*>(buffer.data()) + byte_offset,
                sizeof(value));
    return value;
  }

  [[nodiscard]] std::uint8_t load_u8(const std::vector<Widget>& buffer,
                                     std::size_t byte_offset) {
    return reinterpret_cast<const unsigned char*>(buffer.data())[byte_offset];
  }

  void store_widget_bytes(std::vector<Widget>& buffer, std::size_t byte_offset,
                          std::uint32_t id, std::uint8_t flag) {
    auto* bytes = reinterpret_cast<unsigned char*>(buffer.data());
    std::memcpy(bytes + byte_offset, &id, sizeof(id));
    bytes[byte_offset + sizeof(id)] = flag;
  }

  // Consecutive holey layout blocks: a block reserves byte_extent bytes of address space, so
  // the public element displacement must advance by byte_extent / sizeof(T) elements (MPI 4.1
  // 6.5: a displacement is an extent unit of the recvtype, and the element unit is sizeof(T)).
  template<class T>
  [[nodiscard]] std::vector<mp1::count_t> extent_displacements(int peers,
                                                               const mp1::layout<T>& block) {
    const mp1::count_t stride = block.byte_extent() / static_cast<mp1::count_t>(sizeof(T));
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(peers), 0);
    for (int i = 0; i < peers; ++i)
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * stride;
    return displs;
  }

  // A raw reflected datatype with the same basic-component sequence as
  // native_datatype<Widget>() (uint32 then uint8), resized to sizeof(Widget).  The leaf types
  // come from mp1's traits, because type matching is by predefined name (MPI 4.1 3.3.1): on
  // this ABI uint32_t is `unsigned int`, so MPI_UINT32_T would be a different name.
  class raw_widget_datatype {
  public:
    raw_widget_datatype() {
      const MPI_Count blocklengths[2] = {1, 1};
      const MPI_Count displacements[2] = {0, 4};
      const MPI_Datatype types[2] = {mp1::native_datatype<std::uint32_t>(),
                                     mp1::native_datatype<std::uint8_t>()};
      MPI_Datatype structure = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_struct_c(2, blocklengths, displacements, types,
                                             &structure) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_create_resized(structure, 0, static_cast<MPI_Aint>(sizeof(Widget)),
                                            &handle_) == MPI_SUCCESS);
      (void)MPI_Type_free(&structure);
      BOOST_REQUIRE(MPI_Type_commit(&handle_) == MPI_SUCCESS);
    }

    raw_widget_datatype(const raw_widget_datatype&) = delete;
    raw_widget_datatype& operator=(const raw_widget_datatype&) = delete;

    ~raw_widget_datatype() {
      if (handle_ != MPI_DATATYPE_NULL)
        (void)MPI_Type_free(&handle_);
    }

    [[nodiscard]] MPI_Datatype get() const noexcept {
      return handle_;
    }

  private:
    MPI_Datatype handle_{MPI_DATATYPE_NULL};
  };

}  // namespace

MP1_REFLECTION(Widget, id, flag)
MP1_REFLECTION(shifted_widget, value)
MP1_DATATYPE_TRAITS_PACKED(zero_size_value, ::zero_size_native_datatype())


BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(variable_typed_contract)

// ---------------------------------------------------------------------------------------
// Tail-padded Widget: the trigger facts, then the four families on the tail-padded shape.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(tail_padded_trigger_facts) {
  const mp1::detail::datatype_info& info = mp1::detail::datatype_info_of<Widget>();
  // If a future reflection change made the map non-packed or the extent != sizeof(T), the
  // direct path would silently stop exercising the tail-padded trigger; these assertions keep
  // it honest.
  BOOST_TEST(static_cast<long long>(info.size) == 5LL);
  BOOST_TEST(static_cast<long long>(info.extent) == static_cast<long long>(sizeof(Widget)));
  BOOST_TEST(static_cast<long long>(info.true_lb) == 0LL);
  BOOST_TEST(info.packed);

  const mp1::layout<Widget> two = mp1::layout<Widget>::contiguous(2);
  BOOST_TEST(static_cast<long long>(two.size()) == 2LL);
  BOOST_TEST(static_cast<long long>(two.byte_size()) == 10LL);
  BOOST_TEST(static_cast<long long>(two.byte_extent()) == 16LL);
  BOOST_TEST(two.is_base_repetition());
}

BOOST_AUTO_TEST_CASE(tail_padded_gatherv_all_spellings) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  // Counts 0, 1, 2, 3.  The same call is repeated with a dense counts receive and with a
  // layouts<Widget> receive (a native-T repetition), so both lowerings are exercised.  The
  // third spelling is the mixed pairing: the root receives with layouts<Widget> while the
  // peers send the pointer+count Widget[n] form, so the mixed spelling must produce one wire
  // signature.
  for (int elements = 0; elements <= 3; ++elements) {
    const std::vector<mp1::count_t> counts =
        uniform_counts(size, static_cast<mp1::count_t>(elements));
    const std::vector<mp1::count_t> displs = prefix_displacements(counts);
    std::vector<Widget> send(static_cast<std::size_t>(elements));
    for (int j = 0; j < elements; ++j)
      send[static_cast<std::size_t>(j)] = widget_value(rank, j);

    for (int spelling = 0; spelling < 3; ++spelling) {
      std::vector<Widget> recv(static_cast<std::size_t>(elements * size),
                               Widget{0xFFFFFFFFU, 0xFFU});
      if (rank == root) {
        if (spelling == 0) {
          mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
        } else {
          std::vector<mp1::layout<Widget>> blocks;
          for (int i = 0; i < size; ++i) {
            blocks.push_back(elements == 0 ? mp1::layout<Widget>::contiguous(0)
                                           : mp1::layout<Widget>::contiguous(elements));
          }
          mp1::gatherv(c, root, send, recv, blocks, displs);
        }
        for (int i = 0; i < size; ++i) {
          for (int j = 0; j < elements; ++j) {
            const Widget& value = recv[static_cast<std::size_t>(i * elements + j)];
            const Widget expected = widget_value(i, j);
            BOOST_TEST(value.id == expected.id);
            BOOST_TEST(static_cast<unsigned>(value.flag) ==
                       static_cast<unsigned>(expected.flag));
          }
        }
      } else {
        // Every non-root spelling sends the same single flat block; the descriptor lives on the
        // root, so the documented dummy receive position is never inspected.
        mp1::gatherv(c, root, send, std::span<Widget>{}, mp1::blocks{});
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(tail_padded_scatterv_pointer_and_layout) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<Widget> send(static_cast<std::size_t>(elements * size));
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < elements; ++j)
      send[static_cast<std::size_t>(i * elements + j)] = widget_value(i, j);
  }

  // Root sends from dense counts; non-roots receive the pointer+count form (the observed
  // receive corruption went through this single-block lowering).
  std::vector<Widget> recv(static_cast<std::size_t>(elements), Widget{0xFFFFFFFFU, 0xFFU});
  if (rank == root) {
    mp1::scatterv(c, root, send, mp1::blocks{counts, displs}, recv);
  } else {
    mp1::scatterv(c, root, std::span<const Widget>{}, mp1::blocks{}, recv);
  }
  for (int j = 0; j < elements; ++j) {
    const Widget& value = recv[static_cast<std::size_t>(j)];
    const Widget expected = widget_value(rank, j);
    BOOST_TEST(value.id == expected.id);
    BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
  }

  // Layout spelling on both sides: the root's send map is a native repetition per rank and the
  // non-root receive map is contiguous(elements).
  std::vector<mp1::layout<Widget>> blocks;
  for (int i = 0; i < size; ++i)
    blocks.push_back(mp1::layout<Widget>::contiguous(elements));
  std::vector<Widget> recv2(static_cast<std::size_t>(elements), Widget{0xFFFFFFFFU, 0xFFU});
  if (rank == root) {
    mp1::scatterv(c, root, send, blocks, displs, recv2);
  } else {
    mp1::scatterv(c, root, std::span<const Widget>{}, mp1::blocks{}, recv2);
  }
  for (int j = 0; j < elements; ++j) {
    const Widget& value = recv2[static_cast<std::size_t>(j)];
    const Widget expected = widget_value(rank, j);
    BOOST_TEST(value.id == expected.id);
    BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
  }
}

BOOST_AUTO_TEST_CASE(tail_padded_allgatherv_and_alltoallv) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  constexpr int elements = 2;

  // allgatherv, dense counts on both sides.
  {
    const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
    const std::vector<mp1::count_t> displs = prefix_displacements(counts);
    std::vector<Widget> send(static_cast<std::size_t>(elements));
    for (int j = 0; j < elements; ++j)
      send[static_cast<std::size_t>(j)] = widget_value(rank, j);
    std::vector<Widget> recv(static_cast<std::size_t>(elements * size),
                             Widget{0xFFFFFFFFU, 0xFFU});
    mp1::allgatherv(c, send, recv, mp1::blocks{counts, displs});
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements; ++j) {
        const Widget& value = recv[static_cast<std::size_t>(i * elements + j)];
        const Widget expected = widget_value(i, j);
        BOOST_TEST(value.id == expected.id);
        BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
      }
    }
  }

  // alltoallv, dense counts: rank r sends widget_value(r*10 + peer, j) to peer.
  {
    const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
    const std::vector<mp1::count_t> displs = prefix_displacements(counts);
    std::vector<Widget> send(static_cast<std::size_t>(elements * size));
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j < elements; ++j)
        send[static_cast<std::size_t>(peer * elements + j)] = widget_value(rank * 10 + peer, j);
    }
    std::vector<Widget> recv(static_cast<std::size_t>(elements * size),
                             Widget{0xFFFFFFFFU, 0xFFU});
    mp1::alltoallv(c, send, mp1::blocks{counts, displs}, recv, mp1::blocks{counts, displs});
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j < elements; ++j) {
        const Widget& value = recv[static_cast<std::size_t>(peer * elements + j)];
        const Widget expected = widget_value(peer * 10 + rank, j);
        BOOST_TEST(value.id == expected.id);
        BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
      }
    }
  }

  // alltoallv with a staged (holey) layout on both sides: one Widget at stride 2 extents.
  {
    const mp1::layout<Widget> holey = holey_widget_layout();
    std::vector<mp1::layout<Widget>> blocks(static_cast<std::size_t>(size), holey);
    const std::vector<mp1::count_t> displs = prefix_displacements(uniform_counts(size, 2));
    const std::size_t total = static_cast<std::size_t>(2 * size);
    std::vector<Widget> send(total, Widget{0xFFFFFFFFU, 0xFFU});
    std::vector<Widget> recv(total, Widget{0xFFFFFFFFU, 0xFFU});
    for (int peer = 0; peer < size; ++peer) {
      send[static_cast<std::size_t>(peer * 2)] = widget_value(rank * 10 + peer, 0);
      // The second slot at +1 is the hole of the vector layout and keeps its canary.
    }
    mp1::alltoallv(c, send, blocks, displs, recv, blocks, displs);
    for (int peer = 0; peer < size; ++peer) {
      const Widget& value = recv[static_cast<std::size_t>(peer * 2)];
      const Widget expected = widget_value(peer * 10 + rank, 0);
      BOOST_TEST(value.id == expected.id);
      BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
      BOOST_TEST(recv[static_cast<std::size_t>(peer * 2 + 1)].id == 0xFFFFFFFFU);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Nonzero true lower bound: MPI adds the datatype's lb itself (MPI 4.1 5.1.11), so a native T
// whose only reflected member starts at byte 4 must put the values at object offset 4 and leave
// the leading padding untouched.  A flat byte image or a second manual lb shift corrupts this.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(nonzero_true_lb) {
  const mp1::detail::datatype_info& info = mp1::detail::datatype_info_of<shifted_widget>();
  BOOST_TEST(static_cast<long long>(info.size) == 4LL);
  BOOST_TEST(static_cast<long long>(info.extent) ==
             static_cast<long long>(sizeof(shifted_widget)));
  BOOST_TEST(static_cast<long long>(info.true_lb) == 4LL);
  BOOST_TEST(static_cast<long long>(info.true_extent) == 4LL);

  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  constexpr int elements = 2;
  constexpr std::uint32_t padding_canary = 0xBBBBBBBBU;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);

  std::vector<shifted_widget> send(static_cast<std::size_t>(elements));
  for (int j = 0; j < elements; ++j) {
    send[static_cast<std::size_t>(j)] =
        shifted_widget{0xAAAAAAAAU, static_cast<std::uint32_t>(rank * 100 + j)};
  }

  // Gatherv: the peers send the pointer+count spelling and the root receives dense counts.
  {
    std::vector<shifted_widget> recv(static_cast<std::size_t>(elements * size),
                                     shifted_widget{padding_canary, 0xFFFFFFFFU});
    if (rank == root) {
      mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < elements; ++j) {
          const shifted_widget& value = recv[static_cast<std::size_t>(i * elements + j)];
          BOOST_TEST(value.value == static_cast<std::uint32_t>(i * 100 + j));
          // Only the reflected value member is in the type map; the leading padding is outside
          // it and must not be written (MPI 4.1 5.1.11 advice to implementors).
          BOOST_TEST(value.padding == padding_canary);
        }
      }
    } else {
      mp1::gatherv(c, root, send, std::span<shifted_widget>{}, mp1::blocks{});
    }
  }

  // Allgatherv: contiguous layout send into a layouts<shifted_widget>::contiguous(elements)
  // receive.  The layout is a native repetition whose lb is 4, so this is the layout lowering
  // with a nonzero true lower bound.
  {
    std::vector<mp1::layout<shifted_widget>> blocks;
    for (int i = 0; i < size; ++i)
      blocks.push_back(mp1::layout<shifted_widget>::contiguous(elements));
    std::vector<shifted_widget> recv(static_cast<std::size_t>(elements * size),
                                     shifted_widget{padding_canary, 0xFFFFFFFFU});
    mp1::allgatherv(c, send, recv, blocks, displs);
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements; ++j) {
        const shifted_widget& value = recv[static_cast<std::size_t>(i * elements + j)];
        BOOST_TEST(value.value == static_cast<std::uint32_t>(i * 100 + j));
        BOOST_TEST(value.padding == padding_canary);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------
// Dense-but-not-a-native-repetition layout: an hindexed_parameter with blocklength 1 at
// displacements 0 and 5 tiles 10 bytes (dense() is true) but its second copy starts at
// byte 5, not at the Widget extent stride, so it is not a base_repetition.  The typed wire
// must stage it instead of lowering a flat byte image.  It is also the mixed tail-padded
// spelling: the root receives
// layouts<Widget> while the peers send the pointer+count Widget[2] form (the same type
// signature).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(packed_spaced_layout_stages) {
  const mp1::layout<Widget> spaced = packed_spaced_widget_layout();
  BOOST_TEST(static_cast<long long>(spaced.size()) == 2LL);
  BOOST_TEST(static_cast<long long>(spaced.byte_size()) == 10LL);
  BOOST_TEST(spaced.dense());
  BOOST_TEST(not spaced.is_base_repetition());

  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  constexpr int elements = 2;
  // Two Widget extents (16 bytes) per block: the 10-byte map fits and consecutive blocks do not
  // alias.  Displacements stay in Widget extent units (MPI 4.1 6.5/6.6/6.8).
  constexpr int stride = 2;
  std::vector<mp1::layout<Widget>> blocks;
  for (int i = 0; i < size; ++i)
    blocks.push_back(spaced);
  std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * stride;

  std::vector<Widget> send(static_cast<std::size_t>(elements));
  for (int j = 0; j < elements; ++j)
    send[static_cast<std::size_t>(j)] = widget_value(rank, j);

  // Gatherv: the root stages the spaced receive map; the peers send contiguous Widget[2].
  {
    std::vector<Widget> recv(static_cast<std::size_t>(stride * size));
    std::memset(recv.data(), 0xEE, recv.size() * sizeof(Widget));
    const mp1::layout<Widget> contiguous = mp1::layout<Widget>::contiguous(elements);
    if (rank == root) {
      mp1::gatherv(c, root, send, recv, blocks, displs);
      for (int i = 0; i < size; ++i) {
        const std::size_t base = static_cast<std::size_t>(i * stride * sizeof(Widget));
        const Widget first = widget_value(i, 0);
        const Widget second = widget_value(i, 1);
        BOOST_TEST(load_u32(recv, base) == first.id);
        BOOST_TEST(static_cast<unsigned>(load_u8(recv, base + 4)) ==
                   static_cast<unsigned>(first.flag));
        BOOST_TEST(load_u32(recv, base + 5) == second.id);
        BOOST_TEST(static_cast<unsigned>(load_u8(recv, base + 9)) ==
                   static_cast<unsigned>(second.flag));
        // Bytes [10,16) are outside the type map and keep the caller's canary.
        for (std::size_t canary = 10; canary < sizeof(Widget) * 2; ++canary)
          BOOST_TEST(static_cast<unsigned>(load_u8(recv, base + canary)) == 0xEEU);
      }
    } else {
      mp1::gatherv(c, root, send, std::span<Widget>{}, mp1::blocks{});
    }
  }

  // Scatterv: the root sends the staged spaced map; every rank receives contiguous Widget[2].
  {
    std::vector<Widget> scatter_send(static_cast<std::size_t>(stride * size));
    std::memset(scatter_send.data(), 0xEE, scatter_send.size() * sizeof(Widget));
    for (int i = 0; i < size; ++i) {
      const std::size_t base = static_cast<std::size_t>(i * stride * sizeof(Widget));
      const Widget first = widget_value(i, 0);
      const Widget second = widget_value(i, 1);
      store_widget_bytes(scatter_send, base, first.id, first.flag);
      store_widget_bytes(scatter_send, base + 5, second.id, second.flag);
    }
    const mp1::layout<Widget> contiguous = mp1::layout<Widget>::contiguous(elements);
    std::vector<Widget> local(static_cast<std::size_t>(elements), Widget{0xEEEEEEEEU, 0xEEU});
    if (rank == root) {
      mp1::scatterv(c, root, scatter_send, blocks, displs, local);
    } else {
      mp1::scatterv(c, root, std::span<const Widget>{}, mp1::blocks{}, local);
    }
    for (int j = 0; j < elements; ++j) {
      const Widget expected = widget_value(rank, j);
      BOOST_TEST(local[static_cast<std::size_t>(j)].id == expected.id);
      BOOST_TEST(static_cast<unsigned>(local[static_cast<std::size_t>(j)].flag) ==
                 static_cast<unsigned>(expected.flag));
    }
  }
}

// ---------------------------------------------------------------------------------------
// A custom zero-size native datatype (MPI_Type_size_c == 0, extent == sizeof(T)).  Every block
// is empty by the n * size == 0 rule, so the empty normalization must use a dummy pointer with
// the element count and no byte may be written.  The datatype itself is created by a trait
// because no reflected type can have size zero.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(zero_size_native_datatype) {
  const mp1::detail::datatype_info& info = mp1::detail::datatype_info_of<zero_size_value>();
  BOOST_TEST(static_cast<long long>(info.size) == 0LL);
  BOOST_TEST(static_cast<long long>(info.extent) ==
             static_cast<long long>(sizeof(zero_size_value)));
  BOOST_TEST(static_cast<long long>(info.true_lb) == 0LL);
  BOOST_TEST(static_cast<long long>(info.true_extent) == 0LL);

  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);

  // S == 0 normalization: a positive element count of a zero-size native datatype is an empty
  // type signature, so the plan normalizes the wire count to 0 and uses the dummy pointer.
  // MPICH 4.3.2 is reproduced to abort (SIGFPE) on the equivalent raw positive count of a
  // resized zero-size type in the variable collectives; count 0 is standard-conforming and
  // portable (MPI 4.1 §5.1.11).
  {
    mp1::detail::xfer_plan<zero_size_value> single =
        mp1::detail::xfer_plan<zero_size_value>::one_base(
            nullptr, elements, mp1::detail::transfer_direction::send);
    BOOST_TEST(single.count() == 0);
    BOOST_TEST(single.ptr() != nullptr);
    mp1::detail::xfer_plan<zero_size_value> many =
        mp1::detail::xfer_plan<zero_size_value>::many_counts(
            nullptr, counts, displs, static_cast<std::size_t>(size),
            mp1::detail::transfer_direction::send);
    BOOST_TEST(many.count() == 0);
    BOOST_TEST(many.ptr() != nullptr);

    // A typed zero-byte layout with a positive repetition count is also normalized: the type
    // signature is empty even though size() is 2.
    const mp1::layout<zero_size_value> repeated_empty =
        mp1::layout<zero_size_value>::vector(2, 1, 1);
    BOOST_TEST(repeated_empty.is_valid());
    BOOST_TEST(repeated_empty.size() == 2);
    BOOST_TEST(repeated_empty.byte_size() == 0);
    mp1::detail::xfer_plan<zero_size_value> typed =
        mp1::detail::xfer_plan<zero_size_value>::one(nullptr, repeated_empty,
                                                     mp1::detail::transfer_direction::send);
    BOOST_TEST(typed.count() == 0);
    BOOST_TEST(typed.ptr() != nullptr);
  }

  // The public path with positive dense counts and empty spans: the dense fit check must not
  // compare a zero-byte block against the caller's (empty) span, and the normalized wire count
  // must not abort a provider.  This is the exact shape MPICH 4.3.2 rejected with a raw call.
  {
    std::vector<zero_size_value> recv(1, zero_size_value{0xEEU});
    if (rank == root) {
      mp1::gatherv(c, root, std::span<const zero_size_value>{}, recv,
                   mp1::blocks{counts, displs});
    } else {
      mp1::gatherv(c, root, std::span<const zero_size_value>{}, std::span<zero_size_value>{},
                   mp1::blocks{});
    }
    BOOST_TEST(static_cast<unsigned>(recv[0].tag) == 0xEEU);

    std::vector<zero_size_value> a2a(1, zero_size_value{0xEEU});
    mp1::alltoallv(c, std::span<const zero_size_value>{}, mp1::blocks{counts, displs}, a2a,
                   mp1::blocks{counts, displs});
    BOOST_TEST(static_cast<unsigned>(a2a[0].tag) == 0xEEU);

    if (rank == root) {
      mp1::scatterv(c, root, std::span<const zero_size_value>{}, mp1::blocks{counts, displs},
                    std::span<zero_size_value>{});
    } else {
      mp1::scatterv(c, root, std::span<const zero_size_value>{}, mp1::blocks{},
                    std::span<zero_size_value>{});
    }
  }

  // The typed zero-byte layout spelling: a block whose map addresses no byte, with a positive
  // repetition count and empty caller spans.  The per-block fit check must skip it and the
  // normalized wire count must reach MPI as 0.
  {
    const mp1::layout<zero_size_value> repeated_empty =
        mp1::layout<zero_size_value>::vector(2, 1, 1);
    const mp1::region<const zero_size_value> empty_send{nullptr, repeated_empty};
    std::vector<mp1::layout<zero_size_value>> blocks(static_cast<std::size_t>(size),
                                                     repeated_empty);
    const std::vector<mp1::count_t> zero_displacements(static_cast<std::size_t>(size), 0);
    std::vector<zero_size_value> canary(1, zero_size_value{0xEEU});
    mp1::allgatherv(c, empty_send, canary, blocks, zero_displacements);
    BOOST_TEST(static_cast<unsigned>(canary[0].tag) == 0xEEU);
  }

  // The same zero-size datatype in a fixed collective: the single-block count is normalized
  // to 0, so a positive count of the empty type never reaches the provider.
  {
    std::vector<zero_size_value> fixed_send(1, zero_size_value{0xAAU});
    std::vector<zero_size_value> fixed_recv(1, zero_size_value{0xEEU});
    mp1::bcast(c, 0, fixed_recv);
    BOOST_TEST(static_cast<unsigned>(fixed_recv[0].tag) == 0xEEU);
    std::vector<zero_size_value> gathered(static_cast<std::size_t>(size),
                                          zero_size_value{0xEEU});
    mp1::allgather(c, fixed_send, gathered);
    for (const zero_size_value& value : gathered)
      BOOST_TEST(static_cast<unsigned>(value.tag) == 0xEEU);
  }

  // Portable public path with S == 0: all-zero counts.  The provider handles count 0 of a
  // zero-size datatype on both installed MPIs, and no byte may be written.
  {
    const std::vector<mp1::count_t> zero_counts = uniform_counts(size, 0);
    const std::vector<mp1::count_t> zero_displs = prefix_displacements(zero_counts);
    std::vector<zero_size_value> recv(1, zero_size_value{0xEEU});
    if (rank == root) {
      mp1::gatherv(c, root, std::span<const zero_size_value>{}, recv,
                   mp1::blocks{zero_counts, zero_displs});
    } else {
      mp1::gatherv(c, root, std::span<const zero_size_value>{}, std::span<zero_size_value>{},
                   mp1::blocks{});
    }
    BOOST_TEST(static_cast<unsigned>(recv[0].tag) == 0xEEU);

    std::vector<zero_size_value> a2a(1, zero_size_value{0xEEU});
    mp1::alltoallv(c, a2a, mp1::blocks{zero_counts, zero_displs}, a2a,
                   mp1::blocks{zero_counts, zero_displs});
    BOOST_TEST(static_cast<unsigned>(a2a[0].tag) == 0xEEU);
  }
}

// ---------------------------------------------------------------------------------------
// Raw-MPI peers: the wire must be the native type signature with element counts.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(raw_peer_native_int) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 3;

  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<int> send(static_cast<std::size_t>(elements));
  for (int j = 0; j < elements; ++j)
    send[static_cast<std::size_t>(j)] = rank * 10 + j;

  std::vector<int> recv(static_cast<std::size_t>(elements * size), -1);
  std::vector<MPI_Count> ignored_counts(static_cast<std::size_t>(size), 0);
  std::vector<MPI_Aint> ignored_displs(static_cast<std::size_t>(size), 0);
  std::vector<int> ignored_recv(static_cast<std::size_t>(elements), 0);
  if (rank == root) {
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(i * elements + j)] == i * 10 + j);
    }
  } else {
    // A raw peer uses the native type (MPI_INT) and the same element count.
    BOOST_REQUIRE(MPI_Gatherv_c(send.data(), static_cast<MPI_Count>(elements), MPI_INT,
                                ignored_recv.data(), ignored_counts.data(),
                                ignored_displs.data(), MPI_INT, root,
                                c.native_handle()) == MPI_SUCCESS);
  }

  // The reverse role: a raw root sends and the mp1 side receives pointer+count.
  std::vector<int> scatter_send(static_cast<std::size_t>(elements * size), 0);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < elements; ++j)
      scatter_send[static_cast<std::size_t>(i * elements + j)] = i * 100 + j;
  }
  std::vector<int> local(static_cast<std::size_t>(elements), -1);
  std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(size), elements);
  std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(size), 0);
  for (int i = 0; i < size; ++i)
    raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i * elements);
  // Exactly one collective per step: the raw root sends, the mp1 side receives pointer+count.
  // The root also supplies a real receive buffer for its own segment: MPI_Scatterv delivers the
  // root's block to the root too, so a null receive there is a truncated self-message.
  if (rank == root) {
    BOOST_REQUIRE(MPI_Scatterv_c(scatter_send.data(), raw_counts.data(), raw_displs.data(),
                                 MPI_INT, local.data(), static_cast<MPI_Count>(elements),
                                 MPI_INT, root, c.native_handle()) == MPI_SUCCESS);
  } else {
    mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, local);
  }
  for (int j = 0; j < elements; ++j)
    BOOST_TEST(local[static_cast<std::size_t>(j)] == rank * 100 + j);
}

BOOST_AUTO_TEST_CASE(raw_peer_reflected_widget) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 2;

  const raw_widget_datatype raw_type;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<Widget> send(static_cast<std::size_t>(elements));
  for (int j = 0; j < elements; ++j)
    send[static_cast<std::size_t>(j)] = widget_value(rank, j);

  std::vector<Widget> recv(static_cast<std::size_t>(elements * size),
                           Widget{0xFFFFFFFFU, 0xFFU});
  if (rank == root) {
    // The mp1 side sends native_datatype<Widget>(); the message signature is the two basic
    // components repeated twice.
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements; ++j) {
        const Widget& value = recv[static_cast<std::size_t>(i * elements + j)];
        const Widget expected = widget_value(i, j);
        BOOST_TEST(value.id == expected.id);
        BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
      }
    }
  } else {
    std::vector<Widget> ignored(static_cast<std::size_t>(elements), Widget{0, 0});
    BOOST_REQUIRE(MPI_Gatherv_c(send.data(), static_cast<MPI_Count>(elements), raw_type.get(),
                                ignored.data(), nullptr, nullptr, raw_type.get(), root,
                                c.native_handle()) == MPI_SUCCESS);
  }
}

// ---------------------------------------------------------------------------------------
// A receive must not modify storage outside the type map (MPI 4.1 5.1.11 "Advice to
// implementors": padding space in a structure is not part of the message).  A raw peer
// receives a repeated tail-padded Widget into a sentinel-filled byte buffer and checks that the
// three padding bytes of every element are untouched.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(raw_peer_receive_does_not_write_padding) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 2;

  const raw_widget_datatype raw_type;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<Widget> whole(static_cast<std::size_t>(elements * size),
                            Widget{0xFFFFFFFFU, 0xFFU});
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < elements; ++j)
      whole[static_cast<std::size_t>(i * elements + j)] = widget_value(i, j);
  }
  std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(size), elements);
  std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(size), 0);
  for (int i = 0; i < size; ++i)
    raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i * elements);

  if (rank == root) {
    std::vector<Widget> local(static_cast<std::size_t>(elements), Widget{0, 0});
    mp1::scatterv(c, root, whole, mp1::blocks{counts, displs}, local);
  } else {
    // Over-sized, over-aligned storage prefilled with a sentinel: the receive may only write
    // the id (bytes 0..4) and flag (byte 4) of each element.
    alignas(Widget) unsigned char storage[2 * sizeof(Widget)];
    std::memset(storage, 0xEE, sizeof(storage));
    BOOST_REQUIRE(MPI_Scatterv_c(whole.data(), raw_counts.data(), raw_displs.data(),
                                 raw_type.get(), storage, static_cast<MPI_Count>(elements),
                                 raw_type.get(), root, c.native_handle()) == MPI_SUCCESS);
    const auto* got = reinterpret_cast<const Widget*>(storage);
    for (int j = 0; j < elements; ++j) {
      const Widget expected = widget_value(rank, j);
      BOOST_TEST(got[j].id == expected.id);
      BOOST_TEST(static_cast<unsigned>(got[j].flag) == static_cast<unsigned>(expected.flag));
      const std::size_t base = static_cast<std::size_t>(j) * sizeof(Widget);
      for (std::size_t padding = 5; padding < sizeof(Widget); ++padding)
        BOOST_TEST(static_cast<unsigned>(storage[base + padding]) == 0xEEU);
    }
  }
}

// ---------------------------------------------------------------------------------------
// std::byte: the native datatype is MPI_BYTE, so the wire is a byte message with a byte count.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(std_byte_wire_dense_and_layout) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 3;

  // Dense: mp1 root, raw MPI_BYTE peers.  This is the documented type-erasure path, and it
  // must stay byte-exact.
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<std::byte> send(static_cast<std::size_t>(elements));
  for (int j = 0; j < elements; ++j)
    send[static_cast<std::size_t>(j)] = static_cast<std::byte>((rank * 16 + j) & 0xFF);
  std::vector<std::byte> recv(static_cast<std::size_t>(elements * size),
                              static_cast<std::byte>(0xEE));
  if (rank == root) {
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements; ++j) {
        BOOST_TEST(static_cast<unsigned>(recv[static_cast<std::size_t>(i * elements + j)]) ==
                   static_cast<unsigned>((i * 16 + j) & 0xFF));
      }
    }
  } else {
    std::vector<std::byte> ignored(static_cast<std::size_t>(elements),
                                   static_cast<std::byte>(0));
    BOOST_REQUIRE(MPI_Gatherv_c(send.data(), static_cast<MPI_Count>(elements), MPI_BYTE,
                                ignored.data(), nullptr, nullptr, MPI_BYTE, root,
                                c.native_handle()) == MPI_SUCCESS);
  }

  // Layout with a hole: the hole byte of the send buffer must not reach the receiver and the
  // receiver's hole must keep its canary (staged std::byte lowering, still MPI_BYTE).
  const mp1::layout<std::byte> holey = mp1::layout<std::byte>::vector(2, 1, 2);
  std::vector<mp1::layout<std::byte>> blocks(static_cast<std::size_t>(size), holey);
  const std::vector<mp1::count_t> holey_displs = extent_displacements(size, holey);
  // The send side is one block at this rank's pointer: its own two values live at element
  // offsets 0 and 2, and the hole keeps 0xAB locally (it is never sent).
  std::vector<std::byte> holey_send(3, static_cast<std::byte>(0xAB));
  holey_send[0] = static_cast<std::byte>((rank * 16) & 0xFF);
  holey_send[2] = static_cast<std::byte>((rank * 16 + 8) & 0xFF);
  std::vector<std::byte> holey_recv(static_cast<std::size_t>(3 * size),
                                    static_cast<std::byte>(0xEE));
  mp1::allgatherv(c, mp1::region{holey_send.data(), holey}, holey_recv, blocks, holey_displs);
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(static_cast<unsigned>(holey_recv[static_cast<std::size_t>(3 * i)]) ==
               static_cast<unsigned>((i * 16) & 0xFF));
    BOOST_TEST(static_cast<unsigned>(holey_recv[static_cast<std::size_t>(3 * i + 2)]) ==
               static_cast<unsigned>((i * 16 + 8) & 0xFF));
    BOOST_TEST(static_cast<unsigned>(holey_recv[static_cast<std::size_t>(3 * i + 1)]) == 0xEEU);
  }
}

// ---------------------------------------------------------------------------------------
// Holes and canaries across a staged send and a staged receive, mixed with a dense peer.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(holey_layouts_preserve_canaries) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const mp1::layout<int> holey = holey_int_layout();

  std::vector<mp1::layout<int>> recv_blocks(static_cast<std::size_t>(size), holey);
  const std::vector<mp1::count_t> displs = extent_displacements(size, holey);
  std::vector<int> send(3, 0);
  send[0] = rank * 100 + 1;
  send[1] = -7;  // canary in the hole
  send[2] = rank * 100 + 2;
  // The dense spelling of the same two elements is a contiguous pair: the hole is not part of
  // the message, which is exactly why both spellings produce the same wire signature.
  const std::array<int, 2> dense_send{{rank * 100 + 1, rank * 100 + 2}};

  // Root stages the receive; even-numbered peers send the layout spelling and odd-numbered
  // peers send the dense pointer+count spelling of the same element count.  Both produce the
  // same wire signature, and the root's holes stay untouched.
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
  if (rank == root) {
    mp1::gatherv(c, root, mp1::region{send.data(), holey}, recv, recv_blocks, displs);
  } else if (rank % 2 == 0) {
    mp1::gatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{}, mp1::blocks{});
  } else {
    mp1::gatherv(c, root, dense_send, std::span<int>{}, mp1::blocks{});
  }
  BOOST_TEST(send[1] == -7);
  if (rank == root) {
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 100 + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i * 100 + 2);
    }
  }

  // The same on allgatherv: a staged receive and a staged send in one collective.
  std::vector<int> all_recv(static_cast<std::size_t>(3 * size), -1);
  mp1::allgatherv(c, mp1::region{send.data(), holey}, all_recv, recv_blocks, displs);
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(all_recv[static_cast<std::size_t>(3 * i)] == i * 100 + 1);
    BOOST_TEST(all_recv[static_cast<std::size_t>(3 * i + 1)] == -1);
    BOOST_TEST(all_recv[static_cast<std::size_t>(3 * i + 2)] == i * 100 + 2);
  }
}

// ---------------------------------------------------------------------------------------
// In-place forms: the local contribution is already in the receive buffer, so a staged
// receive must be primed before the call (MPI 4.1 6.5/6.7/6.8) and scatterv must not move the
// root's own segment (6.6).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(in_place_gatherv_and_allgatherv) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const mp1::layout<int> holey = holey_int_layout();
  const std::vector<mp1::count_t> displs = extent_displacements(size, holey);

  // gatherv in-place at the root with a staged receive.  Every rank's two values are placed
  // directly in the root's receive buffer before the call.
  {
    std::vector<int> send(3, -7);
    send[0] = rank * 100 + 1;
    send[2] = rank * 100 + 2;
    std::vector<mp1::layout<int>> blocks(static_cast<std::size_t>(size), holey);
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    if (rank == root) {
      for (int i = 0; i < size; ++i) {
        recv[static_cast<std::size_t>(3 * i)] = i * 100 + 1;
        recv[static_cast<std::size_t>(3 * i + 2)] = i * 100 + 2;
      }
      mp1::gatherv(c, root, mp1::in_place, recv, blocks, displs);
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 100 + 1);
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i * 100 + 2);
      }
    } else {
      mp1::gatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{}, mp1::blocks{});
    }
  }

  // allgatherv in-place with a staged receive: every rank primes its own block.
  {
    std::vector<mp1::layout<int>> blocks(static_cast<std::size_t>(size), holey);
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      recv[static_cast<std::size_t>(3 * i)] = i * 100 + 1;
      recv[static_cast<std::size_t>(3 * i + 2)] = i * 100 + 2;
    }
    mp1::allgatherv(c, mp1::in_place, recv, blocks, displs);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 100 + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i * 100 + 2);
    }
  }
}

BOOST_AUTO_TEST_CASE(in_place_alltoallv_and_scatterv) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  constexpr int elements = 2;

  // alltoallv in-place: the send data is taken from the receive buffer, so a staged receive
  // must copy every block into the staging image before the call (MPI 4.1 6.8).
  {
    const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
    const std::vector<mp1::count_t> displs = prefix_displacements(counts);
    std::vector<int> buffer(static_cast<std::size_t>(elements * size), -1);
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j < elements; ++j)
        buffer[static_cast<std::size_t>(peer * elements + j)] = rank * 100 + peer * 10 + j;
    }
    mp1::alltoallv(c, mp1::in_place, mp1::blocks{}, buffer, mp1::blocks{counts, displs});
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j < elements; ++j) {
        BOOST_TEST(buffer[static_cast<std::size_t>(peer * elements + j)] ==
                   peer * 100 + rank * 10 + j);
      }
    }
  }

  // scatterv in-place: the root's own segment is not moved, and the caller's send buffer is
  // read-only (MPI 4.1 6.6).
  {
    const mp1::layout<int> holey = holey_int_layout();
    std::vector<mp1::layout<int>> blocks(static_cast<std::size_t>(size), holey);
    const std::vector<mp1::count_t> displs = extent_displacements(size, holey);
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = i * 100 + 1;
      send[static_cast<std::size_t>(3 * i + 2)] = i * 100 + 2;
    }
    const std::vector<int> send_copy = send;
    if (rank == root) {
      mp1::scatterv(c, root, send, blocks, displs, mp1::in_place);
    } else {
      std::vector<int> local(3, -1);
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region{local.data(), holey});
      BOOST_TEST(local[0] == rank * 100 + 1);
      BOOST_TEST(local[1] == -1);
      BOOST_TEST(local[2] == rank * 100 + 2);
    }
    BOOST_TEST(send == send_copy);
  }
}

// ---------------------------------------------------------------------------------------
// Lifetimes: three persistent cycles with a staged send and a staged receive, plus a
// nonblocking staged receive whose descriptors die before completion.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(persistent_cycles_repack) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> holey = holey_int_layout();

  // The counts/displacements arrays are destroyed after init: the persistent plan owns its
  // converted arrays (MPI 4.1 6.13 forbids modifying them until free).
  mp1::request persistent;
  {
    const std::vector<mp1::count_t> displs = extent_displacements(size, holey);
    std::vector<mp1::layout<int>> blocks(static_cast<std::size_t>(size), holey);
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    const std::span<int> recv_span{recv};
    persistent =
        mp1::allgatherv_init(c, mp1::region{send.data(), holey}, recv_span, blocks, displs);
    // Three cycles; the staged send is re-packed from the caller's buffer at every start, so
    // the values written between starts must be observed.
    for (int cycle = 0; cycle < 3; ++cycle) {
      // The allgatherv send side is a single block at the caller's pointer (no displacement),
      // so every rank maps its own values at element offsets 0 and 2 of its own send buffer.
      send[0] = rank * 1000 + cycle * 10 + 1;
      send[1] = -7;  // hole canary
      send[2] = rank * 1000 + cycle * 10 + 2;
      persistent.start();
      persistent.wait();
      for (int i = 0; i < size; ++i) {
        const int expected_one = i * 1000 + cycle * 10 + 1;
        const int expected_two = i * 1000 + cycle * 10 + 2;
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == expected_one);
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == expected_two);
        // The hole byte is never written by the staged unpack.
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(nonblocking_lifetime_descriptors) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const mp1::layout<int> holey = holey_int_layout();
  const std::vector<mp1::count_t> displs = extent_displacements(size, holey);

  std::vector<int> send(3, -1);
  send[0] = rank * 100 + 1;
  send[2] = rank * 100 + 2;
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);

  // The layout array is destroyed before the request completes; the payload owns the staged
  // receive descriptors (copy-on-need), so completion still unpacks a valid type map.
  mp1::request pending;
  {
    std::vector<mp1::layout<int>> blocks(static_cast<std::size_t>(size), holey);
    if (rank == root) {
      pending = mp1::igatherv(c, root, mp1::region{send.data(), holey}, recv, blocks, displs);
    } else {
      pending = mp1::igatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{},
                              mp1::blocks{});
    }
  }
  pending.wait();
  if (rank == root) {
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 100 + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i * 100 + 2);
    }
  }
}

// ---------------------------------------------------------------------------------------
// Intercommunicator: the two groups may carry different element counts, and the tail-padded
// type must stay exact (MPI 4.1 6.7 advice to users).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(intercommunicator_widget_counts) {
  const mp1::mpi_communicator world = mp1::comm_world();
  if (world.size() < 2) {
    BOOST_TEST_MESSAGE("intercommunicator test needs at least 2 ranks");
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  const mp1::communicator local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{913});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();

  // Group 0 contributes one Widget, group 1 two, so each group receives the peer group's
  // count for every remote rank.
  const int send_elements = color == 0 ? 1 : 2;
  const int recv_elements = color == 0 ? 2 : 1;
  std::vector<Widget> send(static_cast<std::size_t>(send_elements));
  for (int j = 0; j < send_elements; ++j)
    send[static_cast<std::size_t>(j)] = widget_value(color * 1000 + local_rank, j);
  const std::vector<mp1::count_t> recv_counts = uniform_counts(remote_size, recv_elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(recv_counts);
  std::vector<Widget> recv(static_cast<std::size_t>(recv_elements * remote_size),
                           Widget{0xFFFFFFFFU, 0xFFU});

  mp1::allgatherv(inter, send, recv, mp1::blocks{recv_counts, displs});
  for (int i = 0; i < remote_size; ++i) {
    for (int j = 0; j < recv_elements; ++j) {
      const Widget& value = recv[static_cast<std::size_t>(i * recv_elements + j)];
      const Widget expected = widget_value((1 - color) * 1000 + i, j);
      BOOST_TEST(value.id == expected.id);
      BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
    }
  }

  inter.disconnect();
}

// Intercommunicator alltoallv with the tail-padded type: send and receive counts refer to the
// peer group, so each group may carry a different per-peer count.  The wire stays
// native_datatype<Widget>() with element counts (MPI 4.1 6.8); no root role is involved.
BOOST_AUTO_TEST_CASE(intercommunicator_alltoallv_widget) {
  const mp1::mpi_communicator world = mp1::comm_world();
  if (world.size() < 2) {
    BOOST_TEST_MESSAGE("intercommunicator test needs at least 2 ranks");
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  const mp1::communicator local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{914});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();

  // Group 0 sends one Widget per peer, group 1 sends two; each group therefore receives the
  // peer group's count.  Counts and displacements are indexed by the remote size.
  const int send_elements = color == 0 ? 1 : 2;
  const int recv_elements = color == 0 ? 2 : 1;
  const std::vector<mp1::count_t> send_counts = uniform_counts(remote_size, send_elements);
  const std::vector<mp1::count_t> recv_counts = uniform_counts(remote_size, recv_elements);
  const std::vector<mp1::count_t> send_displs = prefix_displacements(send_counts);
  const std::vector<mp1::count_t> recv_displs = prefix_displacements(recv_counts);

  std::vector<Widget> send(static_cast<std::size_t>(send_elements * remote_size));
  for (int peer = 0; peer < remote_size; ++peer) {
    for (int j = 0; j < send_elements; ++j) {
      send[static_cast<std::size_t>(peer * send_elements + j)] =
          widget_value(color * 1000 + local_rank, peer * 10 + j);
    }
  }
  std::vector<Widget> recv(static_cast<std::size_t>(recv_elements * remote_size),
                           Widget{0xFFFFFFFFU, 0xFFU});
  mp1::alltoallv(inter, send, mp1::blocks{send_counts, send_displs}, recv,
                 mp1::blocks{recv_counts, recv_displs});
  for (int peer = 0; peer < remote_size; ++peer) {
    for (int j = 0; j < recv_elements; ++j) {
      const Widget& value = recv[static_cast<std::size_t>(peer * recv_elements + j)];
      const Widget expected = widget_value((1 - color) * 1000 + peer, local_rank * 10 + j);
      BOOST_TEST(value.id == expected.id);
      BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
    }
  }

  inter.disconnect();
}

// The intercommunicator root forms: the root group passes MPI_ROOT at its root and
// MPI_PROC_NULL elsewhere, the remote group passes the root's rank in the other group, and the
// count arrays are indexed by the remote size (MPI 4.1 6.2.3, 6.5, 6.6).  In-place is invalid
// on an intercommunicator and is rejected locally before any plan is built.
BOOST_AUTO_TEST_CASE(intercommunicator_gatherv_scatterv_roles) {
  const mp1::mpi_communicator world = mp1::comm_world();
  if (world.size() < 2) {
    BOOST_TEST_MESSAGE("intercommunicator test needs at least 2 ranks");
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  const mp1::communicator local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{917});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();
  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform_counts(remote_size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);

  // gatherv: every rank of the remote group sends `elements` Widgets to the root of group 0.
  {
    std::vector<Widget> send(static_cast<std::size_t>(elements));
    for (int j = 0; j < elements; ++j)
      send[static_cast<std::size_t>(j)] = widget_value(color * 1000 + local_rank, j);
    std::vector<Widget> recv(static_cast<std::size_t>(elements * remote_size),
                             Widget{0xFFFFFFFFU, 0xFFU});
    if (color == 0 and local_rank == 0) {
      mp1::gatherv(inter, MPI_ROOT, std::span<const Widget>{}, recv,
                   mp1::blocks{counts, displs});
      for (int i = 0; i < remote_size; ++i) {
        for (int j = 0; j < elements; ++j) {
          const Widget& value = recv[static_cast<std::size_t>(i * elements + j)];
          const Widget expected = widget_value(1000 + i, j);
          BOOST_TEST(value.id == expected.id);
          BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
        }
      }
    } else if (color == 0) {
      mp1::gatherv(inter, MPI_PROC_NULL, std::span<const Widget>{}, std::span<Widget>{},
                   mp1::blocks{});
    } else {
      mp1::gatherv(inter, 0, send, std::span<Widget>{}, mp1::blocks{});
    }
  }

  // scatterv: the root of group 0 sends `elements` Widgets to every remote rank.
  {
    std::vector<Widget> send(static_cast<std::size_t>(elements * remote_size),
                             Widget{0xFFFFFFFFU, 0xFFU});
    for (int i = 0; i < remote_size; ++i) {
      for (int j = 0; j < elements; ++j)
        send[static_cast<std::size_t>(i * elements + j)] = widget_value(2000 + i, j);
    }
    std::vector<Widget> local_recv(static_cast<std::size_t>(elements),
                                   Widget{0xFFFFFFFFU, 0xFFU});
    if (color == 0 and local_rank == 0) {
      mp1::scatterv(inter, MPI_ROOT, send, mp1::blocks{counts, displs}, std::span<Widget>{});
    } else if (color == 0) {
      mp1::scatterv(inter, MPI_PROC_NULL, std::span<const Widget>{}, mp1::blocks{},
                    std::span<Widget>{});
    } else {
      mp1::scatterv(inter, 0, std::span<const Widget>{}, mp1::blocks{}, local_recv);
      for (int j = 0; j < elements; ++j) {
        const Widget& value = local_recv[static_cast<std::size_t>(j)];
        const Widget expected = widget_value(2000 + local_rank, j);
        BOOST_TEST(value.id == expected.id);
        BOOST_TEST(static_cast<unsigned>(value.flag) == static_cast<unsigned>(expected.flag));
      }
    }
  }

  // Intercommunicator in-place is invalid (MPI 4.1 6.2.3); the root rejects it locally and the
  // other ranks skip the call and synchronize.
  if (color == 0 and local_rank == 0) {
    std::vector<Widget> recv(static_cast<std::size_t>(elements * remote_size), Widget{0, 0});
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(inter, MPI_ROOT, mp1::in_place, recv, mp1::blocks{counts, displs});
    }));
  }
  mp1::barrier(inter);

  inter.disconnect();
}

// ---------------------------------------------------------------------------------------
// Dense-path allocation budget: every blocking dense direction up to K peers is allocation
// free (the plan's count/displacement arrays are inline).  Only meaningful with the wrap
// shim's global operator-new counter.
// ---------------------------------------------------------------------------------------

#if MP1_TYPED_HAVE_WRAP
BOOST_AUTO_TEST_CASE(dense_path_allocates_nothing) {
  if (not mpi_wrap::supported()) {
    BOOST_TEST_MESSAGE("allocation counter needs the -Wl,--wrap shim");
    return;
  }
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  constexpr int elements = 2;
  const std::vector<mp1::count_t> counts = uniform_counts(size, elements);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  std::vector<int> send(static_cast<std::size_t>(elements), 0);
  std::vector<int> recv(static_cast<std::size_t>(elements * size), -1);
  std::vector<int> a2a(static_cast<std::size_t>(elements * size), 0);
  std::vector<int> a2a_recv(static_cast<std::size_t>(elements * size), -1);

  const auto measure = [](auto&& operation) {
    mpi_wrap::reset_allocation_count();
    operation();
    return mpi_wrap::allocation_count();
  };

  // Warm up the datatype/plan caches (setup costs are not per-call behavior).
  if (rank == root)
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
  else
    mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
  mp1::allgatherv(c, send, recv, mp1::blocks{counts, displs});
  mp1::alltoallv(c, a2a, mp1::blocks{counts, displs}, a2a_recv, mp1::blocks{counts, displs});

  // The plan's count/displacement arrays are inline up to K = xfer_inline_blocks peers; above K
  // each array allocates once at plan construction (that allocation is expected).  Assert
  // the exact expected spill per direction instead of zero unconditionally: a 17-rank run must
  // see counts+displs = 2 for the many-block gatherv root, allgatherv receive and alltoallv
  // direction (alltoallv has two many-block directions, so 4 altogether), and the non-root
  // scalar gatherv send must stay at 0 at every size.
  const std::size_t spill =
      static_cast<std::size_t>(size) <= mp1::detail::xfer_inline_blocks ? 0 : 2;

  if (rank == root) {
    BOOST_TEST(measure([&] {
                 mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
               }) == spill);
  } else {
    BOOST_TEST(measure([&] { mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{}); }) ==
               0);
  }
  BOOST_TEST(measure([&] { mp1::allgatherv(c, send, recv, mp1::blocks{counts, displs}); }) ==
             spill);
  BOOST_TEST(measure([&] {
               mp1::alltoallv(c, a2a, mp1::blocks{counts, displs}, a2a_recv,
                              mp1::blocks{counts, displs});
             }) == 2 * spill);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
