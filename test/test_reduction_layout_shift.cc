// Regression matrix for collective lowering bugs: a flat reduction layout
// over a reflected T with a non-zero true lower bound, and in-place exscan with a non-zero
// layout true lower bound.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
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

  [[nodiscard]] int prefix_sum(int rank) {
    return rank * (rank + 1) / 2;
  }

  [[nodiscard]] int exclusive_prefix_sum(int rank) {
    // Each rank contributes rank + 1; the exclusive scan on rank r is the sum over
    // ranks 0 .. r-1, i.e. 1 + 2 + ... + r.
    return rank * (rank + 1) / 2;
  }

}  // namespace

// The first *listed* member is at offset 4; native_datatype<T>() therefore has true_lb 4,
// extent 8 and size 4.  The managed callback image preserves that native map (there is no
// MPI_BYTE prefix or full-object image), so the callback's T[len] indexing addresses the
// listed `value` member at its C++ offset and never touches the unlisted `ignored` member.
struct reduction_shift_padded {
  int ignored{-777};
  int value{0};
};

MP1_REFLECTION(reduction_shift_padded, value)

// A reflected T whose unlisted member and natural alignment padding occupy bytes 0..3 and
// whose mapped member is the int at offset 4 (native true_lb 4, extent 8).  A staged
// managed-functor reduction packs only the mapped int; the unlisted tag and the padding are
// not part of the image, so completion preserves the caller's bytes.
struct reduction_padding_probe {
  unsigned char tag{0xab};
  int value{0};
};

MP1_REFLECTION(reduction_padding_probe, value)

namespace {

  struct padded_add {
    reduction_shift_padded operator()(const reduction_shift_padded& left,
                                      const reduction_shift_padded& right) const noexcept {
      return reduction_shift_padded{0, left.value + right.value};
    }
  };

  // Only listed members are inspected: the typed callback image does not transmit the tag
  // or the padding bytes, so the functor contract forbids reading them.
  struct padding_probe_add {
    reduction_padding_probe operator()(const reduction_padding_probe& left,
                                       const reduction_padding_probe& right) const noexcept {
      return reduction_padding_probe{0, left.value + right.value};
    }
  };

  struct guarded_block {
    int before{0x12345678};
    reduction_shift_padded values[2];
    int after{0x5a5a5a5a};
  };

  // Three T values with canaries.  The shifted layouts below select values[1] and
  // values[2] (or values[0] and values[2]), so the first type-map entry is not at the
  // buffer base and the lowering pointer has to move to the first selected T object.
  struct guarded_triple {
    int before{0x12345678};
    reduction_shift_padded values[3];
    int after{0x5a5a5a5a};
  };

  void require_canaries(const guarded_block& block, const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(block.before == 0x12345678);
      BOOST_TEST(block.after == 0x5a5a5a5a);
    }
  }

  void require_canaries(const guarded_triple& block, const char* label) {
    BOOST_TEST_CONTEXT(label) {
      BOOST_TEST(block.before == 0x12345678);
      BOOST_TEST(block.after == 0x5a5a5a5a);
    }
  }

  // A caller-created MPI_Op.  mp1 passes a raw handle native_datatype<T>() unchanged, so this
  // callback must apply the datatype's own type map (MPI 4.1 §6.9.5: the buffer is described
  // by invec/len/datatype, and general datatypes may be passed to the user function).  The
  // native geometry of reduction_shift_padded is one int at offset 4 with extent 8.
  struct raw_padded_add {
    static bool& native_geometry_seen() {
      static bool seen = false;
      return seen;
    }

    static void call(void* in, void* inout, int* len, MPI_Datatype* type) {
      MPI_Count size = 0;
      MPI_Count marker_lb = 0;
      MPI_Count extent = 0;
      MPI_Count true_lb = 0;
      MPI_Count true_extent = 0;
      MPI_Type_size_c(*type, &size);
      MPI_Type_get_extent_c(*type, &marker_lb, &extent);
      MPI_Type_get_true_extent_c(*type, &true_lb, &true_extent);
      if (size == 4 and extent == 8 and true_lb == 4 and true_extent == 4)
        native_geometry_seen() = true;
      const auto* input = static_cast<const char*>(in);
      auto* output = static_cast<char*>(inout);
      for (int i = 0; i < *len; ++i) {
        const std::size_t offset =
            static_cast<std::size_t>(true_lb) +
            static_cast<std::size_t>(i) * static_cast<std::size_t>(extent);
        int left = 0;
        int right = 0;
        std::memcpy(&left, input + offset, sizeof(int));
        std::memcpy(&right, output + offset, sizeof(int));
        const int sum = left + right;
        std::memcpy(output + offset, &sum, sizeof(int));
      }
    }
  };

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(reduction_layout_shift_test)

BOOST_AUTO_TEST_CASE(layout_reductions_with_nonzero_true_lb) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const mp1::layout<reduction_shift_padded> block =
      mp1::layout<reduction_shift_padded>::contiguous(2);
  BOOST_TEST(block.true_byte_lower_bound() == 4);

  const auto make_send = [&] {
    guarded_block block_values;
    block_values.values[0] = reduction_shift_padded{100, rank + 1};
    block_values.values[1] = reduction_shift_padded{200, 10 * (rank + 1)};
    return block_values;
  };
  const auto make_recv = [] {
    guarded_block block_values;
    block_values.values[0] = reduction_shift_padded{-1, -1};
    block_values.values[1] = reduction_shift_padded{-2, -2};
    return block_values;
  };

  // reduce: root receives the per-entry sum; non-roots must not touch their buffer.
  {
    guarded_block send = make_send();
    guarded_block recv = make_recv();
    mp1::reduce(comm, padded_add{}, 0, mp1::region{send.values, block},
                mp1::region{recv.values, block});
    if (rank == 0) {
      BOOST_TEST(recv.values[0].value == size * (size + 1) / 2);
      BOOST_TEST(recv.values[1].value == 10 * size * (size + 1) / 2);
      // The typed image writes only the listed `value` member, so the unlisted member keeps
      // the caller's receive sentinel on every rank.  The canary words detect an
      // out-of-bounds write.
      BOOST_TEST(recv.values[0].ignored == -1);
      BOOST_TEST(recv.values[1].ignored == -2);
    } else {
      BOOST_TEST(recv.values[0].value == -1);
      BOOST_TEST(recv.values[1].value == -2);
    }
    require_canaries(send, "reduce send canaries");
    require_canaries(recv, "reduce recv canaries");
  }

  // allreduce.
  {
    guarded_block send = make_send();
    guarded_block recv = make_recv();
    mp1::allreduce(comm, padded_add{}, mp1::region{send.values, block},
                   mp1::region{recv.values, block});
    BOOST_TEST(recv.values[0].value == size * (size + 1) / 2);
    BOOST_TEST(recv.values[1].value == 10 * size * (size + 1) / 2);
    BOOST_TEST(recv.values[0].ignored == -1);
    BOOST_TEST(recv.values[1].ignored == -2);
    require_canaries(send, "allreduce send canaries");
    require_canaries(recv, "allreduce recv canaries");
  }

  // scan: inclusive prefix.
  {
    guarded_block send = make_send();
    guarded_block recv = make_recv();
    mp1::scan(comm, padded_add{}, mp1::region{send.values, block},
              mp1::region{recv.values, block});
    BOOST_TEST(recv.values[0].value == prefix_sum(rank + 1));
    BOOST_TEST(recv.values[1].value == 10 * prefix_sum(rank + 1));
    require_canaries(send, "scan send canaries");
    require_canaries(recv, "scan recv canaries");
  }

  // exscan out-of-place: rank 0 keeps its sentinel (wrapper guarantee), rank > 0 exclusive.
  {
    guarded_block send = make_send();
    guarded_block recv = make_recv();
    mp1::exscan(comm, padded_add{}, mp1::region{send.values, block},
                mp1::region{recv.values, block});
    if (rank == 0) {
      BOOST_TEST(recv.values[0].value == -1);
      BOOST_TEST(recv.values[1].value == -2);
    } else {
      BOOST_TEST(recv.values[0].value == exclusive_prefix_sum(rank));
      BOOST_TEST(recv.values[1].value == 10 * exclusive_prefix_sum(rank));
    }
    require_canaries(send, "exscan send canaries");
    require_canaries(recv, "exscan recv canaries");
  }
}

BOOST_AUTO_TEST_CASE(inplace_exscan_nonzero_true_lb) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  // One int at element offset 2: flat (size/true_extent 4) with true_lb 8.
  mp1::indexed_parameter block_parameter;
  block_parameter.add(1, 2);
  const mp1::layout<int> block = mp1::layout<int>::indexed(block_parameter);
  BOOST_TEST(block.true_byte_lower_bound() == 8);
  BOOST_TEST(block.is_base_repetition());

  // In-place blocking exscan.
  {
    std::array<int, 5> buffer{0x12345678, -1, rank + 1, -1, 0x5a5a5a5a};
    mp1::exscan(comm, MPI_SUM, mp1::in_place, mp1::region{buffer.data(), block});
    if (rank == 0) {
      BOOST_TEST(buffer[2] == 1);  // rank 0's receive buffer is unchanged
    } else {
      BOOST_TEST(buffer[2] == exclusive_prefix_sum(rank));
    }
    BOOST_TEST(buffer[0] == 0x12345678);
    BOOST_TEST(buffer[1] == -1);
    BOOST_TEST(buffer[3] == -1);
    BOOST_TEST(buffer[4] == 0x5a5a5a5a);
  }

  // The same in-place lowering through the nonblocking request.
  {
    std::array<int, 5> buffer{0x12345678, -1, rank + 1, -1, 0x5a5a5a5a};
    mp1::request request =
        mp1::iexscan(comm, MPI_SUM, mp1::in_place, mp1::region{buffer.data(), block});
    request.wait();
    if (rank == 0) {
      BOOST_TEST(buffer[2] == 1);
    } else {
      BOOST_TEST(buffer[2] == exclusive_prefix_sum(rank));
    }
    BOOST_TEST(buffer[0] == 0x12345678);
    BOOST_TEST(buffer[4] == 0x5a5a5a5a);
  }

  // Persistent in-place exscan: the flat path needs no re-pack recipe; each start reads the
  // current receive buffer through MPI_IN_PLACE.
  {
    std::array<int, 5> buffer{0x12345678, -1, 0, -1, 0x5a5a5a5a};
    mp1::request request =
        mp1::exscan_init(comm, MPI_SUM, mp1::in_place, mp1::region{buffer.data(), block});
    for (int cycle = 0; cycle < 2; ++cycle) {
      buffer[2] = rank + 1 + cycle * 10;
      request.start();
      request.wait();
      if (rank == 0) {
        BOOST_TEST(buffer[2] == 1 + cycle * 10);
      } else {
        BOOST_TEST(buffer[2] == exclusive_prefix_sum(rank) + cycle * 10 * rank);
      }
      BOOST_TEST(buffer[0] == 0x12345678);
      BOOST_TEST(buffer[4] == 0x5a5a5a5a);
    }
  }

  // A null receive buffer for a non-empty in-place layout throws locally instead of
  // dereferencing null.
  BOOST_CHECK_THROW(mp1::exscan(comm, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, block}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(
      (void)mp1::iexscan(comm, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, block}),
      std::invalid_argument);
}

// The same typed-image contract applies to every reduction form that lowers to a flat
// (pointer, count, datatype) call.  The scalar, pointer+count and container forms have no
// layout at all, but the managed callback still receives its typed image; padding and
// unlisted members are never transmitted and a direct receive preserves them.
BOOST_AUTO_TEST_CASE(flat_forms_with_nonzero_true_lb) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;
  const int expected10 = 10 * expected;

  // Scalar reduce.
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{-1, -1};
    mp1::reduce(comm, padded_add{}, 0, send, recv);
    if (rank == 0) {
      BOOST_TEST(recv.value == expected);
      // The typed image transmits only `value`; the unlisted member is preserved.
      BOOST_TEST(recv.ignored == -1);
    } else {
      BOOST_TEST(recv.value == -1);
      BOOST_TEST(recv.ignored == -1);
    }
  }

  // Scalar allreduce, out-of-place and in-place.
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{-1, -1};
    mp1::allreduce(comm, padded_add{}, send, recv);
    BOOST_TEST(recv.value == expected);
    BOOST_TEST(recv.ignored == -1);
  }
  {
    reduction_shift_padded buffer{100, rank + 1};
    mp1::allreduce(comm, padded_add{}, mp1::in_place, buffer);
    BOOST_TEST(buffer.value == expected);
    // In-place: the same object is input and result; the unlisted member stays 100.
    BOOST_TEST(buffer.ignored == 100);
  }

  // Scalar scan (inclusive) and exscan (exclusive; mp1 leaves rank 0's receive untouched).
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{-1, -1};
    mp1::scan(comm, padded_add{}, send, recv);
    BOOST_TEST(recv.value == prefix_sum(rank + 1));
  }
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{-1, -1};
    mp1::exscan(comm, padded_add{}, send, recv);
    if (rank == 0) {
      BOOST_TEST(recv.value == -1);
    } else {
      BOOST_TEST(recv.value == exclusive_prefix_sum(rank));
    }
  }

  // Pointer+count form: two complete T objects at the caller's pointer.
  {
    const std::array<reduction_shift_padded, 2> send{
        reduction_shift_padded{100, rank + 1}, reduction_shift_padded{200, 10 * (rank + 1)}};
    std::array<reduction_shift_padded, 2> recv{reduction_shift_padded{-1, -1},
                                               reduction_shift_padded{-2, -2}};
    if (rank == 0) {
      mp1::reduce(comm, padded_add{}, 0, std::span<const reduction_shift_padded>{send},
                  std::span<reduction_shift_padded>{recv});
      BOOST_TEST(recv[0].value == expected);
      BOOST_TEST(recv[1].value == expected10);
    } else {
      mp1::reduce(comm, padded_add{}, 0, std::span<const reduction_shift_padded>{send},
                  std::span<reduction_shift_padded>{});
    }
  }

  // Container form.
  {
    const std::vector<reduction_shift_padded> send{
        reduction_shift_padded{100, rank + 1}, reduction_shift_padded{200, 10 * (rank + 1)}};
    std::vector<reduction_shift_padded> recv(2, reduction_shift_padded{-1, -1});
    if (rank == 0) {
      mp1::reduce(comm, padded_add{}, 0, send, recv);
      BOOST_TEST(recv[0].value == expected);
      BOOST_TEST(recv[1].value == expected10);
    } else {
      mp1::reduce(comm, padded_add{}, 0, send, std::span<reduction_shift_padded>{});
    }
  }

  // Nonblocking scalar and pointer+count forms.
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{-1, -1};
    mp1::request request = mp1::ireduce(comm, padded_add{}, 0, send, recv);
    request.wait();
    if (rank == 0) {
      BOOST_TEST(recv.value == expected);
    } else {
      BOOST_TEST(recv.value == -1);
    }
  }
  {
    const std::array<reduction_shift_padded, 2> send{
        reduction_shift_padded{100, rank + 1}, reduction_shift_padded{200, 10 * (rank + 1)}};
    std::array<reduction_shift_padded, 2> recv{reduction_shift_padded{-1, -1},
                                               reduction_shift_padded{-2, -2}};
    mp1::request request =
        mp1::iallreduce(comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                        std::span<reduction_shift_padded>{recv});
    request.wait();
    BOOST_TEST(recv[0].value == expected);
    BOOST_TEST(recv[1].value == expected10);
  }

  // reduce_local is a local callback invocation with no communicator; inout = op(in, inout).
  {
    const reduction_shift_padded send{100, rank + 1};
    reduction_shift_padded recv{200, 1000};
    mp1::reduce_local(padded_add{}, send, recv);
    BOOST_TEST(recv.value == 1001 + rank);
    BOOST_TEST(recv.ignored == 200);
  }
}

// A layout whose true lower bound is larger than the base datatype's: the flat lowering
// pointer is p + (true_lb(l) - true_lb(base)) and must still be the first complete T object.
// A non-base-repetition layout stages flat T objects; the typed callback image describes one
// complete T element at extent sizeof(T) with the native true_lb.
BOOST_AUTO_TEST_CASE(shifted_and_staged_layouts_with_nonzero_true_lb) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;
  const int expected10 = 10 * expected;

  // blocklength 2 at base displacement 1: the type map starts at byte 12, so the first
  // selected T object is values[1] and the repetition shift is 8 bytes.
  mp1::indexed_parameter shifted_parameter;
  shifted_parameter.add(2, 1);
  const mp1::layout<reduction_shift_padded> shifted =
      mp1::layout<reduction_shift_padded>::indexed(shifted_parameter);
  BOOST_TEST(shifted.true_byte_lower_bound() == 12);
  BOOST_TEST(shifted.is_base_repetition());

  const auto make_triple_send = [&] {
    guarded_triple values;
    values.values[0] = reduction_shift_padded{300, -300};
    values.values[1] = reduction_shift_padded{100, rank + 1};
    values.values[2] = reduction_shift_padded{200, 10 * (rank + 1)};
    return values;
  };
  const auto make_triple_recv = [] {
    guarded_triple values;
    for (auto& value : values.values)
      value = reduction_shift_padded{-1, -1};
    return values;
  };

  // reduce over the shifted layout: only values[1] and values[2] are part of the map.
  {
    guarded_triple send = make_triple_send();
    guarded_triple recv = make_triple_recv();
    mp1::reduce(comm, padded_add{}, 0, mp1::region{send.values, shifted},
                mp1::region{recv.values, shifted});
    if (rank == 0) {
      BOOST_TEST(recv.values[0].value == -1);
      BOOST_TEST(recv.values[1].value == expected);
      BOOST_TEST(recv.values[2].value == expected10);
    } else {
      for (const auto& value : recv.values)
        BOOST_TEST(value.value == -1);
    }
    require_canaries(send, "shifted reduce send canaries");
    require_canaries(recv, "shifted reduce recv canaries");
  }

  // allreduce and scan over the same shifted layout.
  {
    guarded_triple send = make_triple_send();
    guarded_triple recv = make_triple_recv();
    mp1::allreduce(comm, padded_add{}, mp1::region{send.values, shifted},
                   mp1::region{recv.values, shifted});
    BOOST_TEST(recv.values[0].value == -1);
    BOOST_TEST(recv.values[1].value == expected);
    BOOST_TEST(recv.values[2].value == expected10);
    require_canaries(recv, "shifted allreduce recv canaries");
  }
  {
    guarded_triple send = make_triple_send();
    guarded_triple recv = make_triple_recv();
    mp1::scan(comm, padded_add{}, mp1::region{send.values, shifted},
              mp1::region{recv.values, shifted});
    BOOST_TEST(recv.values[0].value == -1);
    BOOST_TEST(recv.values[1].value == prefix_sum(rank + 1));
    BOOST_TEST(recv.values[2].value == 10 * prefix_sum(rank + 1));
    require_canaries(recv, "shifted scan recv canaries");
  }

  // Root in-place reduce over the shifted layout.
  {
    guarded_triple buffer = make_triple_send();
    if (rank == 0) {
      mp1::reduce(comm, padded_add{}, 0, mp1::in_place, mp1::region{buffer.values, shifted});
      BOOST_TEST(buffer.values[0].value == -300);
      BOOST_TEST(buffer.values[1].value == expected);
      BOOST_TEST(buffer.values[2].value == expected10);
    } else {
      mp1::reduce(comm, padded_add{}, 0, mp1::region{buffer.values, shifted},
                  std::span<reduction_shift_padded>{});
      BOOST_TEST(buffer.values[1].value == rank + 1);
    }
    require_canaries(buffer, "shifted in-place reduce canaries");
  }

  // A vector layout (count 2, blocklength 1, stride 2) is not a base repetition, so both
  // directions stage flat T objects.  values[0] and values[2] are mapped, values[1] is a
  // hole that MPI must not touch.
  const mp1::layout<reduction_shift_padded> holey =
      mp1::layout<reduction_shift_padded>::vector(2, 1, 2);
  BOOST_TEST(holey.true_byte_lower_bound() == 4);
  BOOST_TEST(not holey.is_base_repetition());
  {
    guarded_triple send;
    send.values[0] = reduction_shift_padded{100, rank + 1};
    send.values[1] = reduction_shift_padded{500, -500};
    send.values[2] = reduction_shift_padded{200, 10 * (rank + 1)};
    guarded_triple recv;
    for (auto& value : recv.values)
      value = reduction_shift_padded{-1, -1};
    mp1::reduce(comm, padded_add{}, 0, mp1::region{send.values, holey},
                mp1::region{recv.values, holey});
    if (rank == 0) {
      BOOST_TEST(recv.values[0].value == expected);
      BOOST_TEST(recv.values[1].value == -1);
      BOOST_TEST(recv.values[2].value == expected10);
    } else {
      // The non-root receive buffer is not significant for reduce and stays untouched,
      // including the hole that is outside the layout's type map.
      for (const auto& value : recv.values)
        BOOST_TEST(value.value == -1);
    }
    require_canaries(send, "holey reduce send canaries");
    require_canaries(recv, "holey reduce recv canaries");
  }
  {
    guarded_triple send;
    send.values[0] = reduction_shift_padded{100, rank + 1};
    send.values[1] = reduction_shift_padded{500, -500};
    send.values[2] = reduction_shift_padded{200, 10 * (rank + 1)};
    guarded_triple recv;
    for (auto& value : recv.values)
      value = reduction_shift_padded{-1, -1};
    mp1::allreduce(comm, padded_add{}, mp1::region{send.values, holey},
                   mp1::region{recv.values, holey});
    BOOST_TEST(recv.values[0].value == expected);
    BOOST_TEST(recv.values[1].value == -1);
    BOOST_TEST(recv.values[2].value == expected10);
    require_canaries(recv, "holey allreduce recv canaries");
  }
}

// A staged managed-functor reduction packs only the layout's mapped positions.  The typed
// callback image transmits only the native map's `value` int; the unlisted tag and the
// padding bytes are not part of the image, and completion unpacks only the mapped member, so
// the caller's unmapped bytes are preserved and the callback never needs to inspect them.
BOOST_AUTO_TEST_CASE(staged_user_op_preserves_unmapped_bytes) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;

  // count 2, blocklength 1, stride 2: not a base repetition, so both directions stage.
  const mp1::layout<reduction_padding_probe> holey =
      mp1::layout<reduction_padding_probe>::vector(2, 1, 2);
  BOOST_TEST(not holey.is_base_repetition());
  BOOST_TEST(holey.size() == 2);

  std::array<reduction_padding_probe, 3> send{};
  send[0] = reduction_padding_probe{0xab, rank + 1};
  send[1] = reduction_padding_probe{0xcd, -7};
  send[2] = reduction_padding_probe{0xab, 10 * (rank + 1)};
  // Make the unlisted member and the natural padding bytes 1..3 explicit instead of relying
  // on whatever the default constructor left there.
  for (auto* value : {&send[0], &send[2]}) {
    auto* head = reinterpret_cast<unsigned char*>(value);
    head[0] = 0xab;
    head[1] = 0x11;
    head[2] = 0x22;
    head[3] = 0x33;
  }
  std::array<reduction_padding_probe, 3> recv{};
  for (auto& value : recv) {
    value = reduction_padding_probe{0xab, -1};
    // Give the natural padding bytes an explicit receive sentinel as well.
    auto* head = reinterpret_cast<unsigned char*>(&value);
    head[1] = 0x77;
    head[2] = 0x77;
    head[3] = 0x77;
  }

  mp1::allreduce(comm, padding_probe_add{}, mp1::region{send.data(), holey},
                 mp1::region{recv.data(), holey});

  BOOST_TEST(recv[0].value == expected);
  BOOST_TEST(recv[2].value == 10 * expected);
  // values[1] is a layout hole: never packed and never unpacked.
  BOOST_TEST(recv[1].tag == 0xab);
  BOOST_TEST(recv[1].value == -1);
  // Only the mapped int is unpacked, so the caller's tag byte and padding bytes survive the
  // staged round trip; nothing outside the native type map is overwritten.
  for (const std::size_t index : {std::size_t{0}, std::size_t{2}}) {
    BOOST_TEST(recv[index].tag == 0xab);
    const auto* head = reinterpret_cast<const unsigned char*>(&recv[index]);
    BOOST_TEST(head[1] == 0x77);
    BOOST_TEST(head[2] == 0x77);
    BOOST_TEST(head[3] == 0x77);
  }
}

// Native predefined-op reductions keep the native datatype; a nonzero true_lb layout must
// still address the selected elements (and must not take the user-op image path).
BOOST_AUTO_TEST_CASE(native_predefined_ops_with_nonzero_true_lb_layout) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;
  const int expected10 = 10 * expected;

  // blocklength 2 at base displacement 2: true_lb is 8 and only elements 2 and 3 are mapped.
  mp1::indexed_parameter shifted_parameter;
  shifted_parameter.add(2, 2);
  const mp1::layout<int> shifted = mp1::layout<int>::indexed(shifted_parameter);
  BOOST_TEST(shifted.true_byte_lower_bound() == 8);
  BOOST_TEST(shifted.is_base_repetition());

  {
    const std::array<int, 4> send{-7, -7, rank + 1, 10 * (rank + 1)};
    std::array<int, 4> recv{-1, -1, -2, -2};
    if (rank == 0) {
      mp1::reduce(comm, MPI_SUM, 0, mp1::region{send.data(), shifted},
                  mp1::region{recv.data(), shifted});
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == expected);
      BOOST_TEST(recv[3] == expected10);
    } else {
      mp1::reduce(comm, MPI_SUM, 0, mp1::region{send.data(), shifted}, std::span<int>{});
    }
  }
  {
    const std::array<int, 4> send{-7, -7, rank + 1, 10 * (rank + 1)};
    std::array<int, 4> recv{-1, -1, -2, -2};
    mp1::allreduce(comm, std::plus<int>{}, mp1::region{send.data(), shifted},
                   mp1::region{recv.data(), shifted});
    BOOST_TEST(recv[0] == -1);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected);
    BOOST_TEST(recv[3] == expected10);
  }
  {
    const std::array<int, 4> send{-7, -7, rank + 1, 10 * (rank + 1)};
    std::array<int, 4> recv{-1, -1, -2, -2};
    mp1::scan(comm, MPI_SUM, mp1::region{send.data(), shifted},
              mp1::region{recv.data(), shifted});
    BOOST_TEST(recv[0] == -1);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == prefix_sum(rank + 1));
    BOOST_TEST(recv[3] == 10 * prefix_sum(rank + 1));
  }
}

// User-op reduce_scatter[_block] over a gapped callback image.  The floor provider
// (MPICH 4.3.2) lays the callback temporaries out at the image's true extent, not at the
// sizeof(T) extent, so a T whose native map does not span the whole object (a non-zero true
// lower bound or trailing padding) cannot be presented as a T[len] array on that path.  mp1
// rejects the combination locally on every rank before MPI instead of returning wrong data;
// all other reduction families keep the typed image for such a T.
BOOST_AUTO_TEST_CASE(reduce_scatter_rejects_gapped_callback_image) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::vector<mp1::count_t> recvcounts(static_cast<std::size_t>(size), 1);

  const auto make_flat_send = [&] {
    std::vector<reduction_shift_padded> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = reduction_shift_padded{100, rank + 1 + 10 * j};
    return send;
  };

  // Blocking out-of-place and in-place.
  {
    const std::vector<reduction_shift_padded> send = make_flat_send();
    std::vector<reduction_shift_padded> recv(1, reduction_shift_padded{-1, -1});
    BOOST_CHECK_THROW(
        mp1::reduce_scatter(comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                            std::span<reduction_shift_padded>{recv}, recvcounts),
        std::invalid_argument);
  }
  {
    std::vector<reduction_shift_padded> buffer = make_flat_send();
    BOOST_CHECK_THROW(
        mp1::reduce_scatter(comm, padded_add{}, mp1::in_place,
                            std::span<reduction_shift_padded>{buffer}, recvcounts),
        std::invalid_argument);
  }

  // Nonblocking and persistent init.
  {
    const std::vector<reduction_shift_padded> send = make_flat_send();
    std::vector<reduction_shift_padded> recv(1, reduction_shift_padded{-1, -1});
    BOOST_CHECK_THROW((void)mp1::ireduce_scatter(
                          comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                          std::span<reduction_shift_padded>{recv}, recvcounts),
                      std::invalid_argument);
    BOOST_CHECK_THROW((void)mp1::reduce_scatter_init(
                          comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                          std::span<reduction_shift_padded>{recv}, recvcounts),
                      std::invalid_argument);
  }

  // reduce_scatter_block and its nonblocking/persistent forms.
  {
    const std::vector<reduction_shift_padded> send = make_flat_send();
    std::vector<reduction_shift_padded> recv(1, reduction_shift_padded{-1, -1});
    BOOST_CHECK_THROW(mp1::reduce_scatter_block(comm, padded_add{},
                                                std::span<const reduction_shift_padded>{send},
                                                std::span<reduction_shift_padded>{recv}, 1),
                      std::invalid_argument);
    BOOST_CHECK_THROW((void)mp1::ireduce_scatter_block(
                          comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                          std::span<reduction_shift_padded>{recv}, 1),
                      std::invalid_argument);
    BOOST_CHECK_THROW((void)mp1::reduce_scatter_block_init(
                          comm, padded_add{}, std::span<const reduction_shift_padded>{send},
                          std::span<reduction_shift_padded>{recv}, 1),
                      std::invalid_argument);
  }
}

// Zero count still reaches MPI with count 0 and the callback datatype; no element is
// exchanged and the callback must not run.
BOOST_AUTO_TEST_CASE(user_op_zero_count_and_empty_layout) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const mp1::layout<reduction_shift_padded> empty =
      mp1::layout<reduction_shift_padded>::contiguous(0);
  BOOST_TEST(empty.size() == 0);

  reduction_shift_padded recv{-1, -1};
  mp1::allreduce(comm, padded_add{}, mp1::region<const reduction_shift_padded>{nullptr, empty},
                 mp1::region<reduction_shift_padded>{&recv, empty});
  BOOST_TEST(recv.value == -1);
  mp1::allreduce(comm, padded_add{}, std::span<const reduction_shift_padded>{},
                 std::span<reduction_shift_padded>{&recv, 0});
  BOOST_TEST(recv.value == -1);
  mp1::scan(comm, padded_add{}, std::span<const reduction_shift_padded>{},
            std::span<reduction_shift_padded>{&recv, 0});
  BOOST_TEST(recv.value == -1);
}

// A raw, caller-created MPI_Op keeps the caller's contract: mp1 passes it
// native_datatype<T>() (true_lb 4 here), never the managed callback's typed image.  The
// callback applies the datatype's own type map, as MPI 4.1 §6.9.5 requires.
BOOST_AUTO_TEST_CASE(raw_mpi_op_keeps_native_datatype) {
#if defined(I_MPI_NUMVERSION) && I_MPI_NUMVERSION == 20211900300
  // Intel MPI 2021.19.0 can corrupt the result of a user-defined MPI_Op over a derived
  // datatype (COMPATIBILITY.md, "User-defined MPI_Op over a derived datatype").  This is the
  // configuration observed in the CI stable-intel-2021.19-debug leg; later Intel builds run
  // this case.  It is a provider limitation, not an mp1 contract: the raw handle and the
  // native datatype are forwarded unchanged, and the managed-callback cases above cover the
  // typed image.
  std::fprintf(stderr,
               "raw MPI_Op over a derived datatype skipped on Intel MPI 2021.19.0: provider "
               "can corrupt the reduction result (see COMPATIBILITY.md)\n");
  return;
#endif
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size + 1) / 2;
  const int expected10 = 10 * expected;
  const mp1::layout<reduction_shift_padded> block =
      mp1::layout<reduction_shift_padded>::contiguous(2);

  MPI_Op raw = MPI_OP_NULL;
  BOOST_TEST(MPI_Op_create(&raw_padded_add::call, 1, &raw) == MPI_SUCCESS);
  BOOST_REQUIRE(raw != MPI_OP_NULL);

  guarded_block send;
  send.values[0] = reduction_shift_padded{100, rank + 1};
  send.values[1] = reduction_shift_padded{200, 10 * (rank + 1)};
  guarded_block recv;
  recv.values[0] = reduction_shift_padded{-1, -1};
  recv.values[1] = reduction_shift_padded{-2, -2};

  mp1::reduce(comm, raw, 0, mp1::region{send.values, block}, mp1::region{recv.values, block});
  if (rank == 0) {
    // A one-rank MPI_Reduce need not invoke the user callback at all; the datatype geometry
    // can only be observed when the reduction combines at least two contributions.
    if (size > 1)
      BOOST_TEST(raw_padded_add::native_geometry_seen());
    BOOST_TEST(recv.values[0].value == expected);
    BOOST_TEST(recv.values[1].value == expected10);
  } else {
    BOOST_TEST(recv.values[0].value == -1);
    BOOST_TEST(recv.values[1].value == -2);
  }
  require_canaries(send, "raw op reduce send canaries");
  require_canaries(recv, "raw op reduce recv canaries");

  BOOST_TEST(MPI_Op_free(&raw) == MPI_SUCCESS);
}

// A 2+2 intercommunicator (1+1 at two ranks): MPI_Allreduce on an intercommunicator reduces
// each group's contributions into the other group.  The mp1-managed functor over the
// reflected T with true_lb 4 uses the typed callback image on this path too.
BOOST_AUTO_TEST_CASE(intercommunicator_reduction_with_nonzero_true_lb) {
  const mp1::mpi_communicator world = mp1::comm_world();
  // An intercommunicator needs two non-empty groups; at one rank the parity split would
  // leave the remote group empty, which MPI cannot create.
  if (world.size() < 2)
    return;
  const int group = world.rank() % 2;
  auto local = mp1::communicator::split(world, group);
  const int remote_leader = group == 0 ? 1 : 0;
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{700});
  BOOST_REQUIRE(inter.is_inter());
  const int local_rank = inter.rank();
  const int remote_size = inter.remote_size();
  const int expected = remote_size * (remote_size + 1) / 2;
  const int expected10 = 10 * expected;

  const mp1::layout<reduction_shift_padded> block =
      mp1::layout<reduction_shift_padded>::contiguous(2);
  guarded_block send;
  send.values[0] = reduction_shift_padded{0, local_rank + 1};
  send.values[1] = reduction_shift_padded{0, 10 * (local_rank + 1)};
  guarded_block recv;
  recv.values[0] = reduction_shift_padded{-1, -1};
  recv.values[1] = reduction_shift_padded{-2, -2};

  mp1::allreduce(inter, padded_add{}, mp1::region{send.values, block},
                 mp1::region{recv.values, block});
  BOOST_TEST(recv.values[0].value == expected);
  BOOST_TEST(recv.values[1].value == expected10);
  require_canaries(send, "intercomm send canaries");
  require_canaries(recv, "intercomm recv canaries");

  inter.free();
  local.free();
}

BOOST_AUTO_TEST_SUITE_END()
