// Non-contiguous sequence send: the MPI 4.1 §5.1.12 absolute-address form.
//
// A non-trivially-copyable element type cannot be staged without inventing value semantics, so
// the documented std::list/std::deque send describes the caller's elements with one committed
// hindexed datatype whose displacements are their absolute addresses, passed to MPI as
// {MPI_BOTTOM, 1, map}.  This file checks:
//
//  * std::list and std::deque of a non-trivially-copyable reflected aggregate block and
//    nonblocking sends; the typed receiver sees the right count and values;
//  * the wire is typed (never MPI_BYTE) and the buffer token is MPI_BOTTOM;
//  * the caller's sequence may die after a successful blocking send, and after the request
//    completes for a nonblocking send (the map borrows the elements, it does not copy them);
//  * the trivially-copyable control (std::list<int>) still goes through typed staging: an
//    allocation happens, no hindexed map is built, and the buffer token is not MPI_BOTTOM.
//
// MPI 4.1 basis: §5.1.12 (buffer/datatype association and MPI_BOTTOM), §3.7.2 (nonblocking
// buffer lifetime), §3.2.5 (receive count).

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <type_traits>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_SEQUENCE_HAVE_WRAP 1
#else
#define MP1_SEQUENCE_HAVE_WRAP 0
#endif


namespace {

  struct sequence_fixture {
    sequence_fixture() {
      mp1::init();
    }

    ~sequence_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  // A non-trivially-copyable aggregate: a user-provided destructor keeps the type an
  // aggregate (so MP1_REFLECTION applies) and makes the copy non-trivial.
  struct dtor_member {
    int v{0};
    ~dtor_member() {
    }
  };

  struct non_trivial {
    dtor_member m{};
  };

}  // namespace

MP1_REFLECTION(dtor_member, v)
MP1_REFLECTION(non_trivial, m)

BOOST_TEST_GLOBAL_FIXTURE(sequence_fixture);


namespace {

  static_assert(!std::is_trivially_copyable_v<non_trivial>,
                "the absolute-map coverage requires a non-trivially-copyable element");
  static_assert(std::is_aggregate_v<non_trivial>);

  non_trivial make_value(int value) {
    non_trivial result;
    result.m.v = value;
    return result;
  }

  template<class Sequence>
  Sequence make_sequence(int count, int base) {
    Sequence result;
    for (int i = 0; i < count; ++i)
      result.push_back(make_value(base + i));
    return result;
  }

  std::vector<int> received_values(const std::vector<non_trivial>& values) {
    std::vector<int> result;
    result.reserve(values.size());
    for (const non_trivial& value : values)
      result.push_back(value.m.v);
    return result;
  }

  template<class Handle>
  std::uintptr_t handle_token(Handle handle) {
    // A datatype is an integer on MPICH and a pointer on Open MPI; the token is an integer
    // copy used only for equality comparison, never passed back to MPI.
    if constexpr (std::is_pointer_v<Handle>)
      return reinterpret_cast<std::uintptr_t>(handle);
    else
      return static_cast<std::uintptr_t>(handle);
  }

#if MP1_SEQUENCE_HAVE_WRAP
  bool wrap_active() {
    return mpi_wrap::supported();
  }
#else
  bool wrap_active() {
    return false;
  }
#endif

}  // namespace


BOOST_AUTO_TEST_CASE(list_non_trivial_blocking_send_uses_absolute_map) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  // Force the reflected datatype before a reset so the call capture below sees only the send
  // path (datatype construction performs its own MPI_Get_address calls on first use).
  (void)mp1::native_datatype<non_trivial>();

  if (comm.rank() == 0) {
    const std::list<non_trivial> values = make_sequence<std::list<non_trivial>>(5, 100);
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
#endif
    mp1::send(comm, values, 1, mp1::tag{11});
#if MP1_SEQUENCE_HAVE_WRAP
    if (wrap_active()) {
      BOOST_TEST(mpi_wrap::count("MPI_Send_c") == 1u);
      BOOST_TEST(mpi_wrap::count("MPI_Send") == 0u);
      BOOST_TEST(mpi_wrap::last_buffer("MPI_Send_c") ==
                 reinterpret_cast<std::uintptr_t>(MPI_BOTTOM));
      BOOST_TEST(mpi_wrap::last_datatype("MPI_Send_c") != handle_token(MPI_BYTE));
      BOOST_TEST(mpi_wrap::last_datatype("MPI_Send_c") != handle_token(MPI_DATATYPE_NULL));
      BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 1u);
      BOOST_TEST(mpi_wrap::count("MPI_Get_address") == 5u);
    }
#endif
  } else if (comm.rank() == 1) {
    std::vector<non_trivial> received;
    const mp1::status result = mp1::recv(comm, received, 0, mp1::tag{11});
    BOOST_TEST(result.get_count<non_trivial>() == 5);
    BOOST_TEST(received.size() == 5u);
    const std::vector<int> expected{100, 101, 102, 103, 104};
    BOOST_TEST(received_values(received) == expected, boost::test_tools::per_element());
  }
}


BOOST_AUTO_TEST_CASE(deque_non_trivial_blocking_send_uses_absolute_map) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  (void)mp1::native_datatype<non_trivial>();

  if (comm.rank() == 0) {
    const std::deque<non_trivial> values = make_sequence<std::deque<non_trivial>>(7, 200);
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
#endif
    mp1::send(comm, values, 1, mp1::tag{12});
#if MP1_SEQUENCE_HAVE_WRAP
    if (wrap_active()) {
      BOOST_TEST(mpi_wrap::count("MPI_Send_c") == 1u);
      BOOST_TEST(mpi_wrap::last_buffer("MPI_Send_c") ==
                 reinterpret_cast<std::uintptr_t>(MPI_BOTTOM));
      BOOST_TEST(mpi_wrap::last_datatype("MPI_Send_c") != handle_token(MPI_BYTE));
      BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 1u);
    }
#endif
  } else if (comm.rank() == 1) {
    std::vector<non_trivial> received;
    const mp1::status result = mp1::recv(comm, received, 0, mp1::tag{12});
    BOOST_TEST(result.get_count<non_trivial>() == 7);
    const std::vector<int> expected{200, 201, 202, 203, 204, 205, 206};
    BOOST_TEST(received_values(received) == expected, boost::test_tools::per_element());
  }
}


BOOST_AUTO_TEST_CASE(list_non_trivial_isend_borrows_until_completion) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  (void)mp1::native_datatype<non_trivial>();

  if (comm.rank() == 0) {
    std::list<non_trivial> values = make_sequence<std::list<non_trivial>>(4, 300);
    mp1::request pending;
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
#endif
    {
      // The sequence is an lvalue and must outlive the request; destroying it after
      // completion is legal. The scope makes the borrow window explicit.
      pending = mp1::isend(comm, values, 1, mp1::tag{13});
      BOOST_TEST(pending.is_active());
#if MP1_SEQUENCE_HAVE_WRAP
      if (wrap_active()) {
        BOOST_TEST(mpi_wrap::count("MPI_Isend_c") == 1u);
        BOOST_TEST(mpi_wrap::last_buffer("MPI_Isend_c") ==
                   reinterpret_cast<std::uintptr_t>(MPI_BOTTOM));
        BOOST_TEST(mpi_wrap::last_datatype("MPI_Isend_c") != handle_token(MPI_BYTE));
        BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 1u);
      }
#endif
      (void)pending.wait();
      values.clear();
    }
  } else if (comm.rank() == 1) {
    std::vector<non_trivial> received;
    const mp1::status result = mp1::recv(comm, received, 0, mp1::tag{13});
    BOOST_TEST(result.get_count<non_trivial>() == 4);
    const std::vector<int> expected{300, 301, 302, 303};
    BOOST_TEST(received_values(received) == expected, boost::test_tools::per_element());
  }
}


BOOST_AUTO_TEST_CASE(trivial_list_send_still_stages) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  (void)mp1::native_datatype<int>();

  if (comm.rank() == 0) {
    const std::list<int> values{11, 22, 33};
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
#endif
    mp1::send(comm, values, 1, mp1::tag{14});
#if MP1_SEQUENCE_HAVE_WRAP
    if (wrap_active()) {
      // The staging copy is the documented behavior for a trivially copyable element; it is
      // the fast path with one allocation and no absolute map.
      BOOST_TEST(mpi_wrap::allocation_count() > 0u);
      BOOST_TEST(mpi_wrap::last_buffer("MPI_Send_c") !=
                 reinterpret_cast<std::uintptr_t>(MPI_BOTTOM));
      BOOST_TEST(mpi_wrap::last_datatype("MPI_Send_c") != handle_token(MPI_BYTE));
      BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 0u);
      BOOST_TEST(mpi_wrap::count("MPI_Get_address") == 0u);
    }
#endif
  } else if (comm.rank() == 1) {
    std::vector<int> received;
    const mp1::status result = mp1::recv(comm, received, 0, mp1::tag{14});
    BOOST_TEST(result.get_count<int>() == 3);
    const std::vector<int> expected{11, 22, 33};
    BOOST_TEST(received == expected, boost::test_tools::per_element());
  }
}


BOOST_AUTO_TEST_CASE(isendrecv_non_trivial_send_borrows_until_completion) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  (void)mp1::native_datatype<non_trivial>();

  if (comm.rank() == 0) {
    std::list<non_trivial> outgoing = make_sequence<std::list<non_trivial>>(3, 600);
    std::vector<non_trivial> incoming(2);
    mp1::request pending;
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
#endif
    {
      pending = mp1::isendrecv(comm, outgoing, 1, mp1::tag{17}, incoming, 1, mp1::tag{18});
      BOOST_TEST(pending.is_active());
#if MP1_SEQUENCE_HAVE_WRAP
      if (wrap_active())
        BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 1u);
#endif
      (void)pending.wait();
    }
    const std::vector<int> expected{700, 701};
    BOOST_TEST(received_values(incoming) == expected, boost::test_tools::per_element());
  } else if (comm.rank() == 1) {
    std::list<non_trivial> outgoing = make_sequence<std::list<non_trivial>>(2, 700);
    std::vector<non_trivial> incoming(3);
    (void)mp1::isendrecv(comm, outgoing, 0, mp1::tag{18}, incoming, 0, mp1::tag{17}).wait();
    const std::vector<int> expected{600, 601, 602};
    BOOST_TEST(received_values(incoming) == expected, boost::test_tools::per_element());
  }
}


BOOST_AUTO_TEST_CASE(sendrecv_non_trivial_send_uses_absolute_map) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  if (comm.size() < 2)
    return;
  (void)mp1::native_datatype<non_trivial>();

  if (comm.rank() == 0) {
    const std::list<non_trivial> values = make_sequence<std::list<non_trivial>>(3, 400);
    std::vector<non_trivial> back(3);
#if MP1_SEQUENCE_HAVE_WRAP
    mpi_wrap::reset_counts();
#endif
    (void)mp1::sendrecv(comm, values, 1, mp1::tag{15}, back, 1, mp1::tag{16});
#if MP1_SEQUENCE_HAVE_WRAP
    if (wrap_active()) {
      BOOST_TEST(mpi_wrap::count("MPI_Sendrecv_c") == 1u);
      // The combined wrapper captures only the datatype; the map must still be typed and the
      // send path must not fall back to a byte image.
      BOOST_TEST(mpi_wrap::last_datatype("MPI_Sendrecv_c") != handle_token(MPI_BYTE));
      BOOST_TEST(mpi_wrap::count("MPI_Type_create_hindexed") == 1u);
    }
#endif
  } else if (comm.rank() == 1) {
    std::vector<non_trivial> values(3);
    std::vector<non_trivial> back{make_value(500), make_value(501), make_value(502)};
    (void)mp1::sendrecv(comm, back, 0, mp1::tag{16}, values, 0, mp1::tag{15});
    const std::vector<int> expected{400, 401, 402};
    BOOST_TEST(received_values(values) == expected, boost::test_tools::per_element());
  }
}
