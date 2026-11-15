// Harness self-check for `KIND unit` targets: shared Boost.Test main, MPI wrap
// counters, exact symbol lookup, allocation counter and the after-finalize
// detector.  Registered at ranks 2 and 4.

#include "mpi_wrap.hpp"

#include <boost/test/unit_test.hpp>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <new>
#include <stdexcept>

BOOST_AUTO_TEST_CASE(harness_unit_smoke) {
  if (!mpi_wrap::supported()) {
    // A real skip with a machine-readable diagnostic, not a silent pass.  A
    // job that requires the harness sets MP1_REQUIRE_WRAP=1, in which case a
    // skip would be a false green and is converted to a failure.
    std::fprintf(stderr,
                 "MP1_SKIP: wrap-unavailable -- the --wrap call-count shim is "
                 "not active in this build\n");
    if (std::getenv("MP1_REQUIRE_WRAP") != nullptr) {
      std::fprintf(stderr, "harness_smoke: wrap is required in this job\n");
      std::exit(1);
    }
    std::exit(77);
  }

  int provided = MPI_THREAD_SINGLE;
  BOOST_TEST(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided) == MPI_SUCCESS);
  BOOST_TEST(provided >= MPI_THREAD_SINGLE);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  mpi_wrap::reset_counts();
  int send_token = rank;
  int recv_token = -1;
  MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  BOOST_TEST(MPI_Isend(&send_token, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[0]) ==
             MPI_SUCCESS);
  BOOST_TEST(MPI_Irecv(&recv_token, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[1]) ==
             MPI_SUCCESS);
  BOOST_TEST(MPI_Waitall(2, requests, MPI_STATUSES_IGNORE) == MPI_SUCCESS);
  BOOST_TEST(recv_token == send_token);

  BOOST_TEST(mpi_wrap::count("MPI_Isend") == 1u);
  BOOST_TEST(mpi_wrap::count("MPI_Irecv") == 1u);
  BOOST_TEST(mpi_wrap::count("MPI_Waitall") == 1u);
  BOOST_TEST(mpi_wrap::total() == 3u);
  BOOST_TEST(mpi_wrap::count("MPI_Gatherv") == 0u);

  bool unknown_symbol_threw = false;
  try {
    static_cast<void>(mpi_wrap::count("MPI_Not_A_Symbol"));
  } catch (const std::invalid_argument&) {
    unknown_symbol_threw = true;
  }
  BOOST_TEST(unknown_symbol_threw);

  mpi_wrap::reset_allocation_count();
  // Explicit allocation-function call: a new-expression may be elided when the object does
  // not escape (C++17 [expr.new]/10), which would make this assertion test nothing under
  // -O3.  An explicit ::operator new call is not elidable; int is trivially destructible, so
  // only the matching deallocation call is needed.
  void* storage = ::operator new(sizeof(int));
  int* allocation = new (storage) int(42);
  BOOST_TEST(*allocation == 42);
  ::operator delete(storage);
  BOOST_TEST(mpi_wrap::allocation_count() >= 1u);

  // MPI_Info_free is legal after MPI_Finalize (MPI 4.1 §11.4.1/Table 11.1).
  // The wrapper must forward it and must not count it as an after-finalize
  // error; a World-Model call such as MPI_Comm_free is still recorded.
  MPI_Info always_available = MPI_INFO_NULL;
  BOOST_TEST(MPI_Info_create(&always_available) == MPI_SUCCESS);

  BOOST_TEST(!mpi_wrap::after_finalize());
  BOOST_TEST(MPI_Finalize() == MPI_SUCCESS);
  BOOST_TEST(mpi_wrap::after_finalize());

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_free(&comm);
  BOOST_TEST(mpi_wrap::count("MPI_Comm_free") == 1u);
  BOOST_TEST(mpi_wrap::after_finalize_calls() == 1u);

  BOOST_TEST(MPI_Info_free(&always_available) == MPI_SUCCESS);
  BOOST_TEST(mpi_wrap::count("MPI_Info_free") == 1u);
  BOOST_TEST(mpi_wrap::after_finalize_calls() == 1u);
}
