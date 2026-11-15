// Harness self-check for `KIND scenario` targets: plain main, ENVIRONMENT
// property and wrap counters at ranks 1, 2 and 4.  Returns 77 when the wrap
// shim is unavailable so CTest can skip.

#include "mpi_wrap.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

int main(int argc, char** argv) {
  static_cast<void>(argc);
  static_cast<void>(argv);

  if (!mpi_wrap::supported()) {
    std::fprintf(stderr,
                 "MP1_SKIP: wrap-unavailable -- the --wrap call-count shim is "
                 "not active in this build\n");
    if (std::getenv("MP1_REQUIRE_WRAP") != nullptr) {
      std::fprintf(stderr, "harness_scenario_smoke: wrap is required in this job\n");
      return 1;
    }
    return 77;
  }

  const char* scenario = std::getenv("MP1_TEST_SCENARIO");
  if (scenario == nullptr || std::strcmp(scenario, "smoke") != 0) {
    return 2;
  }

  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SINGLE, &provided) != MPI_SUCCESS) {
    return 3;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  mpi_wrap::reset_counts();
  int send_token = rank;
  int recv_token = -1;
  MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
  if (MPI_Isend(&send_token, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[0]) !=
      MPI_SUCCESS) {
    return 4;
  }
  if (MPI_Irecv(&recv_token, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[1]) !=
      MPI_SUCCESS) {
    return 5;
  }
  if (MPI_Waitall(2, requests, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    return 6;
  }
  if (recv_token != send_token) {
    return 7;
  }
  if (mpi_wrap::count("MPI_Isend") != 1u || mpi_wrap::count("MPI_Irecv") != 1u ||
      mpi_wrap::count("MPI_Waitall") != 1u || mpi_wrap::total() != 3u) {
    return 8;
  }

  MPI_Info always_available = MPI_INFO_NULL;
  if (MPI_Info_create(&always_available) != MPI_SUCCESS) {
    return 9;
  }
  if (mpi_wrap::after_finalize()) {
    return 10;
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    return 11;
  }
  if (!mpi_wrap::after_finalize()) {
    return 12;
  }

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_free(&comm);
  if (mpi_wrap::count("MPI_Comm_free") != 1u || mpi_wrap::after_finalize_calls() != 1u) {
    return 13;
  }

  // MPI_Info_free is always available (MPI 4.1 §11.4.1/Table 11.1): it is
  // forwarded after finalize and is not an after-finalize error.
  if (MPI_Info_free(&always_available) != MPI_SUCCESS) {
    return 14;
  }
  if (mpi_wrap::count("MPI_Info_free") != 1u || mpi_wrap::after_finalize_calls() != 1u) {
    return 15;
  }
  return 0;
}
