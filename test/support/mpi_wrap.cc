#include "mpi_wrap.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#define MP1_WRAP_ACTIVE 1
#include <mpi.h>
#else
#define MP1_WRAP_ACTIVE 0
#endif

namespace {

  struct counter_entry {
    const char* symbol;
    std::atomic<std::size_t> calls;
    // Test-only datatype-handle token capture for the p2p region-ownership gate.  The
    // secondary slot is used only by MPI_Isendrecv_c (recvtype).
    std::atomic<std::uintptr_t> datatype;
    std::atomic<std::uintptr_t> recv_datatype;
    // Test-only buffer-address token for the absolute-address sequence-send gate.
    std::atomic<std::uintptr_t> buffer;
    // Test-only MPI_Op_create commute argument capture (-1 = not observed).
    std::atomic<int> commute;
  };

  counter_entry* counter_table(std::size_t& size) {
    static counter_entry entries[] = {
#define MP1_WRAP_SYMBOL(name) {#name, {}, {}, {}, {}, {-1}},
#include "mpi_wrap_symbols.hpp"
#undef MP1_WRAP_SYMBOL
    };
    size = sizeof(entries) / sizeof(entries[0]);
    return entries;
  }

  counter_entry* find_entry(const char* symbol) {
    std::size_t size = 0;
    counter_entry* entries = counter_table(size);
    for (std::size_t i = 0; i < size; ++i) {
      if (std::strcmp(entries[i].symbol, symbol) == 0) {
        return &entries[i];
      }
    }
    return nullptr;
  }

  std::atomic<bool> finalized{false};
  std::atomic<std::size_t> calls_after_finalize{0};

  // The allocation counter is intentionally independent of MPI so it can be
  // used before init and after finalize.
  std::atomic<std::size_t> allocation_counter{0};

}  // namespace

namespace mpi_wrap {

  bool supported() noexcept {
    return MP1_WRAP_ACTIVE != 0;
  }

  std::size_t count(const char* symbol) {
    if (symbol == nullptr) {
      throw std::invalid_argument("mpi_wrap::count: null symbol");
    }
    const counter_entry* entry = find_entry(symbol);
    if (entry == nullptr) {
      throw std::invalid_argument(std::string("mpi_wrap::count: unknown symbol ") + symbol);
    }
    return entry->calls.load(std::memory_order_relaxed);
  }

  namespace {

    [[noreturn]] void throw_unknown_datatype_symbol(const char* symbol) {
      throw std::invalid_argument(std::string("mpi_wrap::last_datatype: unknown symbol ") +
                                  symbol);
    }

    std::uintptr_t datatype_of(const counter_entry* entry, bool recv) {
      if (entry == nullptr)
        return 0;
      return (recv ? entry->recv_datatype : entry->datatype).load(std::memory_order_relaxed);
    }

  }  // namespace

  std::uintptr_t last_datatype(const char* symbol) {
    if (symbol == nullptr) {
      throw std::invalid_argument("mpi_wrap::last_datatype: null symbol");
    }
    const counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      throw_unknown_datatype_symbol(symbol);
    return datatype_of(entry, false);
  }

  std::uintptr_t last_recv_datatype(const char* symbol) {
    if (symbol == nullptr) {
      throw std::invalid_argument("mpi_wrap::last_recv_datatype: null symbol");
    }
    const counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      throw_unknown_datatype_symbol(symbol);
    return datatype_of(entry, true);
  }

  std::uintptr_t last_buffer(const char* symbol) {
    if (symbol == nullptr) {
      throw std::invalid_argument("mpi_wrap::last_buffer: null symbol");
    }
    const counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      throw std::invalid_argument(std::string("mpi_wrap::last_buffer: unknown symbol ") +
                                  symbol);
    return entry->buffer.load(std::memory_order_relaxed);
  }

  int last_commute(const char* symbol) {
    if (symbol == nullptr) {
      throw std::invalid_argument("mpi_wrap::last_commute: null symbol");
    }
    const counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      throw std::invalid_argument(std::string("mpi_wrap::last_commute: unknown symbol ") +
                                  symbol);
    return entry->commute.load(std::memory_order_relaxed);
  }

  std::size_t total() noexcept {
    std::size_t size = 0;
    counter_entry* entries = counter_table(size);
    std::size_t sum = 0;
    for (std::size_t i = 0; i < size; ++i) {
      sum += entries[i].calls.load(std::memory_order_relaxed);
    }
    return sum;
  }

  void reset_counts() noexcept {
    std::size_t size = 0;
    counter_entry* entries = counter_table(size);
    for (std::size_t i = 0; i < size; ++i) {
      entries[i].calls.store(0, std::memory_order_relaxed);
      entries[i].datatype.store(0, std::memory_order_relaxed);
      entries[i].recv_datatype.store(0, std::memory_order_relaxed);
      entries[i].buffer.store(0, std::memory_order_relaxed);
      entries[i].commute.store(-1, std::memory_order_relaxed);
    }
  }

  bool after_finalize() noexcept {
    return finalized.load(std::memory_order_acquire);
  }

  std::size_t after_finalize_calls() noexcept {
    return calls_after_finalize.load(std::memory_order_relaxed);
  }

  std::size_t allocation_count() noexcept {
    return allocation_counter.load(std::memory_order_relaxed);
  }

  void reset_allocation_count() noexcept {
    allocation_counter.store(0, std::memory_order_relaxed);
  }

  namespace detail {

    // MPI 4.1 Table 11.1 ("always available") entry points are legal before init and after
    // finalize.  The after-finalize detector forwards them to the real MPI and does not count
    // them as violations; they are still counted by the call-count gates.
    bool always_available(const char* symbol) noexcept {
      static const char* const names[] = {
          "MPI_Info_create",     "MPI_Info_create_env", "MPI_Info_set",
          "MPI_Info_delete",     "MPI_Info_get_string", "MPI_Info_get_nkeys",
          "MPI_Info_get_nthkey", "MPI_Info_dup",        "MPI_Info_free",
          "MPI_Error_string",    "MPI_Error_class",     "MPI_Initialized",
          "MPI_Finalized",       "MPI_Get_version",     "MPI_Get_library_version",
      };
      for (const char* name : names) {
        if (std::strcmp(symbol, name) == 0)
          return true;
      }
      return false;
    }

    void note_call(const char* symbol) {
      counter_entry* entry = find_entry(symbol);
      if (entry == nullptr) {
        std::fprintf(stderr, "mpi_wrap: internal error: unregistered symbol %s\n", symbol);
        std::abort();
      }
      entry->calls.fetch_add(1, std::memory_order_relaxed);
      if (finalized.load(std::memory_order_acquire) and not always_available(symbol)) {
        calls_after_finalize.fetch_add(1, std::memory_order_relaxed);
        if (std::getenv("MP1_TEST_ABORT_AFTER_FINALIZE") != nullptr) {
          std::fprintf(stderr, "mpi_wrap: %s called after MPI_Finalize\n", symbol);
          std::abort();
        }
      }
    }

    bool blocks_after_finalize(const char* symbol) noexcept {
      return always_available(symbol) ? false : finalized.load(std::memory_order_acquire);
    }

    void mark_finalized() noexcept {
      finalized.store(true, std::memory_order_release);
    }

  }  // namespace detail

}  // namespace mpi_wrap

// A wrapped call made after MPI_Finalize is recorded and returns MPI_ERR_OTHER
// without touching the (no longer valid) MPI library.  Destructors can
// therefore be checked with a deterministic flag instead of a hang/crash.
#if MP1_WRAP_ACTIVE

// clang-format off
#define MP1_WRAP_INT(NAME, PARAMS, ARGS)                                      \
  extern "C" int __real_##NAME PARAMS;                                        \
  extern "C" int __wrap_##NAME PARAMS {                                       \
    ::mpi_wrap::detail::note_call(#NAME);                                     \
    if (::mpi_wrap::detail::blocks_after_finalize(#NAME)) {                   \
      return MPI_ERR_OTHER;                                                   \
    }                                                                         \
    return __real_##NAME ARGS;                                                \
  }

#define MP1_WRAP_INT_POST(NAME, PARAMS, ARGS, POST)                           \
  extern "C" int __real_##NAME PARAMS;                                        \
  extern "C" int __wrap_##NAME PARAMS {                                       \
    ::mpi_wrap::detail::note_call(#NAME);                                     \
    if (::mpi_wrap::detail::blocks_after_finalize(#NAME)) {                   \
      return MPI_ERR_OTHER;                                                   \
    }                                                                         \
    const int mp1_wrap_result = __real_##NAME ARGS;                           \
    POST;                                                                     \
    return mp1_wrap_result;                                                   \
  }

namespace mpi_wrap::detail {

  // A datatype handle is an integer on MPICH and a pointer on Open MPI; both become an
  // integer token for equality comparison only.  The token is never cast back or passed to
  // MPI.
  template<class Handle>
  std::uintptr_t handle_token(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>)
      return reinterpret_cast<std::uintptr_t>(handle);
    else
      return static_cast<std::uintptr_t>(handle);
  }

  template<class Handle>
  void note_datatype(const char* symbol, Handle handle, bool recv = false) noexcept {
    counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      return;
    auto& slot = recv ? entry->recv_datatype : entry->datatype;
    slot.store(handle_token(handle), std::memory_order_relaxed);
  }

  // A buffer address is a test-only token.  MPI_BOTTOM is a null address on the supported
  // providers, so the token is 0 for the absolute-map form and non-zero for a real buffer.
  inline void note_buffer(const char* symbol, const void* buffer) noexcept {
    counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      return;
    entry->buffer.store(reinterpret_cast<std::uintptr_t>(buffer), std::memory_order_relaxed);
  }

  inline void note_commute(const char* symbol, int commute) noexcept {
    counter_entry* entry = find_entry(symbol);
    if (entry == nullptr)
      return;
    entry->commute.store(commute, std::memory_order_relaxed);
  }

}  // namespace mpi_wrap::detail

MP1_WRAP_INT_POST(MPI_Finalize, (void), (),
  if (mp1_wrap_result == MPI_SUCCESS) ::mpi_wrap::detail::mark_finalized())

MP1_WRAP_INT(MPI_Init, (int* argc, char*** argv), (argc, argv))

MP1_WRAP_INT(MPI_Init_thread,
  (int* argc, char*** argv, int required, int* provided),
  (argc, argv, required, provided))

// Table 11.1 lifecycle queries.  They stay available after MPI_Finalize (always_available
// above), so both wrappers forward the call and only count it for the release-path gates.
MP1_WRAP_INT(MPI_Initialized, (int* flag), (flag))
MP1_WRAP_INT(MPI_Finalized, (int* flag), (flag))

// p2p
MP1_WRAP_INT(MPI_Send,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT_POST(MPI_Send_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm),
  ::mpi_wrap::detail::note_datatype("MPI_Send_c", datatype);
  ::mpi_wrap::detail::note_buffer("MPI_Send_c", buf))
MP1_WRAP_INT(MPI_Recv,
  (void* buf, int count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Status* status),
  (buf, count, datatype, source, tag, comm, status))
MP1_WRAP_INT(MPI_Recv_c,
  (void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Status* status),
  (buf, count, datatype, source, tag, comm, status))
MP1_WRAP_INT(MPI_Isend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT_POST(MPI_Isend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request),
  ::mpi_wrap::detail::note_datatype("MPI_Isend_c", datatype);
  ::mpi_wrap::detail::note_buffer("MPI_Isend_c", buf))
MP1_WRAP_INT(MPI_Irecv,
  (void* buf, int count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, source, tag, comm, request))
MP1_WRAP_INT(MPI_Irecv_c,
  (void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, source, tag, comm, request))
MP1_WRAP_INT(MPI_Send_init,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT_POST(MPI_Send_init_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request),
  ::mpi_wrap::detail::note_datatype("MPI_Send_init_c", datatype))
MP1_WRAP_INT(MPI_Recv_init,
  (void* buf, int count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, source, tag, comm, request))
MP1_WRAP_INT(MPI_Recv_init_c,
  (void* buf, MPI_Count count, MPI_Datatype datatype, int source, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, source, tag, comm, request))
MP1_WRAP_INT(MPI_Sendrecv,
  (const void* sendbuf, int sendcount, MPI_Datatype sendtype, int dest,
   int sendtag, void* recvbuf, int recvcount, MPI_Datatype recvtype,
   int source, int recvtag, MPI_Comm comm, MPI_Status* status),
  (sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
   source, recvtag, comm, status))
MP1_WRAP_INT(MPI_Sendrecv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, int dest,
   int sendtag, void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype,
   int source, int recvtag, MPI_Comm comm, MPI_Status* status),
  (sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
   source, recvtag, comm, status))
MP1_WRAP_INT(MPI_Sendrecv_replace,
  (void* buf, int count, MPI_Datatype datatype, int dest, int sendtag,
   int source, int recvtag, MPI_Comm comm, MPI_Status* status),
  (buf, count, datatype, dest, sendtag, source, recvtag, comm, status))
MP1_WRAP_INT(MPI_Sendrecv_replace_c,
  (void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int sendtag,
   int source, int recvtag, MPI_Comm comm, MPI_Status* status),
  (buf, count, datatype, dest, sendtag, source, recvtag, comm, status))
MP1_WRAP_INT_POST(MPI_Isendrecv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, int dest,
   int sendtag, void* recvbuf, MPI_Count recvcount, MPI_Datatype recvtype,
   int source, int recvtag, MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype,
   source, recvtag, comm, request),
  ::mpi_wrap::detail::note_datatype("MPI_Isendrecv_c", sendtype);
  ::mpi_wrap::detail::note_datatype("MPI_Isendrecv_c", recvtype, true))
MP1_WRAP_INT_POST(MPI_Isendrecv_replace_c,
  (void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int sendtag,
   int source, int recvtag, MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, sendtag, source, recvtag, comm, request),
  ::mpi_wrap::detail::note_datatype("MPI_Isendrecv_replace_c", datatype))

// buffered / synchronous / ready p2p modes
MP1_WRAP_INT(MPI_Bsend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Bsend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Ibsend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Ibsend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Bsend_init,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Bsend_init_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Ssend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Ssend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Issend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Issend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Ssend_init,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Ssend_init_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Rsend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Rsend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm),
  (buf, count, datatype, dest, tag, comm))
MP1_WRAP_INT(MPI_Irsend,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Irsend_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Rsend_init,
  (const void* buf, int count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))
MP1_WRAP_INT(MPI_Rsend_init_c,
  (const void* buf, MPI_Count count, MPI_Datatype datatype, int dest, int tag,
   MPI_Comm comm, MPI_Request* request),
  (buf, count, datatype, dest, tag, comm, request))

// matching / probes
MP1_WRAP_INT(MPI_Probe,
  (int source, int tag, MPI_Comm comm, MPI_Status* status),
  (source, tag, comm, status))
MP1_WRAP_INT(MPI_Iprobe,
  (int source, int tag, MPI_Comm comm, int* flag, MPI_Status* status),
  (source, tag, comm, flag, status))
MP1_WRAP_INT(MPI_Mprobe,
  (int source, int tag, MPI_Comm comm, MPI_Message* message,
   MPI_Status* status),
  (source, tag, comm, message, status))
MP1_WRAP_INT(MPI_Improbe,
  (int source, int tag, MPI_Comm comm, int* flag, MPI_Message* message,
   MPI_Status* status),
  (source, tag, comm, flag, message, status))
MP1_WRAP_INT(MPI_Mrecv,
  (void* buf, int count, MPI_Datatype datatype, MPI_Message* message,
   MPI_Status* status),
  (buf, count, datatype, message, status))
MP1_WRAP_INT(MPI_Imrecv,
  (void* buf, int count, MPI_Datatype datatype, MPI_Message* message,
   MPI_Request* request),
  (buf, count, datatype, message, request))

// request lifecycle
MP1_WRAP_INT(MPI_Cancel, (MPI_Request * request), (request))
MP1_WRAP_INT(MPI_Request_free, (MPI_Request * request), (request))
MP1_WRAP_INT(MPI_Wait, (MPI_Request * request, MPI_Status* status),
  (request, status))
MP1_WRAP_INT(MPI_Test,
  (MPI_Request * request, int* flag, MPI_Status* status),
  (request, flag, status))
MP1_WRAP_INT(MPI_Waitall,
  (int count, MPI_Request requests[], MPI_Status statuses[]),
  (count, requests, statuses))
MP1_WRAP_INT(MPI_Testall,
  (int count, MPI_Request requests[], int* flag, MPI_Status statuses[]),
  (count, requests, flag, statuses))
MP1_WRAP_INT(MPI_Waitany,
  (int count, MPI_Request requests[], int* index, MPI_Status* status),
  (count, requests, index, status))
MP1_WRAP_INT(MPI_Testany,
  (int count, MPI_Request requests[], int* index, int* flag,
   MPI_Status* status),
  (count, requests, index, flag, status))
MP1_WRAP_INT(MPI_Waitsome,
  (int incount, MPI_Request requests[], int* outcount, int indices[],
   MPI_Status statuses[]),
  (incount, requests, outcount, indices, statuses))
MP1_WRAP_INT(MPI_Testsome,
  (int incount, MPI_Request requests[], int* outcount, int indices[],
   MPI_Status statuses[]),
  (incount, requests, outcount, indices, statuses))
MP1_WRAP_INT(MPI_Request_get_status,
  (MPI_Request request, int* flag, MPI_Status* status),
  (request, flag, status))
MP1_WRAP_INT(MPI_Request_get_status_all,
  (int count, const MPI_Request requests[], int* flag, MPI_Status statuses[]),
  (count, requests, flag, statuses))
MP1_WRAP_INT(MPI_Request_get_status_any,
  (int count, const MPI_Request requests[], int* index, int* flag,
   MPI_Status* status),
  (count, requests, index, flag, status))
MP1_WRAP_INT(MPI_Request_get_status_some,
  (int incount, const MPI_Request requests[], int* outcount, int indices[],
   MPI_Status statuses[]),
  (incount, requests, outcount, indices, statuses))
MP1_WRAP_INT(MPI_Start, (MPI_Request * request), (request))
MP1_WRAP_INT(MPI_Startall, (int count, MPI_Request requests[]),
  (count, requests))

// variable collectives
MP1_WRAP_INT(MPI_Gatherv,
  (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
   const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
   MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
   comm))
MP1_WRAP_INT(MPI_Gatherv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype,
   void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint displs[],
   MPI_Datatype recvtype, int root, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
   comm))
MP1_WRAP_INT(MPI_Igatherv,
  (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
   const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root,
   MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
   comm, request))
MP1_WRAP_INT(MPI_Igatherv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype,
   void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint displs[],
   MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root,
   comm, request))
MP1_WRAP_INT(MPI_Scatterv,
  (const void* sendbuf, const int sendcounts[], const int displs[],
   MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
   int root, MPI_Comm comm),
  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
   comm))
MP1_WRAP_INT(MPI_Scatterv_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint displs[],
   MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
   MPI_Datatype recvtype, int root, MPI_Comm comm),
  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
   comm))
MP1_WRAP_INT(MPI_Iscatterv,
  (const void* sendbuf, const int sendcounts[], const int displs[],
   MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype,
   int root, MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
   comm, request))
MP1_WRAP_INT(MPI_Iscatterv_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint displs[],
   MPI_Datatype sendtype, void* recvbuf, MPI_Count recvcount,
   MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
   comm, request))
MP1_WRAP_INT(MPI_Allgatherv,
  (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
   const int recvcounts[], const int displs[], MPI_Datatype recvtype,
   MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm))
MP1_WRAP_INT(MPI_Allgatherv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype,
   void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint displs[],
   MPI_Datatype recvtype, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm))
MP1_WRAP_INT(MPI_Iallgatherv,
  (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
   const int recvcounts[], const int displs[], MPI_Datatype recvtype,
   MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
   request))
MP1_WRAP_INT(MPI_Iallgatherv_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype,
   void* recvbuf, const MPI_Count recvcounts[], const MPI_Aint displs[],
   MPI_Datatype recvtype, MPI_Comm comm, MPI_Request* request),
  (sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm,
   request))
MP1_WRAP_INT(MPI_Alltoallv,
  (const void* sendbuf, const int sendcounts[], const int sdispls[],
   MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
   const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm),
  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
   recvtype, comm))
MP1_WRAP_INT(MPI_Alltoallv_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint sdispls[],
   MPI_Datatype sendtype, void* recvbuf, const MPI_Count recvcounts[],
   const MPI_Aint rdispls[], MPI_Datatype recvtype, MPI_Comm comm),
  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
   recvtype, comm))
MP1_WRAP_INT(MPI_Ialltoallv,
  (const void* sendbuf, const int sendcounts[], const int sdispls[],
   MPI_Datatype sendtype, void* recvbuf, const int recvcounts[],
   const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
   MPI_Request* request),
  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
   recvtype, comm, request))
MP1_WRAP_INT(MPI_Ialltoallv_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint sdispls[],
   MPI_Datatype sendtype, void* recvbuf, const MPI_Count recvcounts[],
   const MPI_Aint rdispls[], MPI_Datatype recvtype, MPI_Comm comm,
   MPI_Request* request),
  (sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
   recvtype, comm, request))

// Alltoallw is the forbidden variable engine.  These wrappers exist so the
// call-count gates can assert exactly zero calls.
MP1_WRAP_INT(MPI_Alltoallw,
  (const void* sendbuf, const int sendcounts[], const int sdispls[],
   const MPI_Datatype sendtypes[], void* recvbuf, const int recvcounts[],
   const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm),
  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
   recvtypes, comm))
MP1_WRAP_INT(MPI_Alltoallw_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint sdispls[],
   const MPI_Datatype sendtypes[], void* recvbuf, const MPI_Count recvcounts[],
   const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm),
  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
   recvtypes, comm))
MP1_WRAP_INT(MPI_Ialltoallw,
  (const void* sendbuf, const int sendcounts[], const int sdispls[],
   const MPI_Datatype sendtypes[], void* recvbuf, const int recvcounts[],
   const int rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
   MPI_Request* request),
  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
   recvtypes, comm, request))
MP1_WRAP_INT(MPI_Ialltoallw_c,
  (const void* sendbuf, const MPI_Count sendcounts[], const MPI_Aint sdispls[],
   const MPI_Datatype sendtypes[], void* recvbuf, const MPI_Count recvcounts[],
   const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm,
   MPI_Request* request),
  (sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls,
   recvtypes, comm, request))

// reductions
MP1_WRAP_INT(MPI_Reduce,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, int root, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, root, comm))
MP1_WRAP_INT(MPI_Reduce_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, root, comm))
MP1_WRAP_INT(MPI_Reduce_local_c,
  (const void* inbuf, void* inoutbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op),
  (inbuf, inoutbuf, count, datatype, op))
MP1_WRAP_INT(MPI_Ireduce,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, int root, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, root, comm, request))
MP1_WRAP_INT(MPI_Ireduce_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
   MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, root, comm, request))
MP1_WRAP_INT(MPI_Allreduce,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Allreduce_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Iallreduce,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Iallreduce_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Reduce_scatter,
  (const void* sendbuf, void* recvbuf, const int recvcounts[],
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, recvcounts, datatype, op, comm))
MP1_WRAP_INT(MPI_Reduce_scatter_c,
  (const void* sendbuf, void* recvbuf, const MPI_Count recvcounts[],
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, recvcounts, datatype, op, comm))
MP1_WRAP_INT(MPI_Ireduce_scatter,
  (const void* sendbuf, void* recvbuf, const int recvcounts[],
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, recvcounts, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Ireduce_scatter_c,
  (const void* sendbuf, void* recvbuf, const MPI_Count recvcounts[],
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, recvcounts, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Reduce_scatter_block,
  (const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, recvcount, datatype, op, comm))
MP1_WRAP_INT(MPI_Reduce_scatter_block_c,
  (const void* sendbuf, void* recvbuf, MPI_Count recvcount,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, recvcount, datatype, op, comm))
MP1_WRAP_INT(MPI_Ireduce_scatter_block,
  (const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, recvcount, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Ireduce_scatter_block_c,
  (const void* sendbuf, void* recvbuf, MPI_Count recvcount,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, recvcount, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Scan,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Scan_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Iscan,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Iscan_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Exscan,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Exscan_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm),
  (sendbuf, recvbuf, count, datatype, op, comm))
MP1_WRAP_INT(MPI_Iexscan,
  (const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
   MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))
MP1_WRAP_INT(MPI_Iexscan_c,
  (const void* sendbuf, void* recvbuf, MPI_Count count,
   MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, MPI_Request* request),
  (sendbuf, recvbuf, count, datatype, op, comm, request))

// layout / staging
MP1_WRAP_INT_POST(MPI_Type_dup,
  (MPI_Datatype oldtype, MPI_Datatype* newtype),
  (oldtype, newtype),
  if (mp1_wrap_result == MPI_SUCCESS)
    ::mpi_wrap::detail::note_datatype("MPI_Type_dup", *newtype))
MP1_WRAP_INT(MPI_Type_commit, (MPI_Datatype * datatype), (datatype))
MP1_WRAP_INT(MPI_Type_free, (MPI_Datatype * datatype), (datatype))
MP1_WRAP_INT(MPI_Get_address, (const void* location, MPI_Aint* address),
  (location, address))
MP1_WRAP_INT(MPI_Get_count_c,
  (const MPI_Status* status, MPI_Datatype datatype, MPI_Count* count),
  (status, datatype, count))
MP1_WRAP_INT(MPI_Type_create_hindexed,
  (int count, const int array_of_blocklengths[],
   const MPI_Aint array_of_displacements[], MPI_Datatype oldtype,
   MPI_Datatype* newtype),
  (count, array_of_blocklengths, array_of_displacements, oldtype, newtype))
MP1_WRAP_INT(MPI_Pack,
  (const void* inbuf, int incount, MPI_Datatype datatype, void* outbuf,
   int outsize, int* position, MPI_Comm comm),
  (inbuf, incount, datatype, outbuf, outsize, position, comm))
MP1_WRAP_INT(MPI_Pack_c,
  (const void* inbuf, MPI_Count incount, MPI_Datatype datatype, void* outbuf,
   MPI_Count outsize, MPI_Count* position, MPI_Comm comm),
  (inbuf, incount, datatype, outbuf, outsize, position, comm))
MP1_WRAP_INT(MPI_Pack_size,
  (int incount, MPI_Datatype datatype, MPI_Comm comm, int* size),
  (incount, datatype, comm, size))
MP1_WRAP_INT(MPI_Pack_size_c,
  (MPI_Count incount, MPI_Datatype datatype, MPI_Comm comm, MPI_Count* size),
  (incount, datatype, comm, size))
MP1_WRAP_INT(MPI_Unpack,
  (const void* inbuf, int insize, int* position, void* outbuf, int outcount,
   MPI_Datatype datatype, MPI_Comm comm),
  (inbuf, insize, position, outbuf, outcount, datatype, comm))
MP1_WRAP_INT(MPI_Unpack_c,
  (const void* inbuf, MPI_Count insize, MPI_Count* position, void* outbuf,
   MPI_Count outcount, MPI_Datatype datatype, MPI_Comm comm),
  (inbuf, insize, position, outbuf, outcount, datatype, comm))

// RMA accumulating data movement counted by the homogeneous-contract test: a rejected
// untrusted annotation must not reach any of these.
MP1_WRAP_INT(MPI_Accumulate_c,
  (const void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype,
   int target_rank, MPI_Aint target_disp, MPI_Count target_count,
   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win),
  (origin_addr, origin_count, origin_datatype, target_rank, target_disp,
   target_count, target_datatype, op, win))
MP1_WRAP_INT(MPI_Raccumulate_c,
  (const void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype,
   int target_rank, MPI_Aint target_disp, MPI_Count target_count,
   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request),
  (origin_addr, origin_count, origin_datatype, target_rank, target_disp,
   target_count, target_datatype, op, win, request))
MP1_WRAP_INT(MPI_Get_accumulate_c,
  (const void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype,
   void* result_addr, MPI_Count result_count, MPI_Datatype result_datatype,
   int target_rank, MPI_Aint target_disp, MPI_Count target_count,
   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win),
  (origin_addr, origin_count, origin_datatype, result_addr, result_count,
   result_datatype, target_rank, target_disp, target_count, target_datatype,
   op, win))
MP1_WRAP_INT(MPI_Rget_accumulate_c,
  (const void* origin_addr, MPI_Count origin_count, MPI_Datatype origin_datatype,
   void* result_addr, MPI_Count result_count, MPI_Datatype result_datatype,
   int target_rank, MPI_Aint target_disp, MPI_Count target_count,
   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win, MPI_Request* request),
  (origin_addr, origin_count, origin_datatype, result_addr, result_count,
   result_datatype, target_rank, target_disp, target_count, target_datatype,
   op, win, request))

// The one-time untrusted-annotation validator's metadata walk.  Counting the contents
// queries proves that a second accumulating call reuses the cached decision.
MP1_WRAP_INT(MPI_Type_get_envelope_c,
  (MPI_Datatype type, MPI_Count* num_integers, MPI_Count* num_addresses,
   MPI_Count* num_large_counts, MPI_Count* num_datatypes, int* combiner),
  (type, num_integers, num_addresses, num_large_counts, num_datatypes, combiner))
MP1_WRAP_INT(MPI_Type_get_contents_c,
  (MPI_Datatype mtype, MPI_Count max_integers, MPI_Count max_addresses,
   MPI_Count max_large_counts, MPI_Count max_datatypes, int array_of_integers[],
   MPI_Aint array_of_addresses[], MPI_Count array_of_large_counts[],
   MPI_Datatype array_of_datatypes[]),
  (mtype, max_integers, max_addresses, max_large_counts, max_datatypes,
   array_of_integers, array_of_addresses, array_of_large_counts,
   array_of_datatypes))

// communicator metadata
MP1_WRAP_INT(MPI_Comm_size, (MPI_Comm comm, int* size), (comm, size))
MP1_WRAP_INT(MPI_Comm_rank, (MPI_Comm comm, int* rank), (comm, rank))
MP1_WRAP_INT(MPI_Comm_test_inter, (MPI_Comm comm, int* flag), (comm, flag))

// process buffer
MP1_WRAP_INT(MPI_Buffer_attach_c,
  (void* buffer, MPI_Count size),
  (buffer, size))
MP1_WRAP_INT(MPI_Buffer_detach_c,
  (void* buffer_addr, MPI_Count* size),
  (buffer_addr, size))

// file data access (dense pointer+count/container forms)
MP1_WRAP_INT(MPI_File_read_at_c,
  (MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Status* status),
  (fh, offset, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_at_c,
  (MPI_File fh, MPI_Offset offset, const void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Status* status),
  (fh, offset, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_shared_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_shared_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_at_all_c,
  (MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Status* status),
  (fh, offset, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_at_all_c,
  (MPI_File fh, MPI_Offset offset, const void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Status* status),
  (fh, offset, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_all_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_all_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_ordered_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_write_ordered_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Status* status),
  (fh, buf, count, datatype, status))
MP1_WRAP_INT(MPI_File_read_at_all_begin_c,
  (MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count,
   MPI_Datatype datatype),
  (fh, offset, buf, count, datatype))
MP1_WRAP_INT_POST(MPI_File_write_at_all_begin_c,
  (MPI_File fh, MPI_Offset offset, const void* buf, MPI_Count count,
   MPI_Datatype datatype),
  (fh, offset, buf, count, datatype),
  if (mp1_wrap_result == MPI_SUCCESS)
    ::mpi_wrap::detail::note_datatype("MPI_File_write_at_all_begin_c", datatype))
MP1_WRAP_INT(MPI_File_read_all_begin_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype),
  (fh, buf, count, datatype))
MP1_WRAP_INT(MPI_File_write_all_begin_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype),
  (fh, buf, count, datatype))
MP1_WRAP_INT(MPI_File_read_ordered_begin_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype),
  (fh, buf, count, datatype))
MP1_WRAP_INT(MPI_File_write_ordered_begin_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype),
  (fh, buf, count, datatype))
MP1_WRAP_INT_POST(MPI_File_iread_at_c,
  (MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Request* request),
  (fh, offset, buf, count, datatype, request),
  if (mp1_wrap_result == MPI_SUCCESS)
    ::mpi_wrap::detail::note_datatype("MPI_File_iread_at_c", datatype))
MP1_WRAP_INT_POST(MPI_File_iwrite_at_c,
  (MPI_File fh, MPI_Offset offset, const void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Request* request),
  (fh, offset, buf, count, datatype, request),
  if (mp1_wrap_result == MPI_SUCCESS)
    ::mpi_wrap::detail::note_datatype("MPI_File_iwrite_at_c", datatype))
MP1_WRAP_INT(MPI_File_iread_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iwrite_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iread_shared_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iwrite_shared_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iread_at_all_c,
  (MPI_File fh, MPI_Offset offset, void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Request* request),
  (fh, offset, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iwrite_at_all_c,
  (MPI_File fh, MPI_Offset offset, const void* buf, MPI_Count count,
   MPI_Datatype datatype, MPI_Request* request),
  (fh, offset, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iread_all_c,
  (MPI_File fh, void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))
MP1_WRAP_INT(MPI_File_iwrite_all_c,
  (MPI_File fh, const void* buf, MPI_Count count, MPI_Datatype datatype,
   MPI_Request* request),
  (fh, buf, count, datatype, request))

// fixed collectives
MP1_WRAP_INT(MPI_Barrier, (MPI_Comm comm), (comm))
MP1_WRAP_INT(MPI_Gather_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
   MPI_Count recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm))
MP1_WRAP_INT(MPI_Scatter_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
   MPI_Count recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm))
MP1_WRAP_INT(MPI_Allgather_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
   MPI_Count recvcount, MPI_Datatype recvtype, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm))
MP1_WRAP_INT(MPI_Alltoall_c,
  (const void* sendbuf, MPI_Count sendcount, MPI_Datatype sendtype, void* recvbuf,
   MPI_Count recvcount, MPI_Datatype recvtype, MPI_Comm comm),
  (sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm))

// ownership
MP1_WRAP_INT(MPI_Bcast,
  (void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm),
  (buffer, count, datatype, root, comm))
MP1_WRAP_INT(MPI_Comm_free, (MPI_Comm * comm), (comm))
MP1_WRAP_INT(MPI_Comm_disconnect, (MPI_Comm * comm), (comm))
MP1_WRAP_INT(MPI_Comm_spawn,
  (const char* command, char* argv[], int maxprocs, MPI_Info info, int root,
   MPI_Comm comm, MPI_Comm* intercomm, int array_of_errcodes[]),
  (command, argv, maxprocs, info, root, comm, intercomm, array_of_errcodes))
MP1_WRAP_INT(MPI_Comm_spawn_multiple,
  (int count, char* array_of_commands[], char** array_of_argv[],
   const int array_of_maxprocs[], const MPI_Info array_of_info[], int root,
   MPI_Comm comm, MPI_Comm* intercomm, int array_of_errcodes[]),
  (count, array_of_commands, array_of_argv, array_of_maxprocs, array_of_info,
   root, comm, intercomm, array_of_errcodes))
MP1_WRAP_INT(MPI_Comm_get_parent, (MPI_Comm * parent), (parent))
MP1_WRAP_INT(MPI_Win_free, (MPI_Win * win), (win))
MP1_WRAP_INT(MPI_Group_free, (MPI_Group * group), (group))
MP1_WRAP_INT(MPI_File_close, (MPI_File * fh), (fh))
MP1_WRAP_INT(MPI_Info_free, (MPI_Info * info), (info))
MP1_WRAP_INT(MPI_Errhandler_free, (MPI_Errhandler * errhandler), (errhandler))
MP1_WRAP_INT_POST(MPI_Op_create,
  (MPI_User_function * user_fn, int commute, MPI_Op* op),
  (user_fn, commute, op),
  ::mpi_wrap::detail::note_commute("MPI_Op_create", commute))
MP1_WRAP_INT(MPI_Op_free, (MPI_Op * op), (op))

#undef MP1_WRAP_INT
#undef MP1_WRAP_INT_POST
// clang-format on

#endif  // MP1_WRAP_ACTIVE

// Allocation counters: replace the global unaligned allocation functions so
// the performance gates can bound heap use without a separate allocator hook.
// Aligned overloads intentionally keep the standard library implementation.
void* operator new(std::size_t size) {
  void* memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) {
    throw std::bad_alloc();
  }
  allocation_counter.fetch_add(1, std::memory_order_relaxed);
  return memory;
}

void* operator new[](std::size_t size) {
  return ::operator new(size);
}

void operator delete(void* memory) noexcept {
  std::free(memory);
}

void operator delete[](void* memory) noexcept {
  std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
  std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
  std::free(memory);
}
