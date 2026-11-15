#if !(defined MP1_DETAIL_REQUEST_PAYLOAD_HPP)

#define MP1_DETAIL_REQUEST_PAYLOAD_HPP

#include "mp1/impl/status.hpp"

#include <atomic>
#include <cstddef>


namespace mp1::detail {

  // Base of every request-owned payload (staging buffers, owned layouts/count arrays, ...).
  // complete() runs once per successful, non-cancelled completion; a persistent payload may see
  // it once per start/wait cycle. The status argument carries the completing request kind's
  // provenance: a native receive status for a data-carrying receive, a normalized status for a
  // collective/RMA completion (MPI 4.1 3.7.5: a single wait/test's MPI_ERROR is undefined;
  // 6.12/12.3.5: collective and RMA status fields other than error are undefined).
  // prepare_start() runs immediately before every MPI_Start / MPI_Startall and must only do
  // local work (re-pack staged sends, re-prime in-place staged receives, reset per-cycle
  // flags); it never calls a collective operation. It may throw, in which case the request
  // stays inactive and the next start prepares again.
  class request_payload {
  public:
    virtual ~request_payload() = default;

    virtual void complete(const status& status) = 0;

    virtual void prepare_start() {
    }
  };

  inline std::atomic<std::size_t>& abandoned_payload_counter() noexcept {
    static std::atomic<std::size_t> counter{0};
    return counter;
  }

  // The operation may still read/write the payload after MPI_Request_free or finalize, so the
  // only safe action is an intentional process-lifetime leak; it is counted for tests.
  inline void abandon_forever(request_payload* payload) noexcept {
    if (payload == nullptr)
      return;
    abandoned_payload_counter().fetch_add(1, std::memory_order_relaxed);
  }

  inline std::size_t abandoned_payload_count() noexcept {
    return abandoned_payload_counter().load(std::memory_order_relaxed);
  }

}  // namespace mp1::detail

#endif
