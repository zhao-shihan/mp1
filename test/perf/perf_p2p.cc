// Point-to-point microbenchmarks: pointer and scalar send/recv, nonblocking isend/irecv
// with wait, request/wait overhead (wait_all of 1024 completed requests), and empty multi-wait.
// The pointer ratio gate applies to the isend/irecv path; everything else is recorded.

#include "perf_common.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <vector>

namespace {

  bool g_gate_failed = false;

  void emit(const mp1_perf::options& opts, mp1_perf::case_result row, bool enforce) {
    mp1_perf::aggregate_row(row);
    g_gate_failed = mp1_perf::finish_row(row, enforce) or g_gate_failed;
    mp1_perf::write_case(opts, row);
  }

  struct context {
    const mp1_perf::options& opts;
    mp1::mpi_communicator comm;
    int rank{0};
    int size{1};
    int dest{0};
    int source{0};
    int tag{0};
  };

  void bench_pointer_blocking(const context& ctx, int count, std::size_t bytes) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank * 100000);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          // Even ranks send first and odd ranks receive first: every ring send then finds
          // its already-posted receive, so a rendezvous-sized message cannot deadlock.
          if (ctx.rank % 2 == 0) {
            mp1::send(ctx.comm, std::span(send), ctx.dest, mp1::tag{ctx.tag});
            mp1::recv(ctx.comm, std::span(recv), ctx.source, mp1::tag{ctx.tag});
          } else {
            mp1::recv(ctx.comm, std::span(recv), ctx.source, mp1::tag{ctx.tag});
            mp1::send(ctx.comm, std::span(send), ctx.dest, mp1::tag{ctx.tag});
          }
        },
        [&] {
          if (ctx.rank % 2 == 0) {
            mp1_perf::raw_send(ctx.comm, send.data(), count, ctx.dest, ctx.tag);
            mp1_perf::raw_recv(ctx.comm, recv.data(), count, ctx.source, ctx.tag);
          } else {
            mp1_perf::raw_recv(ctx.comm, recv.data(), count, ctx.source, ctx.tag);
            mp1_perf::raw_send(ctx.comm, send.data(), count, ctx.dest, ctx.tag);
          }
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "pointer_sendrecv";
    row.family = "p2p_pointer";
    row.reference = "raw_native";
    row.ranks = ctx.size;
    row.bytes = bytes;
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "blocking ring exchange; recorded, not gated";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_pointer_nonblocking(const context& ctx, int count, std::size_t bytes) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank * 100000);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::request receive = mp1::irecv(ctx.comm, std::span(recv), ctx.source, mp1::tag{ctx.tag});
          mp1::request send_request = mp1::isend(ctx.comm, std::span(send), ctx.dest, mp1::tag{ctx.tag});
          send_request.wait();
          receive.wait();
        },
        [&] {
          MPI_Request receive =
              mp1_perf::raw_irecv(ctx.comm, recv.data(), count, ctx.source, ctx.tag);
          MPI_Request send_request =
              mp1_perf::raw_isend(ctx.comm, send.data(), count, ctx.dest, ctx.tag);
          mp1_perf::raw_wait(send_request);
          mp1_perf::raw_wait(receive);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "pointer_isend_irecv_wait";
    row.family = "p2p_pointer";
    row.reference = "raw_native";
    row.ranks = ctx.size;
    row.bytes = bytes;
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::p2p_pointer_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "pointer ring exchange; native MPI_Isend_c/MPI_Irecv_c + MPI_Wait";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_scalar(const context& ctx, std::size_t bytes) {
    int send = ctx.rank * 100000;
    int recv = -1;
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank % 2 == 0) {
            mp1::send(ctx.comm, send, ctx.dest, mp1::tag{ctx.tag});
            mp1::recv(ctx.comm, recv, ctx.source, mp1::tag{ctx.tag});
          } else {
            mp1::recv(ctx.comm, recv, ctx.source, mp1::tag{ctx.tag});
            mp1::send(ctx.comm, send, ctx.dest, mp1::tag{ctx.tag});
          }
        },
        [&] {
          if (ctx.rank % 2 == 0) {
            mp1_perf::raw_send(ctx.comm, &send, 1, ctx.dest, ctx.tag);
            mp1_perf::raw_recv(ctx.comm, &recv, 1, ctx.source, ctx.tag);
          } else {
            mp1_perf::raw_recv(ctx.comm, &recv, 1, ctx.source, ctx.tag);
            mp1_perf::raw_send(ctx.comm, &send, 1, ctx.dest, ctx.tag);
          }
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "scalar_sendrecv";
    row.family = "p2p_scalar";
    row.reference = "raw_native";
    row.ranks = ctx.size;
    row.bytes = bytes;
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "scalar overload; recorded, not gated";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  // Request-overhead shape: all requests are completed by MPI_PROC_NULL receives, so the
  // measurement isolates request bookkeeping and one MPI_Waitall, not data transfer.
  void bench_waitall(const context& ctx) {
    constexpr int request_count = 1024;
    int value = 0;
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          std::vector<mp1::request> requests;
          requests.reserve(request_count);
          for (int i = 0; i < request_count; ++i) {
            requests.push_back(mp1::irecv(ctx.comm, std::span(&value, 1), MPI_PROC_NULL, mp1::tag{0}));
          }
          (void)mp1::wait_all(requests);
        },
        [&] {
          std::vector<MPI_Request> handles(static_cast<std::size_t>(request_count),
                                           MPI_REQUEST_NULL);
          for (int i = 0; i < request_count; ++i) {
            MPI_Irecv(&value, 1, MPI_INT, MPI_PROC_NULL, 0, ctx.comm.native_handle(),
                      &handles[static_cast<std::size_t>(i)]);
          }
          mp1_perf::raw_waitall(handles);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "waitall_1024";
    row.family = "request_overhead";
    row.reference = "raw_native";
    row.ranks = ctx.size;
    row.bytes = 0;
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "1024 PROC_NULL requests + one wait_all; recorded, not gated";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_empty_waitall(const context& ctx) {
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          std::vector<mp1::request> empty;
          (void)mp1::wait_all(empty);
        },
        [&] {
          std::vector<MPI_Request> empty;
          mp1_perf::raw_waitall(empty);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "waitall_empty";
    row.family = "request_overhead";
    row.reference = "raw_native";
    row.ranks = ctx.size;
    row.bytes = 0;
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "empty multi-wait: zero MPI calls by contract; recorded";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void run(const mp1_perf::options& opts) {
    context ctx{opts, mp1::comm_world(), 0, 0, 0, 0, 0};
    ctx.rank = ctx.comm.rank();
    ctx.size = ctx.comm.size();
    ctx.dest = (ctx.rank + 1) % ctx.size;
    ctx.source = (ctx.rank + ctx.size - 1) % ctx.size;
    ctx.tag = 7100;

    bench_scalar(ctx, sizeof(int));
    bench_pointer_blocking(ctx, 1, sizeof(int));
    // The gate names 8 B and 1 KiB explicitly; always include 8 B even if --sizes omits it.
    std::vector<std::size_t> sizes = opts.sizes;
    if (std::find(sizes.begin(), sizes.end(), 8) == sizes.end()) {
      sizes.insert(sizes.begin(), 8);
    }
    for (const std::size_t bytes : sizes) {
      const int count = mp1_perf::elements_for_bytes(bytes);
      bench_pointer_blocking(ctx, count, static_cast<std::size_t>(count) * sizeof(int));
      bench_pointer_nonblocking(ctx, count, static_cast<std::size_t>(count) * sizeof(int));
    }
    bench_waitall(ctx);
    bench_empty_waitall(ctx);
  }

}  // namespace

int main(int argc, char** argv) {
  mp1::init();
  int status = 0;
  try {
    mp1_perf::options opts = mp1_perf::parse_options(argc, argv);
    mp1_perf::write_environment(opts);
    run(opts);
    status = g_gate_failed ? 1 : 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "perf_p2p: %s\n", error.what());
    status = 2;
  }
  try {
    mp1::finalize();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "perf_p2p: finalize: %s\n", error.what());
    status = status == 0 ? 2 : status;
  }
  return status;
}
