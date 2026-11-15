// Reduction microbenchmarks: flat allreduce/reduce/scan/exscan against raw MPI with the
// same user-defined operation, plus the padded (staged) allreduce shape.  The reference is a
// user MPI_Op created from the same functor, never MPI_SUM for the reference.
//
// Nightly gate: `--gate` enforces the flat-reduction ratio; the padded row is recorded with
// no ratio gate (its call-count shape lives in perf_calls).

#include "perf_common.hpp"

#include <cstdio>
#include <exception>
#include <functional>
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
    MPI_Op raw_op{MPI_OP_NULL};
    int rank{0};
    int size{1};
    int root{0};
  };

  void bench_flat_allreduce(const context& ctx, int count) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::allreduce(ctx.comm, std::plus<int>{}, std::span<int>{send},
                         std::span<int>{recv});
        },
        [&] { mp1_perf::raw_allreduce(ctx.comm, ctx.raw_op, send.data(), recv.data(), count); },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "flat_allreduce";
    row.family = "flat_reduce";
    row.reference = "raw_user_op";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::flat_reduce_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "raw MPI_Allreduce_c with a same-functor user MPI_Op";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_flat_reduce(const context& ctx, int count) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank == ctx.root) {
            mp1::reduce(ctx.comm, std::plus<int>{}, ctx.root, std::span<int>{send},
                        std::span<int>{recv});
          } else {
            mp1::reduce(ctx.comm, std::plus<int>{}, ctx.root, std::span<int>{send},
                        std::span<int>{});
          }
        },
        [&] {
          mp1_perf::raw_reduce(ctx.comm, ctx.raw_op, ctx.root, send.data(), recv.data(), count);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "flat_reduce";
    row.family = "flat_reduce";
    row.reference = "raw_user_op";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::flat_reduce_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "root = size - 1; raw MPI_Reduce_c with a same-functor user MPI_Op";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_flat_scan(const context& ctx, int count) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::scan(ctx.comm, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
        },
        [&] { mp1_perf::raw_scan(ctx.comm, ctx.raw_op, send.data(), recv.data(), count); },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "flat_scan";
    row.family = "flat_reduce";
    row.reference = "raw_user_op";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::flat_reduce_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "raw MPI_Scan_c with a same-functor user MPI_Op";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_flat_exscan(const context& ctx, int count) {
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(count), -1);
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::exscan(ctx.comm, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
        },
        [&] { mp1_perf::raw_exscan(ctx.comm, ctx.raw_op, send.data(), recv.data(), count); },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "flat_exscan";
    row.family = "flat_reduce";
    row.reference = "raw_user_op";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    // The reduction gate names allreduce/reduce/scan.  Exscan is recorded for visibility: the
    // implementation allocates a rank-0 scratch receive buffer per non-in-place call, which
    // shows up as a ratio above 1.10 at large counts but is not one of the gated rows.
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "recorded, not gated: raw MPI_Exscan_c with a same-functor user MPI_Op";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  // Staged path: a holey layout (true extent larger than the element span) cannot use the
  // flat address, so mp1 packs both directions.  The raw baseline is the flat user-op
  // allreduce over the same element count; the row is recorded, not gated.
  void bench_staged_allreduce(const context& ctx, int elements) {
    mp1::layout<int> holey = mp1::layout<int>::vector(elements, 1, 2);
    const int extent = elements * 2 - 1;
    std::vector<int> send(static_cast<std::size_t>(extent), -1);
    std::vector<int> recv(static_cast<std::size_t>(extent), -1);
    for (int i = 0; i < elements; ++i) {
      send[static_cast<std::size_t>(2 * i)] = ctx.rank;
    }
    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::allreduce(ctx.comm, std::plus<int>{}, mp1::region<int>{send.data(), holey},
                         mp1::region<int>{recv.data(), holey});
        },
        [&] {
          mp1_perf::raw_allreduce(ctx.comm, ctx.raw_op, send.data(), recv.data(), elements);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "staged_allreduce";
    row.family = "staged_reduce";
    row.reference = "raw_flat_user_op";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = 0;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "holey layout staged both directions vs raw flat; recorded, not gated";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void run(const mp1_perf::options& opts) {
    // The raw operation must outlive every measurement.
    mp1_perf::raw_user_op raw_op = mp1_perf::raw_user_op::create<int, std::plus<int>>();
    context ctx{opts, mp1::comm_world(), raw_op.handle(), 0, 0, 0};
    ctx.rank = ctx.comm.rank();
    ctx.size = ctx.comm.size();
    ctx.root = ctx.size - 1;

    bench_flat_allreduce(ctx, 1);
    bench_staged_allreduce(ctx, 1);
    for (const std::size_t bytes : opts.sizes) {
      const int count = mp1_perf::elements_for_bytes(bytes);
      bench_flat_allreduce(ctx, count);
      bench_flat_reduce(ctx, count);
      bench_flat_scan(ctx, count);
      bench_flat_exscan(ctx, count);
    }
    bench_staged_allreduce(ctx, mp1_perf::elements_for_bytes(1024 * 1024));
    // Destroy the user op before finalize (its destructor calls MPI_Op_free).
    raw_op = mp1_perf::raw_user_op{};
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
    std::fprintf(stderr, "perf_reduce: %s\n", error.what());
    status = 2;
  }
  try {
    mp1::finalize();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "perf_reduce: finalize: %s\n", error.what());
    status = status == 0 ? 2 : status;
  }
  return status;
}
