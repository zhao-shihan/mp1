// Collective microbenchmarks: dense variable families and holey/padded typed paths.
// Ratios are measured against raw MPI in the same process; absolute
// wall times are recorded but never gated (see perf_common.hpp).
//
// Nightly gate: run with `--gate`; a ratio above the strict threshold makes the process exit
// nonzero (raw-baseline noise above 20 % is reported as `skip`, not `fail`).

#include "perf_common.hpp"

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
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
    int root{0};
  };

  // ---------------------------------------------------------------------------------------
  // Dense variable families: the reference is one native typed MPI_*v_c call
  // (native_datatype<int>(), element counts, extent-unit displacements) with every
  // count/displacement array precomputed outside the timed lambda.  The mp1 side uses the dense
  // counts spelling, so both sides pass the same element counts and extent-unit displacements.
  // This is the honest typed-wire comparison; the previous MPI_BYTE same-bytes reference was
  // not.
  // ---------------------------------------------------------------------------------------

  void bench_dense_gatherv(const context& ctx, std::size_t bytes) {
    const int count = mp1_perf::elements_for_bytes(bytes);
    const MPI_Count elements = static_cast<MPI_Count>(count);
    const mp1::count_t count_t = static_cast<mp1::count_t>(count);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(ctx.size), count_t);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), elements);
    std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(ctx.size), 0);
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * count;
      raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i) * elements;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(count), 0);
    std::vector<int> recv(static_cast<std::size_t>(count) * static_cast<std::size_t>(ctx.size),
                          -1);
    for (int i = 0; i < count; ++i) {
      send[static_cast<std::size_t>(i)] = ctx.rank * 100000 + i;
    }

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank == ctx.root) {
            mp1::gatherv(ctx.comm, ctx.root, send, recv, dense);
          } else {
            mp1::gatherv(ctx.comm, ctx.root, send, std::span<int>{}, mp1::blocks{});
          }
        },
        [&] {
          mp1_perf::raw_gatherv_dense(ctx.comm, ctx.root, send.data(), elements, recv.data(),
                                      raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "dense_gatherv";
    row.family = "dense_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::dense_variable_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "per-rank send bytes; raw native typed reference (arrays precomputed)";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_dense_scatterv(const context& ctx, std::size_t bytes) {
    const int count = mp1_perf::elements_for_bytes(bytes);
    const MPI_Count elements = static_cast<MPI_Count>(count);
    const mp1::count_t count_t = static_cast<mp1::count_t>(count);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(ctx.size), count_t);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), elements);
    std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(ctx.size), 0);
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * count;
      raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i) * elements;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(count) * static_cast<std::size_t>(ctx.size),
                          -1);
    for (int i = 0; i < ctx.size; ++i) {
      for (int j = 0; j < count; ++j) {
        send[static_cast<std::size_t>(i) * static_cast<std::size_t>(count) +
             static_cast<std::size_t>(j)] = i * 100000 + j;
      }
    }
    std::vector<int> recv(static_cast<std::size_t>(count), -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank == ctx.root) {
            mp1::scatterv(ctx.comm, ctx.root, send, dense, recv);
          } else {
            mp1::scatterv(ctx.comm, ctx.root, std::span<const int>{}, mp1::blocks{}, recv);
          }
        },
        [&] {
          mp1_perf::raw_scatterv_dense(ctx.comm, ctx.root, send.data(), raw_counts, raw_displs,
                                       recv.data(), elements);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "dense_scatterv";
    row.family = "dense_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::dense_variable_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "per-rank receive bytes; raw native typed reference (arrays precomputed)";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_dense_allgatherv(const context& ctx, std::size_t bytes) {
    const int count = mp1_perf::elements_for_bytes(bytes);
    const MPI_Count elements = static_cast<MPI_Count>(count);
    const mp1::count_t count_t = static_cast<mp1::count_t>(count);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(ctx.size), count_t);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), elements);
    std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(ctx.size), 0);
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * count;
      raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i) * elements;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(count), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(count) * static_cast<std::size_t>(ctx.size),
                          -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] { mp1::allgatherv(ctx.comm, send, recv, dense); },
        [&] {
          mp1_perf::raw_allgatherv_dense(ctx.comm, send.data(), elements, recv.data(),
                                         raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "dense_allgatherv";
    row.family = "dense_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::dense_variable_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "per-rank contribution bytes; raw native typed reference (arrays precomputed)";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_dense_alltoallv(const context& ctx, std::size_t bytes) {
    const int count = mp1_perf::elements_for_bytes(bytes);
    const MPI_Count elements = static_cast<MPI_Count>(count);
    const mp1::count_t count_t = static_cast<mp1::count_t>(count);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(ctx.size), count_t);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), elements);
    std::vector<MPI_Aint> raw_displs(static_cast<std::size_t>(ctx.size), 0);
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * count;
      raw_displs[static_cast<std::size_t>(i)] = static_cast<MPI_Aint>(i) * elements;
    }
    const mp1::blocks dense{counts, displs};
    const std::size_t block =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(ctx.size);
    std::vector<int> send(block, 0);
    std::vector<int> recv(block, -1);
    for (int peer = 0; peer < ctx.size; ++peer) {
      for (int j = 0; j < count; ++j) {
        send[static_cast<std::size_t>(peer) * static_cast<std::size_t>(count) +
             static_cast<std::size_t>(j)] = ctx.rank * 100000 + peer * 100 + j;
      }
    }

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] { mp1::alltoallv(ctx.comm, send, dense, recv, dense); },
        [&] {
          mp1_perf::raw_alltoallv_dense(ctx.comm, send.data(), raw_counts, raw_displs,
                                        recv.data(), raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "dense_alltoallv";
    row.family = "dense_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(count) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold = mp1_perf::dense_variable_ratio;
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "per-peer block bytes; raw native typed reference (arrays precomputed)";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  // ---------------------------------------------------------------------------------------
  // Holey path: the layout has a true extent larger than its true size, so mp1 stages while
  // the reference uses the same derived datatype natively (count 1 per block).  This is the
  // typed-derived comparison; it checks the ratio, not absolute time.
  // ---------------------------------------------------------------------------------------

  struct holey_fixture {
    mp1::layout<int> block;
    int block_elements{0};   // elements per holey block
    int extent_elements{0};  // stride between two block starts, in int elements
    int total_elements{0};   // exact wire elements per block (Type_size / sizeof(int))
  };

  // An 2-block vector layout: a handful of pack/unpack calls per direction, not one per int.
  // The target element count is split evenly; the true extent is about twice the wire size.
  [[nodiscard]] holey_fixture make_holey(int target_elements) {
    constexpr int blocks = 2;
    holey_fixture result;
    result.block_elements = std::max(target_elements / blocks, 1);
    result.block =
        mp1::layout<int>::vector(blocks, result.block_elements, result.block_elements * 2);
    result.total_elements = blocks * result.block_elements;
    result.extent_elements = (blocks - 1) * result.block_elements * 2 + result.block_elements;
    return result;
  }

  void bench_holey_gatherv(const context& ctx, std::size_t bytes) {
    const holey_fixture fixture = make_holey(mp1_perf::elements_for_bytes(bytes));
    const int extent = fixture.extent_elements;
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(ctx.size),
                                               fixture.block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), MPI_Count{1});
    const std::vector<MPI_Aint> raw_displs = mp1_perf::typed_displacements(
        displs, static_cast<std::size_t>(ctx.size), fixture.block.byte_extent());
    std::vector<int> send(static_cast<std::size_t>(extent), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(extent) * static_cast<std::size_t>(ctx.size),
                          -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank == ctx.root) {
            mp1::gatherv(ctx.comm, ctx.root, mp1::region{send.data(), fixture.block}, recv,
                         recv_layouts, displs);
          } else {
            mp1::gatherv(ctx.comm, ctx.root, mp1::region{send.data(), fixture.block},
                         std::span<int>{}, mp1::blocks{});
          }
        },
        [&] {
          mp1_perf::raw_gatherv_typed(ctx.comm, ctx.root, send.data(),
                                      fixture.block.native_handle(), recv.data(),
                                      fixture.block.native_handle(), raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "holey_gatherv";
    row.family = "holey_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(fixture.total_elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold =
        bytes >= 65536 ? mp1_perf::holey_variable_ratio : 0;  // holey/padded gate at 64 KiB
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "2-block holey layout; staged vs derived-datatype native";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_holey_allgatherv(const context& ctx, std::size_t bytes) {
    const holey_fixture fixture = make_holey(mp1_perf::elements_for_bytes(bytes));
    const int extent = fixture.extent_elements;
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(ctx.size),
                                               fixture.block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), MPI_Count{1});
    const std::vector<MPI_Aint> raw_displs = mp1_perf::typed_displacements(
        displs, static_cast<std::size_t>(ctx.size), fixture.block.byte_extent());
    std::vector<int> send(static_cast<std::size_t>(extent), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(extent) * static_cast<std::size_t>(ctx.size),
                          -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          mp1::allgatherv(ctx.comm, mp1::region{send.data(), fixture.block}, recv, recv_layouts,
                          displs);
        },
        [&] {
          mp1_perf::raw_allgatherv_typed(ctx.comm, send.data(), fixture.block.native_handle(),
                                         recv.data(), fixture.block.native_handle(), raw_counts,
                                         raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "holey_allgatherv";
    row.family = "holey_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(fixture.total_elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold =
        bytes >= 65536 ? mp1_perf::holey_variable_ratio : 0;  // holey/padded gate at 64 KiB
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "2-block holey layout; staged vs derived-datatype native";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_holey_alltoallv(const context& ctx, std::size_t bytes) {
    const holey_fixture fixture = make_holey(mp1_perf::elements_for_bytes(bytes));
    const int extent = fixture.extent_elements;
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(ctx.size), fixture.block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), MPI_Count{1});
    const std::vector<MPI_Aint> raw_displs = mp1_perf::typed_displacements(
        displs, static_cast<std::size_t>(ctx.size), fixture.block.byte_extent());
    const std::size_t total =
        static_cast<std::size_t>(extent) * static_cast<std::size_t>(ctx.size);
    std::vector<int> send(total, ctx.rank);
    std::vector<int> recv(total, -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] { mp1::alltoallv(ctx.comm, send, layouts, displs, recv, layouts, displs); },
        [&] {
          mp1_perf::raw_alltoallv_typed(ctx.comm, send.data(), fixture.block.native_handle(),
                                        raw_counts, raw_displs, recv.data(),
                                        fixture.block.native_handle(), raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "holey_alltoallv";
    row.family = "holey_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(fixture.total_elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold =
        bytes >= 65536 ? mp1_perf::holey_variable_ratio : 0;  // holey/padded gate at 64 KiB
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "2-block holey layout; staged vs derived-datatype native";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  // Padded layout: true size = blocks * sizeof(int), extent = 2 * true size.  Same staging
  // rule as the holey vector case but with contiguous payload bytes (padded shape).
  void bench_padded_gatherv(const context& ctx, std::size_t bytes) {
    const int block_elements = mp1_perf::elements_for_bytes(bytes);
    mp1::layout<int> block = mp1::layout<int>::contiguous(block_elements);
    block.byte_resize(0, block_elements * 2 * sizeof(int));
    const int extent = block_elements * 2;
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(ctx.size), block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), MPI_Count{1});
    const std::vector<MPI_Aint> raw_displs = mp1_perf::typed_displacements(
        displs, static_cast<std::size_t>(ctx.size), block.byte_extent());
    std::vector<int> send(static_cast<std::size_t>(block_elements), ctx.rank);
    std::vector<int> recv(static_cast<std::size_t>(extent) * static_cast<std::size_t>(ctx.size),
                          -1);

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] {
          if (ctx.rank == ctx.root) {
            mp1::gatherv(ctx.comm, ctx.root, mp1::region{send.data(), block}, recv,
                         recv_layouts, displs);
          } else {
            mp1::gatherv(ctx.comm, ctx.root, mp1::region{send.data(), block}, std::span<int>{},
                         mp1::blocks{});
          }
        },
        [&] {
          mp1_perf::raw_gatherv_typed(ctx.comm, ctx.root, send.data(), block.native_handle(),
                                      recv.data(), block.native_handle(), raw_counts,
                                      raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "padded_gatherv";
    row.family = "holey_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(block_elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold =
        bytes >= 65536 ? mp1_perf::holey_variable_ratio : 0;  // holey/padded gate at 64 KiB
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "padded outer extent; staged vs derived-datatype native";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void bench_padded_alltoallv(const context& ctx, std::size_t bytes) {
    const int block_elements = mp1_perf::elements_for_bytes(bytes);
    mp1::layout<int> block = mp1::layout<int>::contiguous(block_elements);
    block.byte_resize(0, block_elements * 2 * sizeof(int));
    const int extent = block_elements * 2;
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(ctx.size), block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(ctx.size));
    for (int i = 0; i < ctx.size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::vector<MPI_Count> raw_counts(static_cast<std::size_t>(ctx.size), MPI_Count{1});
    const std::vector<MPI_Aint> raw_displs = mp1_perf::typed_displacements(
        displs, static_cast<std::size_t>(ctx.size), block.byte_extent());
    const std::size_t total =
        static_cast<std::size_t>(extent) * static_cast<std::size_t>(ctx.size);
    std::vector<int> send(total, 0);
    std::vector<int> recv(total, -1);
    for (int peer = 0; peer < ctx.size; ++peer) {
      for (int i = 0; i < block_elements; ++i) {
        send[static_cast<std::size_t>(peer) * static_cast<std::size_t>(extent) +
             static_cast<std::size_t>(i)] = ctx.rank * 100000 + peer * 100 + i;
      }
    }

    std::vector<double> mp1_samples;
    std::vector<double> raw_samples;
    mp1_perf::measure_pair(
        ctx.opts.warmup, ctx.opts.repeats,
        [&] { mp1::alltoallv(ctx.comm, send, layouts, displs, recv, layouts, displs); },
        [&] {
          mp1_perf::raw_alltoallv_typed(ctx.comm, send.data(), block.native_handle(),
                                        raw_counts, raw_displs, recv.data(),
                                        block.native_handle(), raw_counts, raw_displs);
        },
        mp1_samples, raw_samples);

    mp1_perf::case_result row;
    row.suite = ctx.opts.suite;
    row.name = "padded_alltoallv";
    row.family = "holey_variable";
    row.reference = "raw_typed_native";
    row.ranks = ctx.size;
    row.bytes = static_cast<std::size_t>(block_elements) * sizeof(int);
    row.mp1 = mp1_perf::summarize(mp1_samples);
    row.raw = mp1_perf::summarize(raw_samples);
    row.threshold =
        bytes >= 65536 ? mp1_perf::holey_variable_ratio : 0;  // holey/padded gate at 64 KiB
    row.peak_rss_kb = mp1_perf::peak_rss_kb();
    row.note = "padded outer extent; staged vs derived-datatype native";
    emit(ctx.opts, row, ctx.opts.gate);
  }

  void run(const mp1_perf::options& opts) {
    context ctx{opts, mp1::comm_world(), 0, 0, 0};
    ctx.rank = ctx.comm.rank();
    ctx.size = ctx.comm.size();
    ctx.root = ctx.size - 1;

    for (const std::size_t bytes : opts.sizes) {
      bench_dense_gatherv(ctx, bytes);
      bench_dense_scatterv(ctx, bytes);
      bench_dense_allgatherv(ctx, bytes);
      bench_dense_alltoallv(ctx, bytes);
    }
    for (const std::size_t bytes : opts.sizes) {
      if (bytes > 65536) {
        continue;  // holey/padded only at the 64 KiB point (and smaller)
      }
      bench_holey_gatherv(ctx, bytes);
      bench_holey_allgatherv(ctx, bytes);
      bench_holey_alltoallv(ctx, bytes);
      bench_padded_gatherv(ctx, bytes);
      bench_padded_alltoallv(ctx, bytes);
    }
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
    std::fprintf(stderr, "perf_collective: %s\n", error.what());
    status = 2;
  }
  try {
    mp1::finalize();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "perf_collective: finalize: %s\n", error.what());
    status = status == 0 ? 2 : status;
  }
  return status;
}
