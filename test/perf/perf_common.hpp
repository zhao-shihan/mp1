#if !(defined MP1_TEST_PERF_COMMON_HPP)

#define MP1_TEST_PERF_COMMON_HPP

// Shared helpers for the microbenchmarks.  Deliberately dependency free: the only
// non-standard input is MPI and the public mp1 API.  Everything here is header-only so the
// four perf executables stay single-translation-unit programs.
//
// Measurement contract:
//   * one timed call per sample, the barrier is outside the timed span;
//   * every sample is preceded by MPI_Barrier(MPI_COMM_WORLD), so mp1 and raw start together;
//   * median of `--repeats` samples (default 5);
//   * the gate is a ratio against a raw-MPI reference measured in the same process, never an
//     absolute wall time; a sample spread above 20 % on either side marks the row `skip`
//     (flaky).
//
// The thresholds below are the strict performance targets.  They are recorded in every CSV/JSON
// row so a nightly log is self-describing.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mp1_perf {

  // Strict performance ratio thresholds (nightly policy).
  inline constexpr double dense_variable_ratio =
      1.30;  // dense variable vs raw native (1 KiB wrapper overhead)
  inline constexpr double holey_variable_ratio =
      2.50;  // holey/padded vs raw typed-derived (one local pack/unpack)
  inline constexpr double flat_reduce_ratio =
      1.15;  // flat reduction vs raw same-op (1 KiB small-message overhead)
  inline constexpr double p2p_pointer_ratio =
      1.15;  // pointer p2p vs raw native (small-message overhead)
  inline constexpr double spread_skip_pct = 20.0;  // flaky sample-spread threshold (both sides)

  // ---------------------------------------------------------------------------------------
  // Command line.  All options have deterministic defaults so a bare run is meaningful.
  // ---------------------------------------------------------------------------------------

  struct options {
    std::string suite{"perf"};
    std::vector<std::size_t> sizes{1024, 65536, 1048576};
    int repeats{5};
    int warmup{1};
    bool gate{false};
    std::string csv_path;
    std::string json_path;
  };

  [[nodiscard]] inline std::vector<std::size_t> parse_size_list(const std::string& text) {
    std::vector<std::size_t> sizes;
    std::string item;
    std::istringstream stream{text};
    while (std::getline(stream, item, ',')) {
      if (item.empty()) {
        continue;
      }
      std::size_t used{0};
      const unsigned long long value = std::stoull(item, &used);
      if (used != item.size() or value == 0) {
        throw std::invalid_argument{"perf: invalid size '" + item + "'"};
      }
      sizes.push_back(static_cast<std::size_t>(value));
    }
    if (sizes.empty()) {
      throw std::invalid_argument{"perf: empty --sizes list"};
    }
    return sizes;
  }

  [[nodiscard]] inline options parse_options(int argc, char** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
      const std::string arg{argv[i]};
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) {
          throw std::invalid_argument{std::string{"perf: missing value for "} + name};
        }
        return std::string{argv[++i]};
      };
      if (arg == "--suite") {
        result.suite = value("--suite");
      } else if (arg == "--sizes") {
        result.sizes = parse_size_list(value("--sizes"));
      } else if (arg == "--repeats") {
        result.repeats = std::stoi(value("--repeats"));
        if (result.repeats <= 0) {
          throw std::invalid_argument{"perf: --repeats must be positive"};
        }
      } else if (arg == "--warmup") {
        result.warmup = std::stoi(value("--warmup"));
        if (result.warmup < 0) {
          throw std::invalid_argument{"perf: --warmup must be non-negative"};
        }
      } else if (arg == "--csv") {
        result.csv_path = value("--csv");
      } else if (arg == "--json") {
        result.json_path = value("--json");
      } else if (arg == "--gate") {
        result.gate = true;
      } else if (arg == "--help" or arg == "-h") {
        std::printf(
            "usage: %s [--suite NAME] [--sizes BYTES,...] [--repeats N]"
            " [--warmup N] [--csv PATH] [--json PATH] [--gate]\n",
            argv[0]);
        std::exit(0);
      } else {
        throw std::invalid_argument{"perf: unknown option '" + arg + "'"};
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------------------
  // Rank/size and MPI identity.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline int world_rank() {
    return mp1::comm_world().rank();
  }

  [[nodiscard]] inline int world_size() {
    return mp1::comm_world().size();
  }

  [[nodiscard]] inline std::string mpi_library_version() {
    char buffer[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int length{0};
    MPI_Get_library_version(buffer, &length);
    if (length < 0) {
      length = 0;
    }
    if (length > MPI_MAX_LIBRARY_VERSION_STRING) {
      length = MPI_MAX_LIBRARY_VERSION_STRING;
    }
    std::string result{buffer, static_cast<std::size_t>(length)};
    for (char& ch : result) {
      if (ch == '\n' or ch == '\r') {
        ch = ' ';
      }
    }
    return result;
  }

  // ---------------------------------------------------------------------------------------
  // Timing and statistics.
  // ---------------------------------------------------------------------------------------

  struct stats {
    double min_ns{0};
    double median_ns{0};
    double max_ns{0};
    double spread_pct{0};
  };

  [[nodiscard]] inline double median_of(std::vector<double> values) {
    if (values.empty()) {
      return 0;
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    const double high = values[middle];
    if (values.size() % 2 != 0) {
      return high;
    }
    const auto low =
        std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
    return (high + *low) / 2;
  }

  [[nodiscard]] inline stats summarize(const std::vector<double>& samples) {
    stats result;
    if (samples.empty()) {
      return result;
    }
    const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
    result.min_ns = *minimum;
    result.max_ns = *maximum;
    result.median_ns = median_of(samples);
    if (result.median_ns > 0) {
      result.spread_pct = (result.max_ns - result.min_ns) / result.median_ns * 100.0;
    }
    return result;
  }

  // Time one collective/p2p exchange per sample.  The barrier before each sample is outside
  // the timed span; the call itself is what the row reports.  mp1 and raw are interleaved in
  // one loop so long-term CPU frequency drift affects both equally.
  template<class Mp1Call, class RawCall>
  void measure_pair(int warmup, int repeats, Mp1Call&& mp1_call, RawCall&& raw_call,
                    std::vector<double>& mp1_samples, std::vector<double>& raw_samples) {
    using clock_type = std::chrono::steady_clock;
    mp1_samples.clear();
    raw_samples.clear();
    for (int i = 0; i < warmup; ++i) {
      MPI_Barrier(MPI_COMM_WORLD);
      mp1_call();
      MPI_Barrier(MPI_COMM_WORLD);
      raw_call();
    }
    mp1_samples.reserve(static_cast<std::size_t>(repeats));
    raw_samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
      // Alternate which implementation runs after the barrier first: the first collective
      // after a barrier can pay wake-up/frequency costs that would otherwise bias one side.
      const bool mp1_first = i % 2 == 0;
      MPI_Barrier(MPI_COMM_WORLD);
      const auto first_start = clock_type::now();
      if (mp1_first) {
        mp1_call();
      } else {
        raw_call();
      }
      const auto first_stop = clock_type::now();

      MPI_Barrier(MPI_COMM_WORLD);
      const auto second_start = clock_type::now();
      if (mp1_first) {
        raw_call();
      } else {
        mp1_call();
      }
      const auto second_stop = clock_type::now();

      const double first_ns =
          std::chrono::duration<double, std::nano>(first_stop - first_start).count();
      const double second_ns =
          std::chrono::duration<double, std::nano>(second_stop - second_start).count();
      mp1_samples.push_back(mp1_first ? first_ns : second_ns);
      raw_samples.push_back(mp1_first ? second_ns : first_ns);
    }
  }

  // ---------------------------------------------------------------------------------------
  // Rank aggregation and row building.  Every rank computes the same aggregated row, so all
  // ranks agree on the gate decision and the CSV/JSON writer (rank 0) is deterministic.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline double global_max_double(double value) {
    double result{0};
    MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return result;
  }

  [[nodiscard]] inline unsigned long long global_max_ull(unsigned long long value) {
    unsigned long long result{0};
    MPI_Allreduce(&value, &result, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    return result;
  }

  struct case_result {
    std::string suite;
    std::string name;
    std::string family;
    std::string reference;
    std::string note;
    int ranks{0};
    std::size_t bytes{0};
    stats mp1;
    stats raw;
    double ratio{0};
    double threshold{0};
    double raw_spread_pct{0};
    unsigned long long peak_rss_kb{0};
    std::string gate{"info"};
  };

  inline void aggregate_row(case_result& row) {
    if (std::getenv("MP1_PERF_DEBUG") != nullptr) {
      std::fprintf(stderr,
                   "[perf rank %d] %s/%s bytes=%zu mp1(med=%.1f min=%.1f max=%.1f) "
                   "raw(med=%.1f min=%.1f max=%.1f)\n",
                   world_rank(), row.suite.c_str(), row.name.c_str(), row.bytes,
                   row.mp1.median_ns, row.mp1.min_ns, row.mp1.max_ns, row.raw.median_ns,
                   row.raw.min_ns, row.raw.max_ns);
    }
    const double local_ratio =
        row.raw.median_ns > 0 ? row.mp1.median_ns / row.raw.median_ns : 0;
    row.ratio = global_max_double(local_ratio);
    row.mp1.median_ns = global_max_double(row.mp1.median_ns);
    row.raw.median_ns = global_max_double(row.raw.median_ns);
    row.mp1.min_ns = global_max_double(row.mp1.min_ns);
    row.raw.min_ns = global_max_double(row.raw.min_ns);
    row.mp1.max_ns = global_max_double(row.mp1.max_ns);
    row.raw.max_ns = global_max_double(row.raw.max_ns);
    row.mp1.spread_pct = global_max_double(row.mp1.spread_pct);
    row.raw_spread_pct = global_max_double(row.raw.spread_pct);
    row.peak_rss_kb = global_max_ull(row.peak_rss_kb);
  }

  // Returns true when the row fails an enforced gate.  A spread above 20 % on either side is
  // marked `skip`, never `fail`: it means the runner was too noisy to compare.  A threshold of
  // zero (or less) means "record only".
  [[nodiscard]] inline bool finish_row(case_result& row, bool enforce) {
    const double spread = std::max(row.mp1.spread_pct, row.raw_spread_pct);
    if (row.threshold <= 0) {
      row.gate = "info";
    } else if (spread > spread_skip_pct) {
      row.gate = "skip";
    } else {
      row.gate = row.ratio <= row.threshold ? "pass" : "fail";
    }
    return enforce and row.gate == "fail";
  }

  // Peak resident set size of this process in KiB (Linux /proc; 0 elsewhere).  The suite
  // records this for every nightly run; it is never an absolute pass/fail criterion.
  [[nodiscard]] inline unsigned long long peak_rss_kb() {
#if defined(__linux__)
    std::ifstream status{"/proc/self/status"};
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmHWM:", 0) == 0) {
        std::istringstream value{line.substr(6)};
        unsigned long long kb{0};
        value >> kb;
        return kb;
      }
    }
#endif
    return 0;
  }

  // ---------------------------------------------------------------------------------------
  // CSV / JSON output.  CSV is append-only (the rank driver runs several rank counts into one
  // file); the header is written by whoever creates a zero-length file.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline std::string json_escape(const std::string& text) {
    std::string result;
    result.reserve(text.size() + 8);
    for (const char ch : text) {
      switch (ch) {
        case '"':
          result += "\\\"";
          break;
        case '\\':
          result += "\\\\";
          break;
        case '\n':
          result += "\\n";
          break;
        case '\r':
          result += "\\r";
          break;
        case '\t':
          result += "\\t";
          break;
        default:
          result += ch;
      }
    }
    return result;
  }

  [[nodiscard]] inline std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value;
    return stream.str();
  }

  inline void write_json_line(const std::string& path, const std::string& line) {
    if (path.empty()) {
      return;
    }
    std::ofstream out{path, std::ios::app};
    if (not out) {
      throw std::runtime_error{"perf: cannot open JSON output '" + path + "'"};
    }
    out << line << '\n';
  }

  inline void write_environment(const options& opts) {
    if (opts.json_path.empty() or world_rank() != 0) {
      return;
    }
    std::ostringstream line;
    line << "{\"kind\":\"environment\""
         << ",\"suite\":\"" << json_escape(opts.suite) << "\""
         << ",\"ranks\":" << world_size() << ",\"repeats\":" << opts.repeats
         << ",\"mpi_library\":\"" << json_escape(mpi_library_version()) << "\""
         << ",\"thresholds\":{"
         << "\"dense_variable\":" << dense_variable_ratio
         << ",\"holey_variable\":" << holey_variable_ratio
         << ",\"flat_reduce\":" << flat_reduce_ratio << ",\"p2p_pointer\":" << p2p_pointer_ratio
         << ",\"spread_skip_pct\":" << spread_skip_pct << "}}";
    write_json_line(opts.json_path, line.str());
  }

  inline void write_case(const options& opts, const case_result& row) {
    if (world_rank() != 0) {
      return;
    }
    if (not opts.csv_path.empty()) {
      const bool empty = [&] {
        std::ifstream input{opts.csv_path};
        return not input.good() or input.peek() == std::char_traits<char>::eof();
      }();
      std::ofstream out{opts.csv_path, std::ios::app};
      if (not out) {
        throw std::runtime_error{"perf: cannot open CSV output '" + opts.csv_path + "'"};
      }
      if (empty) {
        out << "suite,case,family,reference,ranks,size_bytes,mp1_median_ns,raw_median_ns,"
               "ratio,threshold,mp1_spread_pct,raw_spread_pct,peak_rss_kb,gate,note\n";
      }
      out << row.suite << ',' << row.name << ',' << row.family << ',' << row.reference << ','
          << row.ranks << ',' << row.bytes << ',' << format_double(row.mp1.median_ns) << ','
          << format_double(row.raw.median_ns) << ',' << format_double(row.ratio) << ','
          << format_double(row.threshold) << ',' << format_double(row.mp1.spread_pct) << ','
          << format_double(row.raw_spread_pct) << ',' << row.peak_rss_kb << ',' << row.gate
          << ',' << json_escape(row.note) << '\n';
    }
    if (not opts.json_path.empty()) {
      std::ostringstream line;
      line << "{\"kind\":\"case\""
           << ",\"suite\":\"" << json_escape(row.suite) << "\""
           << ",\"case\":\"" << json_escape(row.name) << "\""
           << ",\"family\":\"" << json_escape(row.family) << "\""
           << ",\"reference\":\"" << json_escape(row.reference) << "\""
           << ",\"ranks\":" << row.ranks << ",\"size_bytes\":" << row.bytes
           << ",\"mp1_median_ns\":" << format_double(row.mp1.median_ns)
           << ",\"raw_median_ns\":" << format_double(row.raw.median_ns)
           << ",\"ratio\":" << format_double(row.ratio)
           << ",\"threshold\":" << format_double(row.threshold)
           << ",\"mp1_spread_pct\":" << format_double(row.mp1.spread_pct)
           << ",\"raw_spread_pct\":" << format_double(row.raw_spread_pct)
           << ",\"peak_rss_kb\":" << row.peak_rss_kb << ",\"gate\":\"" << row.gate << "\""
           << ",\"note\":\"" << json_escape(row.note) << "\"}";
      write_json_line(opts.json_path, line.str());
    }
  }

  // ---------------------------------------------------------------------------------------
  // Dense raw-MPI variable collectives.  Reference: the same native typed call the dense mp1
  // path makes (native_datatype<int>(), element counts, extent-unit displacements) with no
  // staging and no derived datatype.  Every count/displacement array is precomputed by the
  // caller outside the timed lambda, so the measured delta is the wrapper's own conversion and
  // validation, not a byte-count/build-array difference.
  // ---------------------------------------------------------------------------------------

  inline void raw_gatherv_dense(const mp1::mpi_communicator& c, int root, const int* send,
                                MPI_Count sendcount, int* recv,
                                const std::vector<MPI_Count>& recvcounts,
                                const std::vector<MPI_Aint>& recvdispls) {
    MPI_Gatherv_c(send, sendcount, mp1::native_datatype<int>(), recv, recvcounts.data(),
                  recvdispls.data(), mp1::native_datatype<int>(), root, c.native_handle());
  }

  inline void raw_scatterv_dense(const mp1::mpi_communicator& c, int root, const int* send,
                                 const std::vector<MPI_Count>& sendcounts,
                                 const std::vector<MPI_Aint>& senddispls, int* recv,
                                 MPI_Count recvcount) {
    MPI_Scatterv_c(send, sendcounts.data(), senddispls.data(), mp1::native_datatype<int>(),
                   recv, recvcount, mp1::native_datatype<int>(), root, c.native_handle());
  }

  inline void raw_allgatherv_dense(const mp1::mpi_communicator& c, const int* send,
                                   MPI_Count sendcount, int* recv,
                                   const std::vector<MPI_Count>& recvcounts,
                                   const std::vector<MPI_Aint>& recvdispls) {
    MPI_Allgatherv_c(send, sendcount, mp1::native_datatype<int>(), recv, recvcounts.data(),
                     recvdispls.data(), mp1::native_datatype<int>(), c.native_handle());
  }

  inline void raw_alltoallv_dense(const mp1::mpi_communicator& c, const int* send,
                                  const std::vector<MPI_Count>& sendcounts,
                                  const std::vector<MPI_Aint>& senddispls, int* recv,
                                  const std::vector<MPI_Count>& recvcounts,
                                  const std::vector<MPI_Aint>& recvdispls) {
    MPI_Alltoallv_c(send, sendcounts.data(), senddispls.data(), mp1::native_datatype<int>(),
                    recv, recvcounts.data(), recvdispls.data(), mp1::native_datatype<int>(),
                    c.native_handle());
  }

  // ---------------------------------------------------------------------------------------
  // Typed-derived raw-MPI variable collectives (holey/padded reference).  The reference uses
  // the caller's own derived datatype natively, with count 1 per block, so MPI sees exactly
  // MPI_Type_size_c bytes per block and never the padding.  Displacements are converted from
  // public element counts of int to datatype-extent units.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline std::vector<MPI_Aint> typed_displacements(
      const std::vector<mp1::count_t>& displs, std::size_t count, MPI_Count type_extent) {
    if (displs.size() != count) {
      throw std::invalid_argument{"perf: typed displacement array has the wrong length"};
    }
    if (type_extent <= 0) {
      throw std::invalid_argument{"perf: typed reference datatype has a non-positive extent"};
    }
    std::vector<MPI_Aint> result(count);
    for (std::size_t i = 0; i < count; ++i) {
      const long long bytes =
          static_cast<long long>(displs[i]) * static_cast<long long>(sizeof(int));
      if (bytes % static_cast<long long>(type_extent) != 0) {
        throw std::invalid_argument{
            "perf: holed displacement is not a multiple of the datatype extent"};
      }
      result[i] = static_cast<MPI_Aint>(bytes / static_cast<long long>(type_extent));
    }
    return result;
  }

  inline void raw_gatherv_typed(const mp1::mpi_communicator& c, int root, const int* send,
                                MPI_Datatype sendtype, int* recv, MPI_Datatype recvtype,
                                const std::vector<MPI_Count>& recvcounts,
                                const std::vector<MPI_Aint>& recvdispls) {
    MPI_Gatherv_c(send, MPI_Count{1}, sendtype, recv, recvcounts.data(), recvdispls.data(),
                  recvtype, root, c.native_handle());
  }

  inline void raw_scatterv_typed(const mp1::mpi_communicator& c, int root, const int* send,
                                 MPI_Datatype sendtype,
                                 const std::vector<MPI_Count>& sendcounts,
                                 const std::vector<MPI_Aint>& senddispls, int* recv,
                                 MPI_Datatype recvtype) {
    MPI_Scatterv_c(send, sendcounts.data(), senddispls.data(), sendtype, recv, MPI_Count{1},
                   recvtype, root, c.native_handle());
  }

  inline void raw_allgatherv_typed(const mp1::mpi_communicator& c, const int* send,
                                   MPI_Datatype sendtype, int* recv, MPI_Datatype recvtype,
                                   const std::vector<MPI_Count>& recvcounts,
                                   const std::vector<MPI_Aint>& recvdispls) {
    MPI_Allgatherv_c(send, MPI_Count{1}, sendtype, recv, recvcounts.data(), recvdispls.data(),
                     recvtype, c.native_handle());
  }

  inline void raw_alltoallv_typed(const mp1::mpi_communicator& c, const int* send,
                                  MPI_Datatype sendtype,
                                  const std::vector<MPI_Count>& sendcounts,
                                  const std::vector<MPI_Aint>& senddispls, int* recv,
                                  MPI_Datatype recvtype,
                                  const std::vector<MPI_Count>& recvcounts,
                                  const std::vector<MPI_Aint>& recvdispls) {
    MPI_Alltoallv_c(send, sendcounts.data(), senddispls.data(), sendtype, recv,
                    recvcounts.data(), recvdispls.data(), recvtype, c.native_handle());
  }

  // ---------------------------------------------------------------------------------------
  // Raw reductions with the same user-defined operation.  MPI_SUM is deliberately not the
  // reference: mp1 always creates a user MPI_Op, so the raw side creates its own user MPI_Op
  // from the same functor.
  // ---------------------------------------------------------------------------------------

  template<class T, class F>
  void raw_reduction_callback(void* input, void* input_output, int* length, MPI_Datatype*) {
    const T* in = static_cast<const T*>(input);
    T* inout = static_cast<T*>(input_output);
    for (int i = 0; i < *length; ++i) {
      inout[i] = F{}(in[i], inout[i]);
    }
  }

  class raw_user_op {
  public:
    raw_user_op() = default;
    raw_user_op(const raw_user_op&) = delete;
    raw_user_op& operator=(const raw_user_op&) = delete;

    raw_user_op(raw_user_op&& other) noexcept : op_{other.op_} {
      other.op_ = MPI_OP_NULL;
    }

    raw_user_op& operator=(raw_user_op&& other) noexcept {
      if (this != &other) {
        release();
        op_ = other.op_;
        other.op_ = MPI_OP_NULL;
      }
      return *this;
    }

    ~raw_user_op() {
      release();
    }

    template<class T, class F>
    [[nodiscard]] static raw_user_op create() {
      raw_user_op result;
      MPI_Op_create(&raw_reduction_callback<T, F>, 1, &result.op_);
      return result;
    }

    [[nodiscard]] MPI_Op handle() const noexcept {
      return op_;
    }

  private:
    void release() noexcept {
      if (op_ != MPI_OP_NULL) {
        MPI_Op_free(&op_);
        op_ = MPI_OP_NULL;
      }
    }

    MPI_Op op_{MPI_OP_NULL};
  };

  inline void raw_allreduce(const mp1::mpi_communicator& c, MPI_Op op, const int* send,
                            int* recv, int count) {
    MPI_Allreduce_c(send, recv, static_cast<MPI_Count>(count), MPI_INT, op, c.native_handle());
  }

  inline void raw_reduce(const mp1::mpi_communicator& c, MPI_Op op, int root, const int* send,
                         int* recv, int count) {
    MPI_Reduce_c(send, recv, static_cast<MPI_Count>(count), MPI_INT, op, root,
                 c.native_handle());
  }

  inline void raw_scan(const mp1::mpi_communicator& c, MPI_Op op, const int* send, int* recv,
                       int count) {
    MPI_Scan_c(send, recv, static_cast<MPI_Count>(count), MPI_INT, op, c.native_handle());
  }

  inline void raw_exscan(const mp1::mpi_communicator& c, MPI_Op op, const int* send, int* recv,
                         int count) {
    MPI_Exscan_c(send, recv, static_cast<MPI_Count>(count), MPI_INT, op, c.native_handle());
  }

  // ---------------------------------------------------------------------------------------
  // Raw p2p and request completion.
  // ---------------------------------------------------------------------------------------

  inline void raw_send(const mp1::mpi_communicator& c, const int* data, int count, int dest,
                       int tag) {
    MPI_Send_c(data, static_cast<MPI_Count>(count), MPI_INT, dest, tag, c.native_handle());
  }

  inline void raw_recv(const mp1::mpi_communicator& c, int* data, int count, int source,
                       int tag) {
    MPI_Recv_c(data, static_cast<MPI_Count>(count), MPI_INT, source, tag, c.native_handle(),
               MPI_STATUS_IGNORE);
  }

  [[nodiscard]] inline MPI_Request raw_isend(const mp1::mpi_communicator& c, const int* data,
                                             int count, int dest, int tag) {
    MPI_Request handle{MPI_REQUEST_NULL};
    MPI_Isend_c(data, static_cast<MPI_Count>(count), MPI_INT, dest, tag, c.native_handle(),
                &handle);
    return handle;
  }

  [[nodiscard]] inline MPI_Request raw_irecv(const mp1::mpi_communicator& c, int* data,
                                             int count, int source, int tag) {
    MPI_Request handle{MPI_REQUEST_NULL};
    MPI_Irecv_c(data, static_cast<MPI_Count>(count), MPI_INT, source, tag, c.native_handle(),
                &handle);
    return handle;
  }

  inline void raw_wait(MPI_Request& handle) {
    MPI_Wait(&handle, MPI_STATUS_IGNORE);
  }

  inline void raw_waitall(std::vector<MPI_Request>& handles) {
    if (handles.empty()) {
      return;
    }
    // Request the per-request statuses: mp1's wait_all returns them, so comparing against
    // MPI_STATUSES_IGNORE would understate the raw reference work.
    std::vector<MPI_Status> statuses(handles.size());
    MPI_Waitall(static_cast<int>(handles.size()), handles.data(), statuses.data());
  }

  // ---------------------------------------------------------------------------------------
  // Small helpers shared by the three suite mains.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline int elements_for_bytes(std::size_t bytes) {
    const std::size_t count = std::max<std::size_t>(bytes / sizeof(int), 1);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument{"perf: requested size does not fit an int element count"};
    }
    return static_cast<int>(count);
  }

  [[nodiscard]] inline std::string make_note(const std::string& text) {
    return text;
  }

}  // namespace mp1_perf

#endif
