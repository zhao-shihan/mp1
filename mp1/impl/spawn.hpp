#if !(defined MP1_IMPL_SPAWN_HPP)

#define MP1_IMPL_SPAWN_HPP

// Dynamic process management, spawn side only (MPI 4.1 §11.8 Process Manager Interface and
// §11.10.4 Releasing Connections).
//
// Release protocol: MPI_Comm_spawn and MPI_Comm_get_parent expose the *same* intercommunicator
// to the two ends (§11.8.2, pp. 521-524).  MPI_Comm_disconnect is the collective release that
// waits for decoupled activities and makes the processes independent; MPI_Comm_free does not
// disconnect (it is explicitly listed as leaving the processes connected, §11.10.4, p. 543).
// Every communicator returned here therefore has release_policy::disconnect: free() throws
// std::logic_error, the destructor calls MPI_Comm_disconnect, and release() is the no-MPI-call
// escape for a non-cooperating peer (for example a raw-MPI child that only finalizes).  Both
// ends must disconnect for the connection to be torn down; mp1 cannot repair a mismatch.
//
// Default spawn is a raw pass-through at the root: mp1 performs only the all-rank checks
// (valid, intracommunicator comm and an in-range root) and passes the root's
// command/argv/maxprocs/info to MPI unchanged.  Non-root ranks ignore their copy of the
// payload and pass the portable empty form (nullptr/MPI_ARGV_NULL/0/MPI_INFO_NULL), as MPI
// advises.  mp1 does NOT validate command/argv/maxprocs/info content at the root and does NOT
// add a preflight agreement broadcast.  The root-only arguments are significant only at the
// root and are ignored elsewhere (§11.8.2, p. 523: "All arguments before the root argument are
// examined only on the process whose rank in comm is equal to root.  The value of these
// arguments on other processes is ignored"; §11.8.3, p. 525 for the spawn_multiple arrays).
// Because MPI_Comm_spawn is collective over comm, a root-only failure is not turned into a
// defined collective error by MPI: a malformed argument can abort the job or leave the
// non-root ranks blocked inside MPI_Comm_spawn.  That is the documented pass-through contract;
// raw MPI is the escape when different behavior is required.
//
// Surface: four entry points over five public types
// (command_line, command_lines, spawn_options, spawn_multiple_options, spawn_result).  The
// spawn_multiple arrays are built through spawn_multiple_options::add(), so commands,
// maxprocs and infos cannot diverge.  The two *_with_errcodes forms select the return shape by
// name; spawn_result.errcodes is the OUT array_of_errcodes (maxprocs entries for spawn,
// sum(maxprocs) entries for spawn_multiple), never an option field.

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/info.hpp"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace mp1 {

  /// One command line; command.front() is the program and the remaining elements are its
  /// arguments (MPI_Comm_spawn's argv array starts at the first argument, not the program;
  /// MPI 4.1 §11.8.2, p. 522).  An empty vector means "no program": MPI allows an empty
  /// command string when the info argument names the executable (§11.8.2, p. 523), and mp1
  /// passes the empty string through unchanged.  The strings are borrowed only for the call.
  using command_line = std::vector<std::string>;

  /// A list of command lines, one per executable for spawn_multiple.
  using command_lines = std::vector<command_line>;

  /// Payload of spawn/spawn_with_errcodes.  Only the root's value is examined by MPI
  /// (§11.8.2, p. 523); every other rank may pass a default-constructed object.  The fields are
  /// declared in designated-initializer order; C++20 requires the designators in that order.
  struct spawn_options {
    int maxprocs = 1;      // number of processes to start; non-negative at the root
    command_line command;  // program + arguments
    // implementation spawn hints (MPI_INFO_NULL when default).  The initializer is redundant
    // for value-initialization, but -Wmissing-field-initializers warns at
    // designated-initializer call sites that omit .info when it is absent.
    // NOLINTNEXTLINE(readability-redundant-member-init): -Wmissing-field-initializers
    mp1::info info{};
  };

  /// Payload of spawn_multiple/spawn_multiple_with_errcodes.  add() appends one
  /// (command, maxprocs, info) triple, so the three arrays MPI requires to have the same
  /// length cannot diverge by construction; size() is MPI's count (§11.8.3, p. 525).  The
  /// root's size() must be positive; a malformed root-only payload is passed through and left
  /// to MPI (an erroneous program), as documented at the top of this header.
  class spawn_multiple_options {
  public:
    /// Append one executable request; the builder keeps the three arrays aligned.
    spawn_multiple_options& add(command_line command, int maxprocs, info i = {}) {
      commands_.push_back(std::move(command));
      maxprocs_.push_back(maxprocs);
      infos_.push_back(std::move(i));
      return *this;
    }

    /// \return MPI's count for this payload (number of executables); no MPI call
    [[nodiscard]] std::size_t size() const noexcept {
      return commands_.size();
    }

    /// \return borrowed commands; one per executable
    [[nodiscard]] std::span<const command_line> commands() const noexcept {
      return commands_;
    }

    /// \return borrowed maxprocs; same length as commands()
    [[nodiscard]] std::span<const int> maxprocs() const noexcept {
      return maxprocs_;
    }

    /// \return borrowed infos; same length as commands()
    [[nodiscard]] std::span<const info> infos() const noexcept {
      return infos_;
    }

  private:
    command_lines commands_;
    std::vector<int> maxprocs_;
    std::vector<info> infos_;
  };

  /// Result of spawn_with_errcodes/spawn_multiple_with_errcodes.  `errcodes` is the
  /// array_of_errcodes of MPI 4.1 §11.8.2/§11.8.3: one entry per requested process (spawn) or
  /// sum(maxprocs) entries (spawn_multiple).  Its length is the root's maxprocs count, which a
  /// non-root rank cannot derive from its ignored local payload; the optional
  /// `errcodes_length` argument of the two calls lets every rank pass the root's length
  /// explicitly (the safe contract).  Without it, every rank sizes the array from its own
  /// maxprocs, so all ranks must pass the same maxprocs — a caller obligation the wrapper
  /// cannot verify without a collective.  A soft spawn is a success with a possibly smaller
  /// remote_size and a mix of MPI_SUCCESS and implementation error codes; it never throws
  /// (§11.8.4).
  struct spawn_result {
    communicator intercomm;
    std::vector<int> errcodes;
  };

  namespace detail {

    /// All-rank spawn precondition.  Every rank evaluates the same checks, so a malformed
    /// comm/root throws on every rank before MPI_Comm_spawn is entered.  Root-only arguments
    /// are deliberately not inspected.
    inline void require_spawn_comm(const mpi_communicator& comm, int root, const context& ctx) {
      require_valid(comm, ctx);
      require_mpi(ctx);
      if (comm.is_inter()) [[unlikely]]
        throw_invalid(ctx.operation, "MPI_Comm_spawn requires an intracommunicator");
      if (root < 0 or root >= comm.size()) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "root {} is outside [0,{})", root, comm.size());
      }
    }

    // MPI_Comm_spawn's argv is a null-terminated array of arguments excluding the program;
    // command.front() is the program. Strings are copied so the pointers are writable without
    // const_cast and stay valid for the duration of the MPI call.  An empty command line maps
    // to the empty program string; mp1 never dereferences command.front() on an empty vector.
    class spawn_argv {
    public:
      explicit spawn_argv(const command_line& command)
          : program_{command.empty() ? std::string{} : command.front()} {
        if (command.size() > 1)
          arguments_.assign(command.begin() + 1, command.end());
      }

      [[nodiscard]] const char* program() const noexcept {
        return program_.c_str();
      }

      [[nodiscard]] char** data() {
        pointers_.clear();
        pointers_.reserve(arguments_.size() + 1);
        for (auto& argument : arguments_)
          pointers_.push_back(argument.data());
        pointers_.push_back(nullptr);
        return pointers_.data();
      }

    private:
      std::string program_;
      std::vector<std::string> arguments_;
      std::vector<char*> pointers_;
    };

    // The four spawn_multiple arrays.  mp1 does not validate the root-only arrays; it only
    // marshals them and keeps every pointer readable for the duration of the call.  Empty
    // command vectors map to the empty program string instead of dereferencing front().
    class spawn_multiple_argv {
    public:
      explicit spawn_multiple_argv(std::span<const command_line> commands) {
        commands_.reserve(commands.size());
        for (const command_line& command : commands) {
          const std::string program = command.empty() ? std::string{} : command.front();
          commands_.emplace_back(program.begin(), program.end());
          commands_.back().push_back('\0');
          arguments_.emplace_back();
          if (command.size() > 1)
            arguments_.back().assign(command.begin() + 1, command.end());
        }
        command_pointers_.reserve(commands_.size());
        for (auto& command : commands_)
          command_pointers_.push_back(command.data());
        argument_pointers_.reserve(arguments_.size());
        for (auto& arguments : arguments_) {
          std::vector<char*> pointers;
          pointers.reserve(arguments.size() + 1);
          for (auto& argument : arguments)
            pointers.push_back(argument.data());
          pointers.push_back(nullptr);
          argument_pointers_.push_back(std::move(pointers));
        }
        argument_tables_.reserve(argument_pointers_.size());
        for (auto& pointers : argument_pointers_)
          argument_tables_.push_back(pointers.data());
      }

      [[nodiscard]] char** commands() {
        return command_pointers_.data();
      }

      [[nodiscard]] char*** arguments() {
        return argument_tables_.data();
      }

    private:
      std::vector<std::vector<char>> commands_;
      std::vector<std::vector<std::string>> arguments_;
      std::vector<char*> command_pointers_;
      std::vector<std::vector<char*>> argument_pointers_;
      std::vector<char**> argument_tables_;
    };

    /// Size the OUT errcodes array from the local payload: maxprocs entries for the single
    /// form, sum(maxprocs) for the multiple form.  Returns false when an entry is negative or
    /// the sum is not representable; the payload is then erroneous at the root and the call
    /// passes MPI_ERRCODES_IGNORE instead of attempting an unbounded allocation (the value is
    /// not locally rejected; §11.8.2/§11.8.3 leave a malformed root-only payload to MPI).
    [[nodiscard]] inline bool errcodes_total(std::span<const int> maxprocs,
                                             std::size_t& total) {
      total = 0;
      for (const int value : maxprocs) {
        if (value < 0)
          return false;
        const auto entries = static_cast<std::size_t>(value);
        if (entries > std::numeric_limits<std::size_t>::max() - total)
          return false;
        total += entries;
      }
      return true;
    }

    /// Allocate the errcodes buffer for one *_with_errcodes call.  `explicit_length` is the
    /// caller-supplied OUT length (the root's maxprocs or sum(maxprocs)); it must be used when
    /// a non-root rank does not repeat the root's payload, because the local maxprocs is
    /// ignored by MPI and cannot size a portable OUT array.  A zero value keeps the historical
    /// local-payload sizing (all ranks must pass the same sizes).  Negative values are public
    /// input and throw before any MPI call.
    inline int* prepare_errcodes(std::span<const int> maxprocs, std::vector<int>& codes,
                                 count_t explicit_length) {
      if (explicit_length < 0) [[unlikely]]
        throw_invalid("mp1::spawn", "negative errcodes length");
      std::size_t total = 0;
      if (explicit_length > 0) {
        total = static_cast<std::size_t>(explicit_length);
      } else if (not errcodes_total(maxprocs, total) or total == 0) {
        return nullptr;
      }
      codes.resize(total);
      return codes.data();
    }

    // One MPI_Comm_spawn.  errcodes is null for the default MPI_ERRCODES_IGNORE form.  The OUT
    // array length is the root's maxprocs (MPI 4.1 §11.8.2); `errcodes_length` lets the caller
    // supply that length on every rank, which is required when a non-root rank keeps its
    // ignored default payload.  Without it the allocation is driven by the local maxprocs and
    // all ranks must pass the same maxprocs (caller obligation).  The returned
    // intercommunicator has disconnect policy.
    [[nodiscard]] inline communicator spawn_native(const mpi_communicator& comm, int root,
                                                   const spawn_options& options,
                                                   std::vector<int>* errcodes,
                                                   count_t errcodes_length) {
      const bool is_root = comm.rank() == root;
      std::vector<int> codes;
      int* codes_pointer = nullptr;
      if (errcodes != nullptr)
        codes_pointer = prepare_errcodes(std::span<const int>{&options.maxprocs, 1}, codes,
                                         errcodes_length);

      MPI_Comm intercomm{MPI_COMM_NULL};
      int rc{MPI_SUCCESS};
      if (is_root) {
        spawn_argv arguments{options.command};
        rc =
            MPI_Comm_spawn(arguments.program(), arguments.data(), options.maxprocs,
                           options.info.native_handle(), root, comm.native_handle(), &intercomm,
                           codes_pointer == nullptr ? MPI_ERRCODES_IGNORE : codes_pointer);
      } else {
        rc = MPI_Comm_spawn(nullptr, MPI_ARGV_NULL, 0, MPI_INFO_NULL, root,
                            comm.native_handle(), &intercomm,
                            codes_pointer == nullptr ? MPI_ERRCODES_IGNORE : codes_pointer);
      }
      check_mpi(rc, "MPI_Comm_spawn");
      if (errcodes != nullptr)
        *errcodes = std::move(codes);
      if (intercomm == MPI_COMM_NULL) [[unlikely]]
        detail::fail_with<std::runtime_error>(
            "mp1::spawn", std::source_location::current(),
            "MPI_Comm_spawn returned MPI_COMM_NULL after MPI_SUCCESS");
      return communicator::adopt_connected(intercomm);
    }

    // One MPI_Comm_spawn_multiple.  The root-only count/arrays are passed through; mp1 pads
    // maxprocs/infos to the command count (0 and MPI_INFO_NULL respectively) only so its own
    // marshalling never reads outside the caller's payload.  That padding is not validation:
    // a malformed root-only set remains an erroneous program whose behavior is MPI's.  As in
    // spawn_native, `errcodes_length` is the portable way to size the OUT array when a
    // non-root rank does not repeat the root's maxprocs payload.
    [[nodiscard]] inline communicator spawn_multiple_native(
        const mpi_communicator& comm, int root, const spawn_multiple_options& options,
        std::vector<int>* errcodes, count_t errcodes_length) {
      const bool is_root = comm.rank() == root;
      std::vector<int> codes;
      int* codes_pointer = nullptr;
      if (errcodes != nullptr)
        codes_pointer = prepare_errcodes(options.maxprocs(), codes, errcodes_length);

      MPI_Comm intercomm{MPI_COMM_NULL};
      int rc{MPI_SUCCESS};
      if (is_root) {
        const int count = checked_int(options.size(), "mp1::spawn_multiple");
        spawn_multiple_argv arguments{options.commands()};

        std::vector<int> maxprocs_call(options.maxprocs().begin(), options.maxprocs().end());
        maxprocs_call.resize(static_cast<std::size_t>(count), 0);
        std::vector<MPI_Info> info_handles(static_cast<std::size_t>(count), MPI_INFO_NULL);
        const std::span<const info> infos = options.infos();
        for (std::size_t index = 0; index < info_handles.size() and index < infos.size();
             ++index)
          info_handles[index] = infos[index].native_handle();

        rc = MPI_Comm_spawn_multiple(
            count, arguments.commands(), arguments.arguments(), maxprocs_call.data(),
            info_handles.data(), root, comm.native_handle(), &intercomm,
            codes_pointer == nullptr ? MPI_ERRCODES_IGNORE : codes_pointer);
      } else {
        // §11.8.3, p. 526 advice: the leading dimension of array_of_argv is count, so a
        // nonpositive ignored count could trip a runtime bounds check; 1 with
        // MPI_ARGVS_NULL is the portable non-root form.
        rc = MPI_Comm_spawn_multiple(
            1, nullptr, MPI_ARGVS_NULL, nullptr, nullptr, root, comm.native_handle(),
            &intercomm, codes_pointer == nullptr ? MPI_ERRCODES_IGNORE : codes_pointer);
      }
      check_mpi(rc, "MPI_Comm_spawn_multiple");
      if (errcodes != nullptr)
        *errcodes = std::move(codes);
      if (intercomm == MPI_COMM_NULL) [[unlikely]]
        detail::fail_with<std::runtime_error>(
            "mp1::spawn_multiple", std::source_location::current(),
            "MPI_Comm_spawn_multiple returned MPI_COMM_NULL after MPI_SUCCESS");
      return communicator::adopt_connected(intercomm);
    }

  }  // namespace detail

  /// Spawn `options.maxprocs` copies of `options.command`; collective over comm, returns an
  /// owning intercommunicator with disconnect policy (both ends must disconnect; §11.10.4).
  ///
  /// Only the root's maxprocs/command/info are significant; the values on other ranks are
  /// ignored and replaced by the portable empty form (nullptr/MPI_ARGV_NULL/0/MPI_INFO_NULL)
  /// (§11.8.2, p. 523).  mp1 validates only the
  /// all-rank properties (valid intracommunicator, in-range positional root) and does not
  /// inspect the root-only payload: a malformed root-only value is MPI's to diagnose, and the
  /// outcome is not necessarily collective (it can abort the job or leave non-root ranks
  /// blocked).  The result of a successful soft spawn is a valid intercommunicator with a
  /// possibly smaller remote_size (§11.8.4).  The default form passes MPI_ERRCODES_IGNORE.
  ///
  /// The returned owner frees nothing: dropping it calls MPI_Comm_disconnect, which is
  /// collective and may block; call disconnect() explicitly for a visible synchronization
  /// point, or release() + raw MPI for a non-cooperating peer.  free() throws std::logic_error.
  [[nodiscard]] inline communicator spawn(const mpi_communicator& comm, int root,
                                          const spawn_options& options) {
    detail::require_spawn_comm(comm, root, "mp1::spawn");
    return detail::spawn_native(comm, root, options, nullptr, 0);
  }

  /// spawn() with array_of_errcodes.  The OUT array has `errcodes_length` entries on every
  /// rank when it is positive; otherwise it has the local options.maxprocs entries, which
  /// requires every rank to pass the same maxprocs (the root sets the length; a non-root
  /// maxprocs is otherwise ignored, §11.8.2 p. 523).  Use the explicit length when a non-root
  /// rank keeps its default payload and the root spawns more than one process.  A soft spawn is
  /// a success with a possibly smaller remote_size and per-process error codes.
  [[nodiscard]] inline spawn_result spawn_with_errcodes(const mpi_communicator& comm, int root,
                                                        const spawn_options& options,
                                                        count_t errcodes_length = 0) {
    detail::require_spawn_comm(comm, root, "mp1::spawn_with_errcodes");
    spawn_result result;
    result.intercomm =
        detail::spawn_native(comm, root, options, &result.errcodes, errcodes_length);
    return result;
  }

  /// Spawn several executables; collective over comm, returns an owning intercommunicator with
  /// disconnect policy.  count and the four arrays are significant only at the root
  /// (§11.8.3, p. 525); non-root ranks pass count 1/MPI_ARGVS_NULL as MPI advises (p. 526).
  ///
  /// mp1 does not validate the root-only payload: a malformed set is an erroneous program and
  /// its behavior is MPI's.  spawn_multiple_options::add() keeps the three arrays aligned by
  /// construction, so there is no raw-array entry point and no length validation to perform.
  [[nodiscard]] inline communicator spawn_multiple(const mpi_communicator& comm, int root,
                                                   const spawn_multiple_options& options) {
    detail::require_spawn_comm(comm, root, "mp1::spawn_multiple");
    return detail::spawn_multiple_native(comm, root, options, nullptr, 0);
  }

  /// spawn_multiple() with array_of_errcodes: `errcodes_length` entries on every rank when it
  /// is positive (the root's sum(maxprocs)); otherwise the local sum(maxprocs), which requires
  /// every rank to repeat the root's maxprocs payload.  Soft spawn is a success (§11.8.3,
  /// pp. 526-527).
  [[nodiscard]] inline spawn_result spawn_multiple_with_errcodes(
      const mpi_communicator& comm, int root, const spawn_multiple_options& options,
      count_t errcodes_length = 0) {
    detail::require_spawn_comm(comm, root, "mp1::spawn_multiple_with_errcodes");
    spawn_result result;
    result.intercomm =
        detail::spawn_multiple_native(comm, root, options, &result.errcodes, errcodes_length);
    return result;
  }

}  // namespace mp1

#endif
