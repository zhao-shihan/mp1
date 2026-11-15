#if !(defined MP1_IMPL_INFO_HPP)

#define MP1_IMPL_INFO_HPP

#include "mp1/impl/detail/mpi_call.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


namespace mp1 {

  namespace detail {
    struct info_access;
  }  // namespace detail

  namespace detail {

    /// Conservative key validation shared by every key-taking info operation.  The portable
    /// accepted maximum is MPI_MAX_INFO_KEY - 1 bytes (MPICH accepts one more; Open MPI may
    /// abort).
    inline void info_validate_key(
        std::string_view key, const context& ctx,
        std::source_location where = std::source_location::current()) {
      if (key.empty())
        throw_invalid(ctx.operation, "empty info key", where);
      if (key.find('\0') != std::string_view::npos)
        throw_invalid(ctx.operation, "info key contains an embedded NUL byte", where);
      if (key.size() >= static_cast<std::size_t>(MPI_MAX_INFO_KEY)) {
        detail::fail_as<std::invalid_argument>(
            ctx, where, "info key length {} is not less than the MPI_MAX_INFO_KEY limit ({})",
            key.size(), MPI_MAX_INFO_KEY);
      }
    }

    /// Conservative value validation shared by info::set.  Empty values are legal.
    inline void info_validate_value(
        std::string_view value, const context& ctx,
        std::source_location where = std::source_location::current()) {
      if (value.find('\0') != std::string_view::npos)
        throw_invalid(ctx.operation, "info value contains an embedded NUL byte", where);
      if (value.size() >= static_cast<std::size_t>(MPI_MAX_INFO_VAL)) {
        detail::fail_as<std::invalid_argument>(
            ctx, where, "info value length {} is not less than the MPI_MAX_INFO_VAL limit ({})",
            value.size(), MPI_MAX_INFO_VAL);
      }
    }

  }  // namespace detail

  /// Owning wrapper around an MPI_Info key-value table (MPI 4.1 Chapter 10).
  ///
  /// Availability: every MPI_Info_* routine is in MPI 4.1 §11.4.1 Table 11.1 ("always
  /// available"), so
  /// info (including its destructor) works before MPI_Init and after MPI_Finalize and never
  /// calls detail::require_mpi.  It is the one object class whose destructor may call MPI after
  /// finalize; it frees the handle instead of abandoning it.  MPI also allows concurrent
  /// MPI_Info_* calls from several threads, but the wrapper's lazy handle creation in set() is a
  /// write to this object: build the table with create()/create_env() before sharing it, and do
  /// not mutate one info object from two threads (see "Threads and lifecycle" in the README).
  ///
  /// A default-constructed info is null (MPI_INFO_NULL); info::create()/create_env() return a
  /// valid table that may be empty.  Key/value validation is conservative and local:
  /// key.size() < MPI_MAX_INFO_KEY with a non-empty key and no embedded NUL;
  /// value.size() < MPI_MAX_INFO_VAL with no embedded NUL.
  class info {
  public:
    /// MPI_INFO_NULL; no MPI call
    info() noexcept = default;

    /// Copy: null stays null; a non-null handle is duplicated (always available)
    info(const info& other) : info_{MPI_INFO_NULL} {
      if (other.info_ == MPI_INFO_NULL)
        return;
      detail::check_mpi(MPI_Info_dup(other.info_, &info_), "MPI_Info_dup");
    }

    /// Move: steals the handle and leaves the source null
    info(info&& other) noexcept : info_{other.info_} {
      other.info_ = MPI_INFO_NULL;
    }

    /// Copy assignment: copy-and-swap (strong guarantee)
    info& operator=(const info& other) {
      if (this != &other) {
        info copy{other};
        swap(copy);
      }
      return *this;
    }

    /// Move assignment: releases the target's handle, then steals
    info& operator=(info&& other) noexcept {
      if (this != &other) {
        reset();
        info_ = other.info_;
        other.info_ = MPI_INFO_NULL;
      }
      return *this;
    }

    /// Frees a non-null handle unconditionally: MPI_Info_free is always available, so a
    /// surviving info is freed rather than abandoned.  Errors are ignored (destructors never
    /// throw).
    ~info() noexcept {
      reset();
    }

    /// MPI_Info_create: a valid, initially empty table.  Always available.
    [[nodiscard]] static info create() {
      info result;
      detail::check_mpi(MPI_Info_create(&result.info_), "MPI_Info_create");
      return result;
    }

    /// MPI_Info_create_env(0, nullptr): a valid environment table.  Always available; the table
    /// may be empty when the implementation does not report environment info.
    [[nodiscard]] static info create_env() {
      info result;
      detail::check_mpi(MPI_Info_create_env(0, nullptr, &result.info_), "MPI_Info_create_env");
      return result;
    }

    /// MPI_Info_create_env(argc, argv): forward the launch arguments.  Always available.
    [[nodiscard]] static info create_env(int& argc, char**& argv) {
      info result;
      detail::check_mpi(MPI_Info_create_env(argc, argv, &result.info_), "MPI_Info_create_env");
      return result;
    }

    /// Set a key-value pair, creating the handle on first use.  Always available.
    ///
    /// Validates the key and value locally before creating or touching any handle; an existing
    /// key is overridden without changing size().  If MPI_Info_set fails on a handle created by
    /// this call, the handle is freed and the object stays null (strong guarantee).
    void set(std::string_view key, std::string_view value) {
      detail::info_validate_key(key, "mp1::info::set");
      detail::info_validate_value(value, "mp1::info::set");
      const bool created = info_ == MPI_INFO_NULL;
      if (created)
        detail::check_mpi(MPI_Info_create(&info_), "MPI_Info_create");
      const std::string key_string{key};
      const std::string value_string{value};
      try {
        detail::check_mpi(MPI_Info_set(info_, key_string.c_str(), value_string.c_str()),
                          "MPI_Info_set");
      } catch (...) {
        if (created)
          reset();
        throw;
      }
    }

    /// Remove a key and report whether it existed.  A null info returns false; a missing key
    /// returns false without calling MPI_Info_delete (so the default fatal handler is never
    /// entered for the common "remove if present" use).  Always available.
    bool remove(std::string_view key) {
      detail::info_validate_key(key, "mp1::info::remove");
      if (info_ == MPI_INFO_NULL)
        return false;
      const std::string key_string{key};
      int length{0};
      char dummy{'\0'};
      int flag{0};
      detail::check_mpi(MPI_Info_get_string(info_, key_string.c_str(), &length, &dummy, &flag),
                        "MPI_Info_get_string");
      if (flag == 0)
        return false;
      detail::check_mpi(MPI_Info_delete(info_, key_string.c_str()), "MPI_Info_delete");
      return true;
    }

    /// \return the value for key, or nullopt if absent.  A null info returns nullopt without an
    /// MPI call.  Uses the dynamic two-call MPI_Info_get_string form and distinguishes an absent
    /// key from an engaged empty value.  Always available.
    [[nodiscard]] std::optional<std::string> value(std::string_view key) const {
      detail::info_validate_key(key, "mp1::info::value");
      if (info_ == MPI_INFO_NULL)
        return std::nullopt;
      const std::string key_string{key};
      int length{0};
      char dummy{'\0'};
      int flag{0};
      detail::check_mpi(MPI_Info_get_string(info_, key_string.c_str(), &length, &dummy, &flag),
                        "MPI_Info_get_string");
      if (flag == 0)
        return std::nullopt;
      if (length < 1 or length > MPI_MAX_INFO_VAL + 1) {
        detail::fail_as<std::overflow_error>({"mp1::info::value"},
                                             std::source_location::current(),
                                             "MPI_Info_get_string reported length {}", length);
      }
      std::string buffer(static_cast<std::size_t>(length), '\0');
      int buffer_length = length;
      detail::check_mpi(
          MPI_Info_get_string(info_, key_string.c_str(), &buffer_length, buffer.data(), &flag),
          "MPI_Info_get_string");
      if (flag == 0)
        return std::nullopt;
      if (buffer_length < 1 or buffer_length > length)
        return std::string{};
      buffer.resize(static_cast<std::size_t>(buffer_length - 1));
      return buffer;
    }

    /// \return number of key-value pairs; 0 for a null info without an MPI call
    [[nodiscard]] int size() const {
      if (info_ == MPI_INFO_NULL)
        return 0;
      int number_of_keys{0};
      detail::check_mpi(MPI_Info_get_nkeys(info_, &number_of_keys), "MPI_Info_get_nkeys");
      return number_of_keys;
    }

    /// \return n-th key; throws std::invalid_argument for a null info or out-of-range n.
    /// Key order is implementation-defined and stable until the next mutation.  Always
    /// available.
    [[nodiscard]] std::string key(int n) const {
      if (info_ == MPI_INFO_NULL) {
        detail::fail_as<std::invalid_argument>(
            {"mp1::info::key"}, std::source_location::current(), "n {} on a null info", n);
      }
      if (n < 0) {
        detail::fail_as<std::invalid_argument>(
            {"mp1::info::key"}, std::source_location::current(), "n {} is negative", n);
      }
      const int number_of_keys = size();
      if (n >= number_of_keys) {
        detail::fail_as<std::invalid_argument>({"mp1::info::key"},
                                               std::source_location::current(),
                                               "n {} not in [0,{})", n, number_of_keys);
      }
      char buffer[MPI_MAX_INFO_KEY + 1] = {};
      detail::check_mpi(MPI_Info_get_nthkey(info_, n, buffer), "MPI_Info_get_nthkey");
      return std::string{buffer};
    }

    /// \return true if a native handle is held; no MPI call
    [[nodiscard]] bool is_valid() const noexcept {
      return info_ != MPI_INFO_NULL;
    }

    /// \return true if the table is null or has no entries; no MPI call for a null info
    [[nodiscard]] bool empty() const {
      return size() == 0;
    }

    /// \return non-owning native handle; MPI_INFO_NULL for a default info (does not create)
    [[nodiscard]] MPI_Info native_handle() const noexcept {
      return info_;
    }

  private:
    void swap(info& other) noexcept {
      MPI_Info tmp = info_;
      info_ = other.info_;
      other.info_ = tmp;
    }

    void reset() noexcept {
      if (info_ == MPI_INFO_NULL)
        return;
      (void)MPI_Info_free(&info_);
      info_ = MPI_INFO_NULL;
    }

    MPI_Info info_{MPI_INFO_NULL};

    friend struct detail::info_access;
  };

  namespace detail {

    /// Minimal adoption hook for MPI calls that return a newly allocated MPI_Info. The hook
    /// is private to mp1: it transfers ownership without copying the key-value table.
    struct info_access {
      [[nodiscard]] static info adopt(MPI_Info handle) noexcept {
        info result;
        result.info_ = handle;
        return result;
      }
    };

  }  // namespace detail

  /// List of info objects (kept as an alias so brace initialization works).
  using infos = std::vector<info>;

}  // namespace mp1

#endif
