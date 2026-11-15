#if !(defined MP1_DETAIL_VECTOR_HPP)

#define MP1_DETAIL_VECTOR_HPP

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>


namespace mp1::detail {

  // Small inline-capacity array used by the completion helpers so a poll of a small request
  // set performs no heap allocation.  It is deliberately restricted to trivially copyable
  // elements (MPI_Request, MPI_Status, small integer masks); growth preserves the size prefix
  // by byte copy.  Elements are default-initialized by `new T[n]`, so a trivial T is not
  // zeroed.
  template<class T, std::size_t Inline>
  class small_vector {
    static_assert(std::is_trivially_copyable_v<T>,
                  "mp1::detail::small_vector requires a trivially copyable element type");
    static_assert(Inline != 0, "mp1::detail::small_vector requires a nonzero inline capacity");

  public:
    small_vector() noexcept = default;
    explicit small_vector(std::size_t n) {
      resize(n);
    }

    small_vector(const small_vector&) = delete;
    small_vector& operator=(const small_vector&) = delete;

    small_vector(small_vector&& other) noexcept {
      take(other);
    }

    small_vector& operator=(small_vector&& other) noexcept {
      if (this != &other) {
        release();
        take(other);
      }
      return *this;
    }

    ~small_vector() {
      release();
    }

    void resize(std::size_t n) {
      if (n <= Inline) {
        if (heap_ != nullptr) {
          if (size_ != 0 and n != 0)
            std::memcpy(inline_.data(), heap_, (n < size_ ? n : size_) * sizeof(T));
          delete[] heap_;
          heap_ = nullptr;
        }
        // Returning to inline storage invalidates the heap capacity. Leaving it stale would
        // let a later resize(m) with Inline < m <= old capacity take the "already large
        // enough" branch while heap_ stays null, so size() would exceed the inline array.
        capacity_ = 0;
        size_ = n;
        return;
      }
      if (n > capacity_) {
        T* next = new T[n];
        if (size_ != 0)
          std::memcpy(next, data(), (n < size_ ? n : size_) * sizeof(T));
        delete[] heap_;
        heap_ = next;
        capacity_ = n;
      }
      size_ = n;
    }

    [[nodiscard]] T* data() noexcept {
      return heap_ != nullptr ? heap_ : inline_.data();
    }

    [[nodiscard]] const T* data() const noexcept {
      return heap_ != nullptr ? heap_ : inline_.data();
    }

    [[nodiscard]] std::size_t size() const noexcept {
      return size_;
    }

    [[nodiscard]] bool empty() const noexcept {
      return size_ == 0;
    }

    T& operator[](std::size_t i) noexcept {
      return data()[i];
    }

    const T& operator[](std::size_t i) const noexcept {
      return data()[i];
    }

  private:
    void release() noexcept {
      delete[] heap_;
      heap_ = nullptr;
      capacity_ = 0;
      size_ = 0;
    }

    void take(small_vector& other) noexcept {
      if (other.heap_ != nullptr) {
        heap_ = other.heap_;
        capacity_ = other.capacity_;
        size_ = other.size_;
      } else {
        capacity_ = 0;
        if (other.size_ != 0)
          std::memcpy(inline_.data(), other.inline_.data(), other.size_ * sizeof(T));
        size_ = other.size_;
      }
      other.heap_ = nullptr;
      other.capacity_ = 0;
      other.size_ = 0;
    }

    std::array<T, Inline> inline_;
    T* heap_{nullptr};
    std::size_t capacity_{0};
    std::size_t size_{0};
  };

}  // namespace mp1::detail

#endif
