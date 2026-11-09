#ifndef LLMINFER_SUPPORT_CACHELINEALIGNEDVECTOR_H
#define LLMINFER_SUPPORT_CACHELINEALIGNEDVECTOR_H

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>

namespace llminfer {

/// Alignment for host allocations on Apple Silicon.
inline constexpr std::size_t cacheLineAlignment = 128;

template <typename T> class CacheLineAlignedAllocator {
public:
  using value_type = T;
  using is_always_equal = std::true_type;

  CacheLineAlignedAllocator() noexcept = default;
  template <typename U>
  CacheLineAlignedAllocator(const CacheLineAlignedAllocator<U> &) noexcept {}

  [[nodiscard]] T *allocate(std::size_t count) {
    // A vector allocator must return valid storage. Fail fast in the
    // exception-free build if the size overflows or memory is exhausted.
    if (count > max_size())
      std::abort();
    void *memory = ::operator new(count * sizeof(T), alignment(), std::nothrow);
    if (!memory)
      std::abort();
    return static_cast<T *>(memory);
  }

  void deallocate(T *memory, std::size_t) noexcept {
    ::operator delete(memory, alignment());
  }

  constexpr std::size_t max_size() const noexcept {
    return std::numeric_limits<std::size_t>::max() / sizeof(T);
  }

private:
  static constexpr std::align_val_t alignment() {
    return std::align_val_t{
        alignof(T) > cacheLineAlignment ? alignof(T) : cacheLineAlignment};
  }
};

template <typename T, typename U>
constexpr bool operator==(const CacheLineAlignedAllocator<T> &,
                          const CacheLineAlignedAllocator<U> &) noexcept {
  return true;
}

template <typename T, typename U>
constexpr bool operator!=(const CacheLineAlignedAllocator<T> &lhs,
                          const CacheLineAlignedAllocator<U> &rhs) noexcept {
  return !(lhs == rhs);
}

/// Aligns the allocation while retaining contiguous, unpadded elements.
template <typename T>
using CacheLineAlignedVector = std::vector<T, CacheLineAlignedAllocator<T>>;

} // namespace llminfer

#endif
