#include "llminfer/Support/CacheLineAlignedVector.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <utility>

using namespace llminfer;

namespace {
bool isAligned(const void *address, std::size_t alignment) {
  return address && reinterpret_cast<std::uintptr_t>(address) % alignment == 0;
}

struct alignas(2 * cacheLineAlignment) OverAligned {
  int value;
};
} // namespace

int main() {
  CacheLineAlignedVector<int> values{1, 2};
  if (!isAligned(values.data(), cacheLineAlignment)) {
    std::fputs("Initial storage is not cache-line aligned\n", stderr);
    return EXIT_FAILURE;
  }
  values.reserve(values.capacity() + 1);
  values.push_back(3);
  if (!isAligned(values.data(), cacheLineAlignment) || values[0] != 1 ||
      values[2] != 3) {
    std::fputs("Growth did not preserve alignment and values\n", stderr);
    return EXIT_FAILURE;
  }

  CacheLineAlignedVector<std::unique_ptr<int>> pointers;
  pointers.push_back(std::make_unique<int>(42));
  CacheLineAlignedVector<std::unique_ptr<int>> moved(std::move(pointers));
  if (!isAligned(moved.data(), cacheLineAlignment) || *moved.front() != 42) {
    std::fputs("Move did not preserve alignment and ownership\n", stderr);
    return EXIT_FAILURE;
  }

  CacheLineAlignedVector<OverAligned> overAligned(2);
  if (!isAligned(overAligned.data(), alignof(OverAligned)) ||
      !isAligned(&overAligned[1], alignof(OverAligned))) {
    std::fputs("Element alignment exceeds the allocator's alignment\n", stderr);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
