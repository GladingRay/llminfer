#ifndef LLMINFER_SUPPORT_CHECKEDARITHMETIC_H
#define LLMINFER_SUPPORT_CHECKEDARITHMETIC_H

#include <cstdint>

namespace llminfer {

inline bool checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > UINT64_MAX - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

inline bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

inline bool alignTo(uint64_t value, uint64_t alignment, uint64_t &result) {
  return alignment != 0 &&
         checkedAdd(value, (alignment - value % alignment) % alignment, result);
}

} // namespace llminfer

#endif
