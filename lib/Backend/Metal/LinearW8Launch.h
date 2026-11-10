#ifndef LLMINFER_BACKEND_METAL_LINEARW8LAUNCH_H
#define LLMINFER_BACKEND_METAL_LINEARW8LAUNCH_H

#include "Kernels/GraphParams.h"
#include <cstdint>

namespace llminfer::metal {

struct LinearW8Launch {
  const char *kernel;
  uint64_t groupsX, groupsY;
};

inline LinearW8Launch getLinearW8Launch(uint64_t rows, uint64_t columns) {
  if (rows == 1)
    return {"linearW8A8Gemv",
            (columns + linearW8GemvColumns - 1) / linearW8GemvColumns, 1};
  return {"linearW8A8Gemm",
          (columns + linearW8GemmColumns - 1) / linearW8GemmColumns,
          (rows + linearW8GemmRows - 1) / linearW8GemmRows};
}

} // namespace llminfer::metal

#endif
