#ifndef LLMINFER_TOOLS_METALBENCH_GEMMBENCH_H
#define LLMINFER_TOOLS_METALBENCH_GEMMBENCH_H

#include "llminfer/Support/Result.h"

namespace llminfer::metal {

struct GemmShape {
  unsigned m = 128, n = 9216, k = 2560;
};

Error benchmarkGemm(GemmShape shape, unsigned samples);

} // namespace llminfer::metal

#endif
