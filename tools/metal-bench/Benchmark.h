#ifndef LLMINFER_TOOLS_METALBENCH_BENCHMARK_H
#define LLMINFER_TOOLS_METALBENCH_BENCHMARK_H

#include "MetalContext.h"
#include "llminfer/Support/CacheLineAlignedVector.h"
#include <algorithm>
#include <cmath>

namespace llminfer::metal {

template <typename Params>
Result<double> timeDispatch(MetalContext &context, const char *kernel,
                            std::initializer_list<Buffer> buffers,
                            const Params &params, MTLSize groups,
                            MTLSize threads) {
  @autoreleasepool {
    Command command(context);
    command.encode(kernel, buffers.begin(), buffers.size(), &params,
                   sizeof(params), groups, threads);
    bool submitted;
    if (auto error = command.finish(submitted))
      return error;
    double seconds = command.getGpuSeconds();
    if (!std::isfinite(seconds) || seconds <= 0)
      return Error("Metal did not report a positive GPU execution time");
    return seconds;
  }
}

struct Timing {
  double bestSeconds, medianSeconds;
};

template <typename Launch>
Result<Timing> measureGpu(unsigned samples, Launch launch) {
  CacheLineAlignedVector<double> times(samples);
  for (double &time : times) {
    auto sample = launch();
    if (!sample)
      return sample.getError();
    time = *sample;
  }
  std::sort(times.begin(), times.end());
  double median = (times[(samples - 1) / 2] + times[samples / 2]) / 2;
  return Timing{times.front(), median};
}

} // namespace llminfer::metal

#endif
