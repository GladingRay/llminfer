#ifndef LLMINFER_GRAPHIR_COST_H
#define LLMINFER_GRAPHIR_COST_H

#include "llminfer/GraphIR/Graph.h"

namespace llminfer::ir {

struct OpCost {
  uint64_t int8Macs = 0, fpMacs = 0, scalarOps = 0;
  uint64_t exp = 0, rsqrt = 0, sigmoid = 0, silu = 0, softplus = 0;
  uint64_t sinCosPairs = 0, roundEven = 0;
  uint64_t logicalReadBytes = 0, logicalWriteBytes = 0;
};

struct CostReport {
  OpCost total;
  CacheLineAlignedVector<OpCost> perOp;
};

/// Counts model operations and logical tensor access for this step.
Result<CostReport> estimateCost(const Graph &graph);

} // namespace llminfer::ir

#endif
