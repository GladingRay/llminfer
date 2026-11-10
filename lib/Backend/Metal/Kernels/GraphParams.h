#ifndef LLMINFER_BACKEND_METAL_GRAPHPARAMS_H
#define LLMINFER_BACKEND_METAL_GRAPHPARAMS_H

struct HalfViewParams {
  unsigned count, offset, dims[3], strides[3];
};

struct GroupedEmbedParams {
  unsigned tokens, hidden, firstRow, rowCount, groups;
};

struct QuantizeParams {
  unsigned rows, width, groups;
};

enum : unsigned {
  linearW8WorkgroupSize = 128,
  linearW8GemvLanes = 16,
  linearW8GemmLanes = 8,
  linearW8GemvColumns = linearW8WorkgroupSize / linearW8GemvLanes,
  linearW8GemmColumns = linearW8WorkgroupSize / linearW8GemmLanes,
  linearW8GemmRows = 4,
};

struct LinearW8Params {
  unsigned outputWidth, firstRow, groups, rows, columns;
};

struct L2Params {
  unsigned rows, width;
  float epsilon, scale;
};

struct PartialRopeParams {
  unsigned rows, heads, width, rotaryWidth, past;
  float theta;
};

enum : unsigned {
  gqaSoftmaxWorkgroupSize = 128,
  flashGqaSimdWidth = 32,
  flashGqaQueryTile = 4,
  flashGqaKeyTile = 16,
  flashGqaVectorWidth = 4,
  flashGqaMaxHeadDim = 256,
  flashGqaThreads = flashGqaSimdWidth * flashGqaQueryTile,
};

struct GraphDeltaParams {
  unsigned tokens, keyHeads, keyDim, valueHeads, valueDim;
};

#endif
