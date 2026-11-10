#ifndef LLMINFER_BACKEND_METAL_REFERENCEPARAMS_H
#define LLMINFER_BACKEND_METAL_REFERENCEPARAMS_H

struct EmbedParams {
  unsigned tokens, hidden, firstRow, rowCount;
};

enum LinearEpilogue : unsigned {
  LinearNone,
  LinearSilu,
  LinearSiluMul,
  LinearSigmoidColumns,
  LinearSiluF32,
  LinearDeltaBeta,
  LinearDeltaDecay,
};

struct LinearParams {
  unsigned tokens, inputWidth, outputWidth, firstRow, rowCount;
  unsigned epilogue = LinearNone, epilogueBlock = 0;
};

struct RowParams {
  unsigned rows, width;
  float epsilon;
};

struct ElementParams {
  unsigned count;
};

struct ConvParams {
  unsigned tokens, channels, kernelSize, applySilu;
};

struct DeltaNormParams {
  unsigned tokens, keyHeads, keyDim, qkvWidth;
};

struct DeltaParams {
  unsigned tokens, keyHeads, keyDim, valueHeads, valueDim, qkvWidth;
};

struct RopeParams {
  unsigned tokens, heads, headDim, rotaryDim, headStride, past, outputStart;
  float epsilon, theta;
};

struct CacheParams {
  unsigned count, offset;
};

struct GqaParams {
  unsigned tokens, queryHeads, kvHeads, headDim, past;
};

#endif
