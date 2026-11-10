#include "Kernels/GraphParams.h"
#include "Kernels/ReferenceParams.h"
#include "MetalContext.h"
#include "llminfer/ModelFormat/Quantization.h"
#include "llminfer/Support/CacheLineAlignedVector.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

using namespace llminfer;
using namespace llminfer::metal;

namespace {
Error checkFlash(MetalContext &context, GqaParams params) {
  unsigned queryElements = params.tokens * params.queryHeads * params.headDim;
  unsigned cacheElements =
      (params.past + params.tokens) * params.kvHeads * params.headDim;
  Command command(context);
  Buffer query = command.allocate(queryElements, 2);
  Buffer keys = command.allocate(cacheElements, 2);
  Buffer values = command.allocate(cacheElements, 2);
  Buffer output = command.allocate(queryElements, 2);
  bool submitted;
  if (!query || !keys || !values || !output)
    return command.finish(submitted);

  uint32_t state = 179531u;
  auto fill = [&](Buffer buffer, unsigned count) {
    auto *data = static_cast<uint16_t *>(buffer.contents);
    for (unsigned i = 0; i < count; ++i) {
      state = state * 1664525u + 1013904223u;
      data[i] = floatToHalf(float(int(state >> 24) - 128) / 64.0f);
    }
    return data;
  };
  const auto *q = fill(query, queryElements);
  const auto *k = fill(keys, cacheElements);
  const auto *v = fill(values, cacheElements);
  auto *out = static_cast<uint16_t *>(output.contents);
  std::fill_n(out, queryElements, uint16_t{0x7e00});

  const std::array<Buffer, 4> buffers{query, keys, values, output};
  command.encode(
      "gqaFlashAttention", buffers.data(), buffers.size(), &params,
      sizeof(params),
      MTLSizeMake((params.tokens + flashGqaQueryTile - 1) / flashGqaQueryTile,
                  params.queryHeads, 1),
      MTLSizeMake(flashGqaThreads, 1, 1));
  if (auto error = command.finish(submitted))
    return error;

  CacheLineAlignedVector<double> probabilities(params.past + params.tokens);
  unsigned kvStride = params.kvHeads * params.headDim;
  for (unsigned row = 0; row < params.tokens * params.queryHeads; ++row) {
    unsigned visible = params.past + row / params.queryHeads + 1;
    unsigned kvOffset = (row % params.queryHeads) /
                        (params.queryHeads / params.kvHeads) * params.headDim;
    for (unsigned i = 0; i < visible; ++i) {
      double dot = 0.0;
      for (unsigned d = 0; d < params.headDim; ++d)
        dot += double(halfToFloat(q[row * params.headDim + d])) *
               halfToFloat(k[i * kvStride + kvOffset + d]);
      probabilities[i] = dot / std::sqrt(double(params.headDim));
    }
    double maximum = *std::max_element(probabilities.begin(),
                                       probabilities.begin() + visible);
    double denominator = 0.0;
    for (unsigned i = 0; i < visible; ++i) {
      probabilities[i] = std::exp(probabilities[i] - maximum);
      denominator += probabilities[i];
    }
    for (unsigned d = 0; d < params.headDim; ++d) {
      double expected = 0.0;
      for (unsigned i = 0; i < visible; ++i)
        expected +=
            probabilities[i] * halfToFloat(v[i * kvStride + kvOffset + d]);
      expected /= denominator;
      double actual = halfToFloat(out[row * params.headDim + d]);
      if (!(std::abs(actual - expected) <=
            0.0002 + 0.0005 * std::abs(expected)))
        return Error("FlashAttention disagrees with dense causal GQA at row " +
                     std::to_string(row) + ", channel " + std::to_string(d));
    }
  }
  return {};
}
} // namespace

int main() {
  @autoreleasepool {
    if (!MTLCreateSystemDefaultDevice())
      return 77;
    auto context = MetalContext::create();
    if (!context) {
      std::cerr << context.getError().getMessage() << '\n';
      return 1;
    }
    auto pipeline = context->getPipeline("gqaFlashAttention");
    if (!pipeline) {
      std::cerr << pipeline.getError().getMessage() << '\n';
      return 1;
    }
    if ((*pipeline).threadExecutionWidth != flashGqaSimdWidth ||
        (*pipeline).maxTotalThreadsPerThreadgroup < flashGqaThreads)
      return 77;
    for (GqaParams params :
         {GqaParams{19, 4, 2, 12, 0}, GqaParams{7, 4, 2, 256, 33}})
      if (auto error = checkFlash(*context, params)) {
        std::cerr << error.getMessage() << '\n';
        return 1;
      }
    return 0;
  }
}
