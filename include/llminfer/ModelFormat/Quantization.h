#ifndef LLMINFER_MODELFORMAT_QUANTIZATION_H
#define LLMINFER_MODELFORMAT_QUANTIZATION_H

#include "llminfer/Support/Result.h"
#include <cstdint>

namespace llminfer {

constexpr uint64_t weightGroupSize = 32;

uint16_t floatToHalf(float value);
float halfToFloat(uint16_t bits);

/// Quantizes one group, zero-padding the tail. Returns the stored FP16 scale.
Result<uint16_t> quantizeWeightGroup(const float *values, unsigned count,
                                     int8_t *quantized);

} // namespace llminfer

#endif
