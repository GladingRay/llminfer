#include "llminfer/ModelFormat/Quantization.h"
#include "File.h"
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace llminfer;

namespace {
uint32_t roundShiftEven(uint32_t value, unsigned shift) {
  uint32_t rounded = value >> shift;
  uint32_t remainder = value & ((1u << shift) - 1);
  uint32_t midpoint = 1u << (shift - 1);
  return rounded +
         (remainder > midpoint || (remainder == midpoint && (rounded & 1)));
}

int roundEven(float value) {
  float lower = std::floor(value);
  float fraction = value - lower;
  int rounded = static_cast<int>(lower);
  return rounded + (fraction > 0.5f || (fraction == 0.5f && rounded % 2 != 0));
}
} // namespace

uint16_t llminfer::floatToHalf(float value) {
  uint32_t bits;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  uint16_t sign = (bits >> 16) & 0x8000;
  int exponent = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent < -25)
    return sign;
  if (exponent < -14)
    return sign | roundShiftEven(mantissa | 0x800000, -exponent - 1);
  if (exponent > 15)
    return sign | 0x7c00 | (exponent == 128 && mantissa != 0 ? 0x200 : 0);
  return sign | (((exponent + 15) << 10) + roundShiftEven(mantissa, 13));
}

float llminfer::halfToFloat(uint16_t bits) {
  unsigned exponent = (bits >> 10) & 0x1f;
  unsigned mantissa = bits & 0x3ff;
  uint32_t result;
  if (exponent == 0) {
    float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    return (bits & 0x8000) ? -magnitude : magnitude;
  }
  unsigned exponent32 = exponent == 31 ? 255 : exponent + 112;
  result =
      (uint32_t(bits & 0x8000) << 16) | (exponent32 << 23) | (mantissa << 13);
  float value;
  std::memcpy(&value, &result, sizeof(value));
  return value;
}

uint16_t llminfer::quantizeWeightGroup(const float *values, unsigned count,
                                       int8_t *quantized) {
  if (count == 0 || count > weightGroupSize)
    fail("Invalid quantization group length");
  float maxAbs = 0.0f;
  for (unsigned i = 0; i < count; ++i) {
    if (!std::isfinite(values[i]))
      fail("Non-finite weight");
    maxAbs = std::max(maxAbs, std::abs(values[i]));
  }

  std::fill(quantized + count, quantized + weightGroupSize, int8_t{0});
  if (maxAbs == 0.0f) {
    std::fill_n(quantized, count, int8_t{0});
    return floatToHalf(1.0f);
  }

  uint16_t storedScale = floatToHalf(std::max(maxAbs / 127.0f, 0x1p-14f));
  float scale = halfToFloat(storedScale);
  if (!std::isfinite(scale))
    fail("Weight scale exceeds FP16 range");
  for (unsigned i = 0; i < count; ++i)
    quantized[i] = roundEven(std::clamp(values[i] / scale, -127.0f, 127.0f));
  return storedScale;
}
