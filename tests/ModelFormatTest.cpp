#include "llminfer/ModelFormat/ModelFile.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace llminfer;

namespace {
void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void testQuantization() {
  std::array<int8_t, weightGroupSize> output;
  const float ties[] = {127.0f, -127.0f, 0.5f, 1.5f, 2.5f, -1.5f, -2.5f};
  const std::array<int8_t, 7> expected = {127, -127, 0, 2, 2, -2, -2};
  auto scale = quantizeWeightGroup(ties, expected.size(), output.data());
  check(scale && *scale == 0x3c00 &&
            std::equal(expected.begin(), expected.end(), output.begin()),
        "Unexpected scale or ties-to-even rounding");
  check(std::all_of(output.begin() + expected.size(), output.end(),
                    [](int8_t value) { return value == 0; }),
        "Tail must be zero padded");

  std::array<float, weightGroupSize> zeros{};
  scale = quantizeWeightGroup(zeros.data(), zeros.size(), output.data());
  check(scale && *scale == 0x3c00 &&
            std::all_of(output.begin(), output.end(),
                        [](int8_t q) { return q == 0; }),
        "Invalid all-zero group");
  const float tiny[] = {0x1p-14f};
  scale = quantizeWeightGroup(tiny, 1, output.data());
  check(scale && *scale == 0x0400 && output[0] == 1, "Invalid minimum scale");
  const float invalid[] = {std::numeric_limits<float>::infinity()};
  check(!quantizeWeightGroup(invalid, 1, output.data()),
        "Non-finite weights must fail");
  const float tooLarge[] = {std::numeric_limits<float>::max()};
  check(!quantizeWeightGroup(tooLarge, 1, output.data()),
        "Overflowing scales must fail");

  const float roundedScale[] = {1.0f, 0.019684f};
  scale = quantizeWeightGroup(roundedScale, 2, output.data());
  check(scale && *scale == floatToHalf(1.0f / 127) && output[1] == 3,
        "Weights must use the stored scale");
}

void testHalf() {
  for (uint32_t bits = 0; bits <= UINT16_MAX; ++bits) {
    if ((bits & 0x7c00) != 0x7c00)
      check(floatToHalf(halfToFloat(bits)) == bits, "FP16 roundtrip failed");
  }
  check(floatToHalf(0x1p-25f) == 0 && floatToHalf(0x1.8p-24f) == 2 &&
            floatToHalf(1.0f + 0x1p-11f) == 0x3c00 &&
            floatToHalf(65520.0f) == 0x7c00,
        "FP16 conversion did not round ties to even");
}

void testRowSlicing() {
  TensorRecord tensor{{"embedding", {5, 33}, DType::I8},
                      Layout::RowMajorI8Group32,
                      {5, 64},
                      {1024, 320},
                      {2048, 20}};
  auto slices = splitRows(tensor, 129);
  check(slices && slices->size() == 3 && (*slices)[0].rowCount == 2 &&
            (*slices)[1].firstRow == 2 && (*slices)[1].data.offset == 1152 &&
            (*slices)[1].data.size == 128 &&
            (*slices)[1].scales.offset == 2056 &&
            (*slices)[1].scales.size == 8 && (*slices)[2].firstRow == 4 &&
            (*slices)[2].rowCount == 1 && (*slices)[2].data.size == 64 &&
            (*slices)[2].scales.size == 4,
        "Incorrect data or scale slice boundary");
  auto exact = splitRows(tensor, 320);
  check(exact && exact->size() == 1, "Exact buffer fit failed");
  check(!splitRows(tensor, 63), "A single row must fit the buffer limit");

  TensorRecord dense{
      {"embedding", {2, 3}, DType::F16}, Layout::Dense, {2, 3}, {64, 12}, {}};
  slices = splitRows(dense, 6);
  check(slices && slices->size() == 2 && (*slices)[1].data.offset == 70 &&
            (*slices)[1].scales.size == 0,
        "Invalid dense row slice");
}
} // namespace

int main() {
  testHalf();
  testQuantization();
  testRowSlicing();
  return EXIT_SUCCESS;
}
