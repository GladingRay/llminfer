#include "GemmBench.h"
#include "Benchmark.h"
#include "Kernels/GraphParams.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <array>
#include <iomanip>
#include <iostream>

using namespace llminfer;
using namespace llminfer::metal;

namespace {
struct GemmBuffers {
  Buffer activation, activationScales, weight, weightScales, output;
};

Result<GemmBuffers> allocateBuffers(MetalContext &context, GemmShape shape,
                                    unsigned groups, unsigned paddedK) {
  std::array<uint64_t, 5> sizes{uint64_t(shape.m) * paddedK,
                                uint64_t(shape.m) * groups * sizeof(float),
                                uint64_t(shape.n) * paddedK,
                                uint64_t(shape.n) * groups * sizeof(uint16_t),
                                uint64_t(shape.m) * shape.n * sizeof(uint16_t)};
  std::array<Buffer, 5> buffers{};
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto buffer = context.makeBuffer(sizes[i]);
    if (!buffer)
      return buffer.getError();
    buffers[i] = *buffer;
  }
  return GemmBuffers{buffers[0], buffers[1], buffers[2], buffers[3],
                     buffers[4]};
}

void fillRows(Buffer buffer, unsigned rows, unsigned width, unsigned stride,
              uint32_t seed) {
  auto *data = static_cast<int8_t *>(buffer.contents);
  for (unsigned row = 0; row < rows; ++row) {
    for (unsigned k = 0; k < width; ++k) {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      data[k] = static_cast<int8_t>(int(seed % 15) - 7);
    }
    std::fill(data + width, data + stride, int8_t{0});
    data += stride;
  }
}

Error verify(GemmShape shape, unsigned groups, const GemmBuffers &buffers) {
  const auto *a = static_cast<const int8_t *>(buffers.activation.contents);
  const auto *w = static_cast<const int8_t *>(buffers.weight.contents);
  const auto *as =
      static_cast<const float *>(buffers.activationScales.contents);
  const auto *ws = static_cast<const uint16_t *>(buffers.weightScales.contents);
  const auto *out = static_cast<const uint16_t *>(buffers.output.contents);
  auto check = [&](uint64_t index) -> Error {
    uint64_t aGroup = (index / shape.n) * groups;
    uint64_t wGroup = (index % shape.n) * groups;
    float expected = 0;
    for (unsigned g = 0; g < groups; ++g) {
      unsigned length =
          std::min<uint64_t>(weightGroupSize, shape.k - g * weightGroupSize);
      int dot = 0;
      for (unsigned j = 0; j < length; ++j)
        dot += int(a[(aGroup + g) * weightGroupSize + j]) *
               int(w[(wGroup + g) * weightGroupSize + j]);
      expected += float(dot) * as[aGroup + g] * halfToFloat(ws[wGroup + g]);
    }
    if (out[index] != floatToHalf(expected))
      return Error("GEMM CPU verification failed at row " +
                   std::to_string(index / shape.n) + ", column " +
                   std::to_string(index % shape.n));
    return {};
  };

  uint64_t count = uint64_t(shape.m) * shape.n;
  if (count <= 64) {
    for (uint64_t i = 0; i < count; ++i)
      if (auto error = check(i))
        return error;
    return {};
  }
  for (uint64_t index : {uint64_t{0}, uint64_t(shape.n - 1),
                         uint64_t(shape.m / 2) * shape.n + shape.n / 2,
                         count - shape.n, count - 1})
    if (auto error = check(index))
      return error;
  return {};
}
} // namespace

Error llminfer::metal::benchmarkGemm(GemmShape shape, unsigned samples) {
  unsigned groups = (shape.k + weightGroupSize - 1) / weightGroupSize;
  unsigned paddedK = groups * weightGroupSize;
  uint64_t outputElements = uint64_t(shape.m) * shape.n;
  if (outputElements > UINT32_MAX || uint64_t(shape.m) * paddedK > UINT32_MAX ||
      uint64_t(shape.n) * paddedK > UINT32_MAX)
    return Error("GEMM shape exceeds the kernel's 32-bit index range");
  if (outputElements * paddedK > (uint64_t{1} << 34))
    return Error("GEMM workload exceeds 2^34 padded MACs per dispatch");

  auto context = MetalContext::create();
  if (!context)
    return context.getError();
  auto pipeline = context->getPipeline("linearW8A8");
  if (!pipeline)
    return pipeline.getError();
  auto buffers = allocateBuffers(*context, shape, groups, paddedK);
  if (!buffers)
    return buffers.getError();

  // Small values and power-of-two scales keep the FP32 reference exact even
  // when Metal sums the groups in a different order.
  fillRows(buffers->activation, shape.m, shape.k, paddedK, 0x7135a109u);
  fillRows(buffers->weight, shape.n, shape.k, paddedK, 0x05df26b7u);
  auto *as = static_cast<float *>(buffers->activationScales.contents);
  auto *ws = static_cast<uint16_t *>(buffers->weightScales.contents);
  for (uint64_t i = 0; i < uint64_t(shape.m) * groups; ++i)
    as[i] = (i % 2) ? 0x1p-5f : 0x1p-6f;
  for (uint64_t i = 0; i < uint64_t(shape.n) * groups; ++i)
    ws[i] = floatToHalf((i % 3) ? 0x1p-6f : 0x1p-5f);
  std::fill_n(static_cast<uint16_t *>(buffers->output.contents), outputElements,
              uint16_t{0x7e00});

  LinearW8Params params{shape.n, 0, groups};
  auto launch = [&]() {
    return timeDispatch(*context, "linearW8A8",
                        {buffers->activation, buffers->activationScales,
                         buffers->weight, buffers->weightScales,
                         buffers->output},
                        params, MTLSizeMake(shape.n, shape.m, 1),
                        MTLSizeMake(linearW8WorkgroupSize, 1, 1));
  };
  auto warmup = launch();
  if (!warmup)
    return warmup.getError();
  if (auto error = verify(shape, groups, *buffers))
    return error;
  auto timing = measureGpu(samples, launch);
  if (!timing)
    return timing.getError();

  uint64_t macs = outputElements * shape.k;
  double gigaOps = 2.0 * macs / 1e9;
  std::cout << "GPU: " << (*pipeline).device.name.UTF8String << '\n'
            << "Kernel: linearW8A8 (Group32, FP16 output)\n"
            << "M: " << shape.m << ", N: " << shape.n << ", K: " << shape.k
            << " (padded: " << paddedK << "), samples: " << samples << '\n'
            << "Logical MACs per dispatch: " << macs << " (1 MAC = 2 ops)\n"
            << std::fixed << std::setprecision(2)
            << "Best: " << gigaOps / timing->bestSeconds << " GOPS ("
            << timing->bestSeconds * 1e3 << " ms)\n"
            << "Median: " << gigaOps / timing->medianSeconds << " GOPS ("
            << timing->medianSeconds * 1e3 << " ms)\n";
  return {};
}
