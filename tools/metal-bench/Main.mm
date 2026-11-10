#include "Benchmark.h"
#include "GemmBench.h"
#include "Kernels/MicrobenchParams.h"
#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string_view>

using namespace llminfer;
using namespace llminfer::metal;

namespace {
constexpr unsigned macsPerIteration = int8BenchChains * 4;

struct Options {
  unsigned threads = 32768;
  unsigned iterations = 1024;
  unsigned samples = 7;
  std::optional<GemmShape> gemm;
};

Result<Options> parseOptions(int argc, char **argv) {
  Options options;
  int first = 1;
  if (argc > 1 && std::string_view(argv[1]) == "gemm") {
    options.gemm.emplace();
    ++first;
  }
  for (int i = first; i < argc; ++i) {
    std::string_view flag = argv[i];
    unsigned *target;
    unsigned limit;
    if (flag == "--samples") {
      target = &options.samples;
      limit = 101;
    } else if (options.gemm &&
               (flag == "--m" || flag == "--n" || flag == "--k")) {
      if (flag == "--m")
        target = &options.gemm->m;
      else if (flag == "--n")
        target = &options.gemm->n;
      else
        target = &options.gemm->k;
      limit = 1 << 16;
    } else if (!options.gemm && flag == "--threads") {
      target = &options.threads;
      limit = 1 << 20;
    } else if (!options.gemm && flag == "--iterations") {
      target = &options.iterations;
      limit = 1 << 16;
    } else {
      return Error("Unknown option: " + std::string(flag));
    }
    if (++i == argc)
      return Error("Missing value for " + std::string(flag));
    std::string_view value = argv[i];
    auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), *target);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        *target == 0 || *target > limit)
      return Error(std::string(flag) + " must be in [1, " +
                   std::to_string(limit) + "]");
  }
  if (!options.gemm &&
      uint64_t(options.threads) * options.iterations * macsPerIteration >
          (uint64_t{1} << 34))
    return Error("Benchmark workload exceeds 2^34 MACs per dispatch");
  return options;
}

int signedByte(uint32_t bits, unsigned offset) {
  int value = (bits >> offset) & 0xff;
  return value < 128 ? value : value - 256;
}

Error verify(const Int8BenchParams &params, const uint32_t *output) {
  for (unsigned threadId : {0u, params.threads / 2, params.threads - 1}) {
    for (unsigned chain = 0; chain < int8BenchChains; ++chain) {
      uint32_t expected = int8BenchSeed(threadId, chain);
      uint32_t weight = params.weights[chain];
      for (unsigned i = 0; i < params.iterations; ++i) {
        int dot = 0;
        for (unsigned offset = 0; offset < 32; offset += 8)
          dot += signedByte(expected, offset) * signedByte(weight, offset);
        expected += static_cast<uint32_t>(dot);
      }
      if (output[threadId * int8BenchChains + chain] != expected)
        return Error("CPU verification failed for thread " +
                     std::to_string(threadId) + ", chain " +
                     std::to_string(chain));
    }
  }
  return {};
}

Error benchmarkMac(const Options &options) {
  auto context = MetalContext::create();
  if (!context)
    return context.getError();
  auto pipeline = context->getPipeline("benchInt8Mac");
  if (!pipeline)
    return pipeline.getError();
  unsigned simdWidth = (*pipeline).threadExecutionWidth;
  unsigned groupLimit = (*pipeline).maxTotalThreadsPerThreadgroup;
  if (!simdWidth || simdWidth > groupLimit)
    return Error("Metal reported an invalid SIMD width");

  Int8BenchParams params{options.threads,
                         options.iterations,
                         {0x3d917b81u, 0xf3ac077fu, 0x7ebd29c5u, 0x9b43a521u,
                          0x158d6be7u, 0xb149639fu, 0x37c7810du, 0x51e9916du}};
  auto output = context->makeBuffer(uint64_t(params.threads) * int8BenchChains *
                                    sizeof(uint32_t));
  if (!output)
    return output.getError();
  uint64_t macs =
      uint64_t(params.threads) * params.iterations * macsPerIteration;
  double gigaOps = 2.0 * macs / 1e9;
  std::cout << "GPU: " << (*pipeline).device.name.UTF8String << '\n'
            << "INT8 x INT8 -> INT32, 1 MAC = 2 ops\n"
            << "Threads: " << options.threads
            << ", iterations: " << options.iterations
            << ", samples per group: " << options.samples
            << ", MACs per dispatch: " << macs << '\n';

  double peak = 0;
  unsigned peakGroup = 0;
  auto *results = static_cast<uint32_t *>((*output).contents);
  for (unsigned group = simdWidth; group <= groupLimit; group *= 2) {
    auto launch = [&]() {
      return timeDispatch(
          *context, "benchInt8Mac", {*output}, params,
          MTLSizeMake((params.threads + group - 1) / group, 1, 1),
          MTLSizeMake(group, 1, 1));
    };
    std::fill_n(results, size_t(params.threads) * int8BenchChains, 0u);
    auto warmup = launch();
    if (!warmup)
      return warmup.getError();
    if (auto error = verify(params, results))
      return error;
    auto timing = measureGpu(options.samples, launch);
    if (!timing)
      return timing.getError();
    double best = gigaOps / timing->bestSeconds;
    std::cout << std::fixed << std::setprecision(2) << "Group " << group
              << ": best " << best << " GOPS, median "
              << gigaOps / timing->medianSeconds << " GOPS (best GPU time "
              << timing->bestSeconds * 1e3 << " ms)\n";
    if (best > peak) {
      peak = best;
      peakGroup = group;
    }
  }
  std::cout << "Peak measured: " << peak << " GOPS (group " << peakGroup
            << ")\n";
  return {};
}
} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if ((argc == 2 && std::string_view(argv[1]) == "--help") ||
        (argc == 3 && std::string_view(argv[1]) == "gemm" &&
         std::string_view(argv[2]) == "--help")) {
      std::cout << "Usage: llmi-metal-bench [--threads N] [--iterations N] "
                   "[--samples N]\n"
                << "       llmi-metal-bench gemm [--m M] [--n N] [--k K] "
                   "[--samples N]\n";
      return EXIT_SUCCESS;
    }
    auto options = parseOptions(argc, argv);
    if (!options) {
      std::cerr << "llmi-metal-bench: " << options.getError().getMessage()
                << '\n';
      return EXIT_FAILURE;
    }
    if (!MTLCreateSystemDefaultDevice()) {
      std::cerr << "llmi-metal-bench: No Metal device available\n";
      return 77;
    }
    auto error = options->gemm ? benchmarkGemm(*options->gemm, options->samples)
                               : benchmarkMac(*options);
    if (error) {
      std::cerr << "llmi-metal-bench: " << error.getMessage() << '\n';
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
  }
}
