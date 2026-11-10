#include "llminfer/Backend/Metal/MetalSession.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

using namespace llminfer;

namespace {
using Clock = std::chrono::steady_clock;

struct Options {
  uint64_t prefillTokens = 128;
  uint64_t decodeTokens = 32;
};

void printUsage(std::ostream &out) {
  out << "Usage: llmi-benchmark <model.llmi> [--prefill-tokens N] "
         "[--decode-tokens N]\n"
      << "Defaults: 128 prefill tokens and 32 decode tokens. "
         "Use zero to skip a phase.\n";
}

Result<Options> parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 2; i < argc; ++i) {
    std::string_view flag = argv[i];
    uint64_t *count;
    if (flag == "--prefill-tokens")
      count = &options.prefillTokens;
    else if (flag == "--decode-tokens")
      count = &options.decodeTokens;
    else
      return Error("Unknown option: " + std::string(flag));
    if (++i == argc)
      return Error("Missing value for " + std::string(flag));
    std::string_view value = argv[i];
    auto parsed =
        std::from_chars(value.data(), value.data() + value.size(), *count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
      return Error("Expected a nonnegative token count for " +
                   std::string(flag));
  }
  return options;
}

void report(std::string_view phase, uint64_t tokens, Clock::duration elapsed) {
  double seconds = std::chrono::duration<double>(elapsed).count();
  std::cout << phase << ": " << tokens << " tokens in " << std::fixed
            << std::setprecision(6) << seconds << " s, " << std::setprecision(2)
            << tokens / seconds << " tok/s\n"
            << std::flush;
}

Error benchmark(const std::filesystem::path &modelPath, Options options) {
  uint64_t capacity;
  if (!checkedAdd(options.prefillTokens, options.decodeTokens, capacity) ||
      capacity > UINT32_MAX)
    return Error("Total token count exceeds the Metal session capacity limit");
  if (!capacity)
    return Error("Specify at least one prefill or decode token");

  auto session = MetalSession::open(modelPath, capacity);
  if (!session)
    return session.getError();
  uint64_t vocabulary = session->getVocabularySize();
  CacheLineAlignedVector<uint32_t> prompt(options.prefillTokens);
  for (size_t i = 0; i < prompt.size(); ++i)
    prompt[i] = static_cast<uint32_t>(i % vocabulary);
  CacheLineAlignedVector<uint32_t> token(1, 0);

  // Warm up both requested graph shapes without processing the full workload.
  if (!prompt.empty()) {
    CacheLineAlignedVector<uint32_t> warmup(
        prompt.begin(), prompt.begin() + std::min<size_t>(prompt.size(), 2));
    auto result = session->step(warmup, LogitsMode::Last);
    if (!result)
      return Error("Prefill warmup: " + result.getError().getMessage());
  }
  if (options.decodeTokens) {
    auto result = session->step(token, LogitsMode::Last);
    if (!result)
      return Error("Decode warmup: " + result.getError().getMessage());
  }
  session->reset();

  if (!prompt.empty()) {
    auto start = Clock::now();
    auto result = session->step(prompt, LogitsMode::Last);
    auto elapsed = Clock::now() - start;
    if (!result)
      return Error("Prefill: " + result.getError().getMessage());
    report("Prefill", options.prefillTokens, elapsed);
  }
  if (options.decodeTokens) {
    auto start = Clock::now();
    for (uint64_t i = 0; i < options.decodeTokens; ++i) {
      token.front() =
          static_cast<uint32_t>((options.prefillTokens + i) % vocabulary);
      auto result = session->step(token, LogitsMode::Last);
      if (!result)
        return Error("Decode step " + std::to_string(i + 1) + ": " +
                     result.getError().getMessage());
    }
    report("Decode", options.decodeTokens, Clock::now() - start);
  }
  return {};
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    printUsage(std::cout);
    return EXIT_SUCCESS;
  }
  if (argc < 2) {
    printUsage(std::cerr);
    return EXIT_FAILURE;
  }
  auto options = parseOptions(argc, argv);
  if (!options) {
    std::cerr << "llmi-benchmark: " << options.getError().getMessage() << '\n';
    return EXIT_FAILURE;
  }
  if (auto error = benchmark(argv[1], *options)) {
    std::cerr << "llmi-benchmark: " << error.getMessage() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
