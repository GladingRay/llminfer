#include "../TokenChunk.h"
#include "llminfer/Backend/Metal/MetalSession.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace llminfer;
using nlohmann::json;

namespace {
void printUsage(std::ostream &out) {
  out << "Usage: llmi-run <model.llmi> <token[,token...]>... [--capacity N] "
         "[--logits none|last|all] [--plan]\n";
}

Error run(int argc, char **argv) {
  uint64_t tokens = 0, capacity = 0;
  LogitsMode mode = LogitsMode::Last;
  bool tracePlan = false;
  CacheLineAlignedVector<CacheLineAlignedVector<uint32_t>> chunks;
  for (int i = 2; i < argc; ++i) {
    std::string_view argument = argv[i];
    if (argument == "--plan") {
      tracePlan = true;
    } else if (argument == "--logits") {
      if (++i == argc)
        return Error("Expected --logits none|last|all");
      std::string_view value = argv[i];
      if (value == "none")
        mode = LogitsMode::None;
      else if (value == "last")
        mode = LogitsMode::Last;
      else if (value == "all")
        mode = LogitsMode::All;
      else
        return Error("Expected --logits none|last|all");
    } else if (argument == "--capacity") {
      if (++i == argc)
        return Error("Expected a positive --capacity");
      std::string_view value = argv[i];
      auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), capacity);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || capacity == 0)
        return Error("Expected a positive --capacity");
    } else {
      auto ids = parseChunk(argument);
      if (!ids)
        return ids.getError();
      if (!checkedAdd(tokens, ids->size(), tokens))
        return Error("Token count overflow");
      chunks.push_back(std::move(*ids));
    }
  }
  if (chunks.empty())
    return Error("Expected at least one token chunk");
  if (!capacity)
    capacity = tokens;
  if (tokens > capacity)
    return Error("Token chunks exceed session capacity");

  auto session = MetalSession::open(argv[1], capacity);
  if (!session)
    return session.getError();
  for (const auto &chunk : chunks) {
    auto result = session->step(chunk, mode, tracePlan);
    if (!result)
      return result.getError();
    json output{{"past", result->past},
                {"logit_rows", result->logitRows},
                {"scratch_bytes", result->scratchBytes},
                {"dispatch_count", result->dispatchCount},
                {"logits", std::move(result->logits)}};
    if (tracePlan) {
      auto dispatches = json::array();
      for (const auto &dispatch : result->dispatches)
        dispatches.push_back(
            {{"op_index", dispatch.opIndex}, {"kernel", dispatch.kernel}});
      output["dispatches"] = std::move(dispatches);
    }
    std::cout << output.dump() << '\n';
  }
  return {};
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    printUsage(std::cout);
    return EXIT_SUCCESS;
  }
  if (argc < 3) {
    printUsage(std::cerr);
    return EXIT_FAILURE;
  }
  if (auto error = run(argc, argv)) {
    std::cerr << "llmi-run: " << error.getMessage() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
