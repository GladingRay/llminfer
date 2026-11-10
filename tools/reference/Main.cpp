#include "../TokenChunk.h"
#include "llminfer/Models/Qwen35/MetalReferenceGraph.h"
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string_view>

using namespace llminfer;
using nlohmann::json;

namespace {
json describeBlock(const BlockTrace &trace) {
  return {{"attention_residual", trace.attentionResidual},
          {"swiglu", trace.swiglu},
          {"output", trace.output}};
}

json describe(const ReferenceTrace &trace) {
  return {{"embedding", trace.embedding},
          {"delta", describeBlock(trace.delta)},
          {"gqa", describeBlock(trace.gqa)},
          {"conv_history", trace.convHistory},
          {"recurrent_state", trace.recurrentState},
          {"rotated_keys", trace.rotatedKeys},
          {"logits", trace.logits}};
}

Error run(int argc, char **argv) {
  if (argc < 3)
    return Error("Usage: llmi-reference <fp16.llmi> <token[,token...]>...");
  uint32_t capacity = 0;
  CacheLineAlignedVector<CacheLineAlignedVector<uint32_t>> chunks;
  for (int i = 2; i < argc; ++i) {
    auto ids = parseChunk(argv[i]);
    if (!ids)
      return ids.getError();
    if (ids->size() > UINT32_MAX - capacity)
      return Error("Too many reference tokens");
    capacity += ids->size();
    chunks.push_back(std::move(*ids));
  }

  auto graph = MetalReferenceGraph::open(argv[1], capacity);
  if (!graph)
    return graph.getError();
  for (const auto &chunk : chunks) {
    auto trace = graph->step(chunk);
    if (!trace)
      return trace.getError();
    std::cout << describe(*trace).dump() << '\n';
  }
  return {};
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--probe")
    return MetalReferenceGraph::hasDevice() ? EXIT_SUCCESS : 77;
  if (auto error = run(argc, argv)) {
    std::cerr << "llmi-reference: " << error.getMessage() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
