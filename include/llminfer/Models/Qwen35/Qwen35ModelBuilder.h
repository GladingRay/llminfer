#ifndef LLMINFER_MODELS_QWEN35_QWEN35MODELBUILDER_H
#define LLMINFER_MODELS_QWEN35_QWEN35MODELBUILDER_H

#include "llminfer/GraphIR/Graph.h"

namespace llminfer {

class ModelFile;

enum class LogitsMode { None, Last, All };

struct Qwen35BuildOptions {
  uint64_t tokens = 1;
  uint64_t past = 0;
  /// Zero uses exactly past + tokens.
  uint64_t capacity = 0;
  LogitsMode logits = LogitsMode::Last;
};

class Qwen35ModelBuilder {
public:
  static Result<ir::Graph> build(const ModelFile &model,
                                 Qwen35BuildOptions options = {});
};

} // namespace llminfer

#endif
