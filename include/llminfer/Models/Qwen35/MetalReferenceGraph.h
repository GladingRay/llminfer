#ifndef LLMINFER_MODELS_QWEN35_METALREFERENCEGRAPH_H
#define LLMINFER_MODELS_QWEN35_METALREFERENCEGRAPH_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <cstdint>
#include <filesystem>
#include <memory>

namespace llminfer {

struct BlockTrace {
  CacheLineAlignedVector<float> attentionResidual;
  CacheLineAlignedVector<float> swiglu;
  CacheLineAlignedVector<float> output;
};

struct ReferenceTrace {
  CacheLineAlignedVector<float> embedding;
  BlockTrace delta;
  BlockTrace gqa;
  CacheLineAlignedVector<float> convHistory;
  CacheLineAlignedVector<float> recurrentState;
  CacheLineAlignedVector<float> rotatedKeys;
  CacheLineAlignedVector<float> logits;
};

/// Runs the first DeltaNet layer and first GQA layer on a single sequence.
class MetalReferenceGraph {
public:
  static bool hasDevice();
  static Result<MetalReferenceGraph> open(const std::filesystem::path &path,
                                          uint32_t capacity);
  ~MetalReferenceGraph();
  MetalReferenceGraph(MetalReferenceGraph &&) noexcept;
  MetalReferenceGraph &operator=(MetalReferenceGraph &&) noexcept;

  /// Appends a nonempty chunk and returns activations and current cache state.
  /// Calls on the same graph must be serialized.
  Result<ReferenceTrace> step(const CacheLineAlignedVector<uint32_t> &tokenIds);

private:
  struct Impl;
  explicit MetalReferenceGraph(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

} // namespace llminfer

#endif
