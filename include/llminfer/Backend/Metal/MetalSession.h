#ifndef LLMINFER_BACKEND_METAL_METALSESSION_H
#define LLMINFER_BACKEND_METAL_METALSESSION_H

#include "llminfer/Models/Qwen35/Qwen35ModelBuilder.h"
#include <filesystem>
#include <memory>

namespace llminfer {

struct MetalDispatchInfo {
  size_t opIndex;
  std::string_view kernel;
};

struct MetalStepResult {
  uint64_t past;
  uint64_t logitRows;
  uint64_t scratchBytes;
  size_t dispatchCount;
  CacheLineAlignedVector<float> logits;
  CacheLineAlignedVector<MetalDispatchInfo> dispatches;
};

/// Executes the full Qwen3.5 text graph for one sequence.
class MetalSession {
public:
  static bool hasDevice();
  static Result<MetalSession> open(const std::filesystem::path &path,
                                   uint64_t capacity);
  ~MetalSession();
  MetalSession(MetalSession &&) noexcept;
  MetalSession &operator=(MetalSession &&) noexcept;

  /// Calls on the same session must be serialized.
  Result<MetalStepResult> step(const CacheLineAlignedVector<uint32_t> &tokenIds,
                               LogitsMode mode = LogitsMode::Last,
                               bool tracePlan = false);
  void reset();

private:
  struct Impl;
  explicit MetalSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

} // namespace llminfer

#endif
