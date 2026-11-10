#ifndef LLMINFER_BACKEND_METAL_METALEXECUTIONPLAN_H
#define LLMINFER_BACKEND_METAL_METALEXECUTIONPLAN_H

#include "llminfer/GraphIR/Graph.h"
#include "llminfer/ModelFormat/ModelFile.h"
#include <functional>
#include <initializer_list>

namespace llminfer::metal {

using WeightLayout = CacheLineAlignedVector<CacheLineAlignedVector<RowSlice>>;

Result<WeightLayout> planWeights(const ModelFile &model,
                                 uint64_t maxBufferLength);
Result<uint64_t> getStateBytes(const ir::StateDesc &state);

enum class BufferSpace { Scratch, Weight, Scale, State };
struct BufferRef {
  BufferSpace space = BufferSpace::Scratch;
  size_t index = 0;
  size_t shard = 0;
};

struct Dispatch {
  const char *kernel;
  size_t opIndex;
  std::initializer_list<BufferRef> buffers;
  const void *params;
  size_t paramSize;
  unsigned groupsX, groupsY, threadsX, threadsY;
  const char *epilogue = nullptr;
  size_t fusedOpIndex = 0;
};

struct MetalExecutionPlan {
  using PrepareScratch =
      std::function<Error(const CacheLineAlignedVector<uint64_t> &)>;
  // Dispatch buffers and parameters are borrowed for the duration of the call.
  using EncodeDispatch = std::function<void(const Dispatch &)>;

  CacheLineAlignedVector<uint64_t> scratchBytes;
  BufferRef tokens;
  std::optional<BufferRef> invalidActivation;
  std::optional<BufferRef> logits;
  uint64_t logitRows = 0, logitElements = 0;

  static Result<MetalExecutionPlan>
  lower(const ir::Graph &graph, const ModelFile &model,
        const WeightLayout &weights, uint64_t maxBufferLength,
        bool flashAttentionSupported, const PrepareScratch &prepare,
        const EncodeDispatch &encode);
};

} // namespace llminfer::metal

#endif
