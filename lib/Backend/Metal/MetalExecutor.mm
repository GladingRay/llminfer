#include "Kernels/GraphParams.h"
#include "MetalContext.h"
#include "MetalExecutionPlan.h"
#include "llminfer/Backend/Metal/MetalSession.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <cstring>

using namespace llminfer;
using namespace llminfer::metal;
using namespace llminfer::ir;

namespace {
struct WeightBuffers {
  Buffer data = nil;
  Buffer scale = nil;
};

class MetalExecutor {
public:
  explicit MetalExecutor(MetalContext context) : context(std::move(context)) {}

  Error initialize(const ModelFile &model, const Graph &graph);
  Result<MetalStepResult> execute(const Graph &graph, const ModelFile &model,
                                  const CacheLineAlignedVector<uint32_t> &ids,
                                  uint64_t newPast, bool tracePlan,
                                  bool flashAttentionSupported,
                                  bool &submitted);
  void reset();

  uint64_t maxBufferLength() const { return context.maxBufferLength(); }
  Result<bool> supportsFlashAttention();
  bool matchesStates(const Graph &graph) const;

private:
  Buffer resolve(BufferRef ref) const;
  Result<Buffer> upload(const ModelFile &model, ByteRange range);

  MetalContext context;
  WeightLayout layout;
  CacheLineAlignedVector<CacheLineAlignedVector<WeightBuffers>> weights;
  CacheLineAlignedVector<StateDesc> stateTypes;
  CacheLineAlignedVector<Buffer> states;
  CacheLineAlignedVector<Buffer> scratch;
};

Result<Buffer> MetalExecutor::upload(const ModelFile &model, ByteRange range) {
  auto buffer = context.makeBuffer(range.size);
  if (!buffer)
    return buffer.getError();
  if (auto error = model.read(range, (*buffer).contents))
    return error;
  return *buffer;
}

Error MetalExecutor::initialize(const ModelFile &model, const Graph &graph) {
  auto planned = planWeights(model, maxBufferLength());
  if (!planned)
    return planned.getError();
  layout = std::move(*planned);

  stateTypes = graph.getStates();
  for (const auto &state : stateTypes) {
    auto bytes = getStateBytes(state);
    if (!bytes)
      return bytes.getError();
    auto buffer = context.makeBuffer(*bytes);
    if (!buffer)
      return buffer.getError();
    states.push_back(*buffer);
  }
  reset();

  for (size_t i = 0; i < layout.size(); ++i) {
    CacheLineAlignedVector<WeightBuffers> buffers;
    for (const auto &slice : layout[i]) {
      auto data = upload(model, slice.data);
      if (!data)
        return Error(model.getTensors()[i].spec.id + ": " +
                     data.getError().getMessage());
      WeightBuffers pair{*data, nil};
      if (slice.scales.size) {
        auto scale = upload(model, slice.scales);
        if (!scale)
          return Error(model.getTensors()[i].spec.id + ": " +
                       scale.getError().getMessage());
        pair.scale = *scale;
      }
      buffers.push_back(pair);
    }
    weights.push_back(std::move(buffers));
  }
  return {};
}

bool MetalExecutor::matchesStates(const Graph &graph) const {
  const auto &slots = graph.getStates();
  return std::equal(slots.begin(), slots.end(), stateTypes.begin(),
                    stateTypes.end(),
                    [](const StateDesc &lhs, const StateDesc &rhs) {
                      return lhs.kind == rhs.kind && lhs.layer == rhs.layer &&
                             lhs.type.element == rhs.type.element &&
                             lhs.type.shape == rhs.type.shape;
                    });
}

Result<bool> MetalExecutor::supportsFlashAttention() {
  auto pipeline = context.getPipeline("gqaFlashAttention");
  if (!pipeline)
    return pipeline.getError();
  return (*pipeline).threadExecutionWidth == flashGqaSimdWidth &&
         (*pipeline).maxTotalThreadsPerThreadgroup >= flashGqaThreads;
}

void MetalExecutor::reset() {
  for (Buffer buffer : states)
    std::memset(buffer.contents, 0, buffer.length);
}

Buffer MetalExecutor::resolve(BufferRef ref) const {
  switch (ref.space) {
  case BufferSpace::Scratch:
    return scratch[ref.index];
  case BufferSpace::Weight:
    return weights[ref.index][ref.shard].data;
  case BufferSpace::Scale:
    return weights[ref.index][ref.shard].scale;
  case BufferSpace::State:
    return states[ref.index];
  }
  return nil;
}

Result<MetalStepResult>
MetalExecutor::execute(const Graph &graph, const ModelFile &model,
                       const CacheLineAlignedVector<uint32_t> &ids,
                       uint64_t newPast, bool tracePlan,
                       bool flashAttentionSupported, bool &submitted) {
  submitted = false;
  MetalStepResult result{newPast, 0, 0, 0, {}, {}};
  std::optional<Command> command;
  auto prepare = [&](const CacheLineAlignedVector<uint64_t> &sizes) -> Error {
    scratch.resize(sizes.size(), nil);
    for (size_t i = 0; i < scratch.size(); ++i) {
      if (scratch[i].length < sizes[i]) {
        scratch[i] = nil;
        auto buffer = context.makeBuffer(sizes[i]);
        if (!buffer)
          return buffer.getError();
        scratch[i] = *buffer;
      }
      result.scratchBytes += scratch[i].length;
    }
    command.emplace(context);
    return {};
  };
  CacheLineAlignedVector<Buffer> buffers;
  auto encode = [&](const Dispatch &dispatch) {
    buffers.clear();
    for (auto ref : dispatch.buffers)
      buffers.push_back(resolve(ref));
    command->encode(dispatch.kernel, buffers.data(), buffers.size(),
                    dispatch.params, dispatch.paramSize,
                    MTLSizeMake(dispatch.groupsX, dispatch.groupsY, 1),
                    MTLSizeMake(dispatch.threadsX, dispatch.threadsY, 1));
    ++result.dispatchCount;
    if (tracePlan)
      result.dispatches.push_back({dispatch.opIndex, dispatch.kernel,
                                   dispatch.epilogue ? dispatch.epilogue : "",
                                   dispatch.fusedOpIndex});
  };
  auto plan =
      MetalExecutionPlan::lower(graph, model, layout, maxBufferLength(),
                                flashAttentionSupported, prepare, encode);
  if (!plan)
    return plan.getError();

  std::memcpy(resolve(plan->tokens).contents, ids.data(),
              ids.size() * sizeof(uint32_t));
  if (plan->invalidActivation)
    *static_cast<uint32_t *>(resolve(*plan->invalidActivation).contents) = 0;
  if (auto error = command->finish(submitted))
    return error;
  if (plan->invalidActivation &&
      *static_cast<const uint32_t *>(
          resolve(*plan->invalidActivation).contents))
    return Error("Non-finite activation during W8A8 quantization");

  result.logitRows = plan->logitRows;
  if (plan->logits) {
    const auto *half =
        static_cast<const uint16_t *>(resolve(*plan->logits).contents);
    result.logits.resize(plan->logitElements);
    std::transform(half, half + plan->logitElements, result.logits.begin(),
                   halfToFloat);
  }
  return result;
}
} // namespace

struct MetalSession::Impl {
  Impl(ModelFile model, MetalContext context, uint64_t capacity, uint64_t vocab)
      : model(std::move(model)), executor(std::move(context)),
        capacity(capacity), vocabulary(vocab) {}

  ModelFile model;
  MetalExecutor executor;
  uint64_t capacity, vocabulary, past = 0;
  bool invalid = false;
};

MetalSession::MetalSession(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}
MetalSession::~MetalSession() = default;
MetalSession::MetalSession(MetalSession &&) noexcept = default;
MetalSession &MetalSession::operator=(MetalSession &&) noexcept = default;

uint64_t MetalSession::getVocabularySize() const { return impl->vocabulary; }

std::optional<uint32_t> MetalSession::getEosTokenId() const {
  return impl->model.getEosTokenId();
}

bool MetalSession::hasDevice() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}

Result<MetalSession> MetalSession::open(const std::filesystem::path &path,
                                        uint64_t capacity) {
  @autoreleasepool {
    if (!capacity || capacity > UINT32_MAX)
      return Error("Invalid Metal session capacity");
    auto model = ModelFile::open(path);
    if (!model)
      return model.getError();
    auto graph =
        Qwen35ModelBuilder::build(*model, {1, 0, capacity, LogitsMode::None});
    if (!graph)
      return graph.getError();
    auto context = MetalContext::create();
    if (!context)
      return context.getError();
    uint64_t vocabulary =
        model->getTensor("embed_tokens.weight")->spec.shape[0];
    auto impl = std::make_unique<Impl>(std::move(*model), std::move(*context),
                                       capacity, vocabulary);
    if (auto error = impl->executor.initialize(impl->model, *graph))
      return error;
    return MetalSession(std::move(impl));
  }
}

Result<MetalStepResult>
MetalSession::step(const CacheLineAlignedVector<uint32_t> &ids, LogitsMode mode,
                   bool tracePlan) {
  @autoreleasepool {
    if (impl->invalid)
      return Error("Metal session is invalid after a failed GPU step");
    if (ids.empty() || ids.size() > impl->capacity - impl->past)
      return Error("Token chunk is empty or exceeds session capacity");
    if (std::any_of(ids.begin(), ids.end(),
                    [&](uint32_t id) { return id >= impl->vocabulary; }))
      return Error("Token ID is outside the vocabulary");

    auto graph = Qwen35ModelBuilder::build(
        impl->model, {ids.size(), impl->past, impl->capacity, mode});
    if (!graph)
      return graph.getError();
    auto &executor = impl->executor;
    if (!executor.matchesStates(*graph))
      return Error("Graph state layout changed within the Metal session");
    bool flashAttentionSupported = false;
    if (ids.size() > 1) {
      auto supported = executor.supportsFlashAttention();
      if (!supported)
        return supported.getError();
      flashAttentionSupported = *supported;
    }
    uint64_t newPast = impl->past + ids.size();
    bool submitted;
    auto result = executor.execute(*graph, impl->model, ids, newPast, tracePlan,
                                   flashAttentionSupported, submitted);
    if (!result) {
      impl->invalid = submitted;
      return result.getError();
    }
    impl->past = newPast;
    return std::move(*result);
  }
}

void MetalSession::reset() {
  impl->executor.reset();
  impl->past = 0;
  impl->invalid = false;
}
