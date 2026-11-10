#include "MetalExecutionPlan.h"
#include "Kernels/GraphParams.h"
#include "Kernels/ReferenceParams.h"
#include "LinearW8Launch.h"
#include "llminfer/ModelFormat/Quantization.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <algorithm>
#include <cstring>

using namespace llminfer;
using namespace llminfer::ir;
using namespace llminfer::metal;

namespace {
constexpr unsigned linearTile = 16;
constexpr unsigned elementWorkgroup = 64;
static_assert(weightGroupSize == 32, "Metal kernels use Group32 weights");

unsigned elementBytes(ElementType type) {
  switch (type) {
  case ElementType::I8:
    return 1;
  case ElementType::F16:
    return 2;
  case ElementType::F32:
  case ElementType::U32:
    return 4;
  }
  return 0;
}

Result<uint64_t> elements(const Shape &shape) {
  uint64_t count = 1;
  for (uint64_t dim : shape)
    if (!checkedMul(count, dim, count))
      return Error("Metal tensor element count overflow");
  if (count > UINT32_MAX)
    return Error("Metal tensor exceeds the 32-bit index range");
  return count;
}

bool isContiguous(const ValueDesc &value) {
  uint64_t stride = 1;
  for (size_t i = value.type.shape.size(); i != 0; --i) {
    auto dim = value.type.shape[i - 1];
    if (dim > 1 && value.strides[i - 1] != stride)
      return false;
    stride *= dim;
  }
  return true;
}

struct Lifetime {
  size_t root;
  uint64_t offset = 0;
  size_t lastUse = 0;
  BufferRef buffer;
};

class Lowering {
public:
  Lowering(const Graph &graph, const ModelFile &model,
           const WeightLayout &weights, uint64_t maxBufferLength)
      : graph(graph), model(model), weights(weights),
        maxBufferLength(maxBufferLength), step(graph.getStep()),
        bindings(graph.getValues().size()) {}

  Result<MetalExecutionPlan> run();

private:
  Error prepareValues();
  Error lower(const Operation &op);
  Error linear(const Operation &op);
  Error embedding(const Operation &op);
  Error attention(const Operation &op);

  const ValueDesc &value(ValueId id) const {
    return graph.getValues()[id.index];
  }
  const Shape &shape(ValueId id) const { return value(id).type.shape; }
  unsigned count(ValueId id) const {
    return static_cast<unsigned>(valueCounts[id.index]);
  }

  BufferRef allocate(uint64_t bytes, size_t lastUse);
  BufferRef dense(ValueId id);
  BufferRef state(StateId id) const { return {BufferSpace::State, id.index}; }

  template <typename Params>
  void dispatch(const char *kernel, std::initializer_list<BufferRef> buffers,
                const Params &params, unsigned groupsX, unsigned groupsY,
                unsigned threadsX, unsigned threadsY) {
    static_assert(sizeof(params) <= sizeof(Dispatch::params));
    if (failure)
      return;
    Dispatch command{
        kernel,  opIndex,        CacheLineAlignedVector<BufferRef>(buffers),
        {},      sizeof(params), groupsX,
        groupsY, threadsX,       threadsY};
    std::memcpy(command.params.data(), &params, sizeof(params));
    plan.dispatches.push_back(std::move(command));
  }

  template <typename Params>
  void dispatch(const char *kernel, std::initializer_list<BufferRef> buffers,
                const Params &params, uint64_t threads) {
    dispatch(kernel, buffers, params,
             (threads + elementWorkgroup - 1) / elementWorkgroup, 1,
             elementWorkgroup, 1);
  }

  const Graph &graph;
  const ModelFile &model;
  const WeightLayout &weights;
  uint64_t maxBufferLength;
  StepShape step;
  CacheLineAlignedVector<Lifetime> bindings;
  CacheLineAlignedVector<uint64_t> valueCounts;
  CacheLineAlignedVector<size_t> slotLastUse;
  MetalExecutionPlan plan;
  size_t opIndex = 0;
  Error failure;
};
} // namespace

Result<WeightLayout> metal::planWeights(const ModelFile &model,
                                        uint64_t maxBufferLength) {
  WeightLayout layout;
  constexpr uint64_t uploadLimit = 128 << 20;
  for (const auto &tensor : model.getTensors()) {
    if (tensor.spec.shape.size() == 2) {
      auto slices = splitRows(tensor, std::min(maxBufferLength, uploadLimit));
      if (!slices)
        return Error(tensor.spec.id + ": " + slices.getError().getMessage());
      layout.push_back(std::move(*slices));
    } else {
      if (tensor.data.size > maxBufferLength)
        return Error(tensor.spec.id + ": weight exceeds Metal maxBufferLength");
      layout.push_back({RowSlice{0, 0, tensor.data, tensor.scales}});
    }
  }
  return layout;
}

Result<uint64_t> metal::getStateBytes(const StateDesc &state) {
  auto count = elements(state.type.shape);
  if (!count)
    return count.getError();
  return *count * elementBytes(state.type.element);
}

BufferRef Lowering::allocate(uint64_t bytes, size_t lastUse) {
  if (failure)
    return {};
  if (bytes > maxBufferLength) {
    failure = Error("Temporary tensor exceeds Metal maxBufferLength");
    return {};
  }

  lastUse = std::max(lastUse, opIndex);
  std::optional<size_t> bestFit, largestFree;
  for (size_t i = 0; i < plan.scratchBytes.size(); ++i) {
    if (slotLastUse[i] >= opIndex)
      continue;
    uint64_t size = plan.scratchBytes[i];
    if (size >= bytes && (!bestFit || size < plan.scratchBytes[*bestFit]))
      bestFit = i;
    if (!largestFree || size > plan.scratchBytes[*largestFree])
      largestFree = i;
  }
  auto reusable = bestFit ? bestFit : largestFree;
  if (reusable) {
    auto i = *reusable;
    plan.scratchBytes[i] = std::max(plan.scratchBytes[i], bytes);
    slotLastUse[i] = lastUse;
    return {BufferSpace::Scratch, i};
  }
  size_t slot = plan.scratchBytes.size();
  plan.scratchBytes.push_back(bytes);
  slotLastUse.push_back(lastUse);
  return {BufferSpace::Scratch, slot};
}

Error Lowering::prepareValues() {
  const auto &values = graph.getValues();
  for (size_t i = 0; i < values.size(); ++i) {
    const auto &desc = values[i];
    auto &binding = bindings[i];
    binding.root = i;
    auto count = elements(desc.type.shape);
    if (!count)
      return count.getError();
    for (uint64_t stride : desc.strides)
      if (stride > UINT32_MAX)
        return Error("Metal tensor stride exceeds the 32-bit index range");
    valueCounts.push_back(*count);

    if (desc.viewOf) {
      const auto &base = bindings[desc.viewOf->index];
      binding.root = base.root;
      if (!checkedAdd(base.offset, desc.viewOffset, binding.offset) ||
          binding.offset > UINT32_MAX)
        return Error("Metal view offset exceeds the 32-bit index range");
    } else if (desc.kind == ValueKind::Weight) {
      const auto *tensor = model.getTensor(desc.tensorId);
      if (!tensor || tensor->spec.shape != desc.type.shape)
        return Error("Invalid Metal weight binding: " + desc.tensorId);
      ElementType type =
          tensor->spec.dtype == DType::I8
              ? ElementType::I8
              : (tensor->spec.dtype == DType::F16 ? ElementType::F16
                                                  : ElementType::F32);
      if (type != desc.type.element ||
          (type == ElementType::I8 && desc.groupSize != weightGroupSize))
        return Error("Invalid Metal weight dtype or group size: " +
                     desc.tensorId);
      size_t index = tensor - model.getTensors().data();
      binding.buffer = {BufferSpace::Weight, index};
    }
  }
  for (size_t i = 0; i < graph.getOps().size(); ++i)
    for (auto id : graph.getOps()[i].inputs)
      bindings[bindings[id.index].root].lastUse = i;
  for (auto id : graph.getOutputs())
    bindings[bindings[id.index].root].lastUse = graph.getOps().size();

  bool hasInput = false;
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i].kind != ValueKind::Input)
      continue;
    if (hasInput || values[i].type.element != ElementType::U32 ||
        values[i].type.shape != Shape{step.tokens})
      return Error("Metal graph expects a single token ID input");
    hasInput = true;
    bindings[i].buffer =
        allocate(step.tokens * sizeof(uint32_t), bindings[i].lastUse);
    plan.tokens = bindings[i].buffer;
  }
  if (!hasInput)
    return Error("Missing token ID input");
  return failure;
}

BufferRef Lowering::dense(ValueId id) {
  const auto &binding = bindings[id.index];
  BufferRef buffer = bindings[binding.root].buffer;
  const auto &desc = value(id);
  if (!desc.viewOf || (!binding.offset && isContiguous(desc)))
    return buffer;
  if (desc.type.element != ElementType::F16 || desc.type.shape.size() > 3) {
    failure = Error("Metal cannot materialize this tensor view");
    return {};
  }

  HalfViewParams params{
      count(id), static_cast<unsigned>(binding.offset), {1, 1, 1}, {0, 0, 0}};
  for (size_t i = 0; i < desc.type.shape.size(); ++i) {
    params.dims[i] = desc.type.shape[i];
    params.strides[i] = desc.strides[i];
  }
  auto copy = allocate(uint64_t(params.count) * 2, opIndex);
  dispatch("copyHalfView", {buffer, copy}, params, params.count);
  return copy;
}

Error Lowering::embedding(const Operation &op) {
  auto ids = dense(op.inputs[0]);
  auto weight = dense(op.inputs[1]);
  auto output = dense(op.outputs[0]);
  unsigned hidden = shape(op.outputs[0])[1];
  unsigned tokens = step.tokens;
  bool quantized = value(op.inputs[1]).groupSize != 0;
  const auto &shards = weights[weight.index];
  for (size_t i = 0; i < shards.size(); ++i) {
    unsigned first = shards[i].firstRow, rows = shards[i].rowCount;
    weight.shard = i;
    if (quantized) {
      unsigned groups = (uint64_t(hidden) + 31) / 32;
      dispatch("embeddingW8",
               {ids, weight, {BufferSpace::Scale, weight.index, i}, output},
               GroupedEmbedParams{tokens, hidden, first, rows, groups},
               count(op.outputs[0]));
    } else {
      dispatch("embedding", {ids, weight, output},
               EmbedParams{tokens, hidden, first, rows}, count(op.outputs[0]));
    }
  }
  return failure;
}

Error Lowering::linear(const Operation &op) {
  bool quantized = op.kind == OpKind::LinearW8A8;
  auto input = dense(op.inputs[0]);
  auto weight = dense(op.inputs[quantized ? 2 : 1]);
  auto output = dense(op.outputs[0]);
  unsigned rows = shape(op.outputs[0])[0];
  unsigned width = shape(op.outputs[0])[1];
  const auto &shards = weights[weight.index];
  for (size_t i = 0; i < shards.size(); ++i) {
    unsigned first = shards[i].firstRow, cols = shards[i].rowCount;
    weight.shard = i;
    if (quantized) {
      unsigned groups = shape(op.inputs[1])[1];
      auto launch = getLinearW8Launch(rows, cols);
      dispatch(launch.kernel,
               {input,
                dense(op.inputs[1]),
                weight,
                {BufferSpace::Scale, weight.index, i},
                output},
               LinearW8Params{width, first, groups, rows, cols}, launch.groupsX,
               launch.groupsY, linearW8WorkgroupSize, 1);
    } else {
      unsigned inputWidth = shape(op.inputs[0])[1];
      dispatch("linearF16", {input, weight, output},
               LinearParams{rows, inputWidth, width, first, cols},
               (uint64_t(cols) + linearTile - 1) / linearTile,
               (uint64_t(rows) + linearTile - 1) / linearTile, linearTile,
               linearTile);
    }
  }
  return failure;
}

Error Lowering::attention(const Operation &op) {
  const auto &attrs = *std::get_if<KvAttr>(&op.attributes);
  auto keys = state(attrs.keys), values = state(attrs.values);
  const auto &cache = graph.getStates()[attrs.keys.index].type.shape;
  unsigned heads = cache[1], headDim = cache[2], past = step.past;
  if (op.kind == OpKind::KvAppend) {
    unsigned width = count(op.inputs[0]);
    unsigned offset = uint64_t(past) * heads * headDim;
    CacheParams params{width, offset};
    dispatch("cacheAppend", {dense(op.inputs[0]), keys}, params, width);
    dispatch("cacheAppend", {dense(op.inputs[1]), values}, params, width);
    return failure;
  }

  const auto &query = shape(op.inputs[0]);
  unsigned tokens = step.tokens, queryHeads = query[1];
  uint64_t queryRows = uint64_t(tokens) * queryHeads;
  uint64_t scoreCount;
  if (!checkedMul(queryRows, step.past + step.tokens, scoreCount) ||
      scoreCount > UINT32_MAX)
    return Error("GQA score buffer exceeds the 32-bit index range");
  auto scores = allocate(scoreCount * 2, opIndex);
  GqaParams params{tokens, queryHeads, heads, headDim, past};
  dispatch("gqaScores", {dense(op.inputs[0]), keys, scores}, params,
           scoreCount);
  dispatch("gqaValue", {scores, values, dense(op.outputs[0])}, params,
           queryRows);
  return failure;
}

Error Lowering::lower(const Operation &op) {
  const auto &inputs = op.inputs;
  const auto &outputs = op.outputs;
  switch (op.kind) {
  case OpKind::Reshape:
  case OpKind::Slice:
    break;
  case OpKind::EmbeddingGather:
    return embedding(op);
  case OpKind::LinearF16:
  case OpKind::LinearW8A8:
    return linear(op);
  case OpKind::KvAppend:
  case OpKind::CausalGqa:
    return attention(op);
  case OpKind::QuantizeActivation: {
    if (value(outputs[0]).groupSize != weightGroupSize)
      return Error("Metal activation quantization requires 32-element groups");
    if (!plan.invalidActivation)
      plan.invalidActivation = allocate(4, graph.getOps().size());
    unsigned rows = shape(inputs[0])[0], width = shape(inputs[0])[1];
    unsigned groups = shape(outputs[1])[1];
    dispatch("quantizeActivation",
             {dense(inputs[0]), dense(outputs[0]), dense(outputs[1]),
              *plan.invalidActivation},
             QuantizeParams{rows, width, groups}, count(outputs[1]));
    break;
  }
  case OpKind::RmsNorm:
  case OpKind::GatedRmsNorm: {
    const auto &attrs = *std::get_if<NormAttr>(&op.attributes);
    unsigned width = shape(inputs.back())[0];
    unsigned rows = count(inputs[0]) / width;
    RowParams params{rows, width, attrs.epsilon};
    if (op.kind == OpKind::RmsNorm)
      dispatch("rmsNorm",
               {dense(inputs[0]), dense(inputs[1]), dense(outputs[0])}, params,
               rows);
    else
      dispatch("gatedRmsNorm",
               {dense(inputs[0]), dense(inputs[1]), dense(inputs[2]),
                dense(outputs[0])},
               params, rows);
    break;
  }
  case OpKind::L2Norm: {
    const auto &attrs = *std::get_if<L2NormAttr>(&op.attributes);
    unsigned width = attrs.groupWidth, rows = count(inputs[0]) / width;
    dispatch("l2NormF32", {dense(inputs[0]), dense(outputs[0])},
             L2Params{rows, width, attrs.epsilon, attrs.scale}, rows);
    break;
  }
  case OpKind::Add:
  case OpKind::SiluMul:
  case OpKind::SigmoidMul: {
    unsigned size = count(outputs[0]);
    const char *kernel =
        op.kind == OpKind::Add
            ? "residualAdd"
            : (op.kind == OpKind::SiluMul ? "swiGlu" : "sigmoidMul");
    dispatch(kernel, {dense(inputs[0]), dense(inputs[1]), dense(outputs[0])},
             ElementParams{size}, size);
    break;
  }
  case OpKind::Silu: {
    unsigned size = count(outputs[0]);
    dispatch("siluF16", {dense(inputs[0]), dense(outputs[0])},
             ElementParams{size}, size);
    break;
  }
  case OpKind::PartialRoPE: {
    const auto &attrs = *std::get_if<RopeAttr>(&op.attributes);
    unsigned heads = shape(inputs[0])[1], width = shape(inputs[0])[2];
    unsigned rows = count(inputs[0]) / width;
    dispatch("partialRope", {dense(inputs[0]), dense(outputs[0])},
             PartialRopeParams{rows, heads, width,
                               static_cast<unsigned>(attrs.rotaryWidth),
                               static_cast<unsigned>(step.past), attrs.theta},
             rows);
    break;
  }
  case OpKind::CausalConv1d: {
    auto history = state(std::get_if<StateAttr>(&op.attributes)->state);
    unsigned channels = shape(inputs[0])[1], kernel = shape(inputs[1])[2];
    dispatch(
        "causalConv",
        {dense(inputs[0]), dense(inputs[1]), history, dense(outputs[0])},
        ConvParams{static_cast<unsigned>(step.tokens), channels, kernel, 0},
        channels);
    break;
  }
  case OpKind::DeltaGates: {
    unsigned rows = shape(inputs[0])[0], width = shape(inputs[0])[1];
    dispatch("deltaGates",
             {dense(inputs[0]), dense(inputs[1]), dense(inputs[2]),
              dense(inputs[3]), dense(outputs[0]), dense(outputs[1])},
             RowParams{rows, width, 0}, count(outputs[0]));
    break;
  }
  case OpKind::DeltaUpdate: {
    auto slot = std::get_if<StateAttr>(&op.attributes)->state;
    const auto &dims = graph.getStates()[slot.index].type.shape;
    unsigned valueHeads = dims[0], keyDim = dims[1], valueDim = dims[2];
    unsigned keyHeads = shape(inputs[0])[1] / keyDim;
    dispatch("graphDeltaUpdate",
             {dense(inputs[0]), dense(inputs[1]), dense(inputs[2]),
              dense(inputs[3]), dense(inputs[4]), state(slot),
              dense(outputs[0])},
             GraphDeltaParams{static_cast<unsigned>(step.tokens), keyHeads,
                              keyDim, valueHeads, valueDim},
             uint64_t(valueHeads) * valueDim);
    break;
  }
  }
  return failure;
}

Result<MetalExecutionPlan> Lowering::run() {
  if (step.capacity > UINT32_MAX || weights.size() != model.getTensors().size())
    return Error("Metal context or weight layout is invalid");
  for (const auto &state : graph.getStates()) {
    auto bytes = getStateBytes(state);
    if (!bytes)
      return bytes.getError();
    if (*bytes > maxBufferLength)
      return Error("Session state exceeds Metal maxBufferLength");
  }
  if (auto error = prepareValues())
    return error;

  for (const auto &op : graph.getOps()) {
    for (auto id : op.outputs) {
      const auto &desc = value(id);
      if (desc.viewOf)
        continue;
      uint64_t stored;
      if (!checkedMul(desc.type.shape[0], desc.strides[0], stored) ||
          stored > UINT32_MAX)
        return Error("Padded Metal tensor exceeds the 32-bit index range");
      auto &binding = bindings[id.index];
      binding.buffer =
          allocate(stored * elementBytes(desc.type.element), binding.lastUse);
    }
    if (auto error = lower(op))
      return Error("Lowering op " + std::to_string(opIndex) + " (" +
                   std::string(getName(op.kind)) + "): " + error.getMessage());
    ++opIndex;
  }

  const auto &outputs = graph.getOutputs();
  if (!outputs.empty()) {
    auto id = outputs.front();
    if (outputs.size() != 1 || value(id).type.element != ElementType::F16 ||
        shape(id).size() != 2 || value(id).viewOf ||
        value(id).kind != ValueKind::Result)
      return Error("Metal graph expects a dense FP16 logits result");
    plan.logits = dense(id);
    plan.logitRows = shape(id)[0];
    plan.logitElements = count(id);
  }
  return std::move(plan);
}

Result<MetalExecutionPlan>
MetalExecutionPlan::lower(const Graph &graph, const ModelFile &model,
                          const WeightLayout &weights,
                          uint64_t maxBufferLength) {
  return Lowering(graph, model, weights, maxBufferLength).run();
}
