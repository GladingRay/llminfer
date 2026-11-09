#include "llminfer/GraphIR/Graph.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <algorithm>
#include <cmath>

using namespace llminfer;
using namespace llminfer::ir;

namespace {
Result<Shape> denseStrides(const Shape &shape) {
  if (shape.empty())
    return Error("A tensor must have positive rank");
  Shape strides(shape.size());
  uint64_t stride = 1;
  for (size_t i = shape.size(); i != 0; --i) {
    strides[i - 1] = stride;
    if (!checkedMul(stride, shape[i - 1], stride))
      return Error("Tensor extent overflow");
  }
  return strides;
}

Result<ValueDesc> makeValue(TensorType type, ValueKind kind,
                            uint64_t groupSize = 0) {
  if (std::find(type.shape.begin(), type.shape.end(), 0) != type.shape.end())
    return Error("Value dimensions must be positive");
  auto strides = denseStrides(type.shape);
  if (!strides)
    return strides.getError();
  if (groupSize) {
    uint64_t rowStride, elements;
    if (type.element != ElementType::I8 || type.shape.size() != 2 ||
        !alignTo(type.shape[1], groupSize, rowStride) ||
        !checkedMul(type.shape[0], rowStride, elements))
      return Error("Invalid grouped INT8 storage");
    (*strides)[0] = rowStride;
  }
  ValueDesc value{};
  value.type = std::move(type);
  value.strides = std::move(*strides);
  value.kind = kind;
  value.groupSize = groupSize;
  return value;
}

bool isDense(const ValueDesc &value) {
  uint64_t stride = 1;
  for (size_t i = value.type.shape.size(); i != 0; --i) {
    uint64_t dim = value.type.shape[i - 1];
    if (dim > 1 && value.strides[i - 1] != stride)
      return false;
    stride *= dim;
  }
  return true;
}

bool isPositive(float value) { return std::isfinite(value) && value > 0; }

template <typename Enum, size_t N>
std::string_view lookup(Enum value, const std::string_view (&names)[N]) {
  auto index = static_cast<size_t>(value);
  return index < N ? names[index] : "unknown";
}

class Inference {
public:
  Inference(const Graph &graph,
            const CacheLineAlignedVector<unsigned> &versions, Operation &op)
      : graph(graph), versions(versions), op(op) {}

  Error run();
  CacheLineAlignedVector<ValueDesc> results;

private:
  template <typename Attr = std::monostate>
  const Attr *signature(size_t operands) const {
    return op.inputs.size() == operands ? std::get_if<Attr>(&op.attributes)
                                        : nullptr;
  }

  const ValueDesc &arg(size_t i) const {
    return graph.getValues()[op.inputs[i].index];
  }

  bool tensor(size_t i, ElementType element, size_t rank) const {
    return arg(i).type.element == element && shape(i).size() == rank;
  }

  bool weight(size_t i, ElementType element, size_t rank) const {
    return arg(i).kind == ValueKind::Weight && tensor(i, element, rank);
  }

  const Shape &shape(size_t i) const { return arg(i).type.shape; }

  Error result(ElementType element, Shape dims, uint64_t group = 0) {
    auto value =
        makeValue({element, std::move(dims)}, ValueKind::Result, group);
    if (!value)
      return value.getError();
    results.push_back(std::move(*value));
    return {};
  }

  Error view(Shape dims, Shape strides, uint64_t offset) {
    ValueDesc value{};
    value.type = {arg(0).type.element, std::move(dims)};
    value.strides = std::move(strides);
    value.kind = ValueKind::View;
    value.viewOf = op.inputs.front();
    value.viewOffset = offset;
    results.push_back(std::move(value));
    return {};
  }

  const StateDesc *state(StateId id, StateKind kind) const {
    if (id.index >= graph.getStates().size())
      return nullptr;
    const auto &desc = graph.getStates()[id.index];
    if (desc.kind != kind || op.layer != desc.layer)
      return nullptr;
    return &desc;
  }

  void effect(StateId id, Access access, uint64_t begin, uint64_t length) {
    unsigned version = versions[id.index];
    op.effects.push_back({id, access, begin, length, version,
                          version + (access != Access::Read)});
  }

  Error linear();
  Error normalize();
  Error views();
  Error deltaUpdate();
  Error kv();

  const Graph &graph;
  const CacheLineAlignedVector<unsigned> &versions;
  Operation &op;
};
} // namespace

Error Inference::linear() {
  if (op.kind == OpKind::LinearF16) {
    if (!signature(2) || !tensor(0, ElementType::F16, 2) ||
        !weight(1, ElementType::F16, 2) || shape(0)[1] != shape(1)[1])
      return Error("Expected compatible FP16 activation and weight matrices");
    return result(ElementType::F16, {shape(0)[0], shape(1)[0]});
  }

  if (!signature(3) || !tensor(0, ElementType::I8, 2) ||
      !tensor(1, ElementType::F32, 2) || !weight(2, ElementType::I8, 2) ||
      shape(0)[1] != shape(2)[1] || !arg(0).definingOp ||
      arg(0).definingOp != arg(1).definingOp)
    return Error("Expected a matching quantized activation and weight");
  const auto &quantize = graph.getOps()[*arg(0).definingOp];
  uint64_t group = arg(2).groupSize;
  if (quantize.kind != OpKind::QuantizeActivation ||
      quantize.outputs[0].index != op.inputs[0].index ||
      quantize.outputs[1].index != op.inputs[1].index ||
      arg(0).groupSize != group)
    return Error("Activation data and scales must come from the same matching "
                 "quantization");
  return result(ElementType::F16, {shape(0)[0], shape(2)[0]});
}

Error Inference::normalize() {
  if (op.kind == OpKind::L2Norm) {
    const auto *attr = signature<L2NormAttr>(1);
    if (!attr || !tensor(0, ElementType::F16, 2) || !attr->groupWidth ||
        shape(0).back() % attr->groupWidth || !isPositive(attr->epsilon) ||
        !isPositive(attr->scale))
      return Error("Invalid grouped L2 normalization");
    return result(ElementType::F32, shape(0));
  }

  bool gated = op.kind == OpKind::GatedRmsNorm;
  const auto *attr = signature<NormAttr>(gated ? 3 : 2);
  size_t weightIndex = gated ? 2 : 1;
  if (!attr || !isPositive(attr->epsilon) ||
      arg(0).type.element != ElementType::F16 ||
      !weight(weightIndex, ElementType::F32, 1))
    return Error("Invalid RMS normalization");
  uint64_t width = shape(weightIndex)[0];
  if (gated) {
    if (arg(1).type.element != ElementType::F16 || shape(1) != shape(0) ||
        shape(0).back() % width)
      return Error("Invalid gated RMS normalization shapes");
  } else if (shape(0).back() != width) {
    return Error("RMS normalization weight does not match the last axis");
  }
  return result(ElementType::F16, shape(0));
}

Error Inference::views() {
  if (op.inputs.size() != 1 || arg(0).kind == ValueKind::Weight ||
      arg(0).groupSize)
    return Error("Expected a single ungrouped activation");
  if (op.kind == OpKind::Slice) {
    const auto *attr = signature<SliceAttr>(1);
    if (!attr || attr->axis >= shape(0).size())
      return Error("Invalid slice axis");
    uint64_t width = shape(0)[attr->axis];
    if (attr->begin > width || !attr->length ||
        attr->length > width - attr->begin)
      return Error("Slice is outside the operand");
    Shape dims = shape(0);
    dims[attr->axis] = attr->length;
    return view(std::move(dims), arg(0).strides,
                attr->begin * arg(0).strides[attr->axis]);
  }

  const auto *attr = signature<ReshapeAttr>(1);
  if (!attr || !isDense(arg(0)) ||
      std::find(attr->shape.begin(), attr->shape.end(), 0) != attr->shape.end())
    return Error("Reshape requires contiguous storage and positive dimensions");
  auto strides = denseStrides(attr->shape);
  if (!strides)
    return strides.getError();
  auto source = denseStrides(shape(0));
  if (!source || source->front() * shape(0).front() !=
                     strides->front() * attr->shape.front())
    return Error("Reshape changes the element count");
  return view(attr->shape, std::move(*strides), 0);
}

Error Inference::deltaUpdate() {
  const auto *attr = signature<StateAttr>(5);
  if (!attr)
    return Error("DeltaUpdate expects Q, K, V, beta and decay");
  auto *recurrent = state(attr->state, StateKind::DeltaRecurrent);
  if (!recurrent || versions[attr->state.index] != 0)
    return Error("Expected an unwritten recurrent state in this layer");

  const auto &dims = recurrent->type.shape;
  uint64_t tokens = graph.getStep().tokens, valueWidth;
  if (!checkedMul(dims[0], dims[2], valueWidth) ||
      !tensor(0, ElementType::F32, 2) || !tensor(1, ElementType::F32, 2) ||
      shape(0)[0] != tokens || shape(1) != shape(0) || shape(0)[1] % dims[1] ||
      dims[0] % (shape(0)[1] / dims[1]) || !tensor(2, ElementType::F16, 2) ||
      shape(2) != Shape{tokens, valueWidth} ||
      !tensor(3, ElementType::F32, 2) || shape(3) != Shape{tokens, dims[0]} ||
      !tensor(4, ElementType::F32, 2) || shape(4) != shape(3))
    return Error("DeltaUpdate operand shapes do not match its recurrent state");
  effect(attr->state, Access::ReadWrite, 0, dims[0]);
  return result(ElementType::F16, shape(2));
}

Error Inference::kv() {
  bool append = op.kind == OpKind::KvAppend;
  const auto *attr = signature<KvAttr>(append ? 2 : 1);
  if (!attr)
    return Error("Expected key and value cache slots");
  const auto *keys = state(attr->keys, StateKind::KeyCache);
  const auto *values = state(attr->values, StateKind::ValueCache);
  if (!keys || !values || keys->type.shape != values->type.shape)
    return Error("Expected compatible key and value caches in this layer");
  auto step = graph.getStep();
  const auto &dims = keys->type.shape;
  if (!tensor(0, ElementType::F16, 3) || shape(0)[0] != step.tokens ||
      shape(0)[2] != dims[2] || shape(0)[1] % dims[1])
    return Error("Invalid attention head shape");

  unsigned version = append ? 0 : 1;
  if (versions[attr->keys.index] != version ||
      versions[attr->values.index] != version)
    return Error(append ? "KV cache is already appended"
                        : "Attention requires the appended KV version");
  if (append) {
    if (!tensor(1, ElementType::F16, 3) || shape(0) != shape(1) ||
        shape(0)[1] != dims[1])
      return Error("New keys and values must match the cache layout");
    effect(attr->keys, Access::Write, step.past, step.tokens);
    effect(attr->values, Access::Write, step.past, step.tokens);
    return {};
  }
  effect(attr->keys, Access::Read, 0, step.past + step.tokens);
  effect(attr->values, Access::Read, 0, step.past + step.tokens);
  return result(ElementType::F16, shape(0));
}

Error Inference::run() {
  switch (op.kind) {
  case OpKind::EmbeddingGather:
    if (!signature(2) || !tensor(0, ElementType::U32, 1) ||
        shape(0)[0] != graph.getStep().tokens ||
        !(weight(1, ElementType::I8, 2) || weight(1, ElementType::F16, 2)))
      return Error("Expected step token IDs and an embedding table");
    return result(ElementType::F16, {shape(0)[0], shape(1)[1]});
  case OpKind::QuantizeActivation: {
    const auto *attr = signature<QuantizeAttr>(1);
    if (!attr || !tensor(0, ElementType::F16, 2) || !attr->groupSize)
      return Error("Invalid activation quantization");
    uint64_t width = shape(0)[1];
    uint64_t groups = 1 + (width - 1) / attr->groupSize;
    if (auto error = result(ElementType::I8, shape(0), attr->groupSize))
      return error;
    return result(ElementType::F32, {shape(0)[0], groups});
  }
  case OpKind::LinearF16:
  case OpKind::LinearW8A8:
    return linear();
  case OpKind::RmsNorm:
  case OpKind::GatedRmsNorm:
  case OpKind::L2Norm:
    return normalize();
  case OpKind::Add:
  case OpKind::SiluMul:
  case OpKind::SigmoidMul:
    if (!signature(2) || arg(0).type.element != ElementType::F16 ||
        arg(1).type.element != ElementType::F16 || shape(0) != shape(1))
      return Error("Expected two matching FP16 operands");
    return result(ElementType::F16, shape(0));
  case OpKind::Silu:
    if (!signature(1) || arg(0).type.element != ElementType::F16)
      return Error("Expected a single FP16 operand");
    return result(ElementType::F16, shape(0));
  case OpKind::Reshape:
  case OpKind::Slice:
    return views();
  case OpKind::PartialRoPE: {
    const auto *attr = signature<RopeAttr>(1);
    if (!attr || !tensor(0, ElementType::F16, 3) ||
        shape(0)[0] != graph.getStep().tokens || !attr->rotaryWidth ||
        attr->rotaryWidth % 2 || attr->rotaryWidth > shape(0).back() ||
        !isPositive(attr->theta))
      return Error("Invalid partial RoPE parameters");
    return result(ElementType::F16, shape(0));
  }
  case OpKind::CausalConv1d: {
    const auto *attr = signature<StateAttr>(2);
    if (!attr || !tensor(0, ElementType::F16, 2) ||
        !weight(1, ElementType::F16, 3) || shape(1)[1] != 1 ||
        shape(0) != Shape{graph.getStep().tokens, shape(1)[0]})
      return Error("Expected a step activation and depthwise filter");
    auto *history = state(attr->state, StateKind::ConvHistory);
    if (!history || versions[attr->state.index] != 0 ||
        history->type.shape != Shape{shape(1)[0], shape(1)[2] - 1})
      return Error("Invalid convolution history");
    effect(attr->state, Access::ReadWrite, 0, shape(1)[0]);
    return result(ElementType::F16, shape(0));
  }
  case OpKind::DeltaGates:
    if (!signature(4) || !tensor(0, ElementType::F16, 2) ||
        !tensor(1, ElementType::F16, 2) || shape(0) != shape(1) ||
        shape(0)[0] != graph.getStep().tokens ||
        !weight(2, ElementType::F32, 1) || !weight(3, ElementType::F32, 1) ||
        shape(2) != shape(3) || shape(2)[0] != shape(0)[1])
      return Error("Invalid DeltaNet gate inputs");
    if (auto error = result(ElementType::F32, shape(0)))
      return error;
    return result(ElementType::F32, shape(0));
  case OpKind::DeltaUpdate:
    return deltaUpdate();
  case OpKind::KvAppend:
  case OpKind::CausalGqa:
    return kv();
  }
  return Error("Unknown GraphIR operation");
}

std::string_view ir::getName(ElementType type) {
  return lookup(type, {"i8", "u32", "f16", "f32"});
}
std::string_view ir::getName(ValueKind kind) {
  return lookup(kind, {"input", "weight", "result", "view"});
}
std::string_view ir::getName(StateKind kind) {
  return lookup(
      kind, {"conv_history", "delta_recurrent", "key_cache", "value_cache"});
}
std::string_view ir::getName(Access access) {
  return lookup(access, {"read", "write", "read_write"});
}
std::string_view ir::getName(OpKind kind) {
  return lookup(kind, {"EmbeddingGather", "QuantizeActivation", "LinearF16",
                       "LinearW8A8", "RmsNorm", "GatedRmsNorm", "L2Norm", "Add",
                       "Silu", "SiluMul", "SigmoidMul", "Reshape", "Slice",
                       "PartialRoPE", "CausalConv1d", "DeltaGates",
                       "DeltaUpdate", "KvAppend", "CausalGqa"});
}

Result<Graph> Graph::create(StepShape step) {
  if (!step.tokens || step.past > step.capacity ||
      step.tokens > step.capacity - step.past)
    return Error("Invalid graph step: expected 0 < tokens <= capacity - past");
  return Graph(step);
}

Result<ValueId> Graph::addSource(TensorType type, ValueKind kind,
                                 std::string id, uint64_t groupSize) {
  auto value = makeValue(std::move(type), kind, groupSize);
  if (!value)
    return value.getError();
  value->tensorId = std::move(id);
  ValueId result{values.size()};
  values.push_back(std::move(*value));
  return result;
}

Result<ValueId> Graph::addInput(TensorType type) {
  return addSource(std::move(type), ValueKind::Input, {}, 0);
}

Result<ValueId> Graph::addWeight(std::string tensorId, TensorType type,
                                 uint64_t groupSize) {
  if (tensorId.empty() ||
      (type.element == ElementType::I8) != (groupSize != 0) ||
      type.element == ElementType::U32)
    return Error("Invalid graph weight declaration");
  for (size_t i = 0; i < values.size(); ++i) {
    const auto &value = values[i];
    if (value.kind != ValueKind::Weight || value.tensorId != tensorId)
      continue;
    if (value.type.element != type.element || value.type.shape != type.shape ||
        value.groupSize != groupSize)
      return Error("Conflicting type for weight " + tensorId);
    return ValueId{i};
  }
  return addSource(std::move(type), ValueKind::Weight, std::move(tensorId),
                   groupSize);
}

Result<StateId> Graph::addState(StateKind kind, uint32_t layer,
                                TensorType type) {
  bool history = kind == StateKind::ConvHistory;
  bool recurrent = kind == StateKind::DeltaRecurrent;
  if (type.shape.size() != (history ? 2 : 3) ||
      type.element != (recurrent ? ElementType::F32 : ElementType::F16) ||
      !type.shape[0] ||
      (!history && std::find(type.shape.begin(), type.shape.end(), 0) !=
                       type.shape.end()) ||
      (!history && !recurrent && type.shape[0] != step.capacity))
    return Error("Invalid graph state type");
  if (auto strides = denseStrides(type.shape); !strides)
    return strides.getError();
  for (const auto &slot : states)
    if (slot.kind == kind && slot.layer == layer)
      return Error("Duplicate state slot in layer");
  StateId id{states.size()};
  states.push_back({kind, layer, std::move(type)});
  stateVersions.push_back(0);
  return id;
}

Result<Values> Graph::append(OpKind kind, std::initializer_list<ValueId> inputs,
                             Attributes attributes,
                             std::optional<uint32_t> layer) {
  for (auto input : inputs)
    if (input.index >= values.size())
      return Error("Graph operand has no preceding definition");
  Operation op{kind, layer, std::move(attributes), Values(inputs), {}, {}};
  Inference inference(*this, stateVersions, op);
  if (auto error = inference.run())
    return Error(std::string(getName(kind)) + ": " + error.getMessage());

  for (auto &value : inference.results) {
    op.outputs.push_back({values.size()});
    value.definingOp = operations.size();
    values.push_back(std::move(value));
  }
  for (const auto &effect : op.effects)
    stateVersions[effect.state.index] = effect.versionAfter;
  auto result = op.outputs;
  operations.push_back(std::move(op));
  return result;
}

Error Graph::setOutputs(Values outputs) {
  for (auto value : outputs)
    if (value.index >= values.size())
      return Error("Graph output has no definition");
  this->outputs = std::move(outputs);
  return {};
}
