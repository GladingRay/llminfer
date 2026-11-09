#ifndef LLMINFER_GRAPHIR_GRAPH_H
#define LLMINFER_GRAPHIR_GRAPH_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string_view>
#include <variant>

namespace llminfer::ir {

struct ValueId {
  size_t index;
};
struct StateId {
  size_t index;
};
using Shape = CacheLineAlignedVector<uint64_t>;
using Values = CacheLineAlignedVector<ValueId>;

enum class ElementType { I8, U32, F16, F32 };
enum class ValueKind { Input, Weight, Result, View };
enum class StateKind { ConvHistory, DeltaRecurrent, KeyCache, ValueCache };

struct StepShape {
  uint64_t tokens, past, capacity;
};

struct TensorType {
  ElementType element;
  Shape shape;
};

struct ValueDesc {
  TensorType type;
  Shape strides;
  ValueKind kind;
  std::string tensorId;
  uint64_t groupSize = 0;
  std::optional<size_t> definingOp;
  std::optional<ValueId> viewOf;
  // Element offset relative to viewOf. Strides are also in elements.
  uint64_t viewOffset = 0;
};

struct StateDesc {
  StateKind kind;
  uint32_t layer;
  TensorType type;
};

enum class Access { Read, Write, ReadWrite };
struct StateEffect {
  StateId state;
  Access access;
  // The affected interval along axis zero.
  uint64_t begin, length;
  unsigned versionBefore, versionAfter;
};

enum class OpKind {
  EmbeddingGather,
  // Produces INT8 data followed by FP32 group scales.
  QuantizeActivation,
  LinearF16,
  LinearW8A8,
  // Qwen's zero-centered normalization applies 1 + weight.
  RmsNorm,
  // Consumes input, SiLU gate, and direct RMSNorm weights in that order.
  GatedRmsNorm,
  L2Norm,
  Add,
  Silu,
  SiluMul,
  SigmoidMul,
  Reshape,
  Slice,
  PartialRoPE,
  CausalConv1d,
  // Consumes a, b, A_log and dt_bias. Produces beta and decay in FP32.
  DeltaGates,
  DeltaUpdate,
  KvAppend,
  CausalGqa,
};

struct QuantizeAttr {
  uint64_t groupSize;
};
struct NormAttr {
  float epsilon;
};
struct L2NormAttr {
  uint64_t groupWidth;
  float epsilon, scale;
};
struct ReshapeAttr {
  Shape shape;
};
struct SliceAttr {
  size_t axis;
  uint64_t begin, length;
};
struct RopeAttr {
  uint64_t rotaryWidth;
  float theta;
};
struct StateAttr {
  StateId state;
};
struct KvAttr {
  StateId keys, values;
};

using Attributes =
    std::variant<std::monostate, QuantizeAttr, NormAttr, L2NormAttr,
                 ReshapeAttr, SliceAttr, RopeAttr, StateAttr, KvAttr>;

struct Operation {
  OpKind kind;
  std::optional<uint32_t> layer;
  Attributes attributes;
  Values inputs, outputs;
  CacheLineAlignedVector<StateEffect> effects;
};

std::string_view getName(ElementType type);
std::string_view getName(ValueKind kind);
std::string_view getName(StateKind kind);
std::string_view getName(OpKind kind);
std::string_view getName(Access access);

/// A typed SSA graph specialized for one step shape.
class Graph {
public:
  static Result<Graph> create(StepShape step);

  Result<ValueId> addInput(TensorType type);
  Result<ValueId> addWeight(std::string tensorId, TensorType type,
                            uint64_t groupSize = 0);
  Result<StateId> addState(StateKind kind, uint32_t layer, TensorType type);

  /// Infers results and verifies operands and effects before changing the
  /// graph.
  Result<Values> append(OpKind kind, std::initializer_list<ValueId> inputs,
                        Attributes attributes = {},
                        std::optional<uint32_t> layer = {});
  Error setOutputs(Values outputs);

  StepShape getStep() const { return step; }
  const auto &getValues() const { return values; }
  const auto &getStates() const { return states; }
  const auto &getOps() const { return operations; }
  const Values &getOutputs() const { return outputs; }

private:
  explicit Graph(StepShape step) : step(step) {}
  Result<ValueId> addSource(TensorType type, ValueKind kind, std::string id,
                            uint64_t groupSize);

  StepShape step;
  CacheLineAlignedVector<ValueDesc> values;
  CacheLineAlignedVector<StateDesc> states;
  CacheLineAlignedVector<unsigned> stateVersions;
  CacheLineAlignedVector<Operation> operations;
  Values outputs;
};

} // namespace llminfer::ir

#endif
