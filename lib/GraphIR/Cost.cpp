#include "llminfer/GraphIR/Cost.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <algorithm>

using namespace llminfer;
using namespace llminfer::ir;

namespace {
uint64_t byteWidth(ElementType type) {
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

class Counter {
public:
  template <typename Iterator> uint64_t product(Iterator first, Iterator last) {
    uint64_t value = 1;
    for (; first != last; ++first) {
      if (!checkedMul(value, *first, value)) {
        valid = false;
        return 0;
      }
    }
    return value;
  }

  uint64_t elements(const Shape &shape) {
    return product(shape.begin(), shape.end());
  }

  uint64_t mul(std::initializer_list<uint64_t> factors) {
    return product(factors.begin(), factors.end());
  }

  void add(uint64_t &target, std::initializer_list<uint64_t> factors) {
    uint64_t value = mul(factors);
    if (!checkedAdd(target, value, target))
      valid = false;
  }

  bool valid = true;
};

uint64_t groupCount(const ValueDesc &value) {
  return 1 + (value.type.shape[1] - 1) / value.groupSize;
}

void countTraffic(Counter &count, OpCost &cost, const Graph &graph,
                  const Operation &op) {
  const auto &values = graph.getValues();
  for (size_t i = 0; i < op.inputs.size(); ++i) {
    auto id = op.inputs[i].index;
    auto end = op.inputs.begin() + i;
    if (std::any_of(op.inputs.begin(), end,
                    [id](ValueId earlier) { return earlier.index == id; }))
      continue;
    const auto &value = values[id];
    if (op.kind == OpKind::EmbeddingGather && i == 1) {
      uint64_t rows = graph.getStep().tokens;
      count.add(cost.logicalReadBytes,
                {rows, value.type.shape[1], byteWidth(value.type.element)});
      if (value.groupSize)
        count.add(cost.logicalReadBytes, {rows, groupCount(value), 2});
    } else {
      count.add(cost.logicalReadBytes, {count.elements(value.type.shape),
                                        byteWidth(value.type.element)});
      if (value.kind == ValueKind::Weight && value.groupSize)
        count.add(cost.logicalReadBytes,
                  {value.type.shape[0], groupCount(value), 2});
    }
  }
  for (auto id : op.outputs) {
    const auto &type = values[id.index].type;
    count.add(cost.logicalWriteBytes,
              {count.elements(type.shape), byteWidth(type.element)});
  }
  for (const auto &effect : op.effects) {
    const auto &type = graph.getStates()[effect.state.index].type;
    uint64_t row = count.product(type.shape.begin() + 1, type.shape.end());
    uint64_t bytes = count.mul({effect.length, row, byteWidth(type.element)});
    if (effect.access != Access::Write)
      count.add(cost.logicalReadBytes, {bytes});
    if (effect.access != Access::Read)
      count.add(cost.logicalWriteBytes, {bytes});
  }
}

void countMath(Counter &count, OpCost &cost, const Graph &graph,
               const Operation &op) {
  const auto &values = graph.getValues();
  auto arg = [&](size_t i) -> const ValueDesc & {
    return values[op.inputs[i].index];
  };
  auto shape = [&](size_t i) -> const Shape & { return arg(i).type.shape; };

  switch (op.kind) {
  case OpKind::EmbeddingGather:
    if (arg(1).groupSize)
      count.add(cost.scalarOps, {shape(0)[0], shape(1)[1]});
    break;
  case OpKind::QuantizeActivation: {
    const auto &data = values[op.outputs[0].index];
    uint64_t elems = count.elements(data.type.shape);
    count.add(cost.scalarOps, {5, elems});
    count.add(cost.scalarOps, {shape(0)[0], groupCount(data)});
    cost.roundEven = elems;
    break;
  }
  case OpKind::LinearF16:
    count.add(cost.fpMacs, {shape(0)[0], shape(1)[0], shape(0)[1]});
    break;
  case OpKind::LinearW8A8: {
    uint64_t outputs = count.mul({shape(0)[0], shape(2)[0]});
    uint64_t groups = groupCount(arg(0));
    count.add(cost.int8Macs, {outputs, shape(0)[1]});
    count.add(cost.scalarOps, {2, outputs, groups});
    count.add(cost.scalarOps, {outputs, groups - 1});
    break;
  }
  case OpKind::RmsNorm:
  case OpKind::GatedRmsNorm:
  case OpKind::L2Norm: {
    uint64_t elems = count.elements(shape(0));
    bool l2 = op.kind == OpKind::L2Norm;
    bool gated = op.kind == OpKind::GatedRmsNorm;
    uint64_t width = l2 ? std::get_if<L2NormAttr>(&op.attributes)->groupWidth
                        : shape(gated ? 2 : 1)[0];
    uint64_t groups = elems / width;
    count.add(cost.scalarOps, {elems - groups});
    count.add(cost.scalarOps, {l2 ? 2u : 3u, elems});
    count.add(cost.scalarOps, {l2 ? 1u : 2u, groups});
    if (!l2)
      count.add(cost.scalarOps, {gated ? elems : width});
    else if (std::get_if<L2NormAttr>(&op.attributes)->scale != 1.0f)
      count.add(cost.scalarOps, {groups});
    cost.rsqrt = groups;
    if (gated)
      cost.silu = elems;
    break;
  }
  case OpKind::Add:
  case OpKind::SiluMul:
  case OpKind::SigmoidMul:
    cost.scalarOps = count.elements(shape(0));
    if (op.kind == OpKind::SiluMul)
      cost.silu = cost.scalarOps;
    if (op.kind == OpKind::SigmoidMul)
      cost.sigmoid = cost.scalarOps;
    break;
  case OpKind::Silu:
    cost.silu = count.elements(shape(0));
    break;
  case OpKind::PartialRoPE: {
    uint64_t width = std::get_if<RopeAttr>(&op.attributes)->rotaryWidth;
    count.add(cost.scalarOps, {3, shape(0)[0], shape(0)[1], width});
    cost.sinCosPairs = count.mul({shape(0)[0], width / 2});
    count.add(cost.scalarOps, {cost.sinCosPairs});
    break;
  }
  case OpKind::CausalConv1d:
    count.add(cost.fpMacs, {shape(0)[0], shape(0)[1], shape(1)[2]});
    break;
  case OpKind::DeltaGates: {
    uint64_t gates = count.elements(shape(0));
    cost.sigmoid = gates;
    cost.softplus = gates;
    count.add(cost.exp, {shape(0)[1]});
    count.add(cost.exp, {gates});
    count.add(cost.scalarOps, {3, gates});
    break;
  }
  case OpKind::DeltaUpdate: {
    const auto &state = graph.getStates()[op.effects.front().state.index];
    uint64_t perToken = count.elements(state.type.shape);
    count.add(cost.fpMacs, {3, shape(0)[0], perToken});
    count.add(cost.scalarOps, {shape(0)[0], perToken});
    count.add(cost.scalarOps, {2, count.elements(shape(2))});
    break;
  }
  case OpKind::CausalGqa: {
    auto step = graph.getStep();
    uint64_t t = step.tokens;
    uint64_t pairs = count.mul({t, step.past});
    if (t % 2)
      count.add(pairs, {t, t / 2 + 1});
    else
      count.add(pairs, {t / 2, t + 1});
    uint64_t scores = count.mul({shape(0)[1], pairs});
    count.add(cost.fpMacs, {2, scores, shape(0)[2]});
    count.add(cost.scalarOps, {5, scores});
    cost.exp = scores;
    break;
  }
  case OpKind::Reshape:
  case OpKind::Slice:
  case OpKind::KvAppend:
    break;
  }
}

constexpr uint64_t OpCost::*fields[] = {&OpCost::int8Macs,
                                        &OpCost::fpMacs,
                                        &OpCost::scalarOps,
                                        &OpCost::exp,
                                        &OpCost::rsqrt,
                                        &OpCost::sigmoid,
                                        &OpCost::silu,
                                        &OpCost::softplus,
                                        &OpCost::sinCosPairs,
                                        &OpCost::roundEven,
                                        &OpCost::logicalReadBytes,
                                        &OpCost::logicalWriteBytes};
} // namespace

Result<CostReport> ir::estimateCost(const Graph &graph) {
  CostReport report;
  for (const auto &op : graph.getOps()) {
    OpCost cost;
    Counter count;
    if (op.kind != OpKind::Reshape && op.kind != OpKind::Slice) {
      countTraffic(count, cost, graph, op);
      countMath(count, cost, graph, op);
    }
    for (auto field : fields)
      count.add(report.total.*field, {cost.*field});
    if (!count.valid)
      return Error("Graph cost overflow at " + std::string(getName(op.kind)));
    report.perOp.push_back(cost);
  }
  return report;
}
