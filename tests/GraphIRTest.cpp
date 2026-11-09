#include "llminfer/GraphIR/Cost.h"
#include <cstdlib>
#include <iostream>

using namespace llminfer;
using namespace llminfer::ir;

namespace {
void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <typename T> T take(Result<T> result) {
  if (!result) {
    std::cerr << result.getError().getMessage() << '\n';
    std::exit(EXIT_FAILURE);
  }
  return std::move(*result);
}

void testViewsAndQuantization() {
  auto graph = take(Graph::create({2, 0, 2}));
  auto input = take(graph.addInput({ElementType::F16, {2, 33}}));
  auto matrix = take(graph.addWeight("weight", {ElementType::I8, {7, 33}}, 32));
  auto same = take(graph.addWeight("weight", {ElementType::I8, {7, 33}}, 32));
  check(matrix.index == same.index, "Weight identities must be canonical");
  check(!graph.addWeight("weight", {ElementType::I8, {7, 33}}, 16),
        "Conflicting weight declaration accepted");
  check(!graph.append(OpKind::Add, {input, ValueId{999}}),
        "Undefined operand accepted");

  auto slice =
      take(graph.append(OpKind::Slice, {input}, SliceAttr{1, 2, 4}))[0];
  const auto &desc = graph.getValues()[slice.index];
  check(desc.type.shape == Shape{2, 4} && desc.strides == Shape{33, 1} &&
            desc.viewOf->index == input.index && desc.viewOffset == 2,
        "Slice did not preserve strides and alias");
  check(!graph.append(OpKind::Reshape, {slice}, ReshapeAttr{{8}}),
        "A strided slice must not flatten as a dense view");

  auto first =
      take(graph.append(OpKind::QuantizeActivation, {input}, QuantizeAttr{32}));
  auto second =
      take(graph.append(OpKind::QuantizeActivation, {input}, QuantizeAttr{32}));
  const auto &data = graph.getValues()[first[0].index];
  const auto &scales = graph.getValues()[first[1].index];
  check(data.type.shape == Shape{2, 33} && data.strides == Shape{64, 1} &&
            scales.type.shape == Shape{2, 2},
        "Group32 logical shape, padding or scale shape is wrong");
  size_t before = graph.getOps().size();
  check(!graph.append(OpKind::LinearW8A8, {first[0], second[1], matrix}) &&
            graph.getOps().size() == before,
        "Unrelated activation scales accepted");
  auto wrongGroup =
      take(graph.addWeight("group16", {ElementType::I8, {7, 33}}, 16));
  check(!graph.append(OpKind::LinearW8A8, {first[0], first[1], wrongGroup}),
        "Incompatible group sizes accepted");
  auto output =
      take(graph.append(OpKind::LinearW8A8, {first[0], first[1], matrix}));
  check(!graph.setOutputs(output), "Valid output rejected");
  auto cost = take(estimateCost(graph));
  check(cost.perOp.front().logicalReadBytes == 0 &&
            cost.perOp.front().logicalWriteBytes == 0,
        "A view must have zero logical traffic");
  const auto &linear = cost.perOp.back();
  check(linear.int8Macs == 462 && linear.logicalReadBytes == 341 &&
            linear.logicalWriteBytes == 28,
        "Grouped linear cost does not match logical extents");
}

void testCacheEffects() {
  auto graph = take(Graph::create({3, 2, 8}));
  auto query = take(graph.addInput({ElementType::F16, {3, 2, 8}}));
  auto keys = take(graph.addInput({ElementType::F16, {3, 1, 8}}));
  auto values = take(graph.addInput({ElementType::F16, {3, 1, 8}}));
  KvAttr slots{take(graph.addState(StateKind::KeyCache, 7,
                                   {ElementType::F16, {8, 1, 8}})),
               take(graph.addState(StateKind::ValueCache, 7,
                                   {ElementType::F16, {8, 1, 8}}))};

  check(!graph.append(OpKind::CausalGqa, {query}, slots, 7),
        "Attention read a stale cache version");
  check(!graph.append(OpKind::KvAppend, {keys, values}, slots, 8),
        "Cross-layer cache mutation accepted");
  take(graph.append(OpKind::KvAppend, {keys, values}, slots, 7));
  check(!graph.append(OpKind::KvAppend, {keys, values}, slots, 7),
        "Repeated cache append accepted");
  take(graph.append(OpKind::CausalGqa, {query}, slots, 7));
  const auto &write = graph.getOps()[0].effects.front();
  const auto &read = graph.getOps()[1].effects.front();
  check(write.begin == 2 && write.length == 3 && write.versionBefore == 0 &&
            write.versionAfter == 1 && read.begin == 0 && read.length == 5 &&
            read.versionBefore == 1 && read.versionAfter == 1,
        "KV ranges or versions are incorrect");

  auto cost = take(estimateCost(graph));
  check(cost.perOp[0].logicalWriteBytes == 96 &&
            cost.perOp[1].logicalReadBytes == 256 &&
            cost.perOp[1].fpMacs == 384,
        "Causal GQA cost does not match the visible prefix");
}

void testOverflow() {
  check(!Graph::create({2, UINT64_MAX, UINT64_MAX}),
        "Overflowing step accepted");
  auto graph = take(Graph::create({1, 0, 1}));
  check(!graph.addInput({ElementType::F16, {2, UINT64_MAX}}),
        "Overflowing tensor shape accepted");
  auto ids = take(graph.addInput({ElementType::U32, {1}}));
  auto weight =
      take(graph.addWeight("weight", {ElementType::F16, {1, UINT64_MAX}}));
  take(graph.append(OpKind::EmbeddingGather, {ids, weight}));
  check(!estimateCost(graph), "Overflowing logical byte count accepted");
}
} // namespace

int main() {
  testViewsAndQuantization();
  testCacheEffects();
  testOverflow();
  return EXIT_SUCCESS;
}
