#include "GraphCommand.h"
#include "Qwen35Schema.h"
#include "llminfer/GraphIR/Cost.h"
#include "llminfer/Models/Qwen35/Qwen35ModelBuilder.h"
#include <charconv>
#include <iostream>
#include <type_traits>

using namespace llminfer;
using namespace llminfer::ir;
using nlohmann::json;

namespace {
json describeCost(const OpCost &cost) {
  return {{"int8_macs", cost.int8Macs},
          {"fp_macs", cost.fpMacs},
          {"scalar_ops", cost.scalarOps},
          {"exp", cost.exp},
          {"rsqrt", cost.rsqrt},
          {"sigmoid", cost.sigmoid},
          {"silu", cost.silu},
          {"softplus", cost.softplus},
          {"sin_cos_pairs", cost.sinCosPairs},
          {"round_even", cost.roundEven},
          {"logical_read_bytes", cost.logicalReadBytes},
          {"logical_write_bytes", cost.logicalWriteBytes}};
}

json describeAttrs(const Attributes &attrs) {
  return std::visit(
      [](const auto &attr) -> json {
        using Attr = std::decay_t<decltype(attr)>;
        if constexpr (std::is_same_v<Attr, QuantizeAttr>)
          return {{"group_size", attr.groupSize}};
        else if constexpr (std::is_same_v<Attr, NormAttr>)
          return {{"epsilon", attr.epsilon}};
        else if constexpr (std::is_same_v<Attr, L2NormAttr>)
          return {{"group_width", attr.groupWidth},
                  {"epsilon", attr.epsilon},
                  {"scale", attr.scale}};
        else if constexpr (std::is_same_v<Attr, ReshapeAttr>)
          return {{"shape", attr.shape}};
        else if constexpr (std::is_same_v<Attr, SliceAttr>)
          return {{"axis", attr.axis},
                  {"begin", attr.begin},
                  {"length", attr.length}};
        else if constexpr (std::is_same_v<Attr, RopeAttr>)
          return {{"rotary_width", attr.rotaryWidth}, {"theta", attr.theta}};
        else if constexpr (std::is_same_v<Attr, StateAttr>)
          return {{"state", attr.state.index}};
        else if constexpr (std::is_same_v<Attr, KvAttr>)
          return {{"keys", attr.keys.index}, {"values", attr.values.index}};
        else
          return json::object();
      },
      attrs);
}

json describeIds(const Values &values) {
  auto ids = json::array();
  for (auto value : values)
    ids.push_back(value.index);
  return ids;
}

json describeGraphJson(const Graph &graph, Precision precision,
                       const CostReport &cost) {
  auto step = graph.getStep();
  json result{{"precision", getPrecisionName(precision)},
              {"step",
               {{"tokens", step.tokens},
                {"past", step.past},
                {"capacity", step.capacity}}},
              {"outputs", describeIds(graph.getOutputs())},
              {"cost", describeCost(cost.total)}};
  auto values = json::array();
  for (const auto &value : graph.getValues()) {
    json description{{"kind", getName(value.kind)},
                     {"dtype", getName(value.type.element)},
                     {"shape", value.type.shape},
                     {"strides", value.strides}};
    if (value.kind == ValueKind::Weight)
      description["tensor_id"] = value.tensorId;
    if (value.groupSize)
      description["group_size"] = value.groupSize;
    if (value.definingOp)
      description["defining_op"] = *value.definingOp;
    if (value.viewOf) {
      description["view_of"] = value.viewOf->index;
      description["view_offset"] = value.viewOffset;
    }
    values.push_back(std::move(description));
  }
  result["values"] = std::move(values);

  auto states = json::array();
  for (const auto &state : graph.getStates())
    states.push_back({{"kind", getName(state.kind)},
                      {"layer", state.layer},
                      {"dtype", getName(state.type.element)},
                      {"shape", state.type.shape}});
  result["states"] = std::move(states);

  auto ops = json::array();
  for (size_t i = 0; i < graph.getOps().size(); ++i) {
    const auto &op = graph.getOps()[i];
    auto effects = json::array();
    for (const auto &effect : op.effects)
      effects.push_back({{"state", effect.state.index},
                         {"access", getName(effect.access)},
                         {"begin", effect.begin},
                         {"length", effect.length},
                         {"version_before", effect.versionBefore},
                         {"version_after", effect.versionAfter}});
    json description{{"kind", getName(op.kind)},
                     {"inputs", describeIds(op.inputs)},
                     {"outputs", describeIds(op.outputs)},
                     {"attributes", describeAttrs(op.attributes)},
                     {"effects", std::move(effects)},
                     {"cost", describeCost(cost.perOp[i])}};
    if (op.layer)
      description["layer"] = *op.layer;
    ops.push_back(std::move(description));
  }
  result["ops"] = std::move(ops);
  return result;
}
} // namespace

Error llminfer::describeGraph(int argc, char **argv) {
  Qwen35BuildOptions options;
  bool printJson = false;
  for (int i = 3; i < argc; ++i) {
    std::string_view flag = argv[i];
    if (flag == "--json") {
      printJson = true;
      continue;
    }
    if (i + 1 == argc)
      return Error("Missing graph option value: " + std::string(flag));
    std::string_view argument = argv[++i];
    if (flag == "--logits") {
      if (argument == "none")
        options.logits = LogitsMode::None;
      else if (argument == "last")
        options.logits = LogitsMode::Last;
      else if (argument == "all")
        options.logits = LogitsMode::All;
      else
        return Error("Expected --logits none|last|all");
      continue;
    }
    uint64_t *number;
    if (flag == "--tokens")
      number = &options.tokens;
    else if (flag == "--past")
      number = &options.past;
    else if (flag == "--capacity")
      number = &options.capacity;
    else
      return Error("Unknown graph option: " + std::string(flag));
    auto parsed = std::from_chars(argument.data(),
                                  argument.data() + argument.size(), *number);
    if (parsed.ec != std::errc{} ||
        parsed.ptr != argument.data() + argument.size())
      return Error("Invalid integer for " + std::string(flag));
  }

  auto model = ModelFile::open(argv[2]);
  if (!model)
    return model.getError();
  auto graph = Qwen35ModelBuilder::build(*model, options);
  if (!graph)
    return graph.getError();
  auto cost = estimateCost(*graph);
  if (!cost)
    return cost.getError();

  if (printJson) {
    std::cout << describeGraphJson(*graph, model->getPrecision(), *cost).dump(2)
              << '\n';
    return {};
  }
  auto step = graph->getStep();
  std::cout << "Qwen3.5 GraphIR / " << getPrecisionName(model->getPrecision())
            << "\nTokens: " << step.tokens << ", past: " << step.past
            << ", capacity: " << step.capacity
            << "\nOperations: " << graph->getOps().size()
            << ", state slots: " << graph->getStates().size()
            << "\nINT8 MACs: " << cost->total.int8Macs
            << "\nFloating-point MACs: " << cost->total.fpMacs
            << "\nLogical read bytes: " << cost->total.logicalReadBytes
            << "\nLogical write bytes: " << cost->total.logicalWriteBytes
            << '\n';
  if (!graph->getOutputs().empty()) {
    const auto &shape =
        graph->getValues()[graph->getOutputs().front().index].type.shape;
    std::cout << "Logits: " << shape[0] << " x " << shape[1] << '\n';
  }
  return {};
}
