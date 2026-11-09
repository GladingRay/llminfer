#include "llminfer/Models/Qwen35/Qwen35ModelBuilder.h"
#include "Qwen35Schema.h"
#include "llminfer/ModelFormat/Quantization.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <algorithm>
#include <array>
#include <cmath>

using namespace llminfer;
using namespace llminfer::ir;

namespace {
struct ProjectionInput {
  ValueId activation;
  std::array<ValueId, 2> quantized;
};

class Builder {
public:
  Builder(const ModelFile &model, const Qwen35TextConfig &config, Graph graph)
      : model(model), config(config), graph(std::move(graph)),
        tokens(this->graph.getStep().tokens),
        epsilon(static_cast<float>(config.epsilon)),
        theta(static_cast<float>(config.theta)),
        w8(model.getPrecision() == Precision::W8A8) {}

  Result<Graph> build(LogitsMode logits);

private:
  template <size_t N>
  std::array<ValueId, N> emit(OpKind kind,
                              std::initializer_list<ValueId> inputs,
                              Attributes attrs = {}) {
    std::array<ValueId, N> ids{};
    if (failure)
      return ids;
    auto results = graph.append(kind, inputs, std::move(attrs), layer);
    if (!results)
      failure = results.getError();
    else if (results->size() != N)
      failure = Error("Unexpected result count while building Qwen3.5");
    else
      std::copy(results->begin(), results->end(), ids.begin());
    return ids;
  }

  ValueId op(OpKind kind, std::initializer_list<ValueId> inputs,
             Attributes attrs = {}) {
    return emit<1>(kind, inputs, std::move(attrs)).front();
  }

  ValueId weight(std::string_view name);
  StateId state(StateKind kind, TensorType type);
  ProjectionInput prepare(ValueId input);
  ValueId project(const ProjectionInput &input, std::string_view name);
  ValueId norm(ValueId input, std::string_view name);
  ValueId delta(ValueId input, const std::string &prefix);
  ValueId gqa(ValueId input, const std::string &prefix);
  ValueId mlp(ValueId input, const std::string &prefix);

  const ModelFile &model;
  const Qwen35TextConfig &config;
  Graph graph;
  uint64_t tokens;
  float epsilon, theta;
  bool w8;
  std::optional<uint32_t> layer;
  Error failure;
};

ValueId Builder::weight(std::string_view name) {
  if (failure)
    return {};
  const auto *tensor = model.getTensor(name);
  if (!tensor) {
    failure = Error("Missing graph weight " + std::string(name));
    return {};
  }
  ElementType type = ElementType::F16;
  uint64_t group = 0;
  switch (tensor->spec.dtype) {
  case DType::I8:
    type = ElementType::I8;
    group = weightGroupSize;
    break;
  case DType::F16:
    break;
  case DType::F32:
    type = ElementType::F32;
    break;
  }
  // ModelFile resolves the tied output head to the canonical embedding ID.
  auto value =
      graph.addWeight(tensor->spec.id, {type, tensor->spec.shape}, group);
  if (!value) {
    failure = value.getError();
    return {};
  }
  return *value;
}

StateId Builder::state(StateKind kind, TensorType type) {
  if (failure)
    return {};
  auto slot = graph.addState(kind, *layer, std::move(type));
  if (!slot) {
    failure = slot.getError();
    return {};
  }
  return *slot;
}

ProjectionInput Builder::prepare(ValueId input) {
  if (w8)
    return {input, emit<2>(OpKind::QuantizeActivation, {input},
                           QuantizeAttr{weightGroupSize})};
  return {input, {}};
}

ValueId Builder::project(const ProjectionInput &input, std::string_view name) {
  auto matrix = weight(name);
  if (w8)
    return op(OpKind::LinearW8A8,
              {input.quantized[0], input.quantized[1], matrix});
  return op(OpKind::LinearF16, {input.activation, matrix});
}

ValueId Builder::norm(ValueId input, std::string_view name) {
  return op(OpKind::RmsNorm, {input, weight(name)}, NormAttr{epsilon});
}

ValueId Builder::delta(ValueId input, const std::string &prefix) {
  const auto &c = config;
  auto history =
      state(StateKind::ConvHistory,
            {ElementType::F16, {c.linearQkvWidth, c.convKernel - 1}});
  auto recurrent =
      state(StateKind::DeltaRecurrent,
            {ElementType::F32, {c.valueHeads, c.keyDim, c.valueDim}});

  auto projection = prepare(input);
  auto qkv = project(projection, prefix + "in_proj_qkv.weight");
  auto z = project(projection, prefix + "in_proj_z.weight");
  auto a = project(projection, prefix + "in_proj_a.weight");
  auto b = project(projection, prefix + "in_proj_b.weight");
  qkv = op(OpKind::CausalConv1d, {qkv, weight(prefix + "conv1d.weight")},
           StateAttr{history});
  qkv = op(OpKind::Silu, {qkv});

  auto q = op(OpKind::Slice, {qkv}, SliceAttr{1, 0, c.linearKeyWidth});
  auto k = op(OpKind::Slice, {qkv},
              SliceAttr{1, c.linearKeyWidth, c.linearKeyWidth});
  auto v = op(OpKind::Slice, {qkv},
              SliceAttr{1, 2 * c.linearKeyWidth, c.linearValueWidth});
  q = op(OpKind::L2Norm, {q},
         L2NormAttr{c.keyDim, 1e-6f, 1.0f / std::sqrt(float(c.keyDim))});
  k = op(OpKind::L2Norm, {k}, L2NormAttr{c.keyDim, 1e-6f, 1.0f});

  auto gates = emit<2>(OpKind::DeltaGates, {a, b, weight(prefix + "A_log"),
                                            weight(prefix + "dt_bias")});
  auto mixed = op(OpKind::DeltaUpdate, {q, k, v, gates[0], gates[1]},
                  StateAttr{recurrent});
  auto gated =
      op(OpKind::GatedRmsNorm, {mixed, z, weight(prefix + "norm.weight")},
         NormAttr{epsilon});
  return project(prepare(gated), prefix + "out_proj.weight");
}

ValueId Builder::gqa(ValueId input, const std::string &prefix) {
  const auto &c = config;
  TensorType cacheType{ElementType::F16,
                       {graph.getStep().capacity, c.kvHeads, c.headDim}};
  KvAttr cache{state(StateKind::KeyCache, cacheType),
               state(StateKind::ValueCache, std::move(cacheType))};

  auto projection = prepare(input);
  auto qGate = project(projection, prefix + "q_proj.weight");
  auto k = project(projection, prefix + "k_proj.weight");
  auto v = project(projection, prefix + "v_proj.weight");
  qGate = op(OpKind::Reshape, {qGate},
             ReshapeAttr{{tokens, c.qHeads, 2 * c.headDim}});
  auto q = op(OpKind::Slice, {qGate}, SliceAttr{2, 0, c.headDim});
  auto gate = op(OpKind::Slice, {qGate}, SliceAttr{2, c.headDim, c.headDim});
  k = op(OpKind::Reshape, {k}, ReshapeAttr{{tokens, c.kvHeads, c.headDim}});
  v = op(OpKind::Reshape, {v}, ReshapeAttr{{tokens, c.kvHeads, c.headDim}});

  q = norm(q, prefix + "q_norm.weight");
  k = norm(k, prefix + "k_norm.weight");
  RopeAttr rope{c.rotaryWidth, theta};
  q = op(OpKind::PartialRoPE, {q}, rope);
  k = op(OpKind::PartialRoPE, {k}, rope);
  emit<0>(OpKind::KvAppend, {k, v}, cache);
  auto mixed = op(OpKind::CausalGqa, {q}, cache);
  mixed = op(OpKind::SigmoidMul, {gate, mixed});
  mixed = op(OpKind::Reshape, {mixed}, ReshapeAttr{{tokens, c.queryWidth}});
  return project(prepare(mixed), prefix + "o_proj.weight");
}

ValueId Builder::mlp(ValueId input, const std::string &prefix) {
  auto projection = prepare(input);
  auto gate = project(projection, prefix + "gate_proj.weight");
  auto up = project(projection, prefix + "up_proj.weight");
  auto swiglu = op(OpKind::SiluMul, {gate, up});
  return project(prepare(swiglu), prefix + "down_proj.weight");
}

Result<Graph> Builder::build(LogitsMode logits) {
  auto ids = graph.addInput({ElementType::U32, {tokens}});
  if (!ids)
    return ids.getError();
  auto hidden = op(OpKind::EmbeddingGather, {*ids, weight(embeddingId)});
  for (uint32_t i = 0; i < config.layers.size() && !failure; ++i) {
    layer = i;
    std::string prefix = "layers." + std::to_string(i) + ".";
    auto input = norm(hidden, prefix + "input_layernorm.weight");
    auto mixed = config.layers[i] == Qwen35LayerKind::DeltaNet
                     ? delta(input, prefix + "linear_attn.")
                     : gqa(input, prefix + "self_attn.");
    auto residual = op(OpKind::Add, {hidden, mixed});
    input = norm(residual, prefix + "post_attention_layernorm.weight");
    hidden = op(OpKind::Add, {residual, mlp(input, prefix + "mlp.")});
  }

  layer.reset();
  Values outputs;
  if (logits != LogitsMode::None) {
    if (logits == LogitsMode::Last && tokens > 1)
      hidden = op(OpKind::Slice, {hidden}, SliceAttr{0, tokens - 1, 1});
    hidden = norm(hidden, "norm.weight");
    outputs.push_back(project(prepare(hidden), headId));
  }
  if (failure)
    return failure;
  if (auto error = graph.setOutputs(std::move(outputs)))
    return error;
  return std::move(graph);
}
} // namespace

Result<Graph> Qwen35ModelBuilder::build(const ModelFile &model,
                                        Qwen35BuildOptions options) {
  auto manifest = parseJson(model.getManifest());
  if (!manifest)
    return manifest.getError();
  auto config = getQwen35TextConfig(getField(*manifest, "source_config"));
  if (!config)
    return config.getError();
  if (!options.capacity &&
      !checkedAdd(options.past, options.tokens, options.capacity))
    return Error("Graph context length overflow");
  if (options.capacity > config->maxPositions)
    return Error("Graph capacity exceeds the model context limit");

  auto graph = Graph::create({options.tokens, options.past, options.capacity});
  if (!graph)
    return graph.getError();
  auto epsilon = static_cast<float>(config->epsilon);
  auto theta = static_cast<float>(config->theta);
  if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(theta) ||
      theta <= 0)
    return Error("Graph normalization or RoPE constant exceeds FP32 range");
  return Builder(model, *config, std::move(*graph)).build(options.logits);
}
