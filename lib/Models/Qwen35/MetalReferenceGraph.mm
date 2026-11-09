#include "llminfer/Models/Qwen35/MetalReferenceGraph.h"
#include "Kernels/ReferenceParams.h"
#include "MetalContext.h"
#include "Qwen35Schema.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <string_view>

using namespace llminfer;
using namespace llminfer::metal;

namespace {
static_assert(sizeof(unsigned) == 4,
              "Metal parameters require 32-bit unsigned");

struct Config {
  unsigned hidden, intermediate, vocabulary;
  unsigned keyHeads, keyDim, valueHeads, valueDim, convKernel;
  unsigned queryHeads, kvHeads, headDim, rotaryDim;
  unsigned deltaQkvWidth, deltaValueWidth, queryWidth, kvWidth;
  float epsilon, theta;
  std::string deltaPrefix, gqaPrefix;
};

Result<Config> readConfig(const ModelFile &model, unsigned capacity) {
  auto manifest = parseJson(model.getManifest());
  if (!manifest)
    return manifest.getError();
  const auto &text =
      getField(getField(*manifest, "source_config"), "text_config");
  const auto &maxPositions = getField(text, "max_position_embeddings");
  if (!maxPositions.is_number_unsigned() || capacity == 0 ||
      capacity > maxPositions.get<uint64_t>() || capacity > UINT16_MAX)
    return Error("Invalid reference context capacity");

  Config config{};
  const std::pair<const char *, unsigned &> dimensions[] = {
      {"hidden_size", config.hidden},
      {"intermediate_size", config.intermediate},
      {"vocab_size", config.vocabulary},
      {"linear_num_key_heads", config.keyHeads},
      {"linear_key_head_dim", config.keyDim},
      {"linear_num_value_heads", config.valueHeads},
      {"linear_value_head_dim", config.valueDim},
      {"linear_conv_kernel_dim", config.convKernel},
      {"num_attention_heads", config.queryHeads},
      {"num_key_value_heads", config.kvHeads},
      {"head_dim", config.headDim}};
  for (const auto &dimension : dimensions) {
    const auto &value = getField(text, dimension.first);
    uint64_t limit =
        &dimension.second == &config.vocabulary ? UINT32_MAX : UINT16_MAX;
    if (!value.is_number_unsigned() || value == 0 ||
        value.get<uint64_t>() > limit)
      return Error("Unsupported reference dimension: " +
                   std::string(dimension.first));
    dimension.second = value.get<unsigned>();
  }
  uint64_t keyWidth = uint64_t(config.keyHeads) * config.keyDim;
  uint64_t valueWidth = uint64_t(config.valueHeads) * config.valueDim;
  uint64_t qkvWidth = 2 * keyWidth + valueWidth;
  uint64_t queryWidth = uint64_t(config.queryHeads) * config.headDim;
  if (qkvWidth > UINT32_MAX || 2 * queryWidth > UINT32_MAX)
    return Error("Reference projection width exceeds the 32-bit index range");
  config.deltaQkvWidth = qkvWidth;
  config.deltaValueWidth = valueWidth;
  config.queryWidth = queryWidth;
  config.kvWidth = config.kvHeads * config.headDim;

  // ModelFile has already validated these fields against the text schema.
  const auto &rope = getField(text, "rope_parameters");
  config.rotaryDim =
      config.headDim * getField(rope, "partial_rotary_factor").get<double>();
  config.epsilon = getField(text, "rms_norm_eps").get<float>();
  config.theta = getField(rope, "rope_theta").get<float>();
  if (!std::isfinite(config.epsilon) || config.epsilon <= 0 ||
      !std::isfinite(config.theta) || config.theta <= 0)
    return Error("RoPE or RMSNorm constants exceed FP32 range");

  const auto &layers = getField(text, "layer_types");
  for (size_t i = 0; i < layers.size(); ++i) {
    if (layers[i] == "linear_attention" && config.deltaPrefix.empty())
      config.deltaPrefix = "layers." + std::to_string(i) + ".";
    if (layers[i] == "full_attention" && config.gqaPrefix.empty())
      config.gqaPrefix = "layers." + std::to_string(i) + ".";
  }
  if (config.deltaPrefix.empty() || config.gqaPrefix.empty())
    return Error("Reference graph requires a DeltaNet layer and a GQA layer");
  return config;
}

struct WeightSlice {
  unsigned firstRow, rowCount;
  Buffer buffer;
};

struct Weight {
  unsigned rows = 0, width = 0;
  std::vector<WeightSlice> slices;
};

Result<Weight> loadWeight(const ModelFile &model, MetalContext &metal,
                          const TensorRecord &tensor) {
  Weight weight;
  if (tensor.spec.shape.size() == 2) {
    if (tensor.spec.shape[0] > UINT32_MAX || tensor.spec.shape[1] > UINT32_MAX)
      return Error("Weight exceeds the 32-bit index range");
    weight.rows = tensor.spec.shape[0];
    weight.width = tensor.spec.shape[1];
    constexpr uint64_t uploadLimit = 128 << 20;
    auto slices =
        splitRows(tensor, std::min(metal.maxBufferLength(), uploadLimit));
    if (!slices)
      return slices.getError();
    for (const auto &slice : *slices) {
      auto buffer = metal.makeBuffer(slice.data.size);
      if (!buffer)
        return buffer.getError();
      if (auto error = model.read(slice.data, (*buffer).contents))
        return error;
      weight.slices.push_back({static_cast<unsigned>(slice.firstRow),
                               static_cast<unsigned>(slice.rowCount), *buffer});
    }
  } else {
    auto buffer = metal.makeBuffer(tensor.data.size);
    if (!buffer)
      return buffer.getError();
    if (auto error = model.read(tensor.data, (*buffer).contents))
      return error;
    weight.slices.push_back({0, 0, *buffer});
  }
  return weight;
}

Result<Buffer> allocateState(MetalContext &metal, uint64_t elements,
                             unsigned elementSize) {
  if (elements > UINT32_MAX)
    return Error("Reference state exceeds Metal's 32-bit index range");
  uint64_t bytes = elements * elementSize;
  auto buffer = metal.makeBuffer(bytes);
  if (buffer && bytes)
    std::memset((*buffer).contents, 0, bytes);
  return buffer;
}

std::vector<float> readHalf(Buffer buffer, uint64_t count) {
  std::vector<float> result(count);
  const auto *input = static_cast<const uint16_t *>(buffer.contents);
  std::transform(input, input + count, result.begin(), halfToFloat);
  return result;
}

struct BlockBuffers {
  Buffer attentionResidual;
  Buffer swiglu;
  Buffer output;
};
} // namespace

struct MetalReferenceGraph::Impl {
  Impl(MetalContext metal, Config config, unsigned capacity)
      : metal(std::move(metal)), config(std::move(config)), capacity(capacity) {
  }

  Error load(const ModelFile &model) {
    for (const auto &tensor : model.getTensors()) {
      std::string_view id = tensor.spec.id;
      if (id != embeddingId && id != "norm.weight" &&
          id.substr(0, config.deltaPrefix.size()) != config.deltaPrefix &&
          id.substr(0, config.gqaPrefix.size()) != config.gqaPrefix)
        continue;
      auto weight = loadWeight(model, metal, tensor);
      if (!weight)
        return Error(tensor.spec.id + ": " + weight.getError().getMessage());
      weights.emplace(tensor.spec.id, std::move(*weight));
    }

    auto history = allocateState(metal, historyElements(), 2);
    if (!history)
      return history.getError();
    convHistory = *history;
    auto state = allocateState(metal, stateElements(), 4);
    if (!state)
      return state.getError();
    recurrentState = *state;
    auto keys = allocateState(metal, uint64_t(capacity) * config.kvWidth, 2);
    if (!keys)
      return keys.getError();
    kvKeys = *keys;
    auto values = allocateState(metal, uint64_t(capacity) * config.kvWidth, 2);
    if (!values)
      return values.getError();
    kvValues = *values;
    return {};
  }

  uint64_t historyElements() const {
    return uint64_t(config.deltaQkvWidth) * (config.convKernel - 1);
  }

  uint64_t stateElements() const {
    return uint64_t(config.deltaValueWidth) * config.keyDim;
  }

  const Weight *findWeight(Command &command, std::string_view id) const {
    auto it = weights.find(id == headId ? embeddingId : id);
    if (it == weights.end()) {
      command.reject(Error("Missing reference weight: " + std::string(id)));
      return nullptr;
    }
    return &it->second;
  }

  Buffer denseWeight(Command &command, const std::string &id) const {
    const auto *weight = findWeight(command, id);
    return weight ? weight->slices.front().buffer : nil;
  }

  Buffer linear(Command &command, Buffer input, std::string_view id,
                unsigned tokens) const {
    const auto *weight = findWeight(command, id);
    if (!weight)
      return nil;
    Buffer output = command.allocate(uint64_t(tokens) * weight->rows, 2);
    for (const auto &slice : weight->slices) {
      LinearParams params{tokens, weight->width, weight->rows, slice.firstRow,
                          slice.rowCount};
      command.dispatchMatrix("linearF16", {input, slice.buffer, output}, params,
                             tokens, slice.rowCount);
    }
    return output;
  }

  Buffer norm(Command &command, Buffer input, const std::string &id,
              unsigned tokens) const {
    Buffer output = command.allocate(uint64_t(tokens) * config.hidden, 2);
    command.dispatch("rmsNorm", {input, denseWeight(command, id), output},
                     RowParams{tokens, config.hidden, config.epsilon}, tokens);
    return output;
  }

  Buffer add(Command &command, Buffer input, Buffer mixed,
             unsigned tokens) const {
    uint64_t count = uint64_t(tokens) * config.hidden;
    Buffer output = command.allocate(count, 2);
    command.dispatch("residualAdd", {input, mixed, output},
                     ElementParams{static_cast<unsigned>(count)}, count);
    return output;
  }

  Buffer embed(Command &command, Buffer ids, unsigned tokens) const {
    uint64_t count = uint64_t(tokens) * config.hidden;
    Buffer output = command.allocate(count, 2);
    const auto *weight = findWeight(command, embeddingId);
    if (weight)
      for (const auto &slice : weight->slices)
        command.dispatch(
            "embedding", {ids, slice.buffer, output},
            EmbedParams{tokens, config.hidden, slice.firstRow, slice.rowCount},
            count);
    return output;
  }

  Buffer delta(Command &command, Buffer input, unsigned tokens) const {
    const auto &c = config;
    auto prefix = c.deltaPrefix + "linear_attn.";
    Buffer qkv = linear(command, input, prefix + "in_proj_qkv.weight", tokens);
    Buffer z = linear(command, input, prefix + "in_proj_z.weight", tokens);
    Buffer a = linear(command, input, prefix + "in_proj_a.weight", tokens);
    Buffer b = linear(command, input, prefix + "in_proj_b.weight", tokens);
    Buffer conv = command.allocate(uint64_t(tokens) * c.deltaQkvWidth, 2);
    command.dispatch("causalConv",
                     {qkv, denseWeight(command, prefix + "conv1d.weight"),
                      convHistory, conv},
                     ConvParams{tokens, c.deltaQkvWidth, c.convKernel},
                     c.deltaQkvWidth);

    uint64_t qkRows = 2 * uint64_t(tokens) * c.keyHeads;
    Buffer qk = command.allocate(qkRows * c.keyDim, 4);
    command.dispatch(
        "deltaQkNorm", {conv, qk},
        DeltaNormParams{tokens, c.keyHeads, c.keyDim, c.deltaQkvWidth}, qkRows);
    uint64_t valueElements = uint64_t(tokens) * c.deltaValueWidth;
    Buffer mixed = command.allocate(valueElements, 2);
    command.dispatch("deltaUpdate",
                     {conv, qk, a, b, denseWeight(command, prefix + "A_log"),
                      denseWeight(command, prefix + "dt_bias"), recurrentState,
                      mixed},
                     DeltaParams{tokens, c.keyHeads, c.keyDim, c.valueHeads,
                                 c.valueDim, c.deltaQkvWidth},
                     c.deltaValueWidth);

    Buffer gated = command.allocate(valueElements, 2);
    command.dispatch(
        "gatedRmsNorm",
        {mixed, z, denseWeight(command, prefix + "norm.weight"), gated},
        RowParams{tokens * c.valueHeads, c.valueDim, c.epsilon},
        uint64_t(tokens) * c.valueHeads);
    return linear(command, gated, prefix + "out_proj.weight", tokens);
  }

  Buffer gqa(Command &command, Buffer input, unsigned tokens) const {
    const auto &c = config;
    auto prefix = c.gqaPrefix + "self_attn.";
    Buffer projectedQ =
        linear(command, input, prefix + "q_proj.weight", tokens);
    Buffer k = linear(command, input, prefix + "k_proj.weight", tokens);
    Buffer v = linear(command, input, prefix + "v_proj.weight", tokens);
    uint64_t queryRows = uint64_t(tokens) * c.queryHeads;
    Buffer q = command.allocate(queryRows * c.headDim, 2);
    command.dispatch(
        "normRope",
        {projectedQ, denseWeight(command, prefix + "q_norm.weight"), q},
        RopeParams{tokens, c.queryHeads, c.headDim, c.rotaryDim, 2 * c.headDim,
                   past, 0, c.epsilon, c.theta},
        queryRows);
    command.dispatch(
        "normRope", {k, denseWeight(command, prefix + "k_norm.weight"), kvKeys},
        RopeParams{tokens, c.kvHeads, c.headDim, c.rotaryDim, c.headDim, past,
                   past, c.epsilon, c.theta},
        uint64_t(tokens) * c.kvHeads);
    command.dispatch("cacheAppend", {v, kvValues},
                     CacheParams{tokens * c.kvWidth, past * c.kvWidth},
                     uint64_t(tokens) * c.kvWidth);

    GqaParams params{tokens, c.queryHeads, c.kvHeads, c.headDim, past};
    uint64_t scoreCount = queryRows * (past + tokens);
    Buffer scores = command.allocate(scoreCount, 2);
    Buffer mixed = command.allocate(queryRows * c.headDim, 2);
    command.dispatch("gqaScores", {q, kvKeys, scores}, params, scoreCount);
    command.dispatch("gqaApply", {scores, kvValues, projectedQ, mixed}, params,
                     queryRows);
    return linear(command, mixed, prefix + "o_proj.weight", tokens);
  }

  BlockBuffers block(Command &command, Buffer input, unsigned tokens,
                     bool isDelta) const {
    const auto &prefix = isDelta ? config.deltaPrefix : config.gqaPrefix;
    Buffer normalized =
        norm(command, input, prefix + "input_layernorm.weight", tokens);
    Buffer mixed = isDelta ? delta(command, normalized, tokens)
                           : gqa(command, normalized, tokens);
    Buffer residual = add(command, input, mixed, tokens);
    normalized = norm(command, residual,
                      prefix + "post_attention_layernorm.weight", tokens);
    Buffer gate =
        linear(command, normalized, prefix + "mlp.gate_proj.weight", tokens);
    Buffer up =
        linear(command, normalized, prefix + "mlp.up_proj.weight", tokens);
    uint64_t count = uint64_t(tokens) * config.intermediate;
    Buffer swiglu = command.allocate(count, 2);
    command.dispatch("swiGlu", {gate, up, swiglu},
                     ElementParams{static_cast<unsigned>(count)}, count);
    Buffer mlp =
        linear(command, swiglu, prefix + "mlp.down_proj.weight", tokens);
    return {residual, swiglu, add(command, residual, mlp, tokens)};
  }

  BlockTrace traceBlock(const BlockBuffers &block, unsigned tokens) const {
    uint64_t hidden = uint64_t(tokens) * config.hidden;
    return {readHalf(block.attentionResidual, hidden),
            readHalf(block.swiglu, uint64_t(tokens) * config.intermediate),
            readHalf(block.output, hidden)};
  }

  Result<ReferenceTrace> step(const std::vector<uint32_t> &ids) {
    if (invalid)
      return Error("Reference session is invalid after a failed GPU step");
    if (ids.empty() || ids.size() > capacity - past)
      return Error(
          "Token chunk is empty or exceeds the reference context capacity");
    if (std::any_of(ids.begin(), ids.end(),
                    [&](uint32_t id) { return id >= config.vocabulary; }))
      return Error("Token ID is outside the vocabulary");

    auto input = metal.makeBuffer(ids.size() * sizeof(uint32_t));
    if (!input)
      return input.getError();
    std::memcpy((*input).contents, ids.data(), ids.size() * sizeof(uint32_t));
    unsigned tokens = ids.size();

    Command command(metal);
    Buffer embedding = embed(command, *input, tokens);
    auto deltaBlock = block(command, embedding, tokens, true);
    auto gqaBlock = block(command, deltaBlock.output, tokens, false);
    Buffer normalized = norm(command, gqaBlock.output, "norm.weight", tokens);
    Buffer logits = linear(command, normalized, headId, tokens);
    bool submitted;
    if (auto error = command.finish(submitted)) {
      invalid = submitted;
      return error;
    }
    past += tokens;

    const auto *state = static_cast<const float *>(recurrentState.contents);
    return ReferenceTrace{
        readHalf(embedding, uint64_t(tokens) * config.hidden),
        traceBlock(deltaBlock, tokens),
        traceBlock(gqaBlock, tokens),
        readHalf(convHistory, historyElements()),
        std::vector<float>(state, state + stateElements()),
        readHalf(kvKeys, uint64_t(past) * config.kvWidth),
        readHalf(logits, uint64_t(tokens) * config.vocabulary)};
  }

  MetalContext metal;
  Config config;
  unsigned capacity, past = 0;
  bool invalid = false;
  std::map<std::string, Weight, std::less<>> weights;
  Buffer convHistory = nil, recurrentState = nil, kvKeys = nil, kvValues = nil;
};

MetalReferenceGraph::MetalReferenceGraph(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

MetalReferenceGraph::~MetalReferenceGraph() = default;
MetalReferenceGraph::MetalReferenceGraph(MetalReferenceGraph &&) noexcept =
    default;
MetalReferenceGraph &
MetalReferenceGraph::operator=(MetalReferenceGraph &&) noexcept = default;

bool MetalReferenceGraph::hasDevice() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}

Result<MetalReferenceGraph>
MetalReferenceGraph::open(const std::filesystem::path &path,
                          uint32_t capacity) {
  @autoreleasepool {
    auto model = ModelFile::open(path);
    if (!model)
      return model.getError();
    if (model->getPrecision() != Precision::F16)
      return Error("The Metal reference graph requires an FP16 .llmi model");
    auto config = readConfig(*model, capacity);
    if (!config)
      return config.getError();
    auto metal = MetalContext::create();
    if (!metal)
      return metal.getError();
    auto impl =
        std::make_unique<Impl>(std::move(*metal), std::move(*config), capacity);
    if (auto error = impl->load(*model))
      return error;
    return MetalReferenceGraph(std::move(impl));
  }
}

Result<ReferenceTrace>
MetalReferenceGraph::step(const std::vector<uint32_t> &tokenIds) {
  @autoreleasepool {
    return impl->step(tokenIds);
  }
}
