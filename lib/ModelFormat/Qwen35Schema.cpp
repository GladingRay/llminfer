#include "Qwen35Schema.h"
#include "File.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace llminfer;
using nlohmann::json;

namespace {
Error require(const json &object, const char *key, const json &expected) {
  if (getField(object, key) != expected)
    return Error(std::string("Unsupported model configuration: ") + key);
  return {};
}

double number(const json &value) {
  return value.is_number() ? value.get<double>() : 0.0;
}

Result<CacheLineAlignedVector<TensorSpec>> getTextTensors(const json &config,
                                                          Precision precision) {
  if (auto error = require(config, "model_type", "qwen3_5"))
    return error;
  if (auto error = require(config, "tie_word_embeddings", true))
    return error;

  const auto &text = getField(config, "text_config");
  const std::pair<const char *, json> options[] = {
      {"model_type", "qwen3_5_text"}, {"tie_word_embeddings", true},
      {"attention_bias", false},      {"attn_output_gate", true},
      {"hidden_act", "silu"},         {"mlp_only_layers", json::array()}};
  for (const auto &option : options)
    if (auto error = require(text, option.first, option.second))
      return error;

  uint64_t hidden, intermediate, vocabulary, layerCount, headDim, qHeads,
      kvHeads, keyHeads, valueHeads, linearKeyDim, linearValueDim, convKernel,
      maxPositions;
  const std::pair<const char *, uint64_t &> dimensions[] = {
      {"hidden_size", hidden},
      {"intermediate_size", intermediate},
      {"vocab_size", vocabulary},
      {"num_hidden_layers", layerCount},
      {"head_dim", headDim},
      {"num_attention_heads", qHeads},
      {"num_key_value_heads", kvHeads},
      {"linear_num_key_heads", keyHeads},
      {"linear_num_value_heads", valueHeads},
      {"linear_key_head_dim", linearKeyDim},
      {"linear_value_head_dim", linearValueDim},
      {"linear_conv_kernel_dim", convKernel},
      {"max_position_embeddings", maxPositions}};
  for (const auto &dimension : dimensions) {
    const auto &value = getField(text, dimension.first);
    if (!value.is_number_unsigned() || value == 0)
      return Error(std::string("Invalid model dimension: ") + dimension.first);
    dimension.second = value.get<uint64_t>();
  }

  uint64_t linearKey, linearValue, linearQkv, query, gatedQuery, keyValue;
  if (!checkedMul(keyHeads, linearKeyDim, linearKey) ||
      !checkedMul(valueHeads, linearValueDim, linearValue) ||
      !checkedMul(2, linearKey, linearQkv) ||
      !checkedAdd(linearQkv, linearValue, linearQkv) ||
      !checkedMul(qHeads, headDim, query) ||
      !checkedMul(2, query, gatedQuery) ||
      !checkedMul(kvHeads, headDim, keyValue))
    return Error("Attention dimension overflow");
  if (qHeads % kvHeads || valueHeads % keyHeads)
    return Error("Incompatible attention head counts");

  const auto &rope = getField(text, "rope_parameters");
  if (auto error = require(rope, "rope_type", "default"))
    return error;
  if (auto error = require(rope, "mrope_interleaved", true))
    return error;
  double rotaryFactor = number(getField(rope, "partial_rotary_factor"));
  double rotaryWidth = headDim * rotaryFactor;
  const auto &sections = getField(rope, "mrope_section");
  if (!(rotaryFactor > 0 && rotaryFactor <= 1) || rotaryWidth < 2 ||
      std::fmod(rotaryWidth, 2) != 0 || !sections.is_array() ||
      sections.size() != 3 || !(number(getField(rope, "rope_theta")) > 0) ||
      !(number(getField(text, "rms_norm_eps")) > 0))
    return Error("Invalid RoPE or normalization configuration");
  uint64_t sectionSum = 0;
  for (const auto &section : sections) {
    if (!section.is_number_unsigned() ||
        !checkedAdd(sectionSum, section.get<uint64_t>(), sectionSum))
      return Error("Invalid RoPE section");
  }
  if (static_cast<double>(sectionSum) != rotaryWidth / 2)
    return Error("RoPE section sizes disagree with rotary width");

  const auto &layers = getField(text, "layer_types");
  if (!layers.is_array() || layers.size() != layerCount || layerCount > 1024)
    return Error("Invalid layer_types array");

  DType matrix = precision == Precision::W8A8 ? DType::I8 : DType::F16;
  CacheLineAlignedVector<TensorSpec> tensors;
  tensors.push_back({std::string(embeddingId), {vocabulary, hidden}, matrix});
  tensors.push_back({"norm.weight", {hidden}, DType::F32});

  for (size_t i = 0; i < layers.size(); ++i) {
    const std::string prefix = "layers." + std::to_string(i) + ".";
    auto add = [&](std::string_view name,
                   CacheLineAlignedVector<uint64_t> shape, DType dtype) {
      tensors.push_back({prefix + std::string(name), std::move(shape), dtype});
    };
    add("input_layernorm.weight", {hidden}, DType::F32);
    add("post_attention_layernorm.weight", {hidden}, DType::F32);
    add("mlp.gate_proj.weight", {intermediate, hidden}, matrix);
    add("mlp.up_proj.weight", {intermediate, hidden}, matrix);
    add("mlp.down_proj.weight", {hidden, intermediate}, matrix);

    if (layers[i] == "linear_attention") {
      add("linear_attn.in_proj_qkv.weight", {linearQkv, hidden}, matrix);
      add("linear_attn.in_proj_z.weight", {linearValue, hidden}, matrix);
      add("linear_attn.in_proj_a.weight", {valueHeads, hidden}, matrix);
      add("linear_attn.in_proj_b.weight", {valueHeads, hidden}, matrix);
      add("linear_attn.out_proj.weight", {hidden, linearValue}, matrix);
      add("linear_attn.conv1d.weight", {linearQkv, 1, convKernel}, DType::F16);
      add("linear_attn.norm.weight", {linearValueDim}, DType::F32);
      add("linear_attn.A_log", {valueHeads}, DType::F32);
      add("linear_attn.dt_bias", {valueHeads}, DType::F32);
    } else if (layers[i] == "full_attention") {
      add("self_attn.q_proj.weight", {gatedQuery, hidden}, matrix);
      add("self_attn.k_proj.weight", {keyValue, hidden}, matrix);
      add("self_attn.v_proj.weight", {keyValue, hidden}, matrix);
      add("self_attn.o_proj.weight", {hidden, query}, matrix);
      add("self_attn.q_norm.weight", {headDim}, DType::F32);
      add("self_attn.k_norm.weight", {headDim}, DType::F32);
    } else {
      return Error("Unsupported layer type at layer " + std::to_string(i));
    }
  }

  std::sort(tensors.begin(), tensors.end(),
            [](const TensorSpec &lhs, const TensorSpec &rhs) {
              return lhs.id < rhs.id;
            });
  return tensors;
}
} // namespace

const json &llminfer::getField(const json &object, std::string_view key) {
  static const json missing;
  auto it = object.find(key);
  return it == object.end() ? missing : *it;
}

Result<json> llminfer::parseJson(std::string_view text) {
  if (text.find('\0') != std::string_view::npos)
    return Error("Unexpected NUL in JSON metadata");

  // Bound nesting before parsing. A parser callback can discard a value, but
  // it cannot stop the parser from descending into that value.
  unsigned depth = 0;
  bool quoted = false;
  bool escaped = false;
  for (char ch : text) {
    if (quoted) {
      if (!escaped && ch == '"')
        quoted = false;
      escaped = !escaped && ch == '\\';
    } else if (ch == '"') {
      quoted = true;
    } else if (ch == '[' || ch == '{') {
      if (++depth > 64)
        return Error("JSON nesting exceeds limit");
    } else if ((ch == ']' || ch == '}') && depth != 0) {
      --depth;
    }
  }

  CacheLineAlignedVector<std::unordered_set<std::string>> keys;
  std::string duplicate;
  auto parsed = json::parse(
      text.begin(), text.end(),
      [&](int, json::parse_event_t event, json &value) {
        switch (event) {
        case json::parse_event_t::object_start:
          keys.emplace_back();
          break;
        case json::parse_event_t::object_end:
          keys.pop_back();
          break;
        case json::parse_event_t::key: {
          const auto &key = value.get_ref<const std::string &>();
          if (!keys.back().insert(key).second && duplicate.empty())
            duplicate = "Duplicate JSON key: " + key;
          break;
        }
        default:
          break;
        }
        return true;
      },
      /*allow_exceptions=*/false);
  if (parsed.is_discarded())
    return Error("Invalid JSON metadata");
  if (!duplicate.empty())
    return Error(std::move(duplicate));
  return parsed;
}

std::string_view llminfer::getPrecisionName(Precision precision) {
  return precision == Precision::W8A8 ? "w8a8" : "f16";
}

Result<ModelSchema> llminfer::getModelSchema(std::string_view manifest) {
  auto root = parseJson(manifest);
  if (!root)
    return root.getError();
  if (auto error = require(*root, "quantization_version", 1))
    return error;

  Precision precision;
  if (getField(*root, "precision") == "w8a8") {
    if (auto error = require(*root, "weight_group_size", weightGroupSize))
      return error;
    precision = Precision::W8A8;
  } else if (getField(*root, "precision") == "f16") {
    if (auto error = require(*root, "weight_group_size", 0))
      return error;
    precision = Precision::F16;
  } else {
    return Error("Unsupported model precision");
  }
  if (auto error =
          require(getField(*root, "aliases"), "lm_head.weight", embeddingId))
    return error;
  const auto &skipped = getField(*root, "skipped_tensors");
  if (!skipped.is_array() ||
      !std::all_of(skipped.begin(), skipped.end(),
                   [](const json &name) { return name.is_string(); }))
    return Error("Invalid skipped_tensors array");

  auto tensors = getTextTensors(getField(*root, "source_config"), precision);
  if (!tensors)
    return tensors.getError();
  return ModelSchema{precision, std::move(*tensors)};
}

std::string llminfer::makeManifest(
    const json &sourceConfig, Precision precision,
    const CacheLineAlignedVector<std::string> &skippedTensors) {
  return json{
      {"precision", getPrecisionName(precision)},
      {"quantization_version", 1},
      {"weight_group_size", precision == Precision::W8A8 ? weightGroupSize : 0},
      {"aliases", {{std::string(headId), embeddingId}}},
      {"source_config", sourceConfig},
      {"skipped_tensors", skippedTensors}}
      .dump();
}
