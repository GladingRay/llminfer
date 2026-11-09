#include "Qwen35Schema.h"
#include "File.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace llminfer;
using nlohmann::json;

namespace {
uint64_t dimension(const json &object, const char *key) {
  const auto &value = object.at(key);
  if (!value.is_number_unsigned() || value == 0)
    fail(std::string("Invalid model dimension: ") + key);
  return value.get<uint64_t>();
}

void require(const json &object, const char *key, const json &expected) {
  if (object.at(key) != expected)
    fail(std::string("Unsupported model configuration: ") + key);
}

std::vector<TensorSpec> getTextTensors(const json &config,
                                       Precision precision) {
  require(config, "model_type", "qwen3_5");
  require(config, "tie_word_embeddings", true);
  const auto &text = config.at("text_config");
  require(text, "model_type", "qwen3_5_text");
  require(text, "tie_word_embeddings", true);
  require(text, "attention_bias", false);
  require(text, "attn_output_gate", true);
  require(text, "hidden_act", "silu");
  require(text, "mlp_only_layers", json::array());

  uint64_t hidden = dimension(text, "hidden_size");
  uint64_t intermediate = dimension(text, "intermediate_size");
  uint64_t vocabulary = dimension(text, "vocab_size");
  uint64_t layerCount = dimension(text, "num_hidden_layers");
  uint64_t headDim = dimension(text, "head_dim");
  uint64_t qHeads = dimension(text, "num_attention_heads");
  uint64_t kvHeads = dimension(text, "num_key_value_heads");
  uint64_t keyHeads = dimension(text, "linear_num_key_heads");
  uint64_t valueHeads = dimension(text, "linear_num_value_heads");
  uint64_t linearKey =
      checkedMul(keyHeads, dimension(text, "linear_key_head_dim"));
  uint64_t linearValueDim = dimension(text, "linear_value_head_dim");
  uint64_t linearValue = checkedMul(valueHeads, linearValueDim);
  uint64_t linearQkv = checkedAdd(checkedMul(2, linearKey), linearValue);
  uint64_t convKernel = dimension(text, "linear_conv_kernel_dim");
  uint64_t query = checkedMul(qHeads, headDim);
  uint64_t keyValue = checkedMul(kvHeads, headDim);
  dimension(text, "max_position_embeddings");
  if (qHeads % kvHeads || valueHeads % keyHeads)
    fail("Incompatible attention head counts");

  const auto &rope = text.at("rope_parameters");
  require(rope, "rope_type", "default");
  require(rope, "mrope_interleaved", true);
  double rotaryFactor = rope.at("partial_rotary_factor").get<double>();
  double rotaryWidth = headDim * rotaryFactor;
  const auto &sections = rope.at("mrope_section");
  if (!(rotaryFactor > 0 && rotaryFactor <= 1) || rotaryWidth < 2 ||
      std::fmod(rotaryWidth, 2) != 0 || !sections.is_array() ||
      sections.size() != 3 || !(rope.at("rope_theta").get<double>() > 0) ||
      !(text.at("rms_norm_eps").get<double>() > 0))
    fail("Invalid RoPE or normalization configuration");
  uint64_t sectionSum = 0;
  for (const auto &section : sections) {
    if (!section.is_number_unsigned())
      fail("Invalid RoPE section");
    sectionSum = checkedAdd(sectionSum, section.get<uint64_t>());
  }
  if (static_cast<double>(sectionSum) != rotaryWidth / 2)
    fail("RoPE section sizes disagree with rotary width");

  const auto &layers = text.at("layer_types");
  if (!layers.is_array() || layers.size() != layerCount || layerCount > 1024)
    fail("Invalid layer_types array");

  DType matrix = precision == Precision::W8A8 ? DType::I8 : DType::F16;
  std::vector<TensorSpec> tensors;
  tensors.push_back({std::string(embeddingId), {vocabulary, hidden}, matrix});
  tensors.push_back({"norm.weight", {hidden}, DType::F32});

  for (size_t i = 0; i < layers.size(); ++i) {
    const std::string prefix = "layers." + std::to_string(i) + ".";
    auto add = [&](std::string_view name, std::vector<uint64_t> shape,
                   DType dtype) {
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
      add("self_attn.q_proj.weight", {checkedMul(query, 2), hidden}, matrix);
      add("self_attn.k_proj.weight", {keyValue, hidden}, matrix);
      add("self_attn.v_proj.weight", {keyValue, hidden}, matrix);
      add("self_attn.o_proj.weight", {hidden, query}, matrix);
      add("self_attn.q_norm.weight", {headDim}, DType::F32);
      add("self_attn.k_norm.weight", {headDim}, DType::F32);
    } else {
      fail("Unsupported layer type at layer " + std::to_string(i));
    }
  }

  std::sort(tensors.begin(), tensors.end(),
            [](const TensorSpec &lhs, const TensorSpec &rhs) {
              return lhs.id < rhs.id;
            });
  return tensors;
}
} // namespace

json llminfer::parseJson(std::string_view text) {
  if (text.find('\0') != std::string_view::npos)
    fail("Unexpected NUL in JSON metadata");
  std::vector<std::unordered_set<std::string>> keys;
  return json::parse(
      text.begin(), text.end(),
      [&](int depth, json::parse_event_t event, json &value) {
        if (depth > 64)
          fail("JSON nesting exceeds limit");
        switch (event) {
        case json::parse_event_t::object_start:
          keys.emplace_back();
          break;
        case json::parse_event_t::object_end:
          keys.pop_back();
          break;
        case json::parse_event_t::key:
          if (!keys.back().insert(value.get<std::string>()).second)
            fail("Duplicate JSON key: " + value.get<std::string>());
          break;
        default:
          break;
        }
        return true;
      });
}

std::string_view llminfer::getPrecisionName(Precision precision) {
  return precision == Precision::W8A8 ? "w8a8" : "f16";
}

ModelSchema llminfer::getModelSchema(std::string_view manifest) {
  const auto root = parseJson(manifest);
  require(root, "quantization_version", 1);
  Precision precision;
  if (root.at("precision") == "w8a8") {
    require(root, "weight_group_size", weightGroupSize);
    precision = Precision::W8A8;
  } else if (root.at("precision") == "f16") {
    require(root, "weight_group_size", 0);
    precision = Precision::F16;
  } else {
    fail("Unsupported model precision");
  }
  require(root.at("aliases"), "lm_head.weight", embeddingId);
  const auto &skipped = root.at("skipped_tensors");
  if (!skipped.is_array() ||
      !std::all_of(skipped.begin(), skipped.end(),
                   [](const json &name) { return name.is_string(); }))
    fail("Invalid skipped_tensors array");
  return {precision, getTextTensors(root.at("source_config"), precision)};
}

std::string
llminfer::makeManifest(const json &sourceConfig, Precision precision,
                       const std::vector<std::string> &skippedTensors) {
  return json{
      {"precision", getPrecisionName(precision)},
      {"quantization_version", 1},
      {"weight_group_size", precision == Precision::W8A8 ? weightGroupSize : 0},
      {"aliases", {{std::string(headId), embeddingId}}},
      {"source_config", sourceConfig},
      {"skipped_tensors", skippedTensors}}
      .dump();
}
