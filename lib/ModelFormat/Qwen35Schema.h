#ifndef LLMINFER_MODELFORMAT_QWEN35SCHEMA_H
#define LLMINFER_MODELFORMAT_QWEN35SCHEMA_H

#include "llminfer/ModelFormat/ModelFile.h"
#include <nlohmann/json.hpp>

namespace llminfer {

inline constexpr uint64_t maxManifestSize = 8 << 20;
inline constexpr std::string_view embeddingId = "embed_tokens.weight";
inline constexpr std::string_view headId = "lm_head.weight";
inline constexpr std::string_view hfTextPrefix = "model.language_model.";

struct ModelSchema {
  Precision precision;
  std::vector<TensorSpec> tensors;
};

const nlohmann::json &getField(const nlohmann::json &object,
                               std::string_view key);
Result<nlohmann::json> parseJson(std::string_view text);
Result<ModelSchema> getModelSchema(std::string_view manifest);
std::string makeManifest(const nlohmann::json &sourceConfig,
                         Precision precision,
                         const std::vector<std::string> &skippedTensors);
std::string_view getPrecisionName(Precision precision);

} // namespace llminfer

#endif
