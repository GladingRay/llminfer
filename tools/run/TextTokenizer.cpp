#include "TextTokenizer.h"
#include "File.h"
#include "Qwen35Schema.h"
#include <algorithm>

using namespace llminfer;

namespace {
constexpr uint64_t maxTokenizerBytes = 64 << 20;

bool isValidUtf8(std::string_view text) {
  for (size_t i = 0; i < text.size();) {
    auto lead = static_cast<uint8_t>(text[i++]);
    if (lead < 0x80)
      continue;
    unsigned length;
    uint32_t value, minimum;
    if (lead >= 0xc2 && lead <= 0xdf) {
      length = 1;
      value = lead & 0x1f;
      minimum = 0x80;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      length = 2;
      value = lead & 0x0f;
      minimum = 0x800;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      length = 3;
      value = lead & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (text.size() - i < length)
      return false;
    for (unsigned j = 0; j < length; ++j) {
      auto byte = static_cast<uint8_t>(text[i++]);
      if ((byte & 0xc0) != 0x80)
        return false;
      value = (value << 6) | (byte & 0x3f);
    }
    if (value < minimum || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff))
      return false;
  }
  return true;
}

Result<CacheLineAlignedVector<uint32_t>> readVocabulary(std::string_view blob) {
  auto config = parseJson(blob);
  if (!config)
    return config.getError();
  const auto &model = getField(*config, "model");
  const auto &type = getField(model, "type");
  if (type != "BPE" && type != "WordLevel")
    return Error("Expected a Hugging Face BPE or WordLevel tokenizer.json");

  const auto &vocab = getField(model, "vocab");
  if (!vocab.is_object() || vocab.empty())
    return Error("Tokenizer vocabulary is missing or empty");
  if (type == "BPE" && !getField(model, "merges").is_array())
    return Error("Tokenizer BPE merges are missing");
  const auto &added = getField(*config, "added_tokens");
  if (!added.is_null() && !added.is_array())
    return Error("Invalid tokenizer added_tokens");

  CacheLineAlignedVector<uint32_t> ids;
  ids.reserve(vocab.size() + added.size());
  auto appendId = [&](const nlohmann::json &entry) -> Error {
    if (!entry.is_number_unsigned() || entry.get<uint64_t>() > INT32_MAX)
      return Error("Tokenizer ID exceeds the tokenizers-cpp range");
    ids.push_back(entry.get<uint32_t>());
    return {};
  };
  for (const auto &id : vocab)
    if (auto error = appendId(id))
      return error;
  for (const auto &token : added)
    if (auto error = appendId(getField(token, "id")))
      return error;

  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}
} // namespace

Result<TextTokenizer> TextTokenizer::open(const std::filesystem::path &path) {
  auto blob = readTextFile(path, maxTokenizerBytes);
  if (!blob)
    return blob.getError();
  auto ids = readVocabulary(*blob);
  if (!ids)
    return Error(path.string() + ": " + ids.getError().getMessage());
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(*blob);
  if (!tokenizer)
    return Error("Could not load tokenizer: " + path.string());
  return TextTokenizer(std::move(tokenizer), std::move(*ids));
}

Result<CacheLineAlignedVector<uint32_t>>
TextTokenizer::encode(std::string_view text) {
  if (!isValidUtf8(text))
    return Error("Tokenizer input must be valid UTF-8");
  if (text.empty())
    return CacheLineAlignedVector<uint32_t>{};
  auto encoded = tokenizer->Encode(std::string(text));
  CacheLineAlignedVector<uint32_t> ids;
  ids.reserve(encoded.size());
  for (auto id : encoded) {
    if (id < 0 || !std::binary_search(tokenIds.begin(), tokenIds.end(),
                                      static_cast<uint32_t>(id)))
      return Error("Tokenizer returned an unknown token ID");
    ids.push_back(static_cast<uint32_t>(id));
  }
  return ids;
}

Result<std::string>
TextTokenizer::decode(const CacheLineAlignedVector<uint32_t> &ids) {
  if (ids.empty())
    return std::string{};
  // Match the upstream container at the library boundary.
  decltype(tokenizer->Encode(std::string{})) encoded;
  encoded.reserve(ids.size());
  for (auto id : ids) {
    if (!std::binary_search(tokenIds.begin(), tokenIds.end(), id))
      return Error("Token ID is outside the tokenizer vocabulary: " +
                   std::to_string(id));
    encoded.push_back(static_cast<int32_t>(id));
  }
  return tokenizer->Decode(encoded);
}
