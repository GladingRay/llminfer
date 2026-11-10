#include "TextTokenizer.h"
#include "File.h"
#include "Qwen35Schema.h"
#include <algorithm>
#include <array>

using namespace llminfer;

namespace {
constexpr uint64_t maxTokenizerBytes = 64 << 20;
constexpr std::string_view replacement = "\xef\xbf\xbd";

enum class Utf8Status { Complete, Invalid, Incomplete };

struct Utf8Scalar {
  Utf8Status status;
  size_t size;
  uint32_t value = 0;
};

// On error, size is the number of bytes before the next possible scalar.
Utf8Scalar readUtf8(std::string_view text) {
  auto lead = static_cast<uint8_t>(text.front());
  if (lead < 0x80)
    return {Utf8Status::Complete, 1, lead};
  if (lead < 0xc2 || lead > 0xf4)
    return {Utf8Status::Invalid, 1};

  size_t length = lead < 0xe0 ? 2 : (lead < 0xf0 ? 3 : 4);
  uint32_t value = lead & (0x7f >> length);
  for (size_t i = 1; i < length; ++i) {
    if (i == text.size())
      return {Utf8Status::Incomplete, i};
    auto byte = static_cast<uint8_t>(text[i]);
    if ((byte & 0xc0) != 0x80 ||
        (i == 1 &&
         ((lead == 0xe0 && byte < 0xa0) || (lead == 0xed && byte > 0x9f) ||
          (lead == 0xf0 && byte < 0x90) || (lead == 0xf4 && byte > 0x8f))))
      return {Utf8Status::Invalid, i};
    value = (value << 6) | (byte & 0x3f);
  }
  return {Utf8Status::Complete, length, value};
}

bool isValidUtf8(std::string_view text) {
  while (!text.empty()) {
    auto scalar = readUtf8(text);
    if (scalar.status != Utf8Status::Complete)
      return false;
    text.remove_prefix(scalar.size);
  }
  return true;
}

constexpr auto byteLevelMapping() {
  // ByteLevel maps bytes outside the printable ranges to U+0100 and up.
  std::array<int, 512> mapping{};
  for (auto &entry : mapping)
    entry = -1;
  size_t next = 0x100;
  for (size_t byte = 0; byte < 256; ++byte) {
    bool direct = (byte >= '!' && byte <= '~') ||
                  (byte >= 0xa1 && byte <= 0xff && byte != 0xad);
    mapping[direct ? byte : next++] = static_cast<int>(byte);
  }
  return mapping;
}

void appendByteLevelToken(std::string &bytes, std::string_view token) {
  static constexpr auto mapping = byteLevelMapping();
  size_t start = bytes.size();
  for (auto rest = token; !rest.empty();) {
    auto scalar = readUtf8(rest);
    if (scalar.status != Utf8Status::Complete ||
        scalar.value >= mapping.size() || mapping[scalar.value] < 0) {
      // ByteLevel uses the original UTF-8 for a token with unmapped characters.
      bytes.resize(start);
      bytes.append(token);
      return;
    }
    bytes.push_back(static_cast<char>(mapping[scalar.value]));
    rest.remove_prefix(scalar.size);
  }
}

Result<CacheLineAlignedVector<uint32_t>>
readVocabulary(const nlohmann::json &config) {
  const auto &model = getField(config, "model");
  const auto &type = getField(model, "type");
  if (type != "BPE" && type != "WordLevel")
    return Error("Expected a Hugging Face BPE or WordLevel tokenizer.json");

  const auto &vocab = getField(model, "vocab");
  if (!vocab.is_object() || vocab.empty())
    return Error("Tokenizer vocabulary is missing or empty");
  if (type == "BPE" && !getField(model, "merges").is_array())
    return Error("Tokenizer BPE merges are missing");
  const auto &added = getField(config, "added_tokens");
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
  auto config = parseJson(*blob);
  if (!config)
    return Error(path.string() + ": " + config.getError().getMessage());
  auto ids = readVocabulary(*config);
  if (!ids)
    return Error(path.string() + ": " + ids.getError().getMessage());
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(*blob);
  if (!tokenizer)
    return Error("Could not load tokenizer: " + path.string());
  bool byteLevel =
      getField(getField(*config, "decoder"), "type") == "ByteLevel";
  return TextTokenizer(std::move(tokenizer), std::move(*ids), byteLevel);
}

bool TextTokenizer::contains(uint32_t id) const {
  return std::binary_search(tokenIds.begin(), tokenIds.end(), id);
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
    if (id < 0 || !contains(static_cast<uint32_t>(id)))
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
    if (!contains(id))
      return Error("Token ID is outside the tokenizer vocabulary: " +
                   std::to_string(id));
    encoded.push_back(static_cast<int32_t>(id));
  }
  return tokenizer->Decode(encoded);
}

Result<std::string> TextDecoder::push(uint32_t id) {
  if (!tokenizer.contains(id))
    return Error("Token ID is outside the tokenizer vocabulary: " +
                 std::to_string(id));
  if (tokenizer.byteLevel) {
    appendByteLevelToken(
        pendingBytes, tokenizer.tokenizer->IdToToken(static_cast<int32_t>(id)));
    return drainBytes(false);
  }
  ids.push_back(id);
  return decodePrefix(false);
}

Result<std::string> TextDecoder::finish() {
  if (tokenizer.byteLevel)
    return drainBytes(true);
  return decodePrefix(true);
}

std::string TextDecoder::drainBytes(bool final) {
  std::string chunk;
  std::string_view rest = pendingBytes;
  while (!rest.empty()) {
    auto scalar = readUtf8(rest);
    if (scalar.status == Utf8Status::Incomplete && !final)
      break;
    chunk.append(scalar.status == Utf8Status::Complete
                     ? rest.substr(0, scalar.size)
                     : replacement);
    rest.remove_prefix(scalar.size);
  }
  pendingBytes.erase(0, pendingBytes.size() - rest.size());
  return chunk;
}

Result<std::string> TextDecoder::decodePrefix(bool final) {
  auto text = tokenizer.decode(ids);
  if (!text)
    return text.getError();
  if (text->compare(0, emitted.size(), emitted) != 0)
    return Error("Tokenizer decoder changed text that was already emitted");
  if (!final && text->size() >= replacement.size() &&
      text->compare(text->size() - replacement.size(), replacement.size(),
                    replacement) == 0)
    return std::string{};
  auto chunk = text->substr(emitted.size());
  emitted = std::move(*text);
  return chunk;
}
