#ifndef LLMINFER_TOOLS_RUN_TEXTTOKENIZER_H
#define LLMINFER_TOOLS_RUN_TEXTTOKENIZER_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tokenizers_cpp.h>

namespace llminfer {

class TextTokenizer {
public:
  static Result<TextTokenizer> open(const std::filesystem::path &path);

  Result<CacheLineAlignedVector<uint32_t>> encode(std::string_view text);
  Result<std::string> decode(const CacheLineAlignedVector<uint32_t> &ids);
  std::optional<uint32_t> findTokenId(const std::string &token);
  const auto &getTokenIds() const { return tokenIds; }

private:
  friend class TextDecoder;

  TextTokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer,
                CacheLineAlignedVector<uint32_t> ids, bool byteLevel)
      : tokenizer(std::move(tokenizer)), tokenIds(std::move(ids)),
        byteLevel(byteLevel) {}

  bool contains(uint32_t id) const;

  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  CacheLineAlignedVector<uint32_t> tokenIds;
  bool byteLevel;
};

class TextDecoder {
public:
  explicit TextDecoder(TextTokenizer &tokenizer) : tokenizer(tokenizer) {}

  /// Returns only the newly available text, which may be empty.
  Result<std::string> push(uint32_t id);
  Result<std::string> finish();

private:
  std::string drainBytes(bool final);
  Result<std::string> decodePrefix(bool final);

  TextTokenizer &tokenizer;
  std::string pendingBytes;
  CacheLineAlignedVector<uint32_t> ids;
  std::string emitted;
};

} // namespace llminfer

#endif
