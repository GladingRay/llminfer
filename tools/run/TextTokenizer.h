#ifndef LLMINFER_TOOLS_RUN_TEXTTOKENIZER_H
#define LLMINFER_TOOLS_RUN_TEXTTOKENIZER_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <tokenizers_cpp.h>

namespace llminfer {

class TextTokenizer {
public:
  static Result<TextTokenizer> open(const std::filesystem::path &path);

  Result<CacheLineAlignedVector<uint32_t>> encode(std::string_view text);
  Result<std::string> decode(const CacheLineAlignedVector<uint32_t> &ids);
  const auto &getTokenIds() const { return tokenIds; }

private:
  TextTokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer,
                CacheLineAlignedVector<uint32_t> ids)
      : tokenizer(std::move(tokenizer)), tokenIds(std::move(ids)) {}

  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
  CacheLineAlignedVector<uint32_t> tokenIds;
};

} // namespace llminfer

#endif
