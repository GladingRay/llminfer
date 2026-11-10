#ifndef LLMINFER_TOOLS_RUN_CHATTEMPLATE_H
#define LLMINFER_TOOLS_RUN_CHATTEMPLATE_H

#include "llminfer/Support/Result.h"
#include <filesystem>
#include <string_view>

namespace llminfer {

struct ChatPrompt {
  std::string text;
  std::string eosToken;
};

Result<ChatPrompt> renderChatPrompt(const std::filesystem::path &tokenizer,
                                    const std::filesystem::path &overridePath,
                                    std::string_view prompt);

} // namespace llminfer

#endif
