#include "ChatTemplate.h"
#include "File.h"
#include "Qwen35Schema.h"

using namespace llminfer;

namespace {
constexpr uint64_t maxTemplateBytes = 1 << 20;

struct Input {
  const char *data;
  size_t size;
};

struct ChatInput {
  Input source;
  Input prompt;
  Input bosToken;
  Input eosToken;
  uint8_t enableThinking;
};

extern "C" {
using OutputCallback = void (*)(void *, const char *, size_t);
int32_t llmi_render_chat_template(const ChatInput *input, void *context,
                                  OutputCallback output);
static void storeOutput(void *context, const char *data, size_t size) {
  static_cast<std::string *>(context)->assign(data, size);
}
}

Result<bool> checkFileExists(const std::filesystem::path &path) {
  std::error_code error;
  bool present = std::filesystem::exists(path, error);
  if (error)
    return Error(path.string() + ": " + error.message());
  return present;
}
} // namespace

Result<ChatPrompt>
llminfer::renderChatPrompt(const std::filesystem::path &tokenizer,
                           const std::filesystem::path &overridePath,
                           std::string_view prompt, bool enableThinking) {
  auto configPath = tokenizer.parent_path() / "tokenizer_config.json";
  auto hasConfig = checkFileExists(configPath);
  if (!hasConfig)
    return hasConfig.getError();
  nlohmann::json config;
  if (*hasConfig) {
    auto blob = readTextFile(configPath, maxTemplateBytes);
    if (!blob)
      return blob.getError();
    auto parsed = parseJson(*blob);
    if (!parsed)
      return Error(configPath.string() + ": " + parsed.getError().getMessage());
    config = std::move(*parsed);
  }

  auto path = overridePath;
  if (path.empty()) {
    path = tokenizer.parent_path() / "chat_template.jinja";
    auto hasTemplate = checkFileExists(path);
    if (!hasTemplate)
      return hasTemplate.getError();
    if (!*hasTemplate)
      path.clear();
  }
  std::string source;
  if (!path.empty()) {
    auto blob = readTextFile(path, maxTemplateBytes);
    if (!blob)
      return blob.getError();
    source = std::move(*blob);
  } else {
    path = configPath;
    const auto &field = getField(config, "chat_template");
    if (!field.is_string())
      return Error("Chat template not found next to tokenizer.json. Specify "
                   "--chat-template PATH or provide chat_template.jinja");
    source = field.get<std::string>();
  }
  if (source.empty())
    return Error(path.string() + ": empty chat template");

  const auto &bos = getField(config, "bos_token");
  const auto &eos = getField(config, "eos_token");
  std::string bosToken = bos.is_string() ? bos.get<std::string>() : "";
  std::string eosToken =
      eos.is_string() ? eos.get<std::string>() : "<|im_end|>";
  ChatInput input{{source.data(), source.size()},
                  {prompt.data(), prompt.size()},
                  {bosToken.data(), bosToken.size()},
                  {eosToken.data(), eosToken.size()},
                  static_cast<uint8_t>(enableThinking)};
  std::string output;
  if (llmi_render_chat_template(&input, &output, storeOutput) != 0)
    return Error(path.string() + ": " + output);
  return ChatPrompt{std::move(output), std::move(eosToken)};
}
