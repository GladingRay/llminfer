#include "../TokenChunk.h"
#include "ChatTemplate.h"
#include "TextTokenizer.h"
#include "llminfer/Backend/Metal/MetalSession.h"
#include "llminfer/Support/CheckedArithmetic.h"
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>

using namespace llminfer;
using nlohmann::json;

namespace {
void printUsage(std::ostream &out) {
  out << "Usage:\n"
      << "  llmi-run <model.llmi> <token[,token...]>... [--capacity N] "
         "[--logits none|last|all] [--plan]\n"
      << "  llmi-run <model.llmi> --tokenizer <tokenizer.json> --prompt <text> "
         "[--chat [--chat-template PATH]] [--max-new-tokens N] [--capacity N]\n"
      << "  llmi-run encode <tokenizer.json> <text>\n"
      << "  llmi-run decode <tokenizer.json> <token[,token...]>\n";
}

Result<uint64_t> parsePositive(std::string_view text, std::string_view flag) {
  uint64_t value;
  auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value == 0)
    return Error("Expected a positive integer for " + std::string(flag));
  return value;
}

Error tokenize(int argc, char **argv, bool encode) {
  if (argc != 4)
    return Error(encode
                     ? "Expected encode <tokenizer.json> <text>"
                     : "Expected decode <tokenizer.json> <token[,token...]>");
  auto tokenizer = TextTokenizer::open(argv[2]);
  if (!tokenizer)
    return tokenizer.getError();
  if (encode) {
    auto ids = tokenizer->encode(argv[3]);
    if (!ids)
      return ids.getError();
    std::cout << json(*ids).dump() << '\n';
  } else {
    auto ids = parseChunk(argv[3]);
    if (!ids)
      return ids.getError();
    auto text = tokenizer->decode(*ids);
    if (!text)
      return text.getError();
    std::cout << *text << '\n';
  }
  return {};
}

Error generateText(const std::filesystem::path &modelPath,
                   const std::filesystem::path &tokenizerPath,
                   std::string_view prompt, uint64_t maxNewTokens,
                   uint64_t capacity,
                   const std::optional<std::filesystem::path> &chatTemplate) {
  std::optional<ChatPrompt> chat;
  if (chatTemplate) {
    auto rendered = renderChatPrompt(tokenizerPath, *chatTemplate, prompt);
    if (!rendered)
      return rendered.getError();
    chat = std::move(*rendered);
    prompt = chat->text;
  }
  auto tokenizer = TextTokenizer::open(tokenizerPath);
  if (!tokenizer)
    return tokenizer.getError();
  auto input = tokenizer->encode(prompt);
  if (!input)
    return input.getError();
  if (input->empty())
    return Error("The prompt encodes to zero tokens");

  // The last generated token needs no subsequent decode step.
  uint64_t required;
  if (!checkedAdd(input->size(), maxNewTokens - 1, required))
    return Error("Prompt and generation length overflow");
  if (!capacity)
    capacity = required;
  if (capacity < required)
    return Error("Prompt and generation exceed session capacity");

  auto session = MetalSession::open(modelPath, capacity);
  if (!session)
    return session.getError();
  const auto &candidates = tokenizer->getTokenIds();
  if (candidates.back() >= session->getVocabularySize())
    return Error("Tokenizer contains IDs outside the model vocabulary");

  auto eos = session->getEosTokenId();
  auto chatEos = chat ? tokenizer->findTokenId(chat->eosToken) : std::nullopt;
  auto step = session->step(*input);
  TextDecoder decoder(*tokenizer);
  for (uint64_t i = 0; i < maxNewTokens; ++i) {
    if (!step)
      return step.getError();
    std::optional<uint32_t> best;
    for (auto id : candidates) {
      float score = step->logits[id];
      if (std::isfinite(score) && (!best || score > step->logits[*best]))
        best = id;
    }
    if (!best)
      return Error("No finite logits for the tokenizer vocabulary");
    if (best == eos || best == chatEos)
      break;
    auto chunk = decoder.push(*best);
    if (!chunk)
      return chunk.getError();
    if (!chunk->empty())
      std::cout << *chunk << std::flush;
    if (i + 1 < maxNewTokens)
      step = session->step({*best});
  }

  auto text = decoder.finish();
  if (!text)
    return text.getError();
  std::cout << *text << '\n';
  return {};
}

Error run(int argc, char **argv) {
  std::string_view command = argv[1];
  if (command == "encode" || command == "decode")
    return tokenize(argc, argv, command == "encode");

  uint64_t tokens = 0, capacity = 0;
  std::optional<uint64_t> maxNewTokens;
  std::optional<LogitsMode> mode;
  std::optional<std::string_view> prompt;
  const char *tokenizerPath = nullptr;
  const char *chatTemplatePath = nullptr;
  bool tracePlan = false, chat = false;
  CacheLineAlignedVector<CacheLineAlignedVector<uint32_t>> chunks;
  for (int i = 2; i < argc; ++i) {
    std::string_view argument = argv[i];
    if (argument == "--plan") {
      tracePlan = true;
    } else if (argument == "--chat") {
      chat = true;
    } else if (argument == "--logits") {
      if (++i == argc)
        return Error("Expected --logits none|last|all");
      std::string_view value = argv[i];
      if (value == "none")
        mode = LogitsMode::None;
      else if (value == "last")
        mode = LogitsMode::Last;
      else if (value == "all")
        mode = LogitsMode::All;
      else
        return Error("Expected --logits none|last|all");
    } else if (argument == "--capacity" || argument == "--max-new-tokens") {
      if (++i == argc)
        return Error("Missing value for " + std::string(argument));
      auto value = parsePositive(argv[i], argument);
      if (!value)
        return value.getError();
      if (argument == "--capacity")
        capacity = *value;
      else
        maxNewTokens = *value;
    } else if (argument == "--prompt" || argument == "--tokenizer" ||
               argument == "--chat-template") {
      if (++i == argc)
        return Error("Missing value for " + std::string(argument));
      if (argument == "--prompt")
        prompt = argv[i];
      else if (argument == "--tokenizer")
        tokenizerPath = argv[i];
      else
        chatTemplatePath = argv[i];
    } else {
      auto ids = parseChunk(argument);
      if (!ids)
        return ids.getError();
      if (!checkedAdd(tokens, ids->size(), tokens))
        return Error("Token count overflow");
      chunks.push_back(std::move(*ids));
    }
  }
  if (chatTemplatePath && (!chat || !*chatTemplatePath))
    return Error("--chat-template requires --chat and a nonempty path");
  if (prompt || tokenizerPath || maxNewTokens || chat) {
    if (!prompt || !tokenizerPath)
      return Error("Text generation requires --prompt and --tokenizer");
    if (!chunks.empty() || mode || tracePlan)
      return Error("Use token chunks, --logits and --plan in token ID mode");
    std::optional<std::filesystem::path> chatTemplate;
    if (chat)
      chatTemplate.emplace(chatTemplatePath ? chatTemplatePath : "");
    return generateText(argv[1], tokenizerPath, *prompt,
                        maxNewTokens.value_or(16), capacity, chatTemplate);
  }
  if (chunks.empty())
    return Error("Expected at least one token chunk");
  if (!capacity)
    capacity = tokens;
  if (tokens > capacity)
    return Error("Token chunks exceed session capacity");

  auto session = MetalSession::open(argv[1], capacity);
  if (!session)
    return session.getError();
  for (const auto &chunk : chunks) {
    auto result =
        session->step(chunk, mode.value_or(LogitsMode::Last), tracePlan);
    if (!result)
      return result.getError();
    json output{{"past", result->past},
                {"logit_rows", result->logitRows},
                {"scratch_bytes", result->scratchBytes},
                {"dispatch_count", result->dispatchCount},
                {"logits", std::move(result->logits)}};
    if (tracePlan) {
      auto dispatches = json::array();
      for (const auto &dispatch : result->dispatches)
        dispatches.push_back(
            {{"op_index", dispatch.opIndex}, {"kernel", dispatch.kernel}});
      output["dispatches"] = std::move(dispatches);
    }
    std::cout << output.dump() << '\n';
  }
  return {};
}
} // namespace

int main(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    printUsage(std::cout);
    return EXIT_SUCCESS;
  }
  if (argc < 3) {
    printUsage(std::cerr);
    return EXIT_FAILURE;
  }
  if (auto error = run(argc, argv)) {
    std::cerr << "llmi-run: " << error.getMessage() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
