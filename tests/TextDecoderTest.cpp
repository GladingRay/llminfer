#include "TextTokenizer.h"
#include <cstdlib>
#include <iostream>

using namespace llminfer;

namespace {
struct Chunk {
  uint32_t id;
  std::string_view text;
};

void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void testStream(TextTokenizer &tokenizer) {
  const Chunk chunks[] = {{0, "A"},    {1, " "},   {2, ""},  {3, "你"},
                          {4, ""},     {5, "🌍!"}, {6, "�"}, {7, ""},
                          {8, "�Q"},   {9, "��"},  {10, ""}, {11, "é"},
                          {12, "é🌍"}, {7, ""}};
  TextDecoder decoder(tokenizer);
  CacheLineAlignedVector<uint32_t> ids;
  std::string text;
  for (auto chunk : chunks) {
    auto piece = decoder.push(chunk.id);
    check(piece && *piece == chunk.text, "Unexpected streamed UTF-8 chunk");
    text += *piece;
    ids.push_back(chunk.id);
  }
  auto tail = decoder.finish();
  check(tail && *tail == "�", "Incomplete final scalar was not replaced");
  auto batch = tokenizer.decode(ids);
  check(batch && *batch == text + *tail, "Streaming and batch decoding differ");
}
} // namespace

int main(int argc, char **argv) {
  check(argc == 2, "Expected a tokenizer path");
  auto tokenizer = TextTokenizer::open(argv[1]);
  check(static_cast<bool>(tokenizer), "Failed to open the ByteLevel tokenizer");
  testStream(*tokenizer);
  return EXIT_SUCCESS;
}
