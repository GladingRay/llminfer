#ifndef LLMINFER_TOOLS_TOKENCHUNK_H
#define LLMINFER_TOOLS_TOKENCHUNK_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <charconv>
#include <cstdint>
#include <string_view>

namespace llminfer {

inline Result<CacheLineAlignedVector<uint32_t>>
parseChunk(std::string_view text) {
  CacheLineAlignedVector<uint32_t> ids;
  while (true) {
    auto separator = text.find(',');
    auto item = text.substr(0, separator);
    uint32_t id;
    if (item.empty())
      return Error("Expected comma-separated unsigned token IDs");
    auto result = std::from_chars(item.data(), item.data() + item.size(), id);
    if (result.ec != std::errc{} || result.ptr != item.data() + item.size())
      return Error("Invalid token ID: " + std::string(item));
    ids.push_back(id);
    if (separator == std::string_view::npos)
      return ids;
    text.remove_prefix(separator + 1);
  }
}

} // namespace llminfer

#endif
