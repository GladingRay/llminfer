#include "Safetensors.h"
#include "Qwen35Schema.h"
#include <algorithm>
#include <array>
#include <set>
#include <tuple>

using namespace llminfer;
using nlohmann::json;

namespace {
constexpr uint64_t maxSafetensorsHeader = 100'000'000;

SourceType parseType(const json &value) {
  if (value == "BF16")
    return SourceType::BF16;
  if (value == "F16")
    return SourceType::F16;
  if (value == "F32")
    return SourceType::F32;
  fail("Unsupported safetensors dtype: " + value.dump());
}

uint64_t parseUnsigned(const json &value) {
  if (!value.is_number_unsigned())
    fail("Expected an unsigned safetensors dimension or offset");
  return value.get<uint64_t>();
}

SourceTensor parseTensor(const File &file, const json &entry,
                         uint64_t dataStart, uint64_t payloadSize) {
  SourceType type = parseType(entry.at("dtype"));
  const auto &dims = entry.at("shape");
  if (!dims.is_array() || dims.size() > 8)
    fail("Invalid safetensors shape");
  std::vector<uint64_t> shape;
  uint64_t bytes = getSourceTypeSize(type);
  for (const auto &dim : dims) {
    shape.push_back(parseUnsigned(dim));
    bytes = checkedMul(bytes, shape.back());
  }
  const auto &offsets = entry.at("data_offsets");
  if (!offsets.is_array() || offsets.size() != 2)
    fail("Invalid safetensors data_offsets");
  uint64_t begin = parseUnsigned(offsets[0]);
  uint64_t end = parseUnsigned(offsets[1]);
  if (begin > end || end > payloadSize || end - begin != bytes)
    fail("Invalid safetensors data range");
  return {&file, type, std::move(shape), {checkedAdd(dataStart, begin), bytes}};
}

uint64_t readShard(const File &file, const std::string &shardName,
                   const json &weightMap,
                   std::map<std::string, SourceTensor> &tensors) {
  uint64_t size = file.getSize();
  std::array<uint8_t, 8> prefix;
  file.read({0, prefix.size()}, prefix.data());
  uint64_t headerLength = readLittleEndian(prefix.data(), prefix.size());
  if (headerLength == 0 || headerLength > maxSafetensorsHeader ||
      headerLength > size - prefix.size())
    fail("Invalid safetensors header length in " + shardName);

  std::string header(headerLength, '\0');
  file.read({prefix.size(), headerLength}, header.data());
  if (header.front() != '{')
    fail("Invalid safetensors header in " + shardName);
  auto entries = parseJson(header);
  if (!entries.is_object())
    fail("Expected a safetensors header object");

  uint64_t dataStart = prefix.size() + headerLength;
  std::vector<ByteRange> ranges;
  for (auto it = entries.begin(); it != entries.end(); ++it) {
    if (it.key() == "__metadata__") {
      if (!it->is_object() ||
          !std::all_of(it->begin(), it->end(),
                       [](const json &value) { return value.is_string(); }))
        fail("Invalid safetensors __metadata__");
      continue;
    }
    auto mapping = weightMap.find(it.key());
    if (mapping == weightMap.end() || *mapping != shardName)
      fail("Tensor not indexed in its shard: " + it.key());
    try {
      auto tensor = parseTensor(file, *it, dataStart, size - dataStart);
      ranges.push_back(tensor.data);
      if (!tensors.emplace(it.key(), std::move(tensor)).second)
        fail("Duplicate safetensors tensor");
    } catch (const std::exception &error) {
      fail(it.key() + ": " + error.what());
    }
  }
  std::sort(ranges.begin(), ranges.end(), [](ByteRange lhs, ByteRange rhs) {
    return std::tie(lhs.offset, lhs.size) < std::tie(rhs.offset, rhs.size);
  });
  uint64_t cursor = dataStart;
  for (auto range : ranges) {
    if (range.offset != cursor)
      fail("Overlapping or non-contiguous safetensors ranges in " + shardName);
    cursor += range.size;
  }
  if (cursor != size)
    fail("Incomplete safetensors payload coverage in " + shardName);
  return size - dataStart;
}
} // namespace

unsigned llminfer::getSourceTypeSize(SourceType type) {
  return type == SourceType::F32 ? 4 : 2;
}

Safetensors::Safetensors(const std::filesystem::path &directory) {
  auto index = parseJson(readTextFile(
      directory / "model.safetensors.index.json", maxManifestSize));
  const auto &weightMap = index.at("weight_map");
  if (!weightMap.is_object() || weightMap.empty())
    fail("Invalid safetensors weight_map");

  std::set<std::string> shardNames;
  for (const auto &value : weightMap) {
    auto name = value.get<std::string>();
    std::filesystem::path shard(name);
    if (name.find('\0') != std::string::npos || shard.filename() != shard ||
        shard.extension() != ".safetensors")
      fail("Invalid safetensors shard filename");
    shardNames.insert(std::move(name));
  }
  uint64_t totalSize = 0;
  for (const auto &name : shardNames) {
    auto file = std::make_unique<File>(directory / name);
    totalSize =
        checkedAdd(totalSize, readShard(*file, name, weightMap, tensors));
    shards.push_back(std::move(file));
  }
  if (tensors.size() != weightMap.size())
    fail("Safetensors index contains missing tensors");
  auto metadata = index.find("metadata");
  if (metadata != index.end() && metadata->contains("total_size") &&
      parseUnsigned(metadata->at("total_size")) != totalSize)
    fail("Safetensors index total_size differs from shard payloads");
}
