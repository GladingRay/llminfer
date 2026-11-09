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

Result<SourceType> parseType(const json &value) {
  if (value == "BF16")
    return SourceType::BF16;
  if (value == "F16")
    return SourceType::F16;
  if (value == "F32")
    return SourceType::F32;
  return Error("Unsupported safetensors dtype: " + value.dump());
}

Result<SourceTensor> parseTensor(const File &file, const json &entry,
                                 uint64_t dataStart, uint64_t payloadSize) {
  auto type = parseType(getField(entry, "dtype"));
  if (!type)
    return type.getError();
  const auto &dims = getField(entry, "shape");
  if (!dims.is_array() || dims.size() > 8)
    return Error("Invalid safetensors shape");

  CacheLineAlignedVector<uint64_t> shape;
  uint64_t bytes = getSourceTypeSize(*type);
  for (const auto &dim : dims) {
    if (!dim.is_number_unsigned())
      return Error("Expected an unsigned safetensors dimension");
    shape.push_back(dim.get<uint64_t>());
    if (!checkedMul(bytes, shape.back(), bytes))
      return Error("Safetensors tensor size overflow");
  }

  const auto &offsets = getField(entry, "data_offsets");
  if (!offsets.is_array() || offsets.size() != 2 ||
      !offsets[0].is_number_unsigned() || !offsets[1].is_number_unsigned())
    return Error("Invalid safetensors data_offsets");
  uint64_t begin = offsets[0].get<uint64_t>();
  uint64_t end = offsets[1].get<uint64_t>();
  if (begin > end || end > payloadSize || end - begin != bytes)
    return Error("Invalid safetensors data range");
  return SourceTensor{
      &file, *type, std::move(shape), {dataStart + begin, bytes}};
}

Result<uint64_t> readShard(const File &file, const std::string &shardName,
                           const json &weightMap,
                           std::map<std::string, SourceTensor> &tensors) {
  auto size = file.getSize();
  if (!size)
    return size.getError();
  std::array<uint8_t, 8> prefix;
  if (auto error = file.read({0, prefix.size()}, prefix.data()))
    return error;
  uint64_t headerLength = readLittleEndian(prefix.data(), prefix.size());
  if (*size < prefix.size() || headerLength == 0 ||
      headerLength > maxSafetensorsHeader ||
      headerLength > *size - prefix.size())
    return Error("Invalid safetensors header length in " + shardName);

  std::string header(headerLength, '\0');
  if (auto error = file.read({prefix.size(), headerLength}, header.data()))
    return error;
  if (header.front() != '{')
    return Error("Invalid safetensors header in " + shardName);
  auto entries = parseJson(header);
  if (!entries)
    return Error(shardName + ": " + entries.getError().getMessage());
  if (!entries->is_object())
    return Error("Expected a safetensors header object");

  uint64_t dataStart = prefix.size() + headerLength;
  CacheLineAlignedVector<ByteRange> ranges;
  for (auto it = entries->begin(); it != entries->end(); ++it) {
    if (it.key() == "__metadata__") {
      if (!it->is_object() ||
          !std::all_of(it->begin(), it->end(),
                       [](const json &value) { return value.is_string(); }))
        return Error("Invalid safetensors __metadata__");
      continue;
    }
    auto mapping = weightMap.find(it.key());
    if (mapping == weightMap.end() || *mapping != shardName)
      return Error("Tensor not indexed in its shard: " + it.key());
    auto tensor = parseTensor(file, *it, dataStart, *size - dataStart);
    if (!tensor)
      return Error(it.key() + ": " + tensor.getError().getMessage());
    ranges.push_back(tensor->data);
    if (!tensors.emplace(it.key(), std::move(*tensor)).second)
      return Error("Duplicate safetensors tensor: " + it.key());
  }
  std::sort(ranges.begin(), ranges.end(), [](ByteRange lhs, ByteRange rhs) {
    return std::tie(lhs.offset, lhs.size) < std::tie(rhs.offset, rhs.size);
  });
  uint64_t cursor = dataStart;
  for (auto range : ranges) {
    if (range.offset != cursor)
      return Error("Overlapping or non-contiguous safetensors ranges in " +
                   shardName);
    cursor += range.size;
  }
  if (cursor != *size)
    return Error("Incomplete safetensors payload coverage in " + shardName);
  return *size - dataStart;
}
} // namespace

unsigned llminfer::getSourceTypeSize(SourceType type) {
  return type == SourceType::F32 ? 4 : 2;
}

Result<Safetensors> Safetensors::open(const std::filesystem::path &directory) {
  auto indexText =
      readTextFile(directory / "model.safetensors.index.json", maxManifestSize);
  if (!indexText)
    return indexText.getError();
  auto index = parseJson(*indexText);
  if (!index)
    return Error("Safetensors index: " + index.getError().getMessage());
  const auto &weightMap = getField(*index, "weight_map");
  if (!weightMap.is_object() || weightMap.empty())
    return Error("Invalid safetensors weight_map");

  std::set<std::string> shardNames;
  for (const auto &value : weightMap) {
    const auto *name = value.get_ptr<const std::string *>();
    if (!name || name->find('\0') != std::string::npos)
      return Error("Invalid safetensors shard filename");
    std::filesystem::path shard(*name);
    if (shard.filename() != shard || shard.extension() != ".safetensors")
      return Error("Invalid safetensors shard filename");
    shardNames.insert(*name);
  }

  Safetensors result;
  uint64_t totalSize = 0;
  for (const auto &name : shardNames) {
    auto file = File::open(directory / name);
    if (!file)
      return file.getError();
    auto shard = std::make_unique<File>(std::move(*file));
    auto shardSize = readShard(*shard, name, weightMap, result.tensors);
    if (!shardSize)
      return shardSize.getError();
    if (!checkedAdd(totalSize, *shardSize, totalSize))
      return Error("Safetensors total size overflow");
    result.shards.push_back(std::move(shard));
  }
  if (result.tensors.size() != weightMap.size())
    return Error("Safetensors index contains missing tensors");
  const auto &metadata = getField(*index, "metadata");
  if (metadata.contains("total_size")) {
    const auto &expectedSize = getField(metadata, "total_size");
    if (!expectedSize.is_number_unsigned() ||
        expectedSize.get<uint64_t>() != totalSize)
      return Error("Safetensors index total_size differs from shard payloads");
  }
  return result;
}
