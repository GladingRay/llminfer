#include "Converter.h"
#include "File.h"
#include "Qwen35Schema.h"
#include "Safetensors.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

using namespace llminfer;

namespace {
constexpr uint64_t chunkBytes = 1 << 20;

float decodeFloat(const uint8_t *data, SourceType type) {
  uint32_t bits = data[0] | (uint32_t(data[1]) << 8);
  if (type == SourceType::F16)
    return halfToFloat(bits);
  if (type == SourceType::BF16)
    bits <<= 16;
  else
    bits |= (uint32_t(data[2]) << 16) | (uint32_t(data[3]) << 24);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Error convertDense(const SourceTensor &source, const TensorRecord &target,
                   const PayloadSink &sink) {
  unsigned srcSize = getSourceTypeSize(source.type);
  unsigned dstSize = target.spec.dtype == DType::F16 ? 2 : 4;
  uint64_t elements = source.data.size / srcSize;
  uint64_t step = std::min(elements, chunkBytes / std::max(srcSize, dstSize));
  std::vector<uint8_t> input(step * srcSize);
  std::vector<uint8_t> output(step * dstSize);
  for (uint64_t i = 0; i < elements;) {
    uint64_t count = std::min(step, elements - i);
    if (auto error = source.file->read(
            {source.data.offset + i * srcSize, count * srcSize}, input.data()))
      return error;
    for (uint64_t j = 0; j < count; ++j) {
      float value = decodeFloat(input.data() + j * srcSize, source.type);
      if (!std::isfinite(value))
        return Error("Non-finite weight");
      uint32_t bits;
      if (dstSize == 2) {
        bits = floatToHalf(value);
        if ((bits & 0x7c00) == 0x7c00)
          return Error("Weight exceeds FP16 range");
      } else {
        std::memcpy(&bits, &value, sizeof(bits));
      }
      writeLittleEndian(output.data() + j * dstSize, bits, dstSize);
    }
    if (auto error = sink({target.data.offset + i * dstSize, count * dstSize},
                          output.data()))
      return error;
    i += count;
  }
  return {};
}

Error convertQuantized(const SourceTensor &source, const TensorRecord &target,
                       const PayloadSink &sink) {
  uint64_t rows = source.shape[0];
  uint64_t columns = source.shape[1];
  unsigned elementSize = getSourceTypeSize(source.type);
  uint64_t inputStride = source.data.size / rows;
  uint64_t outputStride = target.storageShape[1];
  uint64_t groups = outputStride / weightGroupSize;
  uint64_t scaleStride = 2 * groups;
  uint64_t rowsPerChunk =
      std::min(rows, std::max<uint64_t>(1, chunkBytes / inputStride));
  std::vector<uint8_t> input(rowsPerChunk * inputStride);
  std::vector<int8_t> output(rowsPerChunk * outputStride);
  std::vector<uint8_t> scales(rowsPerChunk * scaleStride);
  std::array<float, weightGroupSize> values;

  for (uint64_t row = 0; row < rows;) {
    uint64_t count = std::min(rowsPerChunk, rows - row);
    if (auto error = source.file->read(
            {source.data.offset + row * inputStride, count * inputStride},
            input.data()))
      return error;
    for (uint64_t r = 0; r < count; ++r) {
      for (uint64_t group = 0; group < groups; ++group) {
        uint64_t column = group * weightGroupSize;
        unsigned valid = std::min(weightGroupSize, columns - column);
        const auto *groupInput =
            input.data() + r * inputStride + column * elementSize;
        for (unsigned i = 0; i < valid; ++i)
          values[i] = decodeFloat(groupInput + i * elementSize, source.type);
        auto scale = quantizeWeightGroup(
            values.data(), valid, output.data() + r * outputStride + column);
        if (!scale)
          return scale.getError();
        writeLittleEndian(scales.data() + r * scaleStride + 2 * group, *scale,
                          2);
      }
    }
    if (auto error = sink(
            {target.data.offset + row * outputStride, count * outputStride},
            output.data()))
      return error;
    if (auto error = sink(
            {target.scales.offset + row * scaleStride, count * scaleStride},
            scales.data()))
      return error;
    row += count;
  }
  return {};
}
} // namespace

Error llminfer::convertModel(const std::filesystem::path &sourceDirectory,
                             const std::filesystem::path &outputPath,
                             Precision precision) {
  if (outputPath.extension() != ".llmi")
    return Error("Output path must end in .llmi");
  auto configText =
      readTextFile(sourceDirectory / "config.json", maxManifestSize);
  if (!configText)
    return configText.getError();
  auto sourceConfig = parseJson(*configText);
  if (!sourceConfig)
    return Error("config.json: " + sourceConfig.getError().getMessage());
  auto schema = getModelSchema(makeManifest(*sourceConfig, precision, {}));
  if (!schema)
    return schema.getError();
  auto shards = Safetensors::open(sourceDirectory);
  if (!shards)
    return shards.getError();
  const auto &tensors = shards->getTensors();
  std::map<std::string, const SourceTensor *> sourceById;
  for (const auto &expected : schema->tensors) {
    auto source = tensors.find(std::string(hfTextPrefix) + expected.id);
    if (source == tensors.end())
      return Error("Missing tensor: " + expected.id);
    if (source->second.shape != expected.shape)
      return Error("Unexpected shape for tensor: " + expected.id);
    sourceById.emplace(expected.id, &source->second);
  }
  std::vector<std::string> skippedTensors;
  for (const auto &entry : tensors) {
    std::string_view name = entry.first;
    if (name.substr(0, hfTextPrefix.size()) == hfTextPrefix) {
      if (sourceById.count(std::string(name.substr(hfTextPrefix.size()))) == 0)
        return Error("Unexpected text tensor: " + entry.first);
    } else if (name.substr(0, 13) == "model.visual." ||
               name.substr(0, 4) == "mtp.") {
      skippedTensors.push_back(entry.first);
    } else {
      return Error("Unexpected checkpoint tensor: " + entry.first);
    }
  }

  auto manifest = makeManifest(*sourceConfig, precision, skippedTensors);
  return writeModelFile(
      outputPath, manifest,
      [&](const TensorRecord &target, const PayloadSink &sink) {
        auto it = sourceById.find(target.spec.id);
        if (it == sourceById.end())
          return Error("Missing tensor: " + target.spec.id);
        const auto &source = *it->second;
        auto error = target.layout == Layout::RowMajorI8Group32
                         ? convertQuantized(source, target, sink)
                         : convertDense(source, target, sink);
        if (error)
          return Error(target.spec.id + ": " + error.getMessage());
        return Error();
      });
}
