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

void convertDense(const SourceTensor &source, const TensorRecord &target,
                  const PayloadSink &sink) {
  unsigned srcSize = getSourceTypeSize(source.type);
  unsigned dstSize = target.spec.dtype == DType::F16 ? 2 : 4;
  uint64_t elements = source.data.size / srcSize;
  uint64_t step = std::min(elements, chunkBytes / std::max(srcSize, dstSize));
  std::vector<uint8_t> input(step * srcSize);
  std::vector<uint8_t> output(step * dstSize);
  for (uint64_t i = 0; i < elements;) {
    uint64_t count = std::min(step, elements - i);
    source.file->read({source.data.offset + i * srcSize, count * srcSize},
                      input.data());
    for (uint64_t j = 0; j < count; ++j) {
      float value = decodeFloat(input.data() + j * srcSize, source.type);
      if (!std::isfinite(value))
        fail("Non-finite weight");
      uint32_t bits;
      if (dstSize == 2) {
        bits = floatToHalf(value);
        if ((bits & 0x7c00) == 0x7c00)
          fail("Weight exceeds FP16 range");
      } else {
        std::memcpy(&bits, &value, sizeof(bits));
      }
      writeLittleEndian(output.data() + j * dstSize, bits, dstSize);
    }
    sink({target.data.offset + i * dstSize, count * dstSize}, output.data());
    i += count;
  }
}

void convertQuantized(const SourceTensor &source, const TensorRecord &target,
                      const PayloadSink &sink) {
  uint64_t rows = source.shape[0];
  uint64_t columns = source.shape[1];
  unsigned elementSize = getSourceTypeSize(source.type);
  uint64_t inputStride = checkedMul(columns, elementSize);
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
    source.file->read(
        {source.data.offset + row * inputStride, count * inputStride},
        input.data());
    for (uint64_t r = 0; r < count; ++r) {
      for (uint64_t group = 0; group < groups; ++group) {
        uint64_t column = group * weightGroupSize;
        unsigned valid = std::min(weightGroupSize, columns - column);
        const auto *groupInput =
            input.data() + r * inputStride + column * elementSize;
        for (unsigned i = 0; i < valid; ++i)
          values[i] = decodeFloat(groupInput + i * elementSize, source.type);
        uint16_t scale = quantizeWeightGroup(
            values.data(), valid, output.data() + r * outputStride + column);
        writeLittleEndian(scales.data() + r * scaleStride + 2 * group, scale,
                          2);
      }
    }
    sink({target.data.offset + row * outputStride, count * outputStride},
         output.data());
    sink({target.scales.offset + row * scaleStride, count * scaleStride},
         scales.data());
    row += count;
  }
}
} // namespace

void llminfer::convertModel(const std::filesystem::path &sourceDirectory,
                            const std::filesystem::path &outputPath,
                            Precision precision) {
  if (outputPath.extension() != ".llmi")
    fail("Output path must end in .llmi");
  auto sourceConfig =
      parseJson(readTextFile(sourceDirectory / "config.json", maxManifestSize));
  auto schema = getModelSchema(makeManifest(sourceConfig, precision, {}));
  Safetensors shards(sourceDirectory);
  const auto &tensors = shards.getTensors();
  std::map<std::string, const SourceTensor *> sourceById;
  for (const auto &expected : schema.tensors) {
    auto source = tensors.find(std::string(hfTextPrefix) + expected.id);
    if (source == tensors.end())
      fail("Missing tensor: " + expected.id);
    if (source->second.shape != expected.shape)
      fail("Unexpected shape for tensor: " + expected.id);
    sourceById.emplace(expected.id, &source->second);
  }
  std::vector<std::string> skippedTensors;
  for (const auto &entry : tensors) {
    std::string_view name = entry.first;
    if (name.substr(0, hfTextPrefix.size()) == hfTextPrefix) {
      if (sourceById.count(std::string(name.substr(hfTextPrefix.size()))) == 0)
        fail("Unexpected text tensor: " + entry.first);
    } else if (name.substr(0, 13) == "model.visual." ||
               name.substr(0, 4) == "mtp.") {
      skippedTensors.push_back(entry.first);
    } else {
      fail("Unexpected checkpoint tensor: " + entry.first);
    }
  }

  auto manifest = makeManifest(sourceConfig, precision, skippedTensors);
  writeModelFile(outputPath, manifest,
                 [&](const TensorRecord &target, const PayloadSink &sink) {
                   const auto &source = *sourceById.at(target.spec.id);
                   try {
                     if (target.layout == Layout::RowMajorI8Group32)
                       convertQuantized(source, target, sink);
                     else
                       convertDense(source, target, sink);
                   } catch (const std::exception &error) {
                     fail(target.spec.id + ": " + error.what());
                   }
                 });
}
