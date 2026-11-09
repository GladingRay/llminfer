#ifndef LLMINFER_MODELCONVERTER_SAFETENSORS_H
#define LLMINFER_MODELCONVERTER_SAFETENSORS_H

#include "File.h"
#include <map>
#include <memory>

namespace llminfer {

enum class SourceType { BF16, F16, F32 };

struct SourceTensor {
  const File *file;
  SourceType type;
  std::vector<uint64_t> shape;
  ByteRange data;
};

/// Owns all shard descriptors for the lifetime of the tensor views.
class Safetensors {
public:
  explicit Safetensors(const std::filesystem::path &directory);

  const std::map<std::string, SourceTensor> &getTensors() const {
    return tensors;
  }

private:
  std::vector<std::unique_ptr<File>> shards;
  std::map<std::string, SourceTensor> tensors;
};

unsigned getSourceTypeSize(SourceType type);

} // namespace llminfer

#endif
