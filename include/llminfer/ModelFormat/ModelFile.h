#ifndef LLMINFER_MODELFORMAT_MODELFILE_H
#define LLMINFER_MODELFORMAT_MODELFILE_H

#include "llminfer/Support/CacheLineAlignedVector.h"
#include "llminfer/Support/Result.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace llminfer {

enum class DType : uint32_t { I8 = 1, F16, F32 };
enum class Layout : uint32_t { Dense = 1, RowMajorI8Group32 };
enum class Precision { W8A8, F16 };

struct ByteRange {
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct TensorSpec {
  std::string id;
  CacheLineAlignedVector<uint64_t> shape;
  DType dtype;
};

struct TensorRecord {
  TensorSpec spec;
  Layout layout;
  CacheLineAlignedVector<uint64_t> storageShape;
  ByteRange data;
  ByteRange scales;
};

struct RowSlice {
  uint64_t firstRow;
  uint64_t rowCount;
  ByteRange data;
  ByteRange scales;
};

std::string_view getDTypeName(DType dtype);
std::string_view getLayoutName(Layout layout);

/// Splits a rank-two tensor into row ranges whose buffers fit independently.
Result<CacheLineAlignedVector<RowSlice>> splitRows(const TensorRecord &tensor,
                                                   uint64_t maxBufferLength);

class ModelFile {
public:
  /// Validates metadata, tensor layout and the checksum before returning.
  static Result<ModelFile> open(const std::filesystem::path &path);
  ~ModelFile();
  ModelFile(ModelFile &&) noexcept;
  ModelFile &operator=(ModelFile &&) noexcept;

  const CacheLineAlignedVector<TensorRecord> &getTensors() const;
  const TensorRecord *getTensor(std::string_view id) const;
  Precision getPrecision() const;
  const std::string &getManifest() const;
  uint64_t getFileSize() const;
  uint32_t getChecksum() const;

  /// Reads a validated file range into caller-owned storage.
  Error read(ByteRange range, void *destination) const;

private:
  struct Impl;
  explicit ModelFile(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl;
};

/// The callback writes each data and scale range in order, in one or more
/// chunks.
using PayloadSink = std::function<Error(ByteRange, const void *)>;
using TensorWriter =
    std::function<Error(const TensorRecord &, const PayloadSink &)>;

/// Writes and validates a temporary model before publishing it atomically.
Error writeModelFile(const std::filesystem::path &path,
                     std::string_view manifest,
                     const TensorWriter &writeTensor);

} // namespace llminfer

#endif
