#include "llminfer/ModelFormat/ModelFile.h"
#include "File.h"
#include "Qwen35Schema.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <zlib.h>

using namespace llminfer;

namespace {
constexpr std::array<uint8_t, 8> magic = {'L',  'L',  'M',  'I',
                                          '\r', '\n', 0x1a, '\n'};
constexpr uint32_t formatVersion = 1;
constexpr uint32_t qwen35TextArchitecture = 1;
constexpr uint64_t headerSize = 72;
constexpr uint64_t entrySize = 192;
constexpr uint64_t idSize = 96;
constexpr uint64_t alignment = 64;

using Header = std::array<uint8_t, headerSize>;
using Entry = std::array<uint8_t, entrySize>;

uint64_t dtypeSize(DType dtype) {
  switch (dtype) {
  case DType::I8:
    return 1;
  case DType::F16:
    return 2;
  case DType::F32:
    return 4;
  }
  return 0;
}

struct FilePlan {
  ByteRange manifest;
  ByteRange directory;
  ByteRange payload;
  CacheLineAlignedVector<TensorRecord> tensors;

  uint64_t size() const { return payload.offset + payload.size; }
};

bool reserveRange(uint64_t &cursor, uint64_t bytes, ByteRange &range) {
  if (!alignTo(cursor, alignment, range.offset) ||
      !checkedAdd(range.offset, bytes, cursor))
    return false;
  range.size = bytes;
  return true;
}

Result<FilePlan> planFile(uint64_t manifestSize,
                          const CacheLineAlignedVector<TensorSpec> &specs) {
  if (manifestSize == 0 || manifestSize > maxManifestSize)
    return Error("Invalid .llmi manifest size");
  FilePlan plan;
  plan.manifest = {headerSize, manifestSize};
  uint64_t cursor = headerSize + manifestSize;
  uint64_t directoryBytes;
  if (!checkedMul(specs.size(), entrySize, directoryBytes) ||
      !reserveRange(cursor, directoryBytes, plan.directory) ||
      !alignTo(cursor, alignment, plan.payload.offset))
    return Error(".llmi directory size overflow");

  for (const auto &spec : specs) {
    TensorRecord tensor{spec, Layout::Dense, spec.shape, {}, {}};
    if (spec.dtype == DType::I8) {
      tensor.layout = Layout::RowMajorI8Group32;
      if (!alignTo(spec.shape[1], weightGroupSize, tensor.storageShape[1]))
        return Error("Padded tensor width overflow: " + spec.id);
    }
    uint64_t bytes = dtypeSize(spec.dtype);
    for (uint64_t dim : tensor.storageShape)
      if (!checkedMul(bytes, dim, bytes))
        return Error("Tensor size overflow: " + spec.id);
    if (!reserveRange(cursor, bytes, tensor.data))
      return Error("Tensor offset overflow: " + spec.id);
    if (tensor.layout == Layout::RowMajorI8Group32) {
      uint64_t scaleStride = 2 * (tensor.storageShape[1] / weightGroupSize);
      uint64_t scaleBytes;
      if (!checkedMul(spec.shape[0], scaleStride, scaleBytes) ||
          !reserveRange(cursor, scaleBytes, tensor.scales))
        return Error("Tensor scale range overflow: " + spec.id);
    }
    plan.tensors.push_back(std::move(tensor));
  }

  plan.payload.size = cursor - plan.payload.offset;
  return plan;
}

Header encodeHeader(const FilePlan &plan, uint32_t checksum) {
  Header header{};
  std::copy(magic.begin(), magic.end(), header.begin());
  writeLittleEndian(header.data() + 8, formatVersion, 4);
  writeLittleEndian(header.data() + 12, qwen35TextArchitecture, 4);
  writeLittleEndian(header.data() + 16, plan.manifest.offset, 8);
  writeLittleEndian(header.data() + 24, plan.manifest.size, 8);
  writeLittleEndian(header.data() + 32, plan.directory.offset, 8);
  writeLittleEndian(header.data() + 40, plan.tensors.size(), 8);
  writeLittleEndian(header.data() + 48, plan.payload.offset, 8);
  writeLittleEndian(header.data() + 56, plan.payload.size, 8);
  writeLittleEndian(header.data() + 64, checksum, 4);
  return header;
}

Entry encodeEntry(const TensorRecord &tensor) {
  Entry entry{};
  assert(tensor.spec.id.size() < idSize && tensor.spec.shape.size() <= 3);
  std::copy(tensor.spec.id.begin(), tensor.spec.id.end(), entry.begin());
  writeLittleEndian(entry.data() + idSize,
                    static_cast<uint32_t>(tensor.spec.dtype), 4);
  writeLittleEndian(entry.data() + 100, static_cast<uint32_t>(tensor.layout),
                    4);
  writeLittleEndian(entry.data() + 104, tensor.spec.shape.size(), 4);
  for (size_t i = 0; i < tensor.spec.shape.size(); ++i) {
    writeLittleEndian(entry.data() + 112 + 8 * i, tensor.spec.shape[i], 8);
    writeLittleEndian(entry.data() + 136 + 8 * i, tensor.storageShape[i], 8);
  }
  writeLittleEndian(entry.data() + 160, tensor.data.offset, 8);
  writeLittleEndian(entry.data() + 168, tensor.data.size, 8);
  writeLittleEndian(entry.data() + 176, tensor.scales.offset, 8);
  writeLittleEndian(entry.data() + 184, tensor.scales.size, 8);
  return entry;
}

Result<uint32_t> checksumFile(const File &file, uint64_t size) {
  CacheLineAlignedVector<uint8_t> buffer(1 << 20);
  uLong checksum = crc32(0, Z_NULL, 0);
  for (uint64_t offset = headerSize; offset < size;) {
    auto count =
        static_cast<uInt>(std::min<uint64_t>(size - offset, buffer.size()));
    if (auto error = file.read({offset, count}, buffer.data()))
      return error;
    checksum = crc32(checksum, buffer.data(), count);
    offset += count;
  }
  return static_cast<uint32_t>(checksum);
}

bool contains(ByteRange outer, ByteRange inner) {
  if (inner.offset < outer.offset)
    return false;
  uint64_t relative = inner.offset - outer.offset;
  return relative <= outer.size && inner.size <= outer.size - relative;
}
struct TemporaryPath {
  std::string path;
  bool removeOnExit = false;

  ~TemporaryPath() {
    if (removeOnExit)
      ::unlink(path.c_str());
  }
};
} // namespace

std::string_view llminfer::getDTypeName(DType dtype) {
  switch (dtype) {
  case DType::I8:
    return "I8";
  case DType::F16:
    return "F16";
  case DType::F32:
    return "F32";
  }
  return "Unknown";
}

std::string_view llminfer::getLayoutName(Layout layout) {
  switch (layout) {
  case Layout::Dense:
    return "Dense";
  case Layout::RowMajorI8Group32:
    return "RowMajorI8Group32";
  }
  return "Unknown";
}

Result<CacheLineAlignedVector<RowSlice>>
llminfer::splitRows(const TensorRecord &tensor, uint64_t maxBufferLength) {
  if (tensor.spec.shape.size() != 2 || tensor.spec.shape[0] == 0)
    return Error("Row slicing requires a nonempty rank-two tensor");
  uint64_t rows = tensor.spec.shape[0];
  if (tensor.data.size == 0 || tensor.data.size % rows ||
      tensor.scales.size % rows)
    return Error("Invalid tensor row storage");
  uint64_t dataStride = tensor.data.size / rows;
  uint64_t scaleStride = tensor.scales.size / rows;
  uint64_t rowsPerSlice = maxBufferLength / std::max(dataStride, scaleStride);
  if (rowsPerSlice == 0)
    return Error("Tensor row exceeds maximum buffer length");

  if (tensor.data.size > UINT64_MAX - tensor.data.offset ||
      tensor.scales.size > UINT64_MAX - tensor.scales.offset)
    return Error("Tensor row range overflow");
  CacheLineAlignedVector<RowSlice> slices;
  for (uint64_t first = 0; first < rows;) {
    uint64_t count = std::min(rowsPerSlice, rows - first);
    ByteRange data{tensor.data.offset + first * dataStride, count * dataStride};
    ByteRange scales;
    if (scaleStride)
      scales = {tensor.scales.offset + first * scaleStride,
                count * scaleStride};
    slices.push_back({first, count, data, scales});
    first += count;
  }
  return slices;
}

struct ModelFile::Impl {
  explicit Impl(File file) : file(std::move(file)) {}

  Error validate() {
    auto fileSize = file.getSize();
    if (!fileSize)
      return fileSize.getError();
    size = *fileSize;
    Header header;
    if (auto error = file.read({0, header.size()}, header.data()))
      return error;
    if (!std::equal(magic.begin(), magic.end(), header.begin()))
      return Error("Invalid .llmi magic");
    if (readLittleEndian(header.data() + 8, 4) != formatVersion)
      return Error("Unsupported .llmi version");
    if (readLittleEndian(header.data() + 12, 4) != qwen35TextArchitecture)
      return Error("Unsupported .llmi architecture");
    uint64_t manifestSize = readLittleEndian(header.data() + 24, 8);
    if (manifestSize == 0 || manifestSize > maxManifestSize ||
        !contains({0, size}, {headerSize, manifestSize}))
      return Error("Invalid .llmi manifest range");
    manifest.resize(manifestSize);
    if (auto error = file.read({headerSize, manifestSize}, manifest.data()))
      return error;

    auto schema = getModelSchema(manifest);
    if (!schema)
      return schema.getError();
    precision = schema->precision;
    auto plan = planFile(manifestSize, schema->tensors);
    if (!plan)
      return plan.getError();
    checksum = readLittleEndian(header.data() + 64, 4);
    // The canonical layout verifies all encoded ranges and rejects overlaps.
    if (header != encodeHeader(*plan, checksum) || size != plan->size())
      return Error("Invalid .llmi sections or file length");
    uint64_t offset = plan->directory.offset;
    for (const auto &tensor : plan->tensors) {
      Entry entry;
      if (auto error = file.read({offset, entry.size()}, entry.data()))
        return error;
      if (entry != encodeEntry(tensor))
        return Error("Invalid .llmi directory entry for " + tensor.spec.id);
      offset += entrySize;
    }
    auto actualChecksum = checksumFile(file, size);
    if (!actualChecksum)
      return actualChecksum.getError();
    if (checksum != *actualChecksum)
      return Error("Invalid .llmi checksum");
    tensors = std::move(plan->tensors);
    return {};
  }

  File file;
  uint64_t size;
  uint32_t checksum;
  Precision precision;
  std::string manifest;
  CacheLineAlignedVector<TensorRecord> tensors;
};

ModelFile::ModelFile(std::unique_ptr<Impl> impl) : impl(std::move(impl)) {}

Result<ModelFile> ModelFile::open(const std::filesystem::path &path) {
  auto file = File::open(path);
  if (!file)
    return file.getError();
  auto impl = std::make_unique<Impl>(std::move(*file));
  if (auto error = impl->validate())
    return error;
  return ModelFile(std::move(impl));
}

ModelFile::~ModelFile() = default;
ModelFile::ModelFile(ModelFile &&) noexcept = default;
ModelFile &ModelFile::operator=(ModelFile &&) noexcept = default;

const CacheLineAlignedVector<TensorRecord> &ModelFile::getTensors() const {
  return impl->tensors;
}

const TensorRecord *ModelFile::getTensor(std::string_view id) const {
  if (id == headId)
    id = embeddingId;
  auto it =
      std::lower_bound(impl->tensors.begin(), impl->tensors.end(), id,
                       [](const TensorRecord &tensor, std::string_view id) {
                         return tensor.spec.id < id;
                       });
  if (it == impl->tensors.end() || it->spec.id != id)
    return nullptr;
  return &*it;
}

Precision ModelFile::getPrecision() const { return impl->precision; }
const std::string &ModelFile::getManifest() const { return impl->manifest; }
uint64_t ModelFile::getFileSize() const { return impl->size; }
uint32_t ModelFile::getChecksum() const { return impl->checksum; }

Error ModelFile::read(ByteRange range, void *destination) const {
  if (!contains({0, impl->size}, range))
    return Error("Read is outside the model file");
  return impl->file.read(range, destination);
}

Error llminfer::writeModelFile(const std::filesystem::path &path,
                               std::string_view manifest,
                               const TensorWriter &writeTensor) {
  auto schema = getModelSchema(manifest);
  if (!schema)
    return schema.getError();
  auto plan = planFile(manifest.size(), schema->tensors);
  if (!plan)
    return plan.getError();

  TemporaryPath temporary{path.string() + ".tmp.XXXXXX"};
  int fd = ::mkstemp(temporary.path.data());
  if (fd == -1)
    return fileError("Creating temporary model", errno);
  temporary.removeOnExit = true;
  File file(fd);

  if (auto error = file.resize(plan->size()))
    return error;
  if (auto error = file.write(plan->manifest, manifest.data()))
    return error;
  uint64_t offset = plan->directory.offset;
  for (const auto &tensor : plan->tensors) {
    Entry entry = encodeEntry(tensor);
    if (auto error = file.write({offset, entry.size()}, entry.data()))
      return error;
    offset += entrySize;
    ByteRange remainingData = tensor.data;
    ByteRange remainingScales = tensor.scales;
    auto error =
        writeTensor(tensor, [&](ByteRange chunk, const void *bytes) -> Error {
          ByteRange &remaining =
              chunk.offset == remainingData.offset && remainingData.size != 0
                  ? remainingData
                  : remainingScales;
          if (chunk.size == 0 || chunk.offset != remaining.offset ||
              chunk.size > remaining.size)
            return Error("Unexpected tensor write for " + tensor.spec.id);
          if (auto error = file.write(chunk, bytes))
            return error;
          remaining.offset += chunk.size;
          remaining.size -= chunk.size;
          return {};
        });
    if (error)
      return error;
    if (remainingData.size != 0 || remainingScales.size != 0)
      return Error("Incomplete tensor write for " + tensor.spec.id);
  }

  auto checksum = checksumFile(file, plan->size());
  if (!checksum)
    return checksum.getError();
  Header header = encodeHeader(*plan, *checksum);
  if (auto error = file.write({0, header.size()}, header.data()))
    return error;
  if (auto error = file.sync())
    return error;
  auto validated = ModelFile::open(temporary.path);
  if (!validated)
    return validated.getError();

  if (::rename(temporary.path.c_str(), path.c_str()) == -1)
    return fileError("Publishing model", errno);
  temporary.removeOnExit = false;
  return {};
}
