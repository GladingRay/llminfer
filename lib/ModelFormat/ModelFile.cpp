#include "llminfer/ModelFormat/ModelFile.h"
#include "File.h"
#include "Qwen35Schema.h"
#include "llminfer/ModelFormat/Quantization.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  fail("Invalid tensor dtype");
}

struct FilePlan {
  ByteRange manifest;
  ByteRange directory;
  ByteRange payload;
  std::vector<TensorRecord> tensors;

  uint64_t size() const { return checkedAdd(payload.offset, payload.size); }
};

ByteRange reserveRange(uint64_t &cursor, uint64_t bytes) {
  ByteRange range{alignTo(cursor, alignment), bytes};
  cursor = checkedAdd(range.offset, range.size);
  return range;
}

FilePlan planFile(uint64_t manifestSize, const std::vector<TensorSpec> &specs) {
  if (manifestSize == 0 || manifestSize > maxManifestSize)
    fail("Invalid .llmi manifest size");
  FilePlan plan;
  plan.manifest = {headerSize, manifestSize};
  uint64_t cursor = checkedAdd(headerSize, manifestSize);
  plan.directory = reserveRange(cursor, checkedMul(specs.size(), entrySize));
  plan.payload.offset = alignTo(cursor, alignment);

  for (const auto &spec : specs) {
    TensorRecord tensor{spec, Layout::Dense, spec.shape, {}, {}};
    if (spec.dtype == DType::I8) {
      tensor.layout = Layout::RowMajorI8Group32;
      tensor.storageShape[1] = alignTo(spec.shape[1], weightGroupSize);
    }
    uint64_t bytes = dtypeSize(spec.dtype);
    for (uint64_t dim : tensor.storageShape)
      bytes = checkedMul(bytes, dim);
    tensor.data = reserveRange(cursor, bytes);
    if (tensor.layout == Layout::RowMajorI8Group32) {
      uint64_t groups = tensor.storageShape[1] / weightGroupSize;
      tensor.scales = reserveRange(
          cursor, checkedMul(2, checkedMul(spec.shape[0], groups)));
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
  if (tensor.spec.id.size() >= idSize || tensor.spec.shape.size() > 3)
    fail("Tensor exceeds directory entry capacity");
  std::copy(tensor.spec.id.begin(), tensor.spec.id.end(), entry.begin());
  writeLittleEndian(entry.data() + 96, static_cast<uint32_t>(tensor.spec.dtype),
                    4);
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

uint32_t checksumFile(const File &file, uint64_t size) {
  std::vector<uint8_t> buffer(1 << 20);
  uLong checksum = crc32(0, Z_NULL, 0);
  for (uint64_t offset = headerSize; offset < size;) {
    auto count =
        static_cast<uInt>(std::min<uint64_t>(size - offset, buffer.size()));
    file.read({offset, count}, buffer.data());
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
  fail("Invalid tensor dtype");
}

std::string_view llminfer::getLayoutName(Layout layout) {
  switch (layout) {
  case Layout::Dense:
    return "Dense";
  case Layout::RowMajorI8Group32:
    return "RowMajorI8Group32";
  }
  fail("Invalid tensor layout");
}

std::vector<RowSlice> llminfer::splitRows(const TensorRecord &tensor,
                                          uint64_t maxBufferLength) {
  if (tensor.spec.shape.size() != 2 || tensor.spec.shape[0] == 0)
    fail("Row slicing requires a nonempty rank-two tensor");
  uint64_t rows = tensor.spec.shape[0];
  if (tensor.data.size == 0 || tensor.data.size % rows ||
      tensor.scales.size % rows)
    fail("Invalid tensor row storage");
  uint64_t dataStride = tensor.data.size / rows;
  uint64_t scaleStride = tensor.scales.size / rows;
  uint64_t rowsPerSlice = maxBufferLength / std::max(dataStride, scaleStride);
  if (rowsPerSlice == 0)
    fail("Tensor row exceeds maximum buffer length");

  std::vector<RowSlice> slices;
  for (uint64_t first = 0; first < rows;) {
    uint64_t count = std::min(rowsPerSlice, rows - first);
    ByteRange data{checkedAdd(tensor.data.offset, first * dataStride),
                   count * dataStride};
    ByteRange scales;
    if (scaleStride)
      scales = {checkedAdd(tensor.scales.offset, first * scaleStride),
                count * scaleStride};
    slices.push_back({first, count, data, scales});
    first += count;
  }
  return slices;
}

struct ModelFile::Impl {
  explicit Impl(const std::filesystem::path &path) : file(path) {
    size = file.getSize();
    Header header;
    file.read({0, header.size()}, header.data());
    if (!std::equal(magic.begin(), magic.end(), header.begin()))
      fail("Invalid .llmi magic");
    if (readLittleEndian(header.data() + 8, 4) != formatVersion)
      fail("Unsupported .llmi version");
    if (readLittleEndian(header.data() + 12, 4) != qwen35TextArchitecture)
      fail("Unsupported .llmi architecture");
    uint64_t manifestSize = readLittleEndian(header.data() + 24, 8);
    if (manifestSize == 0 || manifestSize > maxManifestSize ||
        !contains({0, size}, {headerSize, manifestSize}))
      fail("Invalid .llmi manifest range");
    manifest.resize(manifestSize);
    file.read({headerSize, manifestSize}, manifest.data());

    auto schema = getModelSchema(manifest);
    precision = schema.precision;
    auto plan = planFile(manifestSize, schema.tensors);
    checksum = readLittleEndian(header.data() + 64, 4);
    // The canonical layout verifies all encoded ranges and rejects overlaps.
    if (header != encodeHeader(plan, checksum) || size != plan.size())
      fail("Invalid .llmi sections or file length");
    uint64_t offset = plan.directory.offset;
    for (const auto &tensor : plan.tensors) {
      Entry entry;
      file.read({offset, entry.size()}, entry.data());
      if (entry != encodeEntry(tensor))
        fail("Invalid .llmi directory entry for " + tensor.spec.id);
      offset += entrySize;
    }
    if (checksum != checksumFile(file, size))
      fail("Invalid .llmi checksum");
    tensors = std::move(plan.tensors);
  }

  File file;
  uint64_t size;
  uint32_t checksum;
  Precision precision;
  std::string manifest;
  std::vector<TensorRecord> tensors;
};

ModelFile::ModelFile(const std::filesystem::path &path)
    : impl(std::make_unique<Impl>(path)) {}

ModelFile::~ModelFile() = default;
ModelFile::ModelFile(ModelFile &&) noexcept = default;
ModelFile &ModelFile::operator=(ModelFile &&) noexcept = default;

const std::vector<TensorRecord> &ModelFile::getTensors() const {
  return impl->tensors;
}

const TensorRecord &ModelFile::getTensor(std::string_view id) const {
  if (id == headId)
    id = embeddingId;
  auto it =
      std::lower_bound(impl->tensors.begin(), impl->tensors.end(), id,
                       [](const TensorRecord &tensor, std::string_view id) {
                         return tensor.spec.id < id;
                       });
  if (it == impl->tensors.end() || it->spec.id != id)
    fail("Unknown tensor: " + std::string(id));
  return *it;
}

Precision ModelFile::getPrecision() const { return impl->precision; }
const std::string &ModelFile::getManifest() const { return impl->manifest; }
uint64_t ModelFile::getFileSize() const { return impl->size; }
uint32_t ModelFile::getChecksum() const { return impl->checksum; }

void ModelFile::read(ByteRange range, void *destination) const {
  if (!contains({0, impl->size}, range))
    fail("Read is outside the model file");
  impl->file.read(range, destination);
}

void llminfer::writeModelFile(const std::filesystem::path &path,
                              std::string_view manifest,
                              const TensorWriter &writeTensor) {
  auto schema = getModelSchema(manifest);
  auto plan = planFile(manifest.size(), schema.tensors);
  std::string temporary = path.string() + ".tmp.XXXXXX";
  int fd = ::mkstemp(temporary.data());
  if (fd == -1)
    fail("Creating temporary model: " + std::string(std::strerror(errno)));
  try {
    File file(fd);
    file.resize(plan.size());
    file.write(plan.manifest, manifest.data());
    uint64_t offset = plan.directory.offset;
    for (const auto &tensor : plan.tensors) {
      Entry entry = encodeEntry(tensor);
      file.write({offset, entry.size()}, entry.data());
      offset += entrySize;
      ByteRange remainingData = tensor.data;
      ByteRange remainingScales = tensor.scales;
      writeTensor(tensor, [&](ByteRange chunk, const void *bytes) {
        ByteRange &remaining =
            chunk.offset == remainingData.offset && remainingData.size != 0
                ? remainingData
                : remainingScales;
        if (chunk.size == 0 || chunk.offset != remaining.offset ||
            chunk.size > remaining.size)
          fail("Unexpected tensor write for " + tensor.spec.id);
        file.write(chunk, bytes);
        remaining.offset += chunk.size;
        remaining.size -= chunk.size;
      });
      if (remainingData.size != 0 || remainingScales.size != 0)
        fail("Incomplete tensor write for " + tensor.spec.id);
    }

    Header header = encodeHeader(plan, checksumFile(file, plan.size()));
    file.write({0, header.size()}, header.data());
    file.sync();
    ModelFile validated(temporary);
    std::filesystem::rename(temporary, path);
  } catch (...) {
    std::remove(temporary.c_str());
    throw;
  }
}
