#include "File.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

using namespace llminfer;

namespace {
constexpr uint64_t maxFileOffset = std::numeric_limits<off_t>::max();
constexpr uint64_t ioChunk = 1 << 20;

bool isFileRange(ByteRange range) {
  return range.offset <= maxFileOffset &&
         range.size <= maxFileOffset - range.offset;
}
} // namespace

Error llminfer::fileError(std::string operation, int errorNumber) {
  return Error(operation + ": " + std::strerror(errorNumber));
}

bool llminfer::checkedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs > UINT64_MAX - lhs)
    return false;
  result = lhs + rhs;
  return true;
}

bool llminfer::checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

bool llminfer::alignTo(uint64_t value, uint64_t alignment, uint64_t &result) {
  return checkedAdd(value, (alignment - value % alignment) % alignment, result);
}

uint64_t llminfer::readLittleEndian(const uint8_t *bytes, unsigned size) {
  uint64_t result = 0;
  for (unsigned i = 0; i < size; ++i)
    result |= uint64_t(bytes[i]) << (8 * i);
  return result;
}

void llminfer::writeLittleEndian(uint8_t *bytes, uint64_t value,
                                 unsigned size) {
  for (unsigned i = 0; i < size; ++i)
    bytes[i] = static_cast<uint8_t>(value >> (8 * i));
}

Result<File> File::open(const std::filesystem::path &path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd == -1) {
    int errorNumber = errno;
    return fileError("Opening " + path.string(), errorNumber);
  }
  return File(fd);
}

File::File(File &&other) noexcept : fd(std::exchange(other.fd, -1)) {}

File::~File() {
  if (fd != -1)
    ::close(fd);
}

Result<uint64_t> File::getSize() const {
  struct stat info{};
  if (::fstat(fd, &info) == -1)
    return fileError("Getting file size", errno);
  if (!S_ISREG(info.st_mode) || info.st_size < 0)
    return Error("Expected a regular file");
  return static_cast<uint64_t>(info.st_size);
}

Error File::resize(uint64_t size) const {
  if (size > maxFileOffset)
    return Error("File offset exceeds platform limit");
  if (::ftruncate(fd, static_cast<off_t>(size)) == -1)
    return fileError("Resizing file", errno);
  return {};
}

Error File::read(ByteRange range, void *destination) const {
  if (!isFileRange(range))
    return Error("Read range exceeds platform file offset limit");
  auto *bytes = static_cast<uint8_t *>(destination);
  while (range.size != 0) {
    ssize_t count = ::pread(fd, bytes, std::min(range.size, ioChunk),
                            static_cast<off_t>(range.offset));
    if (count == -1 && errno == EINTR)
      continue;
    if (count == -1)
      return fileError("Reading file", errno);
    if (count == 0)
      return Error("Unexpected end of file");
    bytes += count;
    range.offset += count;
    range.size -= count;
  }
  return {};
}

Error File::write(ByteRange range, const void *source) const {
  if (!isFileRange(range))
    return Error("Write range exceeds platform file offset limit");
  auto *bytes = static_cast<const uint8_t *>(source);
  while (range.size != 0) {
    ssize_t count = ::pwrite(fd, bytes, std::min(range.size, ioChunk),
                             static_cast<off_t>(range.offset));
    if (count == -1 && errno == EINTR)
      continue;
    if (count == -1)
      return fileError("Writing file", errno);
    if (count == 0)
      return Error("Incomplete write");
    bytes += count;
    range.offset += count;
    range.size -= count;
  }
  return {};
}

Error File::sync() const {
  if (::fsync(fd) == -1)
    return fileError("Syncing file", errno);
  return {};
}

Result<std::string> llminfer::readTextFile(const std::filesystem::path &path,
                                           uint64_t limit) {
  auto file = File::open(path);
  if (!file)
    return file.getError();
  auto size = file->getSize();
  if (!size)
    return size.getError();
  if (*size == 0 || *size > limit)
    return Error("Invalid metadata size for " + path.string());
  std::string content(*size, '\0');
  if (auto error = file->read({0, *size}, content.data()))
    return error;
  return content;
}
