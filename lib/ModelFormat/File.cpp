#include "File.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

using namespace llminfer;

namespace {
[[noreturn]] void failIO(const std::string &operation) {
  fail(operation + ": " + std::strerror(errno));
}

off_t fileOffset(uint64_t offset) {
  if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()))
    fail("File offset exceeds platform limit");
  return static_cast<off_t>(offset);
}

constexpr uint64_t ioChunk = 1 << 20;
} // namespace

uint64_t llminfer::checkedAdd(uint64_t lhs, uint64_t rhs) {
  if (rhs > UINT64_MAX - lhs)
    fail("Integer addition overflow");
  return lhs + rhs;
}

uint64_t llminfer::checkedMul(uint64_t lhs, uint64_t rhs) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs)
    fail("Integer multiplication overflow");
  return lhs * rhs;
}

uint64_t llminfer::alignTo(uint64_t value, uint64_t alignment) {
  return checkedAdd(value, (alignment - value % alignment) % alignment);
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

File::File(const std::filesystem::path &path)
    : fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK)) {
  if (fd == -1)
    failIO("Opening " + path.string());
}

File::~File() { ::close(fd); }

uint64_t File::getSize() const {
  struct stat info{};
  if (::fstat(fd, &info) == -1)
    failIO("Getting file size");
  if (!S_ISREG(info.st_mode) || info.st_size < 0)
    fail("Expected a regular file");
  return static_cast<uint64_t>(info.st_size);
}

void File::resize(uint64_t size) const {
  if (::ftruncate(fd, fileOffset(size)) == -1)
    failIO("Resizing file");
}

void File::read(ByteRange range, void *destination) const {
  auto *bytes = static_cast<uint8_t *>(destination);
  while (range.size != 0) {
    ssize_t count = ::pread(fd, bytes, std::min(range.size, ioChunk),
                            fileOffset(range.offset));
    if (count == -1 && errno == EINTR)
      continue;
    if (count == -1)
      failIO("Reading file");
    if (count == 0)
      fail("Unexpected end of file");
    bytes += count;
    range.offset += count;
    range.size -= count;
  }
}

void File::write(ByteRange range, const void *source) const {
  auto *bytes = static_cast<const uint8_t *>(source);
  while (range.size != 0) {
    ssize_t count = ::pwrite(fd, bytes, std::min(range.size, ioChunk),
                             fileOffset(range.offset));
    if (count == -1 && errno == EINTR)
      continue;
    if (count == -1)
      failIO("Writing file");
    if (count == 0)
      fail("Incomplete write");
    bytes += count;
    range.offset += count;
    range.size -= count;
  }
}

void File::sync() const {
  if (::fsync(fd) == -1)
    failIO("Syncing file");
}

std::string llminfer::readTextFile(const std::filesystem::path &path,
                                   uint64_t limit) {
  File file(path);
  uint64_t size = file.getSize();
  if (size == 0 || size > limit)
    fail("Invalid metadata size for " + path.string());
  std::string content(size, '\0');
  file.read({0, size}, content.data());
  return content;
}
