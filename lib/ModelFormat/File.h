#ifndef LLMINFER_MODELFORMAT_FILE_H
#define LLMINFER_MODELFORMAT_FILE_H

#include "llminfer/ModelFormat/ModelFile.h"
#include <filesystem>
#include <stdexcept>
#include <string>

namespace llminfer {

[[noreturn]] inline void fail(const std::string &message) {
  throw std::runtime_error(message);
}

uint64_t checkedAdd(uint64_t lhs, uint64_t rhs);
uint64_t checkedMul(uint64_t lhs, uint64_t rhs);
uint64_t alignTo(uint64_t value, uint64_t alignment);
uint64_t readLittleEndian(const uint8_t *bytes, unsigned size);
void writeLittleEndian(uint8_t *bytes, uint64_t value, unsigned size);

class File {
public:
  explicit File(const std::filesystem::path &path);
  explicit File(int fd) : fd(fd) {}
  ~File();
  File(const File &) = delete;
  File &operator=(const File &) = delete;

  uint64_t getSize() const;
  void resize(uint64_t size) const;
  void read(ByteRange range, void *destination) const;
  void write(ByteRange range, const void *source) const;
  void sync() const;

private:
  int fd;
};

std::string readTextFile(const std::filesystem::path &path, uint64_t limit);

} // namespace llminfer

#endif
