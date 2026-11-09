#ifndef LLMINFER_MODELFORMAT_FILE_H
#define LLMINFER_MODELFORMAT_FILE_H

#include "llminfer/ModelFormat/ModelFile.h"
#include "llminfer/Support/CheckedArithmetic.h"

namespace llminfer {

Error fileError(std::string operation, int errorNumber);
uint64_t readLittleEndian(const uint8_t *bytes, unsigned size);
void writeLittleEndian(uint8_t *bytes, uint64_t value, unsigned size);

class File {
public:
  static Result<File> open(const std::filesystem::path &path);
  explicit File(int fd) : fd(fd) {}
  ~File();
  File(File &&other) noexcept;
  File(const File &) = delete;
  File &operator=(const File &) = delete;

  Result<uint64_t> getSize() const;
  Error resize(uint64_t size) const;
  Error read(ByteRange range, void *destination) const;
  Error write(ByteRange range, const void *source) const;
  Error sync() const;

private:
  int fd;
};

Result<std::string> readTextFile(const std::filesystem::path &path,
                                 uint64_t limit);

} // namespace llminfer

#endif
