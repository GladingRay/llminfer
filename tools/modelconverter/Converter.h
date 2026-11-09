#ifndef LLMINFER_MODELCONVERTER_CONVERTER_H
#define LLMINFER_MODELCONVERTER_CONVERTER_H

#include "llminfer/ModelFormat/ModelFile.h"

namespace llminfer {
Error convertModel(const std::filesystem::path &sourceDirectory,
                   const std::filesystem::path &outputPath,
                   Precision precision);
} // namespace llminfer

#endif
