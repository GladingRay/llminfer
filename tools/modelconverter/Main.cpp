#include "Converter.h"
#include "File.h"
#include "Qwen35Schema.h"
#include <cstdlib>
#include <iostream>
#include <map>

using namespace llminfer;
using nlohmann::json;

namespace {
void printUsage(std::ostream &out) {
  out << "Usage:\n"
      << "  llmi convert <model-directory> <output.llmi> [--precision "
         "w8a8|f16]\n"
      << "  llmi inspect <model.llmi> [--tensor <id>] [--json]\n";
}

json describeTensor(const TensorRecord &tensor) {
  return {
      {"id", tensor.spec.id},
      {"shape", tensor.spec.shape},
      {"storage_shape", tensor.storageShape},
      {"dtype", getDTypeName(tensor.spec.dtype)},
      {"layout", getLayoutName(tensor.layout)},
      {"data", {{"offset", tensor.data.offset}, {"size", tensor.data.size}}},
      {"scales",
       {{"offset", tensor.scales.offset}, {"size", tensor.scales.size}}}};
}

void inspect(const std::filesystem::path &path, std::string_view tensorId,
             bool printJson) {
  ModelFile model(path);
  auto manifest = parseJson(model.getManifest());
  std::map<std::string, uint64_t> bytesByType;
  for (const auto &tensor : model.getTensors()) {
    bytesByType[std::string(getDTypeName(tensor.spec.dtype))] +=
        tensor.data.size;
    if (tensor.scales.size)
      bytesByType["F16"] += tensor.scales.size;
  }

  json tensor;
  if (!tensorId.empty())
    tensor = describeTensor(model.getTensor(tensorId));
  if (printJson) {
    json result{{"manifest", manifest},
                {"file_size", model.getFileSize()},
                {"crc32", model.getChecksum()},
                {"tensor_count", model.getTensors().size()},
                {"bytes_by_dtype", bytesByType}};
    if (tensor.is_null()) {
      result["tensors"] = json::array();
      for (const auto &entry : model.getTensors())
        result["tensors"].push_back(describeTensor(entry));
    } else {
      result["tensor"] = tensor;
    }
    std::cout << result.dump(2) << '\n';
    return;
  }

  std::cout << "Qwen3.5 text / " << getPrecisionName(model.getPrecision())
            << "\nFile bytes: " << model.getFileSize()
            << "\nCRC32: " << std::hex << model.getChecksum() << std::dec
            << " (verified)\nStored tensors: " << model.getTensors().size()
            << "\nSkipped tensors: " << manifest.at("skipped_tensors").size()
            << "\nAlias: " << headId << " -> " << embeddingId << '\n';
  for (const auto &entry : bytesByType)
    std::cout << entry.first << " payload bytes: " << entry.second << '\n';
  if (!tensor.is_null())
    std::cout << tensor.dump(2) << '\n';
}

int run(int argc, char **argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    printUsage(std::cout);
    return EXIT_SUCCESS;
  }

  if (argc >= 4 && std::string_view(argv[1]) == "convert") {
    Precision precision = Precision::W8A8;
    if (argc == 6 && std::string_view(argv[4]) == "--precision") {
      std::string_view option = argv[5];
      if (option == "f16")
        precision = Precision::F16;
      else if (option != "w8a8")
        fail("Unknown precision: " + std::string(option));
    } else if (argc != 4) {
      fail("Expected --precision w8a8|f16");
    }
    convertModel(argv[2], argv[3], precision);
    std::cout << "Wrote " << argv[3] << '\n';
    return EXIT_SUCCESS;
  }

  if (argc >= 3 && std::string_view(argv[1]) == "inspect") {
    std::string_view tensor;
    bool printJson = false;
    for (int i = 3; i < argc; ++i) {
      std::string_view option = argv[i];
      if (option == "--json" && !printJson)
        printJson = true;
      else if (option == "--tensor" && tensor.empty() && i + 1 < argc &&
               *argv[i + 1] != '\0')
        tensor = argv[++i];
      else
        fail("Invalid inspect option: " + std::string(option));
    }
    inspect(argv[2], tensor, printJson);
    return EXIT_SUCCESS;
  }

  printUsage(std::cerr);
  return EXIT_FAILURE;
}
} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "llmi: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
