#include "Converter.h"
#include "GraphCommand.h"
#include "Qwen35Schema.h"
#ifdef LLMINFER_HAS_METAL
#include "RunCommand.h"
#endif
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
      << "  llmi inspect <model.llmi> [--tensor <id>] [--json]\n"
      << "  llmi graph <model.llmi> [--tokens T] [--past P] [--capacity N] "
         "[--logits none|last|all] [--json]\n";
#ifdef LLMINFER_HAS_METAL
  out << "  llmi run <model.llmi> <token[,token...]>... [--capacity N] "
         "[--logits none|last|all] [--plan]\n";
#endif
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

Error inspect(const std::filesystem::path &path, std::string_view tensorId,
              bool printJson) {
  auto model = ModelFile::open(path);
  if (!model)
    return model.getError();
  auto manifest = parseJson(model->getManifest());
  if (!manifest)
    return manifest.getError();
  std::map<std::string, uint64_t> bytesByType;
  for (const auto &tensor : model->getTensors()) {
    bytesByType[std::string(getDTypeName(tensor.spec.dtype))] +=
        tensor.data.size;
    if (tensor.scales.size)
      bytesByType["F16"] += tensor.scales.size;
  }

  json tensor;
  if (!tensorId.empty()) {
    const auto *record = model->getTensor(tensorId);
    if (!record)
      return Error("Unknown tensor: " + std::string(tensorId));
    tensor = describeTensor(*record);
  }
  if (printJson) {
    json result{{"manifest", *manifest},
                {"file_size", model->getFileSize()},
                {"crc32", model->getChecksum()},
                {"tensor_count", model->getTensors().size()},
                {"bytes_by_dtype", bytesByType}};
    if (tensor.is_null()) {
      result["tensors"] = json::array();
      for (const auto &entry : model->getTensors())
        result["tensors"].push_back(describeTensor(entry));
    } else {
      result["tensor"] = tensor;
    }
    std::cout << result.dump(2) << '\n';
    return {};
  }

  std::cout << "Qwen3.5 text / " << getPrecisionName(model->getPrecision())
            << "\nFile bytes: " << model->getFileSize()
            << "\nCRC32: " << std::hex << model->getChecksum() << std::dec
            << " (verified)\nStored tensors: " << model->getTensors().size()
            << "\nSkipped tensors: "
            << getField(*manifest, "skipped_tensors").size()
            << "\nAlias: " << headId << " -> " << embeddingId << '\n';
  for (const auto &entry : bytesByType)
    std::cout << entry.first << " payload bytes: " << entry.second << '\n';
  if (!tensor.is_null())
    std::cout << tensor.dump(2) << '\n';
  return {};
}

int report(Error error) {
  if (!error)
    return EXIT_SUCCESS;
  std::cerr << "llmi: " << error.getMessage() << '\n';
  return EXIT_FAILURE;
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
        return report(Error("Unknown precision: " + std::string(option)));
    } else if (argc != 4) {
      return report(Error("Expected --precision w8a8|f16"));
    }
    if (auto error = convertModel(argv[2], argv[3], precision))
      return report(std::move(error));
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
        return report(Error("Invalid inspect option: " + std::string(option)));
    }
    return report(inspect(argv[2], tensor, printJson));
  }

  if (argc >= 3 && std::string_view(argv[1]) == "graph")
    return report(describeGraph(argc, argv));

#ifdef LLMINFER_HAS_METAL
  if (argc >= 3 && std::string_view(argv[1]) == "run")
    return report(runMetal(argc, argv));
#endif

  printUsage(std::cerr);
  return EXIT_FAILURE;
}
} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
