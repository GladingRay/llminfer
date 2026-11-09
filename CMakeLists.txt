cmake_minimum_required(VERSION 3.24)
project(llminfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang,GNU>:-fno-exceptions>")

find_package(ZLIB REQUIRED)
find_package(nlohmann_json 3.12.0 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

add_library(llminfer_model_format
  lib/ModelFormat/File.cpp
  lib/ModelFormat/ModelFile.cpp
  lib/ModelFormat/Qwen35Schema.cpp
  lib/ModelFormat/Quantization.cpp
)
target_include_directories(llminfer_model_format PUBLIC include)
target_link_libraries(llminfer_model_format PRIVATE nlohmann_json::nlohmann_json ZLIB::ZLIB)

add_executable(llmi
  tools/modelconverter/Main.cpp
  tools/modelconverter/Converter.cpp
  tools/modelconverter/Safetensors.cpp
)
target_include_directories(llmi PRIVATE lib/ModelFormat)
target_link_libraries(llmi PRIVATE llminfer_model_format nlohmann_json::nlohmann_json)

include(CTest)
if(BUILD_TESTING)
  add_executable(model_format_test tests/ModelFormatTest.cpp)
  target_link_libraries(model_format_test PRIVATE llminfer_model_format)
  add_test(NAME model_format COMMAND model_format_test)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME converter
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ConverterTest.py"
            "$<TARGET_FILE:llmi>")
endif()
