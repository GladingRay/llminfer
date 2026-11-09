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

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS OFF)

  set(METAL_REFERENCE_KERNELS "${CMAKE_CURRENT_SOURCE_DIR}/lib/Backend/Metal/Kernels")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${METAL_REFERENCE_KERNELS}/ReferenceParams.h"
    "${METAL_REFERENCE_KERNELS}/Reference.metal")
  file(READ "${METAL_REFERENCE_KERNELS}/ReferenceParams.h" REFERENCE_PARAMS)
  file(READ "${METAL_REFERENCE_KERNELS}/Reference.metal" REFERENCE_MSL)
  configure_file("${METAL_REFERENCE_KERNELS}/ReferenceSource.h.in"
                 "generated/ReferenceSource.h" @ONLY)

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
  add_library(llminfer_metal_reference
    lib/Backend/Metal/MetalContext.mm
    lib/Models/Qwen35/MetalReferenceGraph.mm)
  target_compile_options(llminfer_metal_reference PRIVATE
    -fobjc-arc -fno-exceptions -fno-objc-exceptions)
  target_include_directories(llminfer_metal_reference PUBLIC include PRIVATE
    lib/Backend/Metal lib/ModelFormat "${CMAKE_CURRENT_BINARY_DIR}/generated")
  target_link_libraries(llminfer_metal_reference PRIVATE
    llminfer_model_format nlohmann_json::nlohmann_json
    "${METAL_FRAMEWORK}" "${FOUNDATION_FRAMEWORK}")

  add_executable(llmi-reference tools/reference/Main.cpp)
  target_link_libraries(llmi-reference PRIVATE
    llminfer_metal_reference nlohmann_json::nlohmann_json)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(model_format_test tests/ModelFormatTest.cpp)
  target_link_libraries(model_format_test PRIVATE llminfer_model_format)
  add_test(NAME model_format COMMAND model_format_test)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME converter
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/ConverterTest.py"
            "$<TARGET_FILE:llmi>")
  if(TARGET llmi-reference)
    add_test(NAME metal_reference
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/MetalReferenceTest.py"
              "$<TARGET_FILE:llmi>" "$<TARGET_FILE:llmi-reference>" --allow-skip)
    set_tests_properties(metal_reference PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
