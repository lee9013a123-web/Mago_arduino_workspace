cmake_minimum_required(VERSION 3.13)

project(campp_native_fbank LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(KALDI_NATIVE_FBANK_SOURCE_DIR "" CACHE PATH
    "Path to the pinned kaldi-native-fbank source checkout")
if(NOT EXISTS "${KALDI_NATIVE_FBANK_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "KALDI_NATIVE_FBANK_SOURCE_DIR must point to kaldi-native-fbank")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_ENABLE_CHECK OFF CACHE BOOL "" FORCE)

# The dependency includes headers as kaldi-native-fbank/csrc/*.h. Its own
# top-level project assumes it is the CMake source root, so expose that root
# explicitly when it is embedded as a subdirectory here.
include_directories("${KALDI_NATIVE_FBANK_SOURCE_DIR}")
add_subdirectory(
    "${KALDI_NATIVE_FBANK_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/kaldi-native-fbank"
    EXCLUDE_FROM_ALL
)

add_executable(campp_fbank campp_fbank_main.cc)
target_include_directories(campp_fbank PRIVATE
    "${KALDI_NATIVE_FBANK_SOURCE_DIR}"
)
target_link_libraries(campp_fbank PRIVATE kaldi-native-fbank-core)
set_target_properties(campp_fbank PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
