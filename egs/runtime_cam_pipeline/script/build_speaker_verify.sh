#!/usr/bin/env bash
# Build the Python/NumPy-free speaker verification orchestrator.

set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${PIPELINE_ROOT}/../.." && pwd)"
SOURCE_ROOT="${PIPELINE_ROOT}/native_pipeline"
BUILD_ROOT="${PIPELINE_ROOT}/build/native_pipeline"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"
CFLAGS="${CFLAGS:--std=c11 -O2 -Wall -Wextra -Wpedantic}"
CXXFLAGS="${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra -Wpedantic}"

if ! command -v "${CC}" >/dev/null 2>&1 ||
   ! command -v "${CXX}" >/dev/null 2>&1; then
    echo "gcc and g++ with C11/C++17 support are required" >&2
    exit 1
fi

mkdir -p "${BUILD_ROOT}"
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS} \
    -I "${SOURCE_ROOT}" \
    "${SOURCE_ROOT}/json_lite.cc" \
    "${SOURCE_ROOT}/test_json_lite.cc" \
    -o "${BUILD_ROOT}/test_json_lite"
"${BUILD_ROOT}/test_json_lite"

# Compile the exact SHA-256 implementation used by the C runtime plan/package
# validators, then exercise both its memory and streaming-file APIs.
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    -I "${REPO_ROOT}/src/c/runtime" \
    -I "${REPO_ROOT}/src/c/runtime/include" \
    "${REPO_ROOT}/src/c/runtime/integrity/sha256.c" \
    "${SOURCE_ROOT}/test_sha256.c" \
    -o "${BUILD_ROOT}/test_sha256"
"${BUILD_ROOT}/test_sha256"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    -I "${REPO_ROOT}/src/c/runtime" \
    -I "${REPO_ROOT}/src/c/runtime/include" \
    -c "${REPO_ROOT}/src/c/runtime/integrity/sha256.c" \
    -o "${BUILD_ROOT}/sha256.o"

# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS} \
    -I "${SOURCE_ROOT}" \
    -I "${REPO_ROOT}/src/c/runtime" \
    -I "${REPO_ROOT}/src/c/runtime/include" \
    "${SOURCE_ROOT}/json_lite.cc" \
    "${SOURCE_ROOT}/process_runner.cc" \
    "${SOURCE_ROOT}/campp_speaker_verify_main.cc" \
    "${BUILD_ROOT}/sha256.o" \
    -o "${BUILD_ROOT}/campp_speaker_verify"

"${BUILD_ROOT}/campp_speaker_verify" --version
echo "Native speaker verification ready: ${BUILD_ROOT}/campp_speaker_verify"
