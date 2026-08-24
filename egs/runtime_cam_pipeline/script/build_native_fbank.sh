#!/usr/bin/env bash
# Download and build the pinned, Torch-free Kaldi-compatible FBank frontend.

set -euo pipefail

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${PIPELINE_ROOT}/.deps/kaldi-native-fbank-v1.22.3"
BUILD_ROOT="${PIPELINE_ROOT}/build/native_fbank"
TAG="v1.22.3"
COMMIT="b09e686fe2084732ddd30d1ef80acfc0f13eaf01"
REPOSITORY="https://github.com/csukuangfj/kaldi-native-fbank.git"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required to download kaldi-native-fbank" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to build campp_fbank" >&2
    exit 1
fi
if ! command -v g++ >/dev/null 2>&1; then
    echo "g++ with C++17 support is required to build campp_fbank" >&2
    exit 1
fi

if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${SOURCE_ROOT}")"
    git clone --depth 1 --branch "${TAG}" "${REPOSITORY}" "${SOURCE_ROOT}"
fi

ACTUAL_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${COMMIT}" ]]; then
    echo "kaldi-native-fbank commit mismatch" >&2
    echo "  expected: ${COMMIT}" >&2
    echo "  actual:   ${ACTUAL_COMMIT}" >&2
    echo "Remove ${SOURCE_ROOT} and run this script again." >&2
    exit 1
fi

cmake \
    -S "${PIPELINE_ROOT}/native_frontend" \
    -B "${BUILD_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DKALDI_NATIVE_FBANK_SOURCE_DIR="${SOURCE_ROOT}" \
    -DKALDI_NATIVE_FBANK_BUILD_PYTHON=OFF \
    -DKALDI_NATIVE_FBANK_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF
cmake --build "${BUILD_ROOT}" --target campp_fbank --parallel

"${BUILD_ROOT}/campp_fbank" --version
echo "Native FBank ready: ${BUILD_ROOT}/campp_fbank"
