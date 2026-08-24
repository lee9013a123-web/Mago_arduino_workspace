#!/usr/bin/env bash

# Build the multibucket mmap candidate without overwriting the incumbent V3.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/weight_streaming_98}"
CC="${CC:-gcc}"
STRIP="${STRIP:-strip}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
FINAL_CFLAGS="-std=c11 -DNDEBUG -Wall -Wextra -O3 -mcpu=cortex-a53 -flto -finline-functions -finline-small-functions -funroll-loops -fomit-frame-pointer -ffunction-sections -fdata-sections"
FINAL_LDFLAGS="-flto -Wl,--gc-sections"
STAMP="${BUILD_DIR}/weight_streaming_98.source.sha256"

build_signature=""
if command -v sha256sum >/dev/null 2>&1; then
    build_signature="$({
        printf '%s\n' "${CC}" "${CPPFLAGS:-}" "${FINAL_CFLAGS}" \
            "${FINAL_LDFLAGS}"
        find "${ROOT}/src/c/runtime" "${ROOT}/src/c/profill" \
            -type f \( -name '*.c' -o -name '*.h' \) -print0 \
            | sort -z | xargs -0 sha256sum
        sha256sum \
            "${ROOT}/tests/runtime/operator_replay_tests/campp_reference_dump.c" \
            "${ROOT}/scripts/4_profill/01_build_profiler.sh" \
            "$0"
    } | sha256sum | cut -d' ' -f1)"
fi
if [[ "${FORCE_REBUILD}" != "1" && -n "${build_signature}" && \
      -f "${BUILD_DIR}/campp_runtime_benchmark_final" && \
      -f "${BUILD_DIR}/campp_reference_dump_final" && \
      -f "${STAMP}" && "$(<"${STAMP}")" == "${build_signature}" ]]; then
    echo "Weight streaming candidate build reused: ${BUILD_DIR}"
    exit 0
fi

BUILD_DIR="${BUILD_DIR}" \
BUILD_VARIANT="weight_streaming_98" \
BUILD_SCOPE="compiler_quick" \
FINAL_SUITE_VARIANT="v3" \
CPPFLAGS="${CPPFLAGS:-} -DCAMPP_ENABLE_WEIGHT_STREAMING=1" \
CC="${CC}" \
CFLAGS="${FINAL_CFLAGS}" \
LDFLAGS="${FINAL_LDFLAGS}" \
STRIP="${STRIP}" \
STRIP_FINAL="1" \
    bash "${ROOT}/scripts/4_profill/01_build_profiler.sh"

if [[ -n "${build_signature}" ]]; then
    printf '%s\n' "${build_signature}" > "${STAMP}"
fi

echo "Multibucket weight streaming candidate build complete: ${BUILD_DIR}"
