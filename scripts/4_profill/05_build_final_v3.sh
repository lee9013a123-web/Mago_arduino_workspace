#!/usr/bin/env bash

# Layer별 QConv/fused-QConv hybrid plan을 적용한 final v3를 v2와 동일한
# aggressive GCC flag로 빌드한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/final_v3_hybrid}"
CC="${CC:-gcc}"
STRIP="${STRIP:-strip}"
FINAL_CFLAGS="-std=c11 -DNDEBUG -Wall -Wextra -O3 -mcpu=cortex-a53 -flto -finline-functions -finline-small-functions -funroll-loops -fomit-frame-pointer -ffunction-sections -fdata-sections"
FINAL_LDFLAGS="-flto -Wl,--gc-sections"

BUILD_DIR="${BUILD_DIR}" \
BUILD_VARIANT="03_layer_hybrid" \
BUILD_SCOPE="compiler_matrix" \
FINAL_SUITE_VARIANT="v3" \
CC="${CC}" \
CFLAGS="${FINAL_CFLAGS}" \
LDFLAGS="${FINAL_LDFLAGS}" \
STRIP="${STRIP}" \
STRIP_FINAL="1" \
    bash "${ROOT}/scripts/4_profill/01_build_profiler.sh"

echo "Final v3 layer-hybrid build complete: ${BUILD_DIR}"
