#!/usr/bin/env bash

# QRB2210 compiler matrix에서 선택된 aggressive packaged flag로 새 final
# runtime/profiler/test/tensor-dump를 동일하게 빌드한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/final_v2_aggressive}"
CC="${CC:-gcc}"
STRIP="${STRIP:-strip}"
FINAL_CFLAGS="-std=c11 -DNDEBUG -Wall -Wextra -O3 -mcpu=cortex-a53 -flto -finline-functions -finline-small-functions -funroll-loops -fomit-frame-pointer -ffunction-sections -fdata-sections"
FINAL_LDFLAGS="-flto -Wl,--gc-sections"

BUILD_DIR="${BUILD_DIR}" \
BUILD_VARIANT="02_final_flags__aggressive_packaged" \
BUILD_SCOPE="compiler_matrix" \
FINAL_SUITE_VARIANT="v2" \
CC="${CC}" \
CFLAGS="${FINAL_CFLAGS}" \
LDFLAGS="${FINAL_LDFLAGS}" \
STRIP="${STRIP}" \
STRIP_FINAL="1" \
    bash "${ROOT}/scripts/4_profill/01_build_profiler.sh"

echo "Final v2 aggressive build complete: ${BUILD_DIR}"
