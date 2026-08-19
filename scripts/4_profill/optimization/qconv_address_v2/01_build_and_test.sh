#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/optimization/qconv_address_v2}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O3 -g -DNDEBUG -Wall -Wextra}"
ADDRESS_DIR="${ROOT}/src/c/profill/optimization/candidates/qlinear_conv_address_v2"

mkdir -p "${BUILD_DIR}"

# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    -I "${ADDRESS_DIR}" \
    "${ADDRESS_DIR}/planning/qconv_address_plan.c" \
    "${ADDRESS_DIR}/planning/qconv_address_schedule.c" \
    "${ADDRESS_DIR}/kernels/one_by_one/qconv_address_1x1.c" \
    "${ADDRESS_DIR}/kernels/three_by_three/qconv_address_3x3.c" \
    "${ADDRESS_DIR}/kernels/generic/qconv_address_generic.c" \
    "${ROOT}/tests/runtime/profill/qconv_address_v2/test_qconv_address_v2.c" \
    -o "${BUILD_DIR}/test_qconv_address_v2"

"${BUILD_DIR}/test_qconv_address_v2"

echo "QConv address-v2 standalone build and tests complete"
