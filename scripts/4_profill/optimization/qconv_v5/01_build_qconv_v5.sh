#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/optimization}"

bash "${ROOT}/scripts/4_profill/optimization/01_build_optimization.sh"
"${BUILD_DIR}/test_qconv_v5_primitives"
"${BUILD_DIR}/test_qconv_candidate"

echo "QConv v5 build and bitwise candidate tests: PASS"
