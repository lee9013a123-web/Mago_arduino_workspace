#!/usr/bin/env bash

# Build and Quick-benchmark the final E7 candidate suite with a staged compiler
# option matrix. Each candidate gets an isolated build and raw-result directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_CONFIG="${ROOT}/configs/runtime/compiler_qrb2210_strict.json"

exec python3 "${ROOT}/scripts/4_profill/compiler/02_run_matrix.py" \
    --config "${DEFAULT_CONFIG}" "$@"
