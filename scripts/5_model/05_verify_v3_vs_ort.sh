#!/usr/bin/env bash

# bucket별 Final V3를 두 축으로 검증한다.
#
#   V2 <-> V3 : bitwise  (회귀 gate -- 최적화가 값을 바꾸지 않았는가)
#   ORT <-> V3: tolerance (정확도 gate -- cosine / max_abs)
#   V2 vs V3  : E2E latency
#
# ORT와 bitwise 일치는 달성 불가능하다.  cam_layer/ReduceMean의 float32 누적 순서가
# ORT와 다르고(~1e-07), 그 차이가 양자화 경계에서 1 LSB 뒤집힘으로 증폭되기
# 때문이다.  이는 V2/V3 공통 특성이며 EER에는 영향이 없음이 확인됐다
# (bucket 298, 780 trial: ORT 5.0000% vs C 5.0000%).
#
# 사용: bash scripts/5_model/05_verify_v3_vs_ort.sh [bucket ...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/arduino/workspace/venv/bin/python3}"
BUCKETS=("$@")
if [[ ${#BUCKETS[@]} -eq 0 ]]; then BUCKETS=(98 298 498 998); fi

exec "${PY}" "${ROOT}/scripts/5_model/05_verify_v3_vs_ort.py" "${BUCKETS[@]}"
