#!/usr/bin/env bash
# Python 없는 native ORT 벤치마크를 빌드한다.
#
# ORT 공유 라이브러리는 보드 venv의 onnxruntime wheel 안에 있고, C API 헤더는
# wheel에 없어서 같은 버전(v1.27.0)을 저장소에 vendoring 했다:
#   third_party/onnxruntime/include/onnxruntime_c_api.h
#
# 헤더의 ORT_API_VERSION과 .so 버전이 어긋나면 런타임에 GetApi가 NULL을 준다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--std=c11 -O3 -DNDEBUG -Wall -Wextra}"

ORT_INCLUDE="${ORT_INCLUDE:-${ROOT}/third_party/onnxruntime/include}"
ORT_LIB_DIR="${ORT_LIB_DIR:-/home/arduino/workspace/venv/lib/python3.13/site-packages/onnxruntime/capi}"
ORT_SO="${ORT_LIB_DIR}/libonnxruntime.so.1.27.0"

for required in "${ORT_INCLUDE}/onnxruntime_c_api.h" "${ORT_SO}"; do
    if [ ! -f "${required}" ]; then
        echo "필요한 파일이 없다: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${BUILD_DIR}"

# wheel에는 SONAME(libonnxruntime.so.1)에 해당하는 심볼릭 링크가 없어서 그대로
# 링크하면 실행 시 로더가 .so.1을 못 찾는다. venv를 건드리지 않고 저장소 안에
# 링크를 만들어 rpath를 그쪽으로 잡는다.
LINK_DIR="${ROOT}/third_party/onnxruntime/lib"
mkdir -p "${LINK_DIR}"
ln -sf "${ORT_SO}" "${LINK_DIR}/libonnxruntime.so.1"
ln -sf "libonnxruntime.so.1" "${LINK_DIR}/libonnxruntime.so"

echo "native ORT 벤치마크 빌드 (${CC})"
# shellcheck disable=SC2086
"${CC}" ${CFLAGS} \
    -I "${ORT_INCLUDE}" \
    "${ROOT}/experiments/baseline_ort_c/ort_benchmark.c" \
    -L "${LINK_DIR}" -lonnxruntime \
    -Wl,-rpath,"${LINK_DIR}" \
    -lm -o "${BUILD_DIR}/campp_ort_benchmark"

{
    printf 'cc=%s\n' "${CC}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'ort_so=%s\n' "${ORT_SO}"
} > "${BUILD_DIR}/campp_ort_benchmark.build.txt"

echo "빌드 완료: ${BUILD_DIR}/campp_ort_benchmark"
