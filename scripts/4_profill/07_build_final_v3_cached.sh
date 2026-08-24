#!/usr/bin/env bash

# ############################################################################
# 경고: 이 빌드로 만든 바이너리를 latency 측정이나 승격에 쓰지 말 것.
#
# 05_build_final_v3.sh와 같은 소스/플래그를 쓰지만 컴파일 방식이 달라 성능이
# 다르다.  2026-08-20 같은 세션 실측 (V2 대비 V3 개선율):
#
#     bucket 98 : 원본 6.54%  vs  이 캐시 빌드 1.56%
#     bucket 298: 원본 5.78%  vs  이 캐시 빌드 0.44%
#
# 원인은 -flto와의 상호작용이다.  원본은 모든 소스를 한 번의 gcc 호출로 넘겨
# 전 프로그램 문맥에서 LTO가 돌지만, 여기서는 -c로 TU마다 -O3 최적화를 끝낸 뒤
# LTO로 합친다.  그 과정에서 인라이닝 결정이 달라지고 layer-hybrid dispatch처럼
# 작은 함수가 hot loop에서 반복 호출되는 구조가 특히 손해를 본다.
#
# 쓸 수 있는 곳: 수치는 정확하다.  4개 bucket 전부 V2와 bitwise 동일함을
# 확인했으므로 **bitwise / 정확도 검증 전용**으로는 유효하다.  그 용도라면
# 182초로 원본 261초보다 빠르다.
#
# latency를 재야 하면 반드시 05_build_final_v3.sh를 쓸 것.
# ############################################################################
#
# 05_build_final_v3.sh와 같은 바이너리를 만들되 오브젝트 캐시를 쓴다.
#
# 01_build_profiler.sh는 타겟마다 runtime+candidate 전체를 다시 컴파일한다.
# layer-hybrid 테이블(static const 배열)만 바꾸는 반복 작업에서는 그 재컴파일이
# 통째로 낭비다.  이 스크립트는 (source, flag set)별로 .o를 캐시해 두고 바뀐
# 파일만 다시 컴파일한 뒤 링크한다.
#
# 사용: bash scripts/4_profill/07_build_final_v3_cached.sh
#       FORCE_REBUILD=1 을 주면 캐시를 무시한다.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build/profill/final_v3_hybrid}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/build/profill/.objcache_v3}"
CC="${CC:-gcc}"
STRIP="${STRIP:-strip}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# 05_build_final_v3.sh와 동일해야 한다.  달라지면 V2 비교가 오염된다.
CFLAGS="-std=c11 -DNDEBUG -Wall -Wextra -O3 -mcpu=cortex-a53 -flto"
CFLAGS="${CFLAGS} -finline-functions -finline-small-functions -funroll-loops"
CFLAGS="${CFLAGS} -fomit-frame-pointer -ffunction-sections -fdata-sections"
LDFLAGS="-flto -Wl,--gc-sections"

# 01_build_profiler.sh의 변수 정의부(소스/인클루드 목록)를 그대로 재사용한다.
# 첫 컴파일 호출 앞의 마지막 배열 종료 지점에서 끊는다.  if 블록 중간에서 자르면
# 구문이 깨지므로 컴파일 라인을 찾은 뒤 그 앞의 "^)" 까지만 가져온다.
PROFILER_SH="${ROOT}/scripts/4_profill/01_build_profiler.sh"
FIRST_CC="$(grep -n '"${CC}"' "${PROFILER_SH}" | head -1 | cut -d: -f1)"
PRELUDE_END="$(awk -v limit="${FIRST_CC}" \
    'NR < limit && /^\)/ { line = NR } END { print line }' "${PROFILER_SH}")"
if [[ -z "${PRELUDE_END}" || "${PRELUDE_END}" -lt 50 ]]; then
    echo "prelude 경계를 찾지 못했다 (FIRST_CC=${FIRST_CC})" >&2
    exit 1
fi
PRELUDE="$(mktemp)"
trap 'rm -f "${PRELUDE}"' EXIT
sed -n "1,${PRELUDE_END}p" \
    "${ROOT}/scripts/4_profill/01_build_profiler.sh" > "${PRELUDE}"
# 원본은 BASH_SOURCE로 ROOT를 잡으므로 여기서 고정해준다.
sed -i "s|^ROOT=\"\$(cd .*|ROOT=\"${ROOT}\"|" "${PRELUDE}"
# shellcheck disable=SC1090
BUILD_SCOPE="compiler_matrix" FINAL_SUITE_VARIANT="v3" \
    CFLAGS="${CFLAGS}" LDFLAGS="${LDFLAGS}" source "${PRELUDE}"

SHARED_SOURCES=(
    "${RUNTIME_SOURCES[@]}"
    "${FINAL_CANDIDATE_SOURCES[@]}"
    "${FINAL_SUITE_SOURCE}"
)
INCLUDES=(
    "${COMMON_INCLUDES[@]}" "${PROFILL_INCLUDES[@]}" "${FINAL_INCLUDES[@]}"
)

compiled=0
reused=0

# $1: flag set 이름  $2: 추가 -D  나머지: 컴파일할 소스
# 캐시 키는 flag set 이름 + 소스 경로다.  소스가 .o보다 새로우면 다시 만든다.
compile_set() {
    local tag="$1"; shift
    local extra="$1"; shift
    local dir="${CACHE_DIR}/${tag}"
    mkdir -p "${dir}"
    OBJECTS=()
    local src obj
    for src in "$@"; do
        obj="${dir}/$(printf '%s' "${src#"${ROOT}/"}" | tr '/.' '__').o"
        if [[ "${FORCE_REBUILD}" != "1" && -f "${obj}" && "${obj}" -nt "${src}" ]]; then
            reused=$((reused + 1))
        else
            # shellcheck disable=SC2086
            "${CC}" ${CFLAGS} ${extra} "${INCLUDES[@]}" -c "${src}" -o "${obj}"
            compiled=$((compiled + 1))
        fi
        OBJECTS+=("${obj}")
    done
}

link_target() {
    local out="$1"; shift
    local extra="$1"; shift
    # shellcheck disable=SC2086
    "${CC}" ${CFLAGS} ${extra} "${OBJECTS[@]}" ${LDFLAGS} -lm -o "${out}"
}

echo "!! 경고: 이 빌드는 latency 측정용이 아니다 (LTO 차이로 V3 이득 상실)"
echo "!!       latency는 05_build_final_v3.sh 로 빌드할 것"
echo "Final v3 cached build (gcc, object cache) -- bitwise 검증 전용"
echo "  build dir: ${BUILD_DIR}"
echo "  cache dir: ${CACHE_DIR}"
mkdir -p "${BUILD_DIR}"

FINAL_D="-DCAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1"
PROF_D="-DCAMPP_ENABLE_OPERATOR_PROFILING=1 -DCAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1"

# 1) final suite 만 켠 조합 -- benchmark_final 과 reference_dump_final 이 공유한다.
compile_set "final" "${FINAL_D}" "${SHARED_SOURCES[@]}"
SHARED_FINAL=("${OBJECTS[@]}")

compile_set "final" "${FINAL_D}" \
    "${ROOT}/src/c/runtime/command_line/campp_runtime_benchmark.c"
OBJECTS=("${SHARED_FINAL[@]}" "${OBJECTS[@]}")
link_target "${BUILD_DIR}/campp_runtime_benchmark_final" "${FINAL_D}"

compile_set "final" "${FINAL_D}" \
    "${ROOT}/tests/runtime/operator_replay_tests/campp_reference_dump.c"
OBJECTS=("${SHARED_FINAL[@]}" "${OBJECTS[@]}")
link_target "${BUILD_DIR}/campp_reference_dump_final" "${FINAL_D}"

# 2) profiling 까지 켠 조합
compile_set "profiling" "${PROF_D}" \
    "${SHARED_SOURCES[@]}" \
    "${ROOT}/src/c/profill/operator_profiler.c" \
    "${ROOT}/src/c/profill/runtime_fixture.c" \
    "${ROOT}/src/c/profill/command_line/campp_e7_profiler.c"
link_target "${BUILD_DIR}/campp_e7_profiler_final" "${PROF_D}"

# 3) 매크로 없는 조합 -- suite 테스트
compile_set "plain" "" \
    "${SHARED_SOURCES[@]}" \
    "${ROOT}/tests/runtime/profill/test_final_candidate_suite.c"
link_target "${BUILD_DIR}/test_final_candidate_suite" ""

cp "${BUILD_DIR}/campp_runtime_benchmark_final" \
    "${BUILD_DIR}/campp_runtime_benchmark_final.unstripped"
"${STRIP}" --strip-unneeded "${BUILD_DIR}/campp_runtime_benchmark_final"

{
    printf 'cc=%s\n' "${CC}"
    printf 'cflags=%s\n' "${CFLAGS}"
    printf 'ldflags=%s\n' "${LDFLAGS}"
    printf 'build_variant=03_layer_hybrid\n'
    printf 'build_scope=compiler_matrix\n'
    printf 'final_suite_variant=v3\n'
    printf 'strip_final=1\n'
    printf 'profiling_macro=CAMPP_ENABLE_OPERATOR_PROFILING=1\n'
    printf 'final_suite_macro=CAMPP_ENABLE_FINAL_CANDIDATE_SUITE=1\n'
    printf 'final_suite=%s\n' "${FINAL_SUITE_CONFIG}"
    printf 'object_cache=%s\n' "${CACHE_DIR}"
    printf 'latency_valid=0  # LTO 분리 컴파일로 성능이 다르다\n'
} > "${BUILD_DIR}/campp_runtime_benchmark_final.build.txt"
cp "${BUILD_DIR}/campp_runtime_benchmark_final.build.txt" \
    "${BUILD_DIR}/campp_e7_profiler_final.build.txt"

echo "  compiled: ${compiled} object(s), reused: ${reused}"
echo "Final v3 cached build complete: ${BUILD_DIR}"
