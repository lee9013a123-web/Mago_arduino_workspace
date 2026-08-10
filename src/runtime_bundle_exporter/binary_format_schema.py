# Python exporter와 C Runtime이 공유할 binary 형식의 기준을 정의할 파일이다.
#
# 구현할 내용:
# - 파일 magic, format version, byte order
# - header, Tensor descriptor, Operator descriptor의 고정 크기 필드
# - weights section과 execution-plan section의 정렬 규칙
# - uint32/uint64 offset 범위와 오류 조건
#
# C 쪽 src/runtime/include/campp_runtime/model_binary_format.h와 반드시 같은 규격을 유지한다.
