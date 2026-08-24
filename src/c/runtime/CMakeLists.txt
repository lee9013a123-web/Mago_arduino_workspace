# CAM++ C Runtime의 build target을 정의할 파일이다.
#
# 예정 target:
# - campp_runtime: 로더, 실행기, 메모리 관리 정적 library
# - campp_cpu_reference: Scalar Reference kernel library
# - campp_compiled_model_inspect: compiled bundle 검증·출력 도구
# - runtime unit tests: 선택적 test target
#
# 초기에는 C11과 cpu_reference만 활성화한다.
# AArch64 NEON과 진단 기능은 CMake option으로 선택 가능하게 추가한다.
