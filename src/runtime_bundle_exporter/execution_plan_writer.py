# bucket별 Tensor table과 Operator table을 plan_*.bin으로 저장하는 모듈이다.
#
# 출력:
# - plan_98.bin
# - plan_298.bin
# - plan_498.bin
# - plan_998.bin
#
# 모든 정수 필드는 binary_format_schema.py 규칙에 따라 명시적으로 직렬화한다.
# Python 객체나 C 구조체의 메모리 표현을 그대로 파일에 쓰지 않는다.
