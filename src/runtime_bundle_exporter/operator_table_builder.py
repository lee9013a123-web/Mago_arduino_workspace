# 정적 그래프의 약 1,438개 runtime node를 실행 명령 table로 만드는 모듈이다.
#
# 각 명령에 기록할 정보:
# - opcode
# - input/output Tensor ID
# - padding, stride, dilation, axis 등의 attribute
# - backend와 kernel 선택 정보가 들어갈 예약 필드
#
# C 함수를 node마다 생성하지 않는다.
# 동일 opcode의 모든 node가 하나의 공통 C kernel을 반복 호출하도록 table만 생성한다.
