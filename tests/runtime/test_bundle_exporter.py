# Runtime bundle exporter의 구조 검증 테스트 파일이다.
#
# 확인할 내용:
# - 정적 그래프의 runtime node와 operator table 수 일치
# - 모든 input/output Tensor ID 존재
# - 네 bucket의 topology 동일성과 shape 차이
# - weights와 plan manifest checksum 일치
