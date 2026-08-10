# Reference Runtime의 end-to-end 테스트 파일이다.
#
# 1, 3, 5, 10초 bucket별 고정 feature를 넣고
# Dense block 경계, statistics pooling, 최종 embedding을 ORT와 비교한다.
# Operator 단위 테스트가 통과해도 전체 연결 순서와 Tensor ID가 틀릴 수 있으므로 별도로 유지한다.
