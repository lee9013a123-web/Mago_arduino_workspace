# C Runtime용 Tensor descriptor table을 만드는 모듈이다.
#
# 각 Tensor에 부여할 정보:
# - 고정 Tensor ID
# - dtype, rank, dimensions, contiguous stride
# - byte size와 storage 종류
# - input, output, constant, activation, view 구분
#
# Phase 3에서는 activation offset을 확정하지 않는다.
# Phase 4에서 arena_offset과 first_use/last_use 필드를 추가한다.
